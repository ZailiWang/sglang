import os
from typing import Optional

import torch
import torch.nn.functional as F


# mamba
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
):
    torch.ops.sgl_kernel.causal_conv1d_fwd(
        x,
        weight,
        bias_,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
):
    torch.ops.sgl_kernel.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias_,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )


def _causal_conv1d_single_seq_torch(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_initial_state: bool,
    silu_activation: bool,
):
    """Torch-native counterpart for one sequence of causal_conv1d_fwd_cpu.

    Args:
        x_seq: (dim, seqlen)
        weight: (dim, width)
        bias: (dim,) or None
        initial_state: (dim, width - 1) or None
        use_initial_state: whether to prepend initial_state, otherwise zero prefix
        silu_activation: apply SiLU on output

    Returns:
        out_seq: (dim, seqlen)
        final_state: (dim, width - 1)
    """
    dim, seqlen = x_seq.shape
    width = weight.size(-1)
    state_len = width - 1

    if state_len > 0:
        if use_initial_state and initial_state is not None:
            prefix = initial_state.to(dtype=x_seq.dtype, device=x_seq.device)
        else:
            prefix = torch.zeros(
                dim, state_len, dtype=x_seq.dtype, device=x_seq.device
            )
        x_cat = torch.cat([prefix, x_seq], dim=-1)
        final_state = x_cat[:, -state_len:].contiguous()
    else:
        x_cat = x_seq
        final_state = x_seq[:, :0].contiguous()

    out_seq = F.conv1d(
        x_cat.unsqueeze(0),
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=dim,
    ).squeeze(0)

    if silu_activation:
        out_seq = F.silu(out_seq)

    return out_seq, final_state


def causal_conv1d_fwd_cpu_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
):
    """Torch-native counterpart of torch.ops.sgl_kernel.causal_conv1d_fwd_cpu.

    Supports both fixed-length and varlen layouts used by SGLang CPU Mamba/GDN:
      - fixed length: x is (batch, dim, seqlen)
      - varlen: x is (dim, total_seqlen), query_start_loc is (batch + 1)

    When conv_states is provided, it is updated in-place.
    """
    if query_start_loc is not None:
        # varlen path: x is transposed as (dim, total_tokens)
        dim, _ = x.shape
        batch = int(query_start_loc.numel() - 1)
        out = torch.empty_like(x)

        for i in range(batch):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            seq = x[:, start:end]

            cache_idx = int(cache_indices[i].item()) if cache_indices is not None else i
            if cache_idx == pad_slot_id:
                if seq.numel() > 0:
                    out[:, start:end] = 0
                continue

            use_init = (
                bool(has_initial_state[i].item())
                if has_initial_state is not None
                else conv_states is not None
            )
            init_state = (
                conv_states[cache_idx] if (conv_states is not None and use_init) else None
            )

            out_seq, final_state = _causal_conv1d_single_seq_torch(
                seq, weight, bias, init_state, use_init, silu_activation
            )
            out[:, start:end] = out_seq

            if conv_states is not None:
                conv_states[cache_idx].copy_(final_state)

        return out

    # fixed-length path: x is transposed as (batch, dim, seqlen)
    batch, _, _ = x.shape
    out = torch.empty_like(x)

    for i in range(batch):
        cache_idx = int(cache_indices[i].item()) if cache_indices is not None else i
        if cache_idx == pad_slot_id:
            out[i].zero_()
            continue

        use_init = (
            bool(has_initial_state[i].item())
            if has_initial_state is not None
            else conv_states is not None
        )
        init_state = (
            conv_states[cache_idx] if (conv_states is not None and use_init) else None
        )

        out_seq, final_state = _causal_conv1d_single_seq_torch(
            x[i], weight, bias, init_state, use_init, silu_activation
        )
        out[i].copy_(out_seq)

        if conv_states is not None:
            conv_states[cache_idx].copy_(final_state)

    return out


def causal_conv1d_fn_cpu(
    mixed_qkv_transposed,
    conv_weights,
    bias,
    activation,
    conv_states,
    has_initial_state,
    cache_indices,
    query_start_loc,
    seq_lens_cpu,
):
    # Default to torch-native fallback path. Set
    # SGLANG_FORCE_NATIVE_CAUSAL_CONV1D_FWD_CPU=0 to force the kernel path.
    use_native = os.environ.get("SGLANG_FORCE_NATIVE_CAUSAL_CONV1D_FWD_CPU", "1") != "0"
    silu_activation = activation == "silu"

    if use_native:
        return causal_conv1d_fwd_cpu_torch(
            mixed_qkv_transposed,
            conv_weights,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            silu_activation,
            -1,
        )

    return torch.ops.sgl_kernel.causal_conv1d_fwd_cpu(
        mixed_qkv_transposed,
        conv_weights,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        -1,
        True,
    )


def causal_conv1d_update_cpu(
    mixed_qkv, conv_states, conv_weights, bias, activation, conv_state_indices
):
    return torch.ops.sgl_kernel.causal_conv1d_update_cpu(
        mixed_qkv.contiguous(),
        conv_states,
        conv_weights,
        bias,
        activation == "silu",
        None,
        conv_state_indices,
        -1,
        True,
    )


def chunk_gated_delta_rule_cpu(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    head_first,
    use_qk_l2norm_in_kernel,
    initial_state_indices,
):
    core_attn_out = torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        True,  # output_final_state
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
        initial_state_indices,
    )
    last_recurrent_state = initial_state
    h = None  # Todo: add return h support
    return core_attn_out, last_recurrent_state, h
