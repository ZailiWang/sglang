# SPDX-License-Identifier: Apache-2.0
"""Torch-native CPU fallbacks for Mamba Triton kernels.

These implementations prioritize correctness and API compatibility on CPU. They
intentionally use straightforward PyTorch recurrences instead of mirroring the
chunked GPU decomposition.
"""

import math

import torch
import torch.nn.functional as F


PAD_SLOT_ID = -1


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _repeat_grouped(x: torch.Tensor, nheads: int) -> torch.Tensor:
    """Expand a (..., ngroups, dstate) tensor over heads."""
    ngroups = x.shape[-2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    return x.repeat_interleave(nheads // ngroups, dim=-2)


def selective_state_update_native(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
    disable_state_update=False,
    intermediate_states_buffer=None,
    cache_steps=None,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
    enable_stochastic_rounding=False,
    cache_philox_rounds=0,
):
    """CPU implementation of ``selective_state_update`` using torch ops.

    The public Triton wrapper mutates ``state`` and writes the sequence output to
    ``out``. This fallback preserves the same side effects for CPU tensors. It
    does not emulate stochastic rounding; CPU stores use PyTorch's regular dtype
    conversion.
    """
    if cache_philox_rounds < 0:
        raise ValueError("cache_philox_rounds must be non-negative.")
    if enable_stochastic_rounding and state.dtype != torch.float16:
        raise ValueError(
            "Stochastic rounding for the Mamba SSM cache requires state dtype "
            f"torch.float16, got {state.dtype}."
        )
    if out is None:
        out = torch.empty_like(x)

    if state.dim() == 3:
        state_v = state.unsqueeze(1)
    else:
        state_v = state
    if x.dim() == 2:
        x_v = x.unsqueeze(1).unsqueeze(1)
    elif x.dim() == 3:
        x_v = x.unsqueeze(1)
    else:
        x_v = x
    if dt.dim() == 2:
        dt_v = dt.unsqueeze(1).unsqueeze(1)
    elif dt.dim() == 3:
        dt_v = dt.unsqueeze(1)
    else:
        dt_v = dt
    if A.dim() == 2:
        A_v = A.unsqueeze(0)
    else:
        A_v = A
    if B.dim() == 2:
        B_v = B.unsqueeze(1).unsqueeze(1)
    elif B.dim() == 3:
        B_v = B.unsqueeze(1)
    else:
        B_v = B
    if C.dim() == 2:
        C_v = C.unsqueeze(1).unsqueeze(1)
    elif C.dim() == 3:
        C_v = C.unsqueeze(1)
    else:
        C_v = C
    if D is not None and D.dim() == 1:
        D_v = D.unsqueeze(0)
    else:
        D_v = D
    if z is not None:
        if z.dim() == 2:
            z_v = z.unsqueeze(1).unsqueeze(1)
        elif z.dim() == 3:
            z_v = z.unsqueeze(1)
        else:
            z_v = z
    else:
        z_v = None
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias_v = dt_bias.unsqueeze(0)
    else:
        dt_bias_v = dt_bias
    if out.dim() == 2:
        out_v = out.unsqueeze(1).unsqueeze(1)
    elif out.dim() == 3:
        out_v = out.unsqueeze(1)
    else:
        out_v = out

    _, nheads, dim, dstate = state_v.shape
    batch, T, _, _ = x_v.shape
    assert x_v.shape == (batch, T, nheads, dim)
    assert dt_v.shape == x_v.shape
    assert A_v.shape == (nheads, dim, dstate)
    ngroups = B_v.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B_v.shape == (batch, T, ngroups, dstate)
    assert C_v.shape == B_v.shape
    if D_v is not None:
        assert D_v.shape == (nheads, dim)
    if z_v is not None:
        assert z_v.shape == x_v.shape
    if dt_bias_v is not None:
        assert dt_bias_v.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    assert out_v.shape == x_v.shape

    nheads_per_group = nheads // ngroups
    for b in range(batch):
        if state_batch_indices is not None:
            state_idx = _as_int(state_batch_indices[b])
        else:
            state_idx = b
        valid_state = state_idx != pad_slot_id
        cache_idx = -1
        if intermediate_states_buffer is not None:
            if intermediate_state_indices is not None:
                cache_idx = _as_int(intermediate_state_indices[b])
            elif state_batch_indices is not None:
                cache_idx = state_idx
            else:
                cache_idx = b

        for h in range(nheads):
            group = h // nheads_per_group
            if valid_state:
                state_cur = state_v[state_idx, h].to(torch.float32)
            else:
                state_cur = torch.zeros(
                    (dim, dstate), device=state.device, dtype=torch.float32
                )

            for t in range(T):
                if (
                    retrieve_parent_token is not None
                    and t != 0
                    and cache_idx >= 0
                    and intermediate_states_buffer is not None
                ):
                    parent_step_idx = _as_int(retrieve_parent_token[b, t])
                    if 0 <= parent_step_idx < T:
                        state_cur = intermediate_states_buffer[
                            cache_idx, parent_step_idx, h
                        ].to(torch.float32)

                dt_t = dt_v[b, t, h].to(torch.float32)
                if dt_bias_v is not None:
                    dt_t = dt_t + dt_bias_v[h].to(torch.float32)
                if dt_softplus:
                    dt_t = F.softplus(dt_t)

                x_t = x_v[b, t, h].to(torch.float32)
                A_h = A_v[h].to(torch.float32)
                B_t = B_v[b, t, group].to(torch.float32)
                C_t = C_v[b, t, group].to(torch.float32)

                state_cur = state_cur * torch.exp(A_h * dt_t[:, None]) + (
                    B_t[None, :] * dt_t[:, None] * x_t[:, None]
                )

                if (
                    intermediate_states_buffer is not None
                    and state_batch_indices is not None
                    and valid_state
                    and cache_idx >= 0
                ):
                    intermediate_states_buffer[cache_idx, t, h].copy_(
                        state_cur.to(intermediate_states_buffer.dtype)
                    )

                out_t = torch.sum(state_cur * C_t[None, :], dim=-1)
                if D_v is not None:
                    out_t = out_t + x_t * D_v[h].to(torch.float32)
                if z_v is not None:
                    out_t = out_t * F.silu(z_v[b, t, h].to(torch.float32))
                out_v[b, t, h].copy_(out_t.to(out_v.dtype))

            if valid_state and not disable_state_update:
                state_v[state_idx, h].copy_(state_cur.to(state_v.dtype))

    return None


def _prepare_dt(dt, A, chunk_size, dt_bias, dt_softplus, dt_limit):
    if dt_bias is not None:
        dt = dt + dt_bias.view(1, 1, -1)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)
    padded = nchunks * chunk_size - seqlen
    if padded:
        dt_pad = F.pad(dt, (0, 0, 0, padded))
    else:
        dt_pad = dt
    dt_chunks = dt_pad.permute(0, 2, 1).reshape(batch, nheads, nchunks, chunk_size)
    dA_cumsum = torch.cumsum(
        dt_chunks.to(torch.float32) * A.view(1, -1, 1, 1), dim=-1
    )
    return dA_cumsum, dt_chunks


def _scan_segment(x, dt, A, B, C, D, z, start, end, state, out):
    nheads = x.shape[1]
    ngroups = B.shape[1]
    for t in range(start, end):
        dt_t = dt[t].to(torch.float32)
        x_t = x[t].to(torch.float32)
        B_t = _repeat_grouped(B[t], nheads).to(torch.float32)
        C_t = _repeat_grouped(C[t], nheads).to(torch.float32)
        state.mul_(torch.exp(A.to(torch.float32)[:, None, None] * dt_t[:, None, None]))
        state.add_(B_t[:, None, :] * x_t[:, :, None] * dt_t[:, None, None])
        out_t = torch.einsum("hpn,hn->hp", state, C_t)
        if D is not None:
            if D.dim() == 1:
                out_t = out_t + x_t * D.to(torch.float32)[:, None]
            else:
                out_t = out_t + x_t * D.to(torch.float32)
        if z is not None:
            out_t = out_t * F.silu(z[t].to(torch.float32))
        out[t].copy_(out_t.to(out.dtype))
    return state


def _state_at_segment_end(x, dt, A, B, start, end, init_state):
    state = init_state.to(torch.float32).clone()
    nheads = x.shape[1]
    for t in range(start, end):
        dt_t = dt[t].to(torch.float32)
        x_t = x[t].to(torch.float32)
        B_t = _repeat_grouped(B[t], nheads).to(torch.float32)
        state.mul_(torch.exp(A.to(torch.float32)[:, None, None] * dt_t[:, None, None]))
        state.add_(B_t[:, None, :] * x_t[:, :, None] * dt_t[:, None, None])
    return state


def mamba_chunk_scan_combined_native(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    out=None,
    return_final_states=False,
    return_varlen_states=False,
    return_intermediate_states=False,
    state_dtype=None,
    return_track_states=False,
    track_seq_idx=None,
    track_end_locs=None,
):
    """CPU implementation of ``mamba_chunk_scan_combined`` using torch ops."""
    assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, (
        "chunk_size must be integer power of 2"
    )
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)

    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert cu_seqlens is not None, (
            "cu_seqlens must be provided if return_varlen_states is True"
        )

    _, dt_chunks = _prepare_dt(dt, A, chunk_size, dt_bias, dt_softplus, dt_limit)
    dt_proc = dt_chunks.reshape(batch, nheads, -1)[:, :, :seqlen].permute(0, 2, 1)

    nchunks = math.ceil(seqlen / chunk_size)
    states_dtype = state_dtype if state_dtype is not None else C.dtype
    states = torch.empty(
        (batch, nchunks, nheads, headdim, dstate),
        device=x.device,
        dtype=states_dtype,
    )
    final_states = torch.empty(
        (batch, nheads, headdim, dstate), device=x.device, dtype=torch.float32
    )

    if cu_seqlens is None:
        if initial_states is not None:
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        for b in range(batch):
            state = (
                initial_states[b].to(torch.float32).clone()
                if initial_states is not None
                else torch.zeros(
                    (nheads, headdim, dstate), device=x.device, dtype=torch.float32
                )
            )
            for ck in range(nchunks):
                states[b, ck].copy_(state.to(states.dtype))
                start = ck * chunk_size
                end = min(seqlen, start + chunk_size)
                state = _scan_segment(
                    x[b],
                    dt_proc[b],
                    A,
                    B[b],
                    C[b],
                    D,
                    z[b] if z is not None else None,
                    start,
                    end,
                    state,
                    out[b],
                )
            final_states[b].copy_(state)
        varlen_states = None
        track_states = None
    else:
        assert batch == 1, (
            "passing cu_seqlens to get the varlen states is only supported if "
            "batch dimension is 1"
        )
        num_sequences = cu_seqlens.numel() - 1
        if initial_states is not None:
            assert initial_states.shape == (num_sequences, nheads, headdim, dstate)
        states.zero_()
        varlen_states = torch.empty(
            (num_sequences, nheads, headdim, dstate),
            device=x.device,
            dtype=torch.float32,
        )
        for seq in range(num_sequences):
            start = _as_int(cu_seqlens[seq])
            end = _as_int(cu_seqlens[seq + 1])
            state = (
                initial_states[seq].to(torch.float32).clone()
                if initial_states is not None
                else torch.zeros(
                    (nheads, headdim, dstate), device=x.device, dtype=torch.float32
                )
            )
            t = start
            while t < end:
                ck = t // chunk_size
                if t % chunk_size == 0:
                    states[0, ck].copy_(state.to(states.dtype))
                next_boundary = min(end, (ck + 1) * chunk_size)
                state = _scan_segment(
                    x[0],
                    dt_proc[0],
                    A,
                    B[0],
                    C[0],
                    D,
                    z[0] if z is not None else None,
                    t,
                    next_boundary,
                    state,
                    out[0],
                )
                t = next_boundary
            varlen_states[seq].copy_(state)
        if num_sequences > 0:
            final_states[0].copy_(varlen_states[-1])
        else:
            final_states.zero_()
        track_states = None
        if (
            track_seq_idx is not None
            and track_end_locs is not None
            and track_end_locs.numel() > 0
        ):
            tracked = []
            for seq_tensor, end_tensor in zip(track_seq_idx, track_end_locs):
                seq = _as_int(seq_tensor)
                start = _as_int(cu_seqlens[seq])
                end = _as_int(end_tensor)
                init = (
                    initial_states[seq]
                    if initial_states is not None
                    else torch.zeros(
                        (nheads, headdim, dstate), device=x.device, dtype=torch.float32
                    )
                )
                tracked.append(
                    _state_at_segment_end(x[0], dt_proc[0], A, B[0], start, end, init)
                )
            track_states = torch.stack(tracked, dim=0).unsqueeze(0)

    if return_track_states:
        assert return_varlen_states, (
            "return_track_states requires return_varlen_states (cu_seqlens mode)"
        )
        return states, varlen_states, track_states
    if return_intermediate_states:
        if return_varlen_states:
            if return_final_states:
                return states, final_states, varlen_states
            return states, varlen_states
        if return_final_states:
            return states, final_states
        return states
    if not return_varlen_states:
        if not return_final_states:
            return None
        return final_states
    if return_final_states:
        return final_states, varlen_states
    return varlen_states
