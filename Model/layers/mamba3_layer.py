# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.rmsnorm import GroupedRMSNorm


try:
    from mamba_ssm.modules.mamba3 import Mamba3 as OfficialMamba3
except ImportError:
    OfficialMamba3 = None


def official_available() -> bool:
    return OfficialMamba3 is not None


def _init_dt_bias(
    nfeatures: int,
    dt_min: float,
    dt_max: float,
    dt_init_floor: float,
) -> torch.Tensor:
    """Upstream Mamba-3 dt_bias initialization.

    Samples ``dt`` uniformly in log space between ``dt_min`` and ``dt_max``,
    floors it, and converts to the additive bias ``dt + log(-expm1(-dt))``
    so that ``softplus(dt_bias) ≈ dt`` at init.
    """

    log_lo = math.log(dt_min)
    log_hi = math.log(dt_max)
    dt = torch.exp(torch.rand(nfeatures) * (log_hi - log_lo) + log_lo)
    dt = torch.clamp(dt, min=dt_init_floor)
    return dt + torch.log(-torch.expm1(-dt))


class NaiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 64,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if expand <= 0:
            raise ValueError("expand must be positive")
        if headdim <= 0:
            raise ValueError("headdim must be positive")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.headdim = headdim
        self.d_conv = d_conv

        if self.d_inner % headdim != 0:
            raise ValueError("d_model * expand must be divisible by headdim")

        self.nheads = self.d_inner // headdim

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.nheads + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.nheads, self.d_inner, bias=True)

        with torch.no_grad():
            self.dt_proj.bias.copy_(
                _init_dt_bias(self.d_inner, dt_min, dt_max, dt_init_floor)
            )
        self.dt_proj.bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.A_log = nn.Parameter(torch.log(torch.rand(self.nheads, d_state) + 0.5))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

        bsz, seq_len, _ = x.shape
        dtype = x.dtype
        incremental = state is not None or return_state

        mask = None
        if attn_mask is not None:
            if attn_mask.shape != (bsz, seq_len):
                raise ValueError("attn_mask must have shape [B, L]")
            if incremental:
                if not bool(attn_mask.bool().all()):
                    raise ValueError(
                        "incremental NaiveSSM requires an all-ones attn_mask"
                    )
            else:
                mask = attn_mask.to(device=x.device, dtype=torch.float32)

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        conv_buf: torch.Tensor | None = None
        ssm_state: torch.Tensor | None = None
        if state is not None:
            conv_buf, ssm_state = state

        u_t = u.transpose(1, 2)
        if incremental:
            if conv_buf is None:
                conv_buf = u_t.new_zeros(bsz, self.d_inner, self.d_conv - 1)
            # exact match with padding=d_conv-1 + right truncation on a fresh buffer
            u_cat = torch.cat([conv_buf, u_t], dim=-1)
            u = F.conv1d(
                u_cat,
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.d_inner,
            ).transpose(1, 2)
            new_conv_buf = u_cat[..., u_cat.shape[-1] - (self.d_conv - 1):]
        else:
            u = self.conv1d(u_t)[..., :seq_len].transpose(1, 2)
            new_conv_buf = None
        u = F.silu(u)

        if mask is not None:
            u = u * mask.unsqueeze(-1).to(dtype=u.dtype)
            z = z * mask.unsqueeze(-1).to(dtype=z.dtype)

        params = self.x_proj(u)
        dt, b_param, c_param = params.split(
            [self.nheads, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))

        u_h = u.reshape(bsz, seq_len, self.nheads, self.headdim)
        dt_h = dt.reshape(bsz, seq_len, self.nheads, self.headdim)

        a = -torch.exp(self.A_log.float())

        if ssm_state is not None:
            state_t = ssm_state.to(device=x.device, dtype=torch.float32)
        else:
            state_t = torch.zeros(
                bsz,
                self.nheads,
                self.headdim,
                self.d_state,
                device=x.device,
                dtype=torch.float32,
            )

        ys: list[torch.Tensor] = []

        for t in range(seq_len):
            dt_t = dt_h[:, t].float()
            u_t = u_h[:, t].float()
            b_t = b_param[:, t].float().view(bsz, 1, 1, self.d_state)
            c_t = c_param[:, t].float().view(bsz, 1, 1, self.d_state)

            da = torch.exp(dt_t.unsqueeze(-1) * a.view(1, self.nheads, 1, self.d_state))
            bu = dt_t.unsqueeze(-1) * b_t * u_t.unsqueeze(-1)

            next_state = state_t * da + bu
            if mask is not None:
                active = mask[:, t].view(bsz, 1, 1, 1).bool()
                state_t = torch.where(active, next_state, state_t)
            else:
                state_t = next_state

            y_t = (state_t * c_t).sum(dim=-1)
            y_t = y_t + self.D.view(1, self.nheads, 1) * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y.reshape(bsz, seq_len, self.d_inner).to(dtype)

        if mask is not None:
            y = y * mask.unsqueeze(-1).to(dtype=y.dtype)

        y = y * F.silu(z)
        out = self.out_proj(y)

        if return_state:
            assert new_conv_buf is not None
            return out, (new_conv_buf, state_t)
        return out


class Mamba3Layer(nn.Module):
    def __init__(
        self,
        cfg,
        layer_idx: int | None = None,
        num_groups: int = 8,
    ):
        super().__init__()

        self.cfg = cfg
        self.layer_idx = layer_idx

        self.norm = GroupedRMSNorm(
            cfg.d_model,
            num_groups=num_groups,
            eps=cfg.rmsnorm_eps,
        )

        use_official = bool(cfg.use_official_mamba and OfficialMamba3 is not None)

        if cfg.use_official_mamba and OfficialMamba3 is None:
            raise RuntimeError(
                "use_official_mamba=True but mamba_ssm is not importable. "
                "Install with `pip install mamba-ssm` (requires CUDA) or set "
                "cfg.use_official_mamba=False to use the NaiveSSM fallback."
            )

        if use_official:
            self.mamba = self._build_official(cfg, layer_idx)
            self.backend = "official"
        else:
            self.mamba = NaiveSSM(
                d_model=cfg.d_model,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim,
                d_conv=cfg.mamba_d_conv,
                dt_min=cfg.mamba_dt_min,
                dt_max=cfg.mamba_dt_max,
                dt_init_floor=cfg.mamba_dt_init_floor,
            )
            self.backend = "fallback"

    def _build_official(self, cfg, layer_idx: int | None):
        """Build the upstream Mamba-3 module with cfg-aligned kwargs.

        Upstream `Mamba3.__init__` does NOT accept ``d_conv`` (it bakes the
        causal short conv into the fused kernel). We forward the parameters
        it does accept; mismatches will surface as a hard ``TypeError`` so
        we never silently degrade.
        """

        return OfficialMamba3(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            chunk_size=cfg.mamba_chunk_size,
            dt_min=cfg.mamba_dt_min,
            dt_max=cfg.mamba_dt_max,
            dt_init_floor=cfg.mamba_dt_init_floor,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache=None,
        **kwargs,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")

        if cache is not None:
            if attn_mask is not None and not bool(attn_mask.bool().all()):
                raise ValueError(
                    "cached decoding requires an all-ones attention mask"
                )
            residual = x
            mamba_input = self.norm(residual)
            return residual + self._forward_cached(mamba_input, cache)

        mask = None
        if attn_mask is not None:
            if attn_mask.shape != x.shape[:2]:
                raise ValueError("attn_mask must have shape [B, L]")
            if (
                not isinstance(self.mamba, NaiveSSM)
                and not bool(attn_mask.all())
                and not _is_right_padding_mask(attn_mask)
            ):
                raise ValueError(
                    "official Mamba backend only supports all-ones or right-padded "
                    "attention masks; use fallback Mamba for left/interior padding"
                )
            mask = attn_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

        residual = x if mask is None else x * mask
        mamba_input = self.norm(residual)

        if isinstance(self.mamba, NaiveSSM):
            y = self.mamba(mamba_input, attn_mask=attn_mask)
        else:
            y = self.mamba(mamba_input)
            if mask is not None:
                y = y * mask

        out = residual + y
        if mask is not None:
            out = out * mask
        return out

    def _forward_cached(self, mamba_input: torch.Tensor, cache) -> torch.Tensor:
        """One incremental chunk through the SSM, persisting state in `cache`."""

        if isinstance(self.mamba, NaiveSSM):
            y, cache.state = self.mamba(
                mamba_input,
                state=cache.state,
                return_state=True,
            )
            return y

        # Official backend: reuse upstream InferenceParams machinery. One
        # InferenceParams per cache slot, so the same shared module can hold
        # independent state per recurrent step.
        if self.layer_idx is None:
            raise RuntimeError(
                "cached decoding with the official Mamba backend requires layer_idx"
            )

        ip = cache.inference_params
        if ip is None:
            try:
                from mamba_ssm.utils.generation import InferenceParams
            except ImportError as exc:
                raise RuntimeError(
                    "cached decoding with the official Mamba backend requires "
                    "mamba_ssm.utils.generation.InferenceParams"
                ) from exc
            ip = InferenceParams(
                max_seqlen=self.cfg.max_seq_len,
                max_batch_size=mamba_input.shape[0],
            )
            cache.inference_params = ip

        ip.seqlen_offset = cache.seqlen_offset
        if cache.seqlen_offset > 0:
            # Upstream Mamba3.forward routes the 3-D [B, L, D] tensor into
            # step(), which expects [B, D] and crashes in its rearrange.
            # Call step() directly; it updates the cached states in place.
            states = self.mamba._get_states_from_cache(ip, mamba_input.shape[0])
            outs = []
            for t in range(mamba_input.shape[1]):
                y_t, *_ = self.mamba.step(mamba_input[:, t], *states)
                outs.append(y_t)
            y = torch.stack(outs, dim=1)
        else:
            y = self.mamba(mamba_input, inference_params=ip)
        cache.seqlen_offset += mamba_input.shape[1]
        return y


def _is_right_padding_mask(attn_mask: torch.Tensor) -> bool:
    if attn_mask.ndim != 2:
        return False
    mask = attn_mask.to(dtype=torch.long)
    if mask.numel() == 0:
        return True
    if not bool(((mask == 0) | (mask == 1)).all()):
        return False
    return bool((mask[:, 1:] <= mask[:, :-1]).all())


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    layer = Mamba3Layer(cfg, layer_idx=0)
    layer.eval()

    print("Mamba3Layer")
    print(f"  backend: {layer.backend}")
    print(f"  official_available: {official_available()}")

    bsz, seq_len = 2, 16
    x = torch.randn(bsz, seq_len, cfg.d_model)

    y = layer(x)

    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in layer.parameters()):,}")

    x2 = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    loss = layer(x2).sum()
    loss.backward()

    print(f"  grad_norm: {x2.grad.norm().item():.6f}")

    x3 = torch.randn(1, 8, cfg.d_model)
    out_full = layer(x3)

    x3_mod = x3.clone()
    x3_mod[0, 5:] += 10.0
    out_mod = layer(x3_mod)

    diff = (out_full[0, :5] - out_mod[0, :5]).abs().max().item()
    print(f"  causal_diff_before_changed_span: {diff:.6e}")

    ignored = layer(
        x,
        word_pos=torch.zeros(bsz, seq_len, dtype=torch.long),
        morph_depth=torch.zeros(bsz, seq_len, dtype=torch.long),
        attn_mask=torch.ones(bsz, seq_len, dtype=torch.long),
    )

    print(f"  kwargs_ignored_shape: {tuple(ignored.shape)}")


if __name__ == "__main__":
    _check()
