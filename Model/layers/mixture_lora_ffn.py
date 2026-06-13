# -*- coding: utf-8 -*-

"""Step-aware Mixture-of-LoRAs SwiGLU for the recurrent depth core.

This is a drop-in replacement for :class:`Model.layers.swiglu.SwiGLU` that adds
*breadth* to the recurrent loop without paying the depth cost. A single shared
base SwiGLU is kept (the parameter-reuse dividend of recurrence), and ``K``
low-rank LoRA experts modulate the gate/up projection per token. A router
(Manifold-Hyper-Connection style: ``bias + alpha*tanh(proj(rms_norm(x)))``)
mixes the experts per token, optionally conditioned on the **recurrent step
index** -- the signal that only the RDT loop has, so the *same* token is routed
to *different* experts at *different* depths (Mixture-of-Recursions / Mixture-of-
LoRAs, arXiv:2507.10524 / 2512.12880), traversing the expert pool along the time
axis of the recurrence.

Cold-start bit-compatibility (the keystone property): with ``lora_b=0``,
``router_alpha=0`` and ``step_embed=0`` the expert delta is exactly zero and the
module reduces to the plain base SwiGLU. So ``use_mol=True`` is byte-identical to
``use_mol=False`` at init, and existing smoke configs are unaffected.

Two design invariants other code depends on -- do not break them without
re-checking the call sites:

* The base down projection is named ``w_down`` so that
  :meth:`RDTForCausalLM._scale_residual_projections` matches it via the
  ``ffn.w_down`` suffix and scales the residual write exactly as for a plain
  SwiGLU. **Do not rename it.**
* LoRA modulates ``w_in`` (upstream of the nonlinearity), NOT ``w_down``. It is
  stored as ``nn.Parameter`` fused tensors (not ``nn.Linear``), so the
  ``isinstance(module, nn.Linear)`` filter in ``_scale_residual_projections``
  never touches it and no LoRA-awareness is needed there. **Do not convert the
  experts to nn.Linear / move them onto w_down** without revisiting residual
  scaling.

``step_embed``/``router_bias``/``router_alpha`` are plain ``nn.Parameter`` (not
``nn.Embedding``) precisely so the model-level ``apply(_init_weights)`` -- which
re-inits every ``nn.Linear``/``nn.Embedding`` -- leaves their zero init intact.

Load balancing (v2). The router exposes the Switch-Transformer / GShard load-
balancing auxiliary loss (arXiv:2101.03961 / 2006.16668)::

    aux = n_experts * sum_i f_i * P_i

where ``f_i`` is the fraction of tokens that *select* expert ``i`` (argmax for
dense routing, top-k membership when ``top_k>0``) and ``P_i`` is the mean router
probability of expert ``i``. ``aux`` is minimised (=1) by perfectly uniform
*population* load and grows toward ``n_experts`` as routing collapses onto one
expert, so adding it to the loss pushes the population usage toward uniform and
prevents dead experts. An optional router z-loss (the mean squared ``logsumexp``
of the router logits) keeps the logits from drifting large.

Scope, precisely (do not over-read this term). ``aux`` is a *population*
balancer: it acts on the token-mean ``P_i`` and is blind to whether the per-token
distributions are sharp or flat. It cannot distinguish (A) every token drawing
the same uniform mix from (B) each token routing sharply to a different expert
with balanced totals -- both score ``aux=1``. So with ``top_k>0`` (discrete
selection) it correctly keeps experts alive without harming the specialisation
the top-k mask already enforces; but with the **dense default** (``top_k=0``) it
does *not* by itself break the averaged-expert degeneracy -- flattening every
token is the cheapest way to flatten ``P_i``, so a strong aux can actually push
dense routing *toward* the uniform per-token mix. Dense breadth therefore relies
on the LoRA experts / step conditioning to differentiate, not on this aux; use
``top_k>0`` if you want the balancer to drive specialisation. To fight the dense
averaged-expert collapse directly you would add a per-token sharpness reward, not
this (population-symmetric) term.

Padding is excluded from these statistics. When an ``attn_mask`` is threaded in
(the recurrent blocks pass the model's ``[B, L]`` mask), masked positions are
dropped from both ``f_i`` and ``P_i`` so a heavily padded batch (SFT / eval) does
not let near-identical pad hidden states all argmax onto one expert and bias the
balancer -- the same valid-token denominator convention the ACT ponder cost and
eval losses already use. With no mask (the bare-FFN path) every position counts,
matching the original behaviour.

These statistics are computed inside :meth:`forward` and *accumulated* on the
module across the recurrent loop's repeated calls (the recurrent FFN is weight-
shared, so one instance is invoked once per step). The accumulators hold live
autograd tensors so the aux/z gradient flows to the router parameters; this is
grad-checkpoint safe under ``use_reentrant=False`` (the recompute reconciles the
cached tensor without double-counting). The model resets the accumulators at the
start of every :meth:`RDTForCausalLM.forward` and reads them back after the core
runs (see ``RDTForCausalLM._mol_router_losses``). They carry zero training
signal on the main logits path -- with ``lora_b=0`` (cold start) the FFN output
is still bit-identical to a plain SwiGLU; only the extra loss term differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.rmsnorm import RMSNorm


class MixtureLoRAFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        n_experts: int,
        rank: int,
        step_table: int,
        *,
        step_aware: bool = True,
        router_temp: float = 1.0,
        top_k: int = 0,
        init_std: float = 0.02,
        rmsnorm_eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if step_table <= 0:
            raise ValueError("step_table must be positive")
        if router_temp <= 0:
            raise ValueError("router_temp must be positive")
        if not (0 <= top_k <= n_experts):
            raise ValueError("top_k must be in [0, n_experts]")

        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.n_experts = n_experts
        self.rank = rank
        self.step_table = step_table
        self.step_aware = step_aware
        self.router_temp = float(router_temp)
        self.top_k = top_k
        self.init_std = init_std

        # Shared base SwiGLU -- identical layout/names to ``SwiGLU`` so residual
        # scaling and weight loading stay compatible.
        self.w_in = nn.Linear(d_model, 2 * ffn_hidden, bias=bias)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=bias)

        # K LoRA experts on the gate/up projection (w_in), fused for a batched
        # GEMM (no Python loop over experts).
        self.lora_a = nn.Parameter(torch.empty(n_experts, d_model, rank))
        self.lora_b = nn.Parameter(torch.empty(n_experts, rank, 2 * ffn_hidden))

        # Per-token router (causal: no cross-position mixing).
        self.router_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.router_proj = nn.Linear(d_model, n_experts, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(n_experts))
        self.router_alpha = nn.Parameter(torch.zeros(()))

        # Recurrent step embedding: the RDT-intrinsic conditioning signal.
        if step_aware:
            self.step_embed = nn.Parameter(torch.zeros(step_table, d_model))
        else:
            self.register_parameter("step_embed", None)

        # Load-balancing accumulators. Plain (non-parameter) attributes holding
        # live autograd tensors, summed across the recurrent loop's repeated
        # calls and read by the model after the core runs. ``None`` means "no
        # routing happened since the last reset" -> the model contributes 0.
        # collect_router_stats is toggled off during cache-free generate /
        # inference so the eval path pays nothing.
        self.collect_router_stats = True
        self._aux_loss: torch.Tensor | None = None
        self._z_loss: torch.Tensor | None = None
        self._router_calls: int = 0

        self.reset_parameters()

    def reset_router_stats(self) -> None:
        """Clear the per-forward load-balancing accumulators. Called by the model
        at the start of each forward so the aux/z statistics never leak across
        forward calls (and so a recompute under grad checkpointing starts clean).
        """

        self._aux_loss = None
        self._z_loss = None
        self._router_calls = 0

    def reset_parameters(self) -> None:
        # LoRA-A nonzero (standard), LoRA-B zero -> expert delta is exactly 0 at
        # init regardless of routing, the cold-start guarantee. router_alpha and
        # step_embed stay zero (constructed as zeros) so the router has no effect
        # at init either. w_in/w_down/router_proj are left for the model-level
        # apply(_init_weights) to initialize like any nn.Linear.
        nn.init.normal_(self.lora_a, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.lora_b)
        # Exempt the routing scalars/table from weight decay. router_bias (1-D)
        # and router_alpha (0-D) are auto-exempt by the optimizer's ndim<=1 rule,
        # but step_embed (2-D) needs the explicit marker; tag all three for
        # clarity. LoRA-A/B intentionally keep weight decay (B starts at 0).
        self.router_bias._no_weight_decay = True
        self.router_alpha._no_weight_decay = True
        if self.step_embed is not None:
            self.step_embed._no_weight_decay = True

    @staticmethod
    def _topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
        # Keep the top-k logits per token, set the rest to -inf so softmax zeroes
        # them. Ties at the boundary keep all tied entries (>= threshold).
        threshold = logits.topk(k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        step: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")

        # Base pre-activation.
        h_in = self.w_in(x)  # [B, L, 2F]

        # Router logits (per token, optionally step-conditioned).
        ref = self.router_norm(x)
        if self.step_aware and step is not None:
            idx = min(max(int(step), 0), self.step_table - 1)
            ref = ref + self.step_embed[idx]
        logits = self.router_bias + self.router_alpha * torch.tanh(self.router_proj(ref))
        logits = logits / self.router_temp
        # Pre-mask logits feed the router z-loss (the canonical ST-MoE form,
        # arXiv:2202.08906): top-k masking sets -inf, which would drop kept
        # experts out of the logsumexp.
        dense_logits = logits
        if 0 < self.top_k < self.n_experts:
            logits = self._topk_mask(logits, self.top_k)
        weights = torch.softmax(logits, dim=-1)  # [B, L, K]

        if self.collect_router_stats:
            self._accumulate_router_stats(weights, dense_logits, attn_mask)

        # Per-token mixed LoRA delta on the w_in pre-activation:
        # expert e contributes (x @ A_e) @ B_e, then mix by router weights.
        xa = torch.einsum("bld,edr->bler", x, self.lora_a)  # [B, L, K, r]
        delta = torch.einsum("bler,erf->blef", xa, self.lora_b)  # [B, L, K, 2F]
        delta = torch.einsum("ble,blef->blf", weights, delta)  # [B, L, 2F]

        gate, up = (h_in + delta).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)

    def _accumulate_router_stats(
        self,
        weights: torch.Tensor,
        dense_logits: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> None:
        """Fold this call's load-balancing statistics into the accumulators.

        ``weights`` is the post-softmax routing distribution ``[B, L, K]`` (already
        top-k-masked when ``top_k>0``); ``dense_logits`` is the pre-mask router
        logits ``[B, L, K]`` for the z-loss. ``attn_mask`` is the optional ``[B, L]``
        valid-token mask: when given, padded positions are dropped from both ``f_i``
        and ``P_i`` (and the z-loss) so a heavily padded batch does not let identical
        pad hidden states all argmax onto one expert and bias the balancer. With no
        mask every position counts (the original Switch/GShard behaviour).
        """

        flat = weights.reshape(-1, self.n_experts)
        dense_flat = dense_logits.reshape(-1, self.n_experts)

        if attn_mask is not None:
            keep = attn_mask.reshape(-1).to(torch.bool)
            if keep.shape[0] != flat.shape[0]:
                raise ValueError("attn_mask must match the [B, L] of the routing input")
            flat = flat[keep]
            dense_flat = dense_flat[keep]

        n_tokens = flat.shape[0]
        if n_tokens == 0:
            return

        # P_i: mean router probability per expert.
        prob_mean = flat.mean(dim=0)  # [K]

        # f_i: fraction of tokens that *select* expert i. Dense routing selects
        # the argmax; top-k selects the k experts with nonzero weight. One-hot /
        # k-hot is detached -- the discrete assignment carries no gradient, the
        # aux gradient flows only through P_i (the standard Switch formulation).
        if 0 < self.top_k < self.n_experts:
            select = (flat > 0).to(flat.dtype)  # k-hot membership
            denom = float(self.top_k)
        else:
            top1 = flat.argmax(dim=-1)
            select = F.one_hot(top1, self.n_experts).to(flat.dtype)
            denom = 1.0
        # Normalise so sum_i f_i == 1 (k-hot rows each contribute k selections).
        load_frac = select.mean(dim=0) / denom  # [K], detached below

        aux = self.n_experts * torch.sum(load_frac.detach() * prob_mean)

        # z-loss: mean over (valid) tokens of logsumexp(logits)^2.
        z = torch.logsumexp(dense_flat, dim=-1)
        z_loss = torch.mean(z * z)

        if self._aux_loss is None:
            self._aux_loss = aux
            self._z_loss = z_loss
        else:
            self._aux_loss = self._aux_loss + aux
            self._z_loss = self._z_loss + z_loss
        self._router_calls += 1

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Mean aux / z load-balancing loss over the calls since the last reset.

        Returns ``(aux, z)`` averaged across the recurrent loop's repeated calls,
        or ``None`` if no routing happened (so the model contributes nothing).
        """

        if self._aux_loss is None or self._router_calls == 0:
            return None
        scale = 1.0 / self._router_calls
        return self._aux_loss * scale, self._z_loss * scale


def _check() -> None:
    torch.manual_seed(0)

    from Model.layers.swiglu import SwiGLU

    d, F_hidden, K, r, table = 512, 1536, 4, 8, 4
    x = torch.randn(2, 16, d)

    base = SwiGLU(d, F_hidden)
    mol = MixtureLoRAFFN(d, F_hidden, K, r, table)
    # Copy base weights so the only difference is the (zero) LoRA path.
    mol.w_in.load_state_dict(base.w_in.state_dict())
    mol.w_down.load_state_dict(base.w_down.state_dict())

    # Cold-start: lora_b=0, router_alpha=0, step_embed=0 -> identical to base.
    with torch.no_grad():
        mol.router_alpha.fill_(2.0)  # even with a hot router...
        mol.step_embed.normal_()  # ...and a random step table...
        # ...lora_b is still zero, so the delta is exactly zero.
        y_mol = mol(x, step=1)
        y_base = base(x)
    max_diff = (y_mol - y_base).abs().max().item()

    print("MixtureLoRAFFN")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y_mol.shape)}")
    print(f"  cold-start max|mol-base| (lora_b=0): {max_diff:.2e}")
    print(f"  params: {sum(p.numel() for p in mol.parameters()):,}")

    # Step-aware routing actually changes the mix across steps.
    with torch.no_grad():
        mol.lora_b.normal_(std=0.02)
        w0 = torch.softmax(
            mol.router_bias
            + mol.router_alpha * torch.tanh(mol.router_proj(mol.router_norm(x) + mol.step_embed[0])),
            dim=-1,
        )
        w1 = torch.softmax(
            mol.router_bias
            + mol.router_alpha * torch.tanh(mol.router_proj(mol.router_norm(x) + mol.step_embed[1])),
            dim=-1,
        )
    print(f"  step0 vs step1 router differ: {not torch.allclose(w0, w1)}")

    x2 = torch.randn(2, 16, d, requires_grad=True)
    mol(x2, step=2).sum().backward()
    print(f"  grad_norm: {x2.grad.norm().item():.6f}")
    print(f"  lora_a.grad finite: {bool(torch.isfinite(mol.lora_a.grad).all())}")


if __name__ == "__main__":
    _check()
