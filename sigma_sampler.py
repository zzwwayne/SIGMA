"""SIGMA: a standalone sigmoid-gated sampler for diffusion language models.

Requires PyTorch only. Call ``sigma_decoding`` from an existing denoising loop
with logits, a boolean mask, and token IDs for the current block::

    from sigma_sampler import sigma_decoding

    proposed_ids, update_mask = sigma_decoding(
        current_logits=logits,       # [batch, block_length, vocabulary]
        current_mask_index=masked,   # [batch, block_length], torch.bool
        current_x=token_ids,         # [batch, block_length], torch.long
        parameter={"lam": 0.12, "rho": 0.6},
    )
    token_ids[update_mask] = proposed_ids[update_mask]

The caller supplies model logits and controls model-specific logit alignment,
block boundaries, stopping criteria, and random seeds. All tensors must be on
the same device. The sampler does not load a model or tokenizer.

``sigma`` exposes the underlying temperature/unmasking-probability solver.
Its risk constraint applies to the expected plan over a renormalized top-k
support. Realized samples, and the forced update used to ensure progress,
are not guaranteed to satisfy that bound on every step.

"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["sigma", "sigma_decoding"]


@torch.no_grad()
def sigma(
    logits: torch.Tensor,
    mask_idx: torch.Tensor,
    lam: float,
    tau_min: float,
    tau_max: float,
    rho: float,
    current_x: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    n_tau: int = 8,
    max_dual_iter: int = 50,
    mu_expand_iter: int = 50,
    tol: float = 1e-6,
    normalize_entropy: bool = False,
    top: int = 512,
) -> Dict[str, torch.Tensor]:
    """Solve per-position temperatures and unmasking probabilities.

    Args:
        logits: Floating-point model logits of shape [B, L, V].
        mask_idx: Mask of shape [B, L]; True marks eligible positions.
        lam: Positive entropy regularization weight.
        tau_min: Positive lower bound of the temperature grid.
        tau_max: Upper bound of the temperature grid.
        rho: Nonnegative expected risk budget per sequence and decoding step.
        current_x: Unused; retained for compatibility with the original solver.
        alpha: Reward coefficient for unmasking a position.
        n_tau: Number of log-spaced candidate temperatures.
        max_dual_iter: Maximum bisection iterations for the dual variable.
        mu_expand_iter: Maximum expansions of the dual search upper bound.
        tol: Absolute tolerance for the risk budget search.
        normalize_entropy: Divide entropy by log(top_k) when True.
        top: Maximum vocabulary support size for the approximate solver.

    For each masked position i, solve:

        tau_i^*(mu) = argmax_tau [lam * H_i(tau) - mu * r_i(tau)]

        s_i^*(mu) = sigmoid(
            alpha / lam + H_i(tau_i^*) - (mu / lam) * r_i(tau_i^*)
        )

    Then search mu so that:

        sum_i s_i r_i <= rho

    Approximation:
        For each position, only top-k logits are used.
        The distribution p_i is the softmax over top-k logits.
        q_i(tau) is also computed over the same top-k support.

    Returns:
        {
            "s":    [B, L], decode probability, zero outside mask_idx
            "tau":  [B, L], selected temperature, zero outside mask_idx
            "tau_logits": [B, L, V], tau-adjusted logits, original logits on non-mask
            "mu":   [B], dual variable per sequence
            "risk": [B], achieved expected risk per sequence
            "H":    [B, L], selected entropy, zero outside mask_idx
            "r":    [B, L], selected risk, zero outside mask_idx
        }
    """

    if logits.ndim != 3:
        raise ValueError(f"`logits` must have shape [B, L, V], got {tuple(logits.shape)}.")

    if mask_idx.shape != logits.shape[:2]:
        raise ValueError(
            f"`mask_idx` must have shape [B, L], got {tuple(mask_idx.shape)}, "
            f"expected {tuple(logits.shape[:2])}."
        )

    if lam <= 0:
        raise ValueError("`lam` must be positive because the sigmoid solution divides by `lam`.")

    if tau_min <= 0:
        raise ValueError("`tau_min` must be positive.")

    if tau_max < tau_min:
        raise ValueError("`tau_max` must be >= `tau_min`.")

    if n_tau <= 0:
        raise ValueError("`n_tau` must be positive.")

    if rho < 0:
        raise ValueError("`rho` must be non-negative.")

    B, L, V = logits.shape
    if int(top) <= 0 or V == 0:
        raise ValueError("`top` and the vocabulary size must be positive.")
    device = logits.device

    mask_idx = mask_idx.bool()
    top_k = min(int(top), V)

    # Use float32 for numerical stability even if logits are fp16/bf16.
    out_dtype = torch.float32

    best_s = torch.zeros((B, L), device=device, dtype=out_dtype)
    best_tau = torch.zeros((B, L), device=device, dtype=out_dtype)
    best_H = torch.zeros((B, L), device=device, dtype=out_dtype)
    best_r = torch.zeros((B, L), device=device, dtype=out_dtype)

    best_mu = torch.zeros((B,), device=device, dtype=out_dtype)
    best_R = torch.zeros((B,), device=device, dtype=out_dtype)
    tau_logits = logits.float().clone()

    # Temperature grid.
    if n_tau == 1:
        taus = torch.tensor([tau_min], device=device, dtype=out_dtype)
    else:
        # Log-space grid is usually more stable because small tau is sensitive.
        log_tau_min = math.log(tau_min)
        log_tau_max = math.log(tau_max)
        taus = torch.exp(
            torch.linspace(
                log_tau_min,
                log_tau_max,
                steps=n_tau,
                device=device,
                dtype=out_dtype,
            )
        )

    # Entropy normalization denominator.
    if normalize_entropy:
        if top_k > 1:
            entropy_denom = math.log(top_k)
        else:
            entropy_denom = 1.0
    else:
        entropy_denom = 1.0

    for b in range(B):
        pos = torch.nonzero(mask_idx[b], as_tuple=False).squeeze(-1)  # [M]
        M = pos.numel()

        if M == 0:
            continue

        # [M, V] -> [M, top_k]
        logits_b = logits[b, pos, :]
        top_logits = torch.topk(logits_b, k=top_k, dim=-1).values.float()  # [M, top_k]

        # p_i over top-k support.
        log_p = F.log_softmax(top_logits, dim=-1)  # [M, top_k]
        p = log_p.exp()                            # [M, top_k]

        # Compute q_i(tau), H_i(tau), r_i(tau) for all taus.
        #
        # log_q: [M, n_tau, top_k]
        log_q = F.log_softmax(
            log_p[:, None, :] / taus[None, :, None],
            dim=-1,
        )
        q = log_q.exp()

        # H: [M, n_tau]
        H = -(q * log_q).sum(dim=-1)
        H = H / entropy_denom

        # risk: [M, n_tau]
        # r_i(tau) = 1 - sum_v p_i(v) q_i^tau(v)
        risk = 1.0 - (p[:, None, :] * q).sum(dim=-1)
        risk = risk.clamp(min=0.0, max=1.0)

        arange_M = torch.arange(M, device=device)

        def solve_given_mu(mu_value: float):
            mu_t = torch.tensor(mu_value, device=device, dtype=out_dtype)

            # phi[i, k] = lam * H[i, k] - mu * risk[i, k]
            phi = lam * H - mu_t * risk  # [M, n_tau]

            best_k = torch.argmax(phi, dim=-1)  # [M]

            H_star = H[arange_M, best_k]        # [M]
            r_star = risk[arange_M, best_k]     # [M]
            tau_star = taus[best_k]             # [M]

            z = alpha / lam + H_star - (mu_t / lam) * r_star
            s_star = torch.sigmoid(z)

            total_risk = torch.sum(s_star * r_star)

            return total_risk, s_star, tau_star, H_star, r_star

        # First check whether the risk constraint is inactive.
        R0, s0, tau0, H0, r0 = solve_given_mu(0.0)

        if R0.item() <= rho + tol:
            chosen_mu = 0.0
            chosen_R = R0
            chosen_s = s0
            chosen_tau = tau0
            chosen_H = H0
            chosen_r = r0
        else:
            # Expand upper bound for mu until risk is feasible.
            mu_low = 0.0
            mu_high = 1.0

            R_high, s_high, tau_high, H_high, r_high = solve_given_mu(mu_high)

            for _ in range(mu_expand_iter):
                if R_high.item() <= rho + tol:
                    break

                mu_high *= 2.0
                R_high, s_high, tau_high, H_high, r_high = solve_given_mu(mu_high)

            # Keep the best feasible result found so far.
            chosen_mu = mu_high
            chosen_R = R_high
            chosen_s = s_high
            chosen_tau = tau_high
            chosen_H = H_high
            chosen_r = r_high

            # Bisection on mu.
            for _ in range(max_dual_iter):
                mu_mid = 0.5 * (mu_low + mu_high)

                R_mid, s_mid, tau_mid, H_mid, r_mid = solve_given_mu(mu_mid)

                if R_mid.item() > rho:
                    # Risk too high, increase mu.
                    mu_low = mu_mid
                else:
                    # Feasible. Try smaller mu.
                    mu_high = mu_mid

                    chosen_mu = mu_mid
                    chosen_R = R_mid
                    chosen_s = s_mid
                    chosen_tau = tau_mid
                    chosen_H = H_mid
                    chosen_r = r_mid

                if abs(R_mid.item() - rho) <= tol:
                    break

        best_s[b, pos] = chosen_s
        best_tau[b, pos] = chosen_tau
        best_H[b, pos] = chosen_H
        best_r[b, pos] = chosen_r
        tau_logits[b, pos, :] = logits[b, pos, :].float() / chosen_tau[:, None]

        best_mu[b] = float(chosen_mu)
        best_R[b] = chosen_R

    return {
        "s": best_s,
        "tau": best_tau,
        "tau_logits": tau_logits,
        "mu": best_mu,
        "risk": best_R,
        "H": best_H,
        "r": best_r,
    }


@torch.no_grad()
def sigma_decoding(
    current_logits: torch.Tensor,
    current_mask_index: torch.Tensor,
    current_x: torch.Tensor,
    parameter: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propose token IDs and select positions for one SIGMA decoding step.

    Args:
        current_logits: Logits for the current block, shape [B, L, V].
        current_mask_index: Boolean mask of eligible positions, shape [B, L].
        current_x: Current token IDs, shape [B, L]. Inputs are not modified.
        parameter: Optional dictionary with the original sampler defaults:
            lam=0.12, rho=0.6, tau_min=0.1, tau_max=0.9, alpha=1.0,
            n_tau=8, top=32. Only these keys affect this wrapper.

    Returns:
        (proposed_ids, update_mask), both of shape [B, L]. Unmasked token
        IDs are preserved, and update_mask selects only eligible positions.
        Each row with eligible positions selects at least one position.

    Positions are sampled independently using the solver probabilities,
    excluding probabilities <= 0.05. Token IDs are sampled from the top-k
    temperature-adjusted distribution. If a row selects no positions, its
    most confident sampled proposal is forced to update. Use PyTorch's RNG
    seed to reproduce stochastic choices. A selected position is only
    unmasked if the sampled token differs from the caller's mask token ID.
    """
    parameter = parameter or {}
    if current_x.shape != current_logits.shape[:2]:
        raise ValueError("`current_x` must match the [B, L] dimensions of the logits.")
    if current_mask_index.dtype != torch.bool:
        raise ValueError("`current_mask_index` must be a boolean tensor.")

    lam = float(parameter.get("lam", 0.12))
    rho = float(parameter.get("rho", 0.6))
    tau_min = float(parameter.get("tau_min", 0.1))
    tau_max = float(parameter.get("tau_max", 0.9))
    alpha = float(parameter.get("alpha", 1.0))
    n_tau = int(parameter.get("n_tau", 8))
    top = int(parameter.get("top", 32))
    decode_plan = sigma(
        logits=current_logits,
        mask_idx=current_mask_index,
        lam=lam,
        tau_min=tau_min,
        tau_max=tau_max,
        rho=rho,
        alpha=alpha,
        n_tau=n_tau,
        top=top,
    )

    decode_prob = decode_plan["s"]
    transfer_index = (
        (torch.rand_like(decode_prob) < decode_prob)
        & current_mask_index
        & (decode_prob > 0.05)
    )

    tau_logits = decode_plan["tau_logits"]
    top_k = min(top, tau_logits.shape[-1])
    top_tau_logits, top_tau_indices = torch.topk(tau_logits.float(), k=top_k, dim=-1)
    top_tau_probs = F.softmax(top_tau_logits, dim=-1)
    sampled_topk_index = torch.multinomial(
        top_tau_probs.view(-1, top_tau_probs.shape[-1]),
        num_samples=1,
    ).view(current_x.shape[0], current_x.shape[1])
    sampled_x0 = torch.gather(
        top_tau_indices,
        dim=-1,
        index=sampled_topk_index.unsqueeze(-1),
    ).squeeze(-1)
    x0 = torch.where(current_mask_index, sampled_x0, current_x)

    # Ensure progress for rows that still have masks.
    masked_any = current_mask_index.any(dim=1)
    none_selected = (~transfer_index.any(dim=1)) & masked_any
    if none_selected.any():
        p = F.softmax(tau_logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        force_conf = torch.where(current_mask_index, x0_p, torch.full_like(x0_p, -1.0))
        max_conf_indices = torch.argmax(force_conf, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | (force_mask & none_selected.unsqueeze(1))

    return x0, transfer_index
