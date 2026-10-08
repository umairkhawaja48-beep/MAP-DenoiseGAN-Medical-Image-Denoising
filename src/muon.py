"""
Muon optimiser (MomentUm Orthogonalised by Newton-Schulz), after Jordan et al.
(2024). Orthogonalises the momentum update for matrix-shaped parameters via a
quintic Newton-Schulz iteration, which equalises the update's singular values.

Applied here to test whether it stabilises adversarial training on an
ultra-small dataset, where Adam-trained MAP-DenoiseGAN produced reconstructions
worse than its noisy inputs.

Per standard practice, only parameters with 2 or more dimensions are handled by
Muon; biases and 1-D normalisation parameters fall back to Adam, since
orthogonalisation is not meaningful for them. Convolutional weights of shape
(out, in, kh, kw) are reshaped to (out, in*kh*kw) for the orthogonalisation.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    """Approximate the orthogonal polar factor of G by a quintic Newton-Schulz
    iteration. Coefficients from the reference implementation; run in float32
    here rather than bfloat16 for numerical headroom on this hardware."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for >=2-D parameters, with an internal Adam fallback for the rest.

    Args mirror the reference implementation: `lr` is the Muon learning rate,
    `momentum` the heavy-ball coefficient, `nesterov` whether to use the
    look-ahead form, `ns_steps` the Newton-Schulz iteration count. Parameters
    with ndim < 2 are routed to a standard Adam step using `adam_lr`/`betas`.
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5,
                 adam_lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adam_lr=adam_lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nesterov, ns_steps = group["nesterov"], group["ns_steps"]
            adam_lr, (b1, b2) = group["adam_lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if p.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    upd = g.add(buf, alpha=mom) if nesterov else buf

                    orig_shape = upd.shape
                    mat = upd.reshape(orig_shape[0], -1)
                    mat = zeropower_via_newtonschulz5(mat, steps=ns_steps)
                    # scale so the update norm is comparable across shapes
                    scale = max(1.0, mat.size(0) / mat.size(1)) ** 0.5
                    upd = (mat * scale).reshape(orig_shape).to(p.dtype)

                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr)
                else:
                    # Adam fallback for biases / 1-D norm parameters
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mhat = m / (1 - b1 ** t)
                    vhat = v / (1 - b2 ** t)
                    p.addcdiv_(mhat, vhat.sqrt().add_(eps), value=-adam_lr)

        return loss
