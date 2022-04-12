import torch

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = keep_grad(fxi.sum(), x)[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.sum(dim=1)


def stein_stats(logp, x, critic, approx_jcb=True, n_samples=1):


    lp = logp
    sq = keep_grad(lp.sum(), x)

    fx = critic(x)
    sq_fx = (sq * fx).sum(-1)

    if approx_jcb==False:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(n_samples)], dim=1).mean(
            dim=1)

    stats = sq_fx + tr_dfdx
    norms = (fx * fx).sum(1)
    grad_norms = (sq * sq).view(x.size(0), -1).sum(1)
    return stats, norms, grad_norms, lp
