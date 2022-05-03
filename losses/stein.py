import torch

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).mean([-1,-2,-3])
    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            for k in range(x.shape[3]):
                fxi = fx[:, i, j, k]
                dfxi_dxi = keep_grad(fxi.sum(), x)[:, i, j, k][:, None]
                vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.mean(dim=1)


def stein_stats(logp, x, critic, approx_jcb=True, n_samples=1):


    lp = logp
    sq = keep_grad(lp.sum(), x)

    fx = critic(x)
    sq_fx = (sq * fx).mean([-1,-2,-3])

    if approx_jcb==False:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(n_samples)], dim=1).mean(
            dim=1)

    stats = sq_fx + tr_dfdx
    norms = (fx * fx).mean([-1,-2,-3])
    grad_norms = (sq * sq).mean([-1,-2,-3])
    return stats, norms, grad_norms, lp


def stein_stats_withscore(score, x,y, critic, approx_jcb=True, n_samples=1):

    sq = score

    fx = critic(x,y)
    sq_fx = (sq * fx).mean([-1,-2,-3])

    if approx_jcb==False:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(n_samples)], dim=1).mean(
            dim=1)

    stats = sq_fx + tr_dfdx
    norms = (fx * fx).mean([-1,-2,-3])
    return stats, norms

def annealed_ssc(basescore,resscore, samples,sigmas, labels=None, lam=1.0, anneal_power=2.0, hook=None, n_particles=1):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    dup_samples = perturbed_samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1,*samples.shape[1:])
    dup_labels = labels.unsqueeze(0).expand(n_particles, *labels.shape).contiguous().view(-1)
    dup_samples.requires_grad_(True)
    
    # use Rademacher
#     if approx_jcb==False:
#         tr_dfdx = exact_jacobian_trace(fx, samples)
#     else:
#         tr_dfdx = torch.cat([approx_jacobian_trace(fx, samples)[:, None] for _ in range(n_samples)], dim=1).mean(
#             dim=1)
    # calc trace J, use Rademacher
    vectors = torch.randn_like(dup_samples)
    grad1 = resscore(dup_samples, dup_labels)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    grad1 = grad1.view(dup_samples.shape[0], -1)
    
    sq = basescore(dup_samples,dup_labels).detach().view(dup_samples.shape[0], -1)
    sq_fx = torch.mean(sq*grad1, dim=-1)
    norm2 = torch.mean(grad1 * grad1, dim=-1)
    tr_dfdx = torch.mean((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)
    
    
    sq_fx = sq_fx.view(n_particles, -1).mean(dim=0)
    norm2 = norm2.view(n_particles, -1).mean(dim=0)
    tr_dfdx = tr_dfdx.view(n_particles, -1).mean(dim=0)

    loss = (-1.0*(sq_fx + tr_dfdx) + 0.5*norm2/lam)/lam
    loss = loss * (used_sigmas.squeeze() ** 2)
    
    return loss.mean(dim=0)
