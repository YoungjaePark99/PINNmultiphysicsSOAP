import sys
import io
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import time
import os
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PI = math.pi


class TeeLogger:
    def __init__(self):
        self._orig = sys.stdout
        self._buf = io.StringIO()
        sys.stdout = self

    def write(self, msg):
        self._orig.write(msg)
        self._buf.write(msg)

    def flush(self):
        self._orig.flush()
        self._buf.flush()

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._buf.getvalue())

    def close(self):
        sys.stdout = self._orig
        self._buf.close()


def parse_args():
    p = argparse.ArgumentParser(description='1D Thermoelasticity PINN')
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--E-modulus', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--N-domain', type=int, default=200)
    p.add_argument('--n-hidden', type=int, default=4)
    p.add_argument('--n-neurons', type=int, default=64)
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='adam')
    p.add_argument('--weighting', choices=['none', 'gradnorm'], default='none')
    p.add_argument('--gn-update-freq', type=int, default=1000)
    p.add_argument('--gn-momentum', type=float, default=0.9)
    p.add_argument('--single-network', action='store_true',
                   help='Use single shared MLP instead of independent networks')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--checkpoint-freq', type=int, default=5000)
    return p.parse_args()


class SubNet(nn.Module):
    def __init__(self, n_hidden, n_neurons):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(2.0 * x - 1.0)


class SingleNet(nn.Module):
    def __init__(self, n_hidden, n_neurons, n_out=2):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(2.0 * x - 1.0)


def exact_T(x):
    return torch.sin(PI * x) + 1.0

def exact_u(x):
    return torch.sin(2 * PI * x)

def compute_sources(x, args):
    sx = torch.sin(PI * x)
    cx = torch.cos(PI * x)
    s2x = torch.sin(2 * PI * x)
    f_T = args.kappa * PI**2 * sx
    f_u = args.E_modulus * (2*PI)**2 * s2x + args.gamma * PI * cx
    return f_T.detach(), f_u.detach()


def compute_residuals(nets, x, src, args):
    T = nets['T'](x)
    u = nets['u'](x)
    ones = torch.ones_like(T)

    T_x = torch.autograd.grad(T, x, ones, create_graph=True, retain_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, ones, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True, retain_graph=True)[0]

    f_T, f_u = src
    R_T = -args.kappa * T_xx - f_T
    R_u = -args.E_modulus * u_xx + args.gamma * T_x - f_u

    return {
        'RT': torch.mean(R_T**2),
        'Ru': torch.mean(R_u**2),
    }


def compute_bc_loss(nets, args, device):
    x0 = torch.zeros(1, 1, device=device)
    x1 = torch.ones(1, 1, device=device)
    bc_T = sum((nets['T'](xb) - exact_T(xb))**2 for xb in [x0, x1]).squeeze()
    bc_u = sum((nets['u'](xb) - exact_u(xb))**2 for xb in [x0, x1]).squeeze()
    return {'BC_T': bc_T, 'BC_u': bc_u}


def get_grad_vec(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs)


def train(args):
    logger = TeeLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"κ={args.kappa}, E={args.E_modulus}, γ={args.gamma}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, epochs={args.epochs}")
    print(f"Network: {args.n_hidden}×{args.n_neurons}, Weighting: {args.weighting}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    net_names = ['T', 'u']
    if args.single_network:
        shared = SingleNet(args.n_hidden, args.n_neurons, n_out=2).to(device)
        class _View:
            def __init__(self, net, idx): self.net, self.idx = net, idx
            def __call__(self, x): return self.net(x)[:, self.idx:self.idx+1]
        nets = {'T': _View(shared, 0), 'u': _View(shared, 1)}
        all_params = list(shared.parameters())
        print(f'  SingleNet: {sum(p.numel() for p in all_params):,} params')
    else:
        nets = {n: SubNet(args.n_hidden, args.n_neurons).to(device) for n in net_names}
        all_params = []
        for n in net_names:
            all_params.extend(list(nets[n].parameters()))
    total_params = sum(p.numel() for p in all_params)
    print(f"Networks: T(1), u(1) — {total_params:,} params total")

    if args.optimizer == 'soap':
        try:
            from soap import SOAP
            optimizer = SOAP(all_params, lr=args.lr, betas=(0.99, 0.999),
                             precondition_frequency=2, precondition_1d=False,
                             weight_decay=0.0)
        except ImportError:
            print("SOAP not found, using Adam")
            optimizer = torch.optim.Adam(all_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(all_params, lr=args.lr)

    def _save_state():
        if args.single_network:
            return {'_model': copy.deepcopy(shared.state_dict())}
        return {n: copy.deepcopy(nets[n].state_dict()) for n in net_names}

    def _load_state(state):
        if args.single_network:
            shared.load_state_dict(state['_model'])
        else:
            for n in net_names:
                nets[n].load_state_dict(state[n])

    def _save_checkpoint(path, epoch, optimizer):
        torch.save({
            'epoch': epoch, 'model': _save_state(), 'optimizer': optimizer.state_dict(),
            'best_state': best_state, 'best_L2_avg': best_L2_avg, 'best_epoch': best_epoch,
            'best_pde_state': best_pde_state, 'best_pde_total': best_pde_total, 'best_pde_epoch': best_pde_epoch,
            'gn_weights': gn_weights,
            'weight_hist': weight_hist, 'loss_hist': loss_hist, 'L2_history': L2_history,
        }, path)

    x_int = torch.linspace(0.01, 0.99, args.N_domain, device=device).reshape(-1, 1)
    x_int.requires_grad_(True)
    src = compute_sources(x_int, args)
    print(f"Collocation: {x_int.shape[0]} interior points")

    PDE_NAMES = ['RT', 'Ru']
    BC_NAMES = ['BC_T', 'BC_u']
    ALL_LOSS_NAMES = PDE_NAMES + BC_NAMES

    loss_hist = {k: [] for k in ALL_LOSS_NAMES + ['total']}
    L2_history = []
    best_L2_avg = float('inf')
    best_epoch = 0
    best_state = None
    best_pde_total = float('inf')
    best_pde_epoch = 0
    best_pde_state = None
    gn_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    weight_hist = {n: [] for n in ALL_LOSS_NAMES}
    backward_times = []
    eval_times = []

    x_eval_mon = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    with torch.no_grad():
        T_e_mon = exact_T(x_eval_mon)
        u_e_mon = exact_u(x_eval_mon)

    t0 = time.time()
    LOG_EVERY = 500
    EVAL_EVERY = args.eval_every

    tag = f"thermo_g{args.gamma:.4g}_s{args.seed}_{args.weighting}_{args.optimizer}"
    if args.single_network:
        tag += '_sn'
    run_dir = os.path.join("runs", f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _load_state(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_state, best_L2_avg, best_epoch = ckpt['best_state'], ckpt['best_L2_avg'], ckpt['best_epoch']
        best_pde_state, best_pde_total, best_pde_epoch = ckpt['best_pde_state'], ckpt['best_pde_total'], ckpt['best_pde_epoch']
        gn_weights = ckpt['gn_weights']
        weight_hist, loss_hist = ckpt['weight_hist'], ckpt['loss_hist']
        L2_history = ckpt['L2_history']
        print(f"  Resumed at epoch {start_epoch}, best L2={best_L2_avg:.4e} @ ep{best_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()

        pde_losses = compute_residuals(nets, x_int, src, args)
        bc_losses = compute_bc_loss(nets, args, device)

        t_back = time.time()
        grads = {}
        for name in PDE_NAMES:
            grads[name] = get_grad_vec(pde_losses[name], all_params)
        for bc_name in BC_NAMES:
            grads[bc_name] = get_grad_vec(bc_losses[bc_name], all_params)

        if args.weighting == 'gradnorm':
            if epoch % args.gn_update_freq == 1 or args.gn_update_freq == 1:
                l2_norms = {n: grads[n].norm().item() for n in ALL_LOSS_NAMES}
                mean_norm = np.mean(list(l2_norms.values()))
                for name in ALL_LOSS_NAMES:
                    if l2_norms[name] > 1e-30:
                        gn_hat = mean_norm / l2_norms[name]
                    else:
                        gn_hat = 1.0
                    gn_weights[name] = (args.gn_momentum * gn_weights[name]
                                        + (1 - args.gn_momentum) * gn_hat)
            g_total = sum(gn_weights[n] * grads[n] for n in ALL_LOSS_NAMES)
            for name in ALL_LOSS_NAMES:
                weight_hist[name].append(gn_weights[name])
        else:
            g_total = sum(grads[n] for n in ALL_LOSS_NAMES)

        idx = 0
        for p in all_params:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].reshape(p.shape).clone()
            idx += numel
        backward_times.append(time.time() - t_back)
        optimizer.step()

        bc_sum = sum(bc_losses[n].item() for n in BC_NAMES)
        total_loss = sum(pde_losses[n].item() for n in PDE_NAMES) + bc_sum
        for n in PDE_NAMES:
            loss_hist[n].append(pde_losses[n].item())
        for n in BC_NAMES:
            loss_hist[n].append(bc_losses[n].item())
        loss_hist['total'].append(total_loss)

        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            _save_checkpoint(os.path.join(run_dir, f"ckpt_ep{epoch}.pt"), epoch, optimizer)

        if total_loss < best_pde_total:
            best_pde_total = total_loss
            best_pde_epoch = epoch
            best_pde_state = _save_state()

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            t_ev = time.time()
            with torch.no_grad():
                T_pred_mon = nets['T'](x_eval_mon)
                u_pred_mon = nets['u'](x_eval_mon)
                l2_T = (torch.norm(T_pred_mon - T_e_mon) / torch.norm(T_e_mon)).item()
                l2_u = (torch.norm(u_pred_mon - u_e_mon) / torch.norm(u_e_mon)).item()
                l2_avg = 0.5 * (l2_T + l2_u)
            eval_times.append(time.time() - t_ev)
            L2_history.append({'epoch': epoch, 'T': l2_T, 'u': l2_u, 'avg': l2_avg})

            if l2_avg < best_L2_avg:
                best_L2_avg = l2_avg
                best_epoch = epoch
                best_state = _save_state()

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60
            improved = ' ***' if epoch == best_epoch else ''
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"L2={l2_avg:.3e}{improved}")
            print(f"  RT={pde_losses['RT'].item():.1e}  Ru={pde_losses['Ru'].item():.1e}  "
                  f"BC={bc_sum:.1e}  [{' '.join(f'{n}:{bc_losses[n].item():.1e}' for n in BC_NAMES)}]")
            print(f"  best={best_L2_avg:.3e}@{best_epoch}  "
                  f"(T={l2_T:.1e} u={l2_u:.1e})")

    _load_state(best_state)
    x_eval = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    with torch.no_grad():
        T_pred = nets['T'](x_eval)
        u_pred = nets['u'](x_eval)
        T_ex = exact_T(x_eval)
        u_ex = exact_u(x_eval)
        l2_T = torch.norm(T_pred - T_ex) / torch.norm(T_ex)
        l2_u = torch.norm(u_pred - u_ex) / torch.norm(u_ex)
        l2_avg = 0.5 * (l2_T.item() + l2_u.item())

    if best_pde_state is not None:
        _load_state(best_pde_state)
        with torch.no_grad():
            T_pred_pde = nets['T'](x_eval)
            u_pred_pde = nets['u'](x_eval)
            l2_T_pde = torch.norm(T_pred_pde - T_ex) / torch.norm(T_ex)
            l2_u_pde = torch.norm(u_pred_pde - u_ex) / torch.norm(u_ex)
            l2_avg_pde = 0.5 * (l2_T_pde.item() + l2_u_pde.item())
        _load_state(best_state)
    else:
        l2_T_pde, l2_u_pde, l2_avg_pde = l2_T, l2_u, l2_avg

    if args.checkpoint_freq > 0:
        _save_checkpoint(os.path.join(run_dir, "ckpt_final.pt"), args.epochs, optimizer)

    final_l2 = L2_history[-1]['avg']
    ratio = final_l2 / best_L2_avg if best_L2_avg > 0 else float('inf')
    total_time = (time.time() - t0) / 60.0

    print()
    print("=" * 70)
    print(f"RESULTS (γ={args.gamma}, {args.weighting}, seed={args.seed}, {args.optimizer})")
    print("=" * 70)
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    print(f"  L2_T   = {l2_T.item():.4e}")
    print(f"  L2_u   = {l2_u.item():.4e}")
    print(f"  L2_avg = {l2_avg:.4e}")
    print(f"  === Best-PDE-loss (epoch {best_pde_epoch}) [practical] ===")
    print(f"  L2_T   = {l2_T_pde.item():.4e}")
    print(f"  L2_u   = {l2_u_pde.item():.4e}")
    print(f"  L2_avg = {l2_avg_pde:.4e}  (PDE_total={best_pde_total:.4e})")
    print(f"  === Final epoch ({args.epochs}) ===")
    print(f"  L2_avg = {final_l2:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {ratio:.2f}x")
    print(f"  PDE-select/Best ratio = {l2_avg_pde / (l2_avg + 1e-30):.2f}x")
    print(f"  Total time: {total_time:.1f} min")

    results = {
        'L2_T_best': l2_T.item(), 'L2_u_best': l2_u.item(), 'L2_avg_best': l2_avg,
        'best_epoch': best_epoch,
        'L2_T_pde_best': l2_T_pde.item(), 'L2_u_pde_best': l2_u_pde.item(),
        'L2_avg_pde_best': l2_avg_pde, 'best_pde_epoch': best_pde_epoch,
        'best_pde_total': best_pde_total,
        'L2_avg_final': final_l2, 'ratio': ratio,
    }
    config = {
        'kappa': args.kappa, 'E_modulus': args.E_modulus, 'gamma': args.gamma,
        'optimizer': args.optimizer, 'weighting': args.weighting,
        'seed': args.seed, 'epochs': args.epochs, 'lr': args.lr,
        'N_domain': args.N_domain, 'n_hidden': args.n_hidden,
        'n_neurons': args.n_neurons, 'single_network': args.single_network,
    }
    if args.weighting == 'gradnorm':
        w_final = gn_weights
        results['final_weights'] = {n: w_final[n] for n in ALL_LOSS_NAMES}
        pde_w = [w_final[n] for n in PDE_NAMES]
        results['final_weight_ratio'] = max(pde_w) / (min(pde_w) + 1e-30)
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)

    SAVE_EVERY = 100
    n_total = len(loss_hist['total'])
    save_idx = list(range(0, n_total, SAVE_EVERY))
    if (n_total - 1) not in save_idx:
        save_idx.append(n_total - 1)

    curves = {'epochs': np.array([i + 1 for i in save_idx])}
    for k in ALL_LOSS_NAMES + ['total']:
        curves[k] = np.array([loss_hist[k][i] for i in save_idx])
    curves['L2_epochs'] = np.array([h['epoch'] for h in L2_history])
    for field in ['T', 'u', 'avg']:
        curves[f'L2_{field}'] = np.array([h[field] for h in L2_history])
    for name in ALL_LOSS_NAMES:
        if weight_hist[name]:
            curves[f'w_{name}'] = np.array([weight_hist[name][i] for i in save_idx])
    np.savez(os.path.join(run_dir, 'loss_curves.npz'), **curves)
    print(f"  Loss curves: {run_dir}/loss_curves.npz ({len(save_idx)} points)")

    if args.single_network:
        torch.save(best_state, os.path.join(run_dir, 'best_checkpoint.pt'))
        torch.save(best_pde_state, os.path.join(run_dir, 'pde_best_checkpoint.pt'))
    else:
        for n in net_names:
            torch.save(best_state[n], os.path.join(run_dir, f'best_{n}.pt'))
            torch.save(best_pde_state[n], os.path.join(run_dir, f'pde_best_{n}.pt'))

    try:
        fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
        fig_lc.suptitle(f'Thermo Loss Curves  |  γ={args.gamma}, {args.optimizer}+{args.weighting}, '
                        f'seed={args.seed}', fontsize=13, fontweight='bold')
        ep = curves['epochs']

        ax = axes_lc[0]
        for k in PDE_NAMES:
            ax.semilogy(ep, curves[k], label=k, alpha=0.8)
        ax.semilogy(ep, curves['total'], 'k-', lw=1.5, alpha=0.5, label='total')
        ax.set_title('PDE Residuals')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes_lc[1]
        for k in BC_NAMES:
            ax.semilogy(ep, curves[k], label=k, alpha=0.8)
        ax.set_title('BC Losses')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes_lc[2]
        l2_ep = curves['L2_epochs']
        for field, label in [('L2_T', 'T'), ('L2_u', 'u')]:
            ax.semilogy(l2_ep, curves[field], label=label, alpha=0.8)
        ax.semilogy(l2_ep, curves['L2_avg'], 'k-', lw=2, label='avg')
        ax.axhline(l2_avg, color='r', ls='--', lw=1, alpha=0.5,
                   label=f'oracle={l2_avg:.1e}')
        ax.set_title('L2 Error')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'loss_curves.png'), dpi=150)
        plt.close()
        print(f"  Loss plot: {run_dir}/loss_curves.png")
    except Exception as e:
        print(f"  Loss plot failed: {e}")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"1D Thermoelasticity  |  γ={args.gamma}  |  "
            f"{args.optimizer}+{args.weighting} | seed={args.seed}",
            fontsize=14, fontweight='bold')

        xx = x_eval.cpu().numpy().ravel()

        ax = axes[0, 0]
        ax.plot(xx, T_ex.cpu().numpy().ravel(), 'k-', lw=2, label='Exact T')
        ax.plot(xx, T_pred.cpu().numpy().ravel(), 'r--', lw=1.5,
                label=f'PINN T (L2={l2_T.item():.2e})')
        ax.set_title('Temperature T(x)')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(xx, u_ex.cpu().numpy().ravel(), 'k-', lw=2, label='Exact u')
        ax.plot(xx, u_pred.cpu().numpy().ravel(), 'b--', lw=1.5,
                label=f'PINN u (L2={l2_u.item():.2e})')
        ax.set_title('Displacement u(x)')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        l2_ep = curves['L2_epochs']
        ax.semilogy(l2_ep, curves['L2_avg'], alpha=0.5, lw=0.5)
        ax.axhline(best_L2_avg, color='r', ls='--', lw=1,
                   label=f'Best={best_L2_avg:.2e} @{best_epoch}')
        ax.set_title('L2_avg History')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(loss_hist['RT'], alpha=0.4, lw=0.5, label='RT (heat)')
        ax.semilogy(loss_hist['Ru'], alpha=0.4, lw=0.5, label='Ru (mech)')
        for bcn in BC_NAMES:
            ax.semilogy(loss_hist[bcn], alpha=0.4, lw=0.5, label=bcn)
        ax.set_title('Loss Components')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        with torch.no_grad():
            err_T = (T_pred - T_ex).abs().cpu().numpy().ravel()
            err_u = (u_pred - u_ex).abs().cpu().numpy().ravel()
        ax.semilogy(xx, err_T, 'r-', lw=1, label='|T_err|')
        ax.semilogy(xx, err_u, 'b-', lw=1, label='|u_err|')
        ax.set_title('Pointwise Error (Best Checkpoint)')
        ax.legend(); ax.grid(True, alpha=0.3)

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150)
        plt.close()
        print(f"  Plot: {run_dir}/summary.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\n  Saved to: {run_dir}/")

    log_path = os.path.join(run_dir, 'training_log.txt')
    logger.save(log_path)
    print(f"  Log: {log_path}")
    logger.close()

    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
