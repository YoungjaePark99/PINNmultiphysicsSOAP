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
    p = argparse.ArgumentParser(description='Reaction-Diffusion 3-Species A⇌B⇌C')
    p.add_argument('--D1', type=float, default=1.0)
    p.add_argument('--D2', type=float, default=1.0)
    p.add_argument('--D3', type=float, default=1.0)
    p.add_argument('--k-rxn', type=float, default=1.0,
                   help='Symmetric reaction rate k=k1=k2=k3=k4')
    p.add_argument('--k1', type=float, default=None)
    p.add_argument('--k2', type=float, default=None)
    p.add_argument('--k3', type=float, default=None)
    p.add_argument('--k4', type=float, default=None)
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


def resolve_rates(args):
    k = args.k_rxn
    args.k1_val = args.k1 if args.k1 is not None else k
    args.k2_val = args.k2 if args.k2 is not None else k
    args.k3_val = args.k3 if args.k3 is not None else k
    args.k4_val = args.k4 if args.k4 is not None else k


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
    def __init__(self, n_hidden, n_neurons, n_out=3):
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


def exact_cA(x):
    return torch.sin(PI * x) + 2.0

def exact_cB(x):
    return torch.cos(PI * x) + 2.0

def exact_cC(x):
    return torch.sin(2 * PI * x) + 2.0

def compute_sources(x, args):
    k1, k2, k3, k4 = args.k1_val, args.k2_val, args.k3_val, args.k4_val
    sx, cx, s2x = torch.sin(PI*x), torch.cos(PI*x), torch.sin(2*PI*x)
    cA, cB, cC = sx + 2.0, cx + 2.0, s2x + 2.0
    lap_cA = -PI**2 * sx
    lap_cB = -PI**2 * cx
    lap_cC = -4*PI**2 * s2x
    f1 = args.D1 * lap_cA - k1 * cA + k2 * cB
    f2 = args.D2 * lap_cB + k1 * cA - (k2 + k3) * cB + k4 * cC
    f3 = args.D3 * lap_cC + k3 * cB - k4 * cC
    return f1.detach(), f2.detach(), f3.detach()


def compute_residuals(nets, x, src, args):
    k1, k2, k3, k4 = args.k1_val, args.k2_val, args.k3_val, args.k4_val
    cA = nets['cA'](x)
    cB = nets['cB'](x)
    cC = nets['cC'](x)
    ones = torch.ones_like(cA)

    cA_x = torch.autograd.grad(cA, x, ones, create_graph=True, retain_graph=True)[0]
    cA_xx = torch.autograd.grad(cA_x, x, ones, create_graph=True, retain_graph=True)[0]
    cB_x = torch.autograd.grad(cB, x, ones, create_graph=True, retain_graph=True)[0]
    cB_xx = torch.autograd.grad(cB_x, x, ones, create_graph=True, retain_graph=True)[0]
    cC_x = torch.autograd.grad(cC, x, ones, create_graph=True, retain_graph=True)[0]
    cC_xx = torch.autograd.grad(cC_x, x, ones, create_graph=True, retain_graph=True)[0]

    f1, f2, f3 = src
    R1 = args.D1 * cA_xx - k1 * cA + k2 * cB - f1
    R2 = args.D2 * cB_xx + k1 * cA - (k2 + k3) * cB + k4 * cC - f2
    R3 = args.D3 * cC_xx + k3 * cB - k4 * cC - f3

    return {
        'R1': torch.mean(R1**2),
        'R2': torch.mean(R2**2),
        'R3': torch.mean(R3**2),
    }


def compute_bc_loss(nets, args, device):
    x0 = torch.zeros(1, 1, device=device)
    x1 = torch.ones(1, 1, device=device)
    bc_cA = sum((nets['cA'](xb) - exact_cA(xb))**2 for xb in [x0, x1]).squeeze()
    bc_cB = sum((nets['cB'](xb) - exact_cB(xb))**2 for xb in [x0, x1]).squeeze()
    bc_cC = sum((nets['cC'](xb) - exact_cC(xb))**2 for xb in [x0, x1]).squeeze()
    return {'BC_cA': bc_cA, 'BC_cB': bc_cB, 'BC_cC': bc_cC}


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
    resolve_rates(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"k1={args.k1_val}, k2={args.k2_val}, k3={args.k3_val}, k4={args.k4_val}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, epochs={args.epochs}")
    print(f"Network: {args.n_hidden}×{args.n_neurons}, Weighting: {args.weighting}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    net_names = ['cA', 'cB', 'cC']
    if args.single_network:
        shared = SingleNet(args.n_hidden, args.n_neurons, n_out=3).to(device)
        class _View:
            def __init__(self, net, idx): self.net, self.idx = net, idx
            def __call__(self, x): return self.net(x)[:, self.idx:self.idx+1]
        nets = {'cA': _View(shared, 0), 'cB': _View(shared, 1), 'cC': _View(shared, 2)}
        all_params = list(shared.parameters())
        print(f'  SingleNet: {sum(p.numel() for p in all_params):,} params')
    else:
        nets = {n: SubNet(args.n_hidden, args.n_neurons).to(device) for n in net_names}
        all_params = []
        for n in net_names:
            all_params.extend(list(nets[n].parameters()))
    total_params = sum(p.numel() for p in all_params)
    print(f"Networks: cA(1), cB(1), cC(1) — {total_params:,} params total")

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
    print(f"Source magnitudes: |f1|={src[0].abs().max().item():.2e}, "
          f"|f2|={src[1].abs().max().item():.2e}, |f3|={src[2].abs().max().item():.2e}")

    PDE_NAMES = ['R1', 'R2', 'R3']
    BC_NAMES = ['BC_cA', 'BC_cB', 'BC_cC']
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
        cA_e_mon = exact_cA(x_eval_mon)
        cB_e_mon = exact_cB(x_eval_mon)
        cC_e_mon = exact_cC(x_eval_mon)

    t0 = time.time()
    LOG_EVERY = 500
    EVAL_EVERY = args.eval_every

    tag = f"rxndiff_k{args.k_rxn:.4g}_s{args.seed}_{args.weighting}_{args.optimizer}"
    if not (args.k1_val == args.k2_val == args.k3_val == args.k4_val):
        tag += "_asym"
    if args.single_network:
        tag += "_sn"
    run_dir = os.path.join('runs', f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
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

        with torch.no_grad():
            vals = {k: pde_losses[k].item() for k in PDE_NAMES}
            for bcn in BC_NAMES:
                vals[bcn] = bc_losses[bcn].item()
            vals['total'] = sum(vals.values())
        for k in vals:
            loss_hist[k].append(vals[k])

        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            _save_checkpoint(os.path.join(run_dir, f"ckpt_ep{epoch}.pt"), epoch, optimizer)

        if vals['total'] < best_pde_total:
            best_pde_total = vals['total']
            best_pde_epoch = epoch
            best_pde_state = _save_state()

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            t_eval = time.time()
            with torch.no_grad():
                cA_p_mon = nets['cA'](x_eval_mon)
                cB_p_mon = nets['cB'](x_eval_mon)
                cC_p_mon = nets['cC'](x_eval_mon)
                L2_cA_ep = (torch.norm(cA_p_mon - cA_e_mon) / torch.norm(cA_e_mon)).item()
                L2_cB_ep = (torch.norm(cB_p_mon - cB_e_mon) / torch.norm(cB_e_mon)).item()
                L2_cC_ep = (torch.norm(cC_p_mon - cC_e_mon) / torch.norm(cC_e_mon)).item()
                L2_avg_ep = (L2_cA_ep + L2_cB_ep + L2_cC_ep) / 3
            eval_times.append(time.time() - t_eval)

            L2_history.append({
                'epoch': epoch, 'cA': L2_cA_ep, 'cB': L2_cB_ep,
                'cC': L2_cC_ep, 'avg': L2_avg_ep,
            })

            if L2_avg_ep < best_L2_avg:
                best_L2_avg = L2_avg_ep
                best_epoch = epoch
                best_state = _save_state()

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60
            improved = ' ***' if epoch == best_epoch else ''
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"L2={L2_avg_ep:.3e}{improved}")
            print(f"  R1={vals['R1']:.1e}  R2={vals['R2']:.1e}  R3={vals['R3']:.1e}  "
                  f"BC={sum(vals[n] for n in BC_NAMES):.1e}")
            print(f"  best={best_L2_avg:.3e}@{best_epoch}  "
                  f"(cA={L2_cA_ep:.1e} cB={L2_cB_ep:.1e} cC={L2_cC_ep:.1e})")

    elapsed_total = (time.time() - t0) / 60

    x_eval = torch.linspace(0, 1, 500, device=device).reshape(-1, 1)
    with torch.no_grad():
        cA_ex = exact_cA(x_eval)
        cB_ex = exact_cB(x_eval)
        cC_ex = exact_cC(x_eval)

        cA_pr_final = nets['cA'](x_eval)
        cB_pr_final = nets['cB'](x_eval)
        cC_pr_final = nets['cC'](x_eval)
        L2_final = {}
        for name, pred, exact in [('cA', cA_pr_final, cA_ex),
                                   ('cB', cB_pr_final, cB_ex),
                                   ('cC', cC_pr_final, cC_ex)]:
            L2_final[name] = (torch.norm(pred - exact) / torch.norm(exact)).item()

    if best_state is not None:
        _load_state(best_state)
        with torch.no_grad():
            cA_pr = nets['cA'](x_eval)
            cB_pr = nets['cB'](x_eval)
            cC_pr = nets['cC'](x_eval)
            L2 = {}
            for name, pred, exact in [('cA', cA_pr, cA_ex),
                                       ('cB', cB_pr, cB_ex),
                                       ('cC', cC_pr, cC_ex)]:
                L2[name] = (torch.norm(pred - exact) / torch.norm(exact)).item()
    else:
        L2 = L2_final
        cA_pr, cB_pr, cC_pr = cA_pr_final, cB_pr_final, cC_pr_final
        best_epoch = args.epochs

    if best_pde_state is not None:
        _load_state(best_pde_state)
        with torch.no_grad():
            L2_pde = {}
            for name, net, exact in [('cA', nets['cA'], cA_ex),
                                      ('cB', nets['cB'], cB_ex),
                                      ('cC', nets['cC'], cC_ex)]:
                pred = net(x_eval)
                L2_pde[name] = (torch.norm(pred - exact) / torch.norm(exact)).item()
        if best_state is not None:
            _load_state(best_state)
    else:
        L2_pde = L2_final

    if args.checkpoint_freq > 0:
        _save_checkpoint(os.path.join(run_dir, "ckpt_final.pt"), args.epochs, optimizer)

    best_avg = (L2['cA'] + L2['cB'] + L2['cC']) / 3
    final_avg = (L2_final['cA'] + L2_final['cB'] + L2_final['cC']) / 3
    pde_avg = (L2_pde['cA'] + L2_pde['cB'] + L2_pde['cC']) / 3
    L2['avg'] = best_avg
    L2_final['avg'] = final_avg
    L2_pde['avg'] = pde_avg

    print(f"\n{'='*70}")
    print(f"RESULTS (k={args.k_rxn}, {args.weighting}, seed={args.seed}, {args.optimizer})")
    print(f"{'='*70}")
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    for name in ['cA', 'cB', 'cC']:
        print(f"  L2_{name} = {L2[name]:.4e}")
    print(f"  L2_avg  = {best_avg:.4e}")
    print(f"  === Best-PDE-loss (epoch {best_pde_epoch}) [practical] ===")
    for name in ['cA', 'cB', 'cC']:
        print(f"  L2_{name} = {L2_pde[name]:.4e}")
    print(f"  L2_avg  = {pde_avg:.4e}  (PDE_total={best_pde_total:.4e})")
    print(f"  === Final epoch ({args.epochs}) ===")
    for name in ['cA', 'cB', 'cC']:
        print(f"  L2_{name} = {L2_final[name]:.4e}")
    print(f"  L2_avg  = {final_avg:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {final_avg / max(best_avg, 1e-30):.2f}x")
    print(f"  PDE-select/Best ratio = {pde_avg / (best_avg + 1e-30):.2f}x")
    print(f"  Total time: {elapsed_total:.1f} min")

    results = {
        'args': {k: v for k, v in vars(args).items()
                 if not k.startswith('k') or k in ['k_rxn','k1_val','k2_val','k3_val','k4_val']},
        'L2': L2,
        'L2_pde_best': L2_pde,
        'L2_final_epoch': L2_final,
        'best_epoch': best_epoch,
        'best_pde_epoch': best_pde_epoch,
        'best_pde_total': best_pde_total,
        'total_time_min': elapsed_total,
    }
    if args.weighting == 'gradnorm':
        w_final = gn_weights
        results['final_weights'] = {n: w_final[n] for n in ALL_LOSS_NAMES}
        pde_w = [w_final[n] for n in PDE_NAMES]
        results['final_weight_ratio'] = max(pde_w) / (min(pde_w) + 1e-30)
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if args.single_network:
        torch.save(best_state, os.path.join(run_dir, 'best_checkpoint.pt'))
        torch.save(best_pde_state, os.path.join(run_dir, 'pde_best_checkpoint.pt'))
    else:
        for n in net_names:
            if best_state is not None:
                torch.save(best_state[n], os.path.join(run_dir, f'best_{n}.pt'))
            if best_pde_state is not None:
                torch.save(best_pde_state[n], os.path.join(run_dir, f'pde_best_{n}.pt'))

    SAVE_EVERY = 100
    n_total = len(loss_hist['total'])
    save_idx = list(range(0, n_total, SAVE_EVERY))
    if (n_total - 1) not in save_idx:
        save_idx.append(n_total - 1)

    curves = {'epochs': np.array([i + 1 for i in save_idx])}
    for k in ALL_LOSS_NAMES + ['total']:
        curves[k] = np.array([loss_hist[k][i] for i in save_idx])
    curves['L2_epochs'] = np.array([h['epoch'] for h in L2_history])
    for field in ['cA', 'cB', 'cC', 'avg']:
        curves[f'L2_{field}'] = np.array([h[field] for h in L2_history])
    for name in ALL_LOSS_NAMES:
        if weight_hist[name]:
            curves[f'w_{name}'] = np.array([weight_hist[name][i] for i in save_idx])
    np.savez(os.path.join(run_dir, 'loss_curves.npz'), **curves)
    print(f"  Loss curves: {run_dir}/loss_curves.npz ({len(save_idx)} points)")

    try:
        fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
        fig_lc.suptitle(f'RxnDiff Loss Curves  |  k={args.k_rxn}, {args.optimizer}+{args.weighting}, '
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
        for field, label in [('L2_cA', 'cA'), ('L2_cB', 'cB'), ('L2_cC', 'cC')]:
            ax.semilogy(l2_ep, curves[field], label=label, alpha=0.8)
        ax.semilogy(l2_ep, curves['L2_avg'], 'k-', lw=2, label='avg')
        ax.axhline(best_avg, color='r', ls='--', lw=1, alpha=0.5,
                   label=f'oracle={best_avg:.1e}')
        ax.set_title('L2 Error')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'loss_curves.png'), dpi=150)
        plt.close()
        print(f"  Loss plot: {run_dir}/loss_curves.png")
    except Exception as e:
        print(f"  Loss plot failed: {e}")

    try:
        if best_state is not None:
            _load_state(best_state)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"Reaction-Diffusion A⇌B⇌C | k={args.k_rxn} | "
                     f"{args.optimizer}+{args.weighting} | seed={args.seed}",
                     fontsize=13, fontweight='bold')

        colors = {'cA': 'tab:red', 'cB': 'tab:blue', 'cC': 'tab:green'}
        x_np = x_eval.cpu().numpy()
        with torch.no_grad():
            pred_dict = {n: nets[n](x_eval).cpu().numpy() for n in net_names}
        exact_dict = {'cA': cA_ex.cpu().numpy(), 'cB': cB_ex.cpu().numpy(),
                      'cC': cC_ex.cpu().numpy()}

        for i, name in enumerate(net_names):
            ax = axes[0, i]
            ax.plot(x_np, exact_dict[name], 'k-', lw=2, label='Exact')
            ax.plot(x_np, pred_dict[name], '--', color=colors[name], lw=1.5,
                    label=f'PINN (best@{best_epoch})')
            ax.set_title(f'{name}  (L2={L2[name]:.2e})')
            ax.legend(); ax.grid(True, alpha=0.3)

        lh = loss_hist
        for key in PDE_NAMES:
            axes[1, 0].semilogy(lh[key], label=key, alpha=0.7)
        axes[1, 0].set_title('PDE Losses')
        axes[1, 0].set_xlabel('epoch')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        l2_ep = [h['epoch'] for h in L2_history]
        stride = max(1, len(L2_history) // 500)
        L2h_plot = L2_history[::stride]
        l2_ep_plot = [h['epoch'] for h in L2h_plot]
        for name in net_names:
            axes[1, 1].semilogy(l2_ep_plot, [h[name] for h in L2h_plot],
                                label=name, color=colors[name], alpha=0.7)
        axes[1, 1].semilogy(l2_ep_plot, [h['avg'] for h in L2h_plot],
                            'k-', label='avg', lw=1.5)
        axes[1, 1].axvline(best_epoch, color='red', ls=':', alpha=0.6,
                           label=f'best ep={best_epoch}')
        axes[1, 1].set_title('L2 history')
        axes[1, 1].set_xlabel('epoch')
        axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].axis('off')
        info = (
            f"k = {args.k_rxn}\n"
            f"optimizer: {args.optimizer}\n"
            f"weighting: {args.weighting}\n"
            f"seed: {args.seed}\n"
            f"time: {elapsed_total:.1f} min\n\n"
            f"Best (ep {best_epoch}):\n"
        )
        for name in net_names:
            info += f"  L2_{name} = {L2[name]:.3e}\n"
        info += f"\nFinal (ep {args.epochs}):\n"
        for name in net_names:
            info += f"  L2_{name} = {L2_final[name]:.3e}\n"
        axes[1, 2].text(0.05, 0.95, info, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Summary')

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150)
        plt.close()
        print(f"  Plot: {run_dir}/summary.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"  Saved to: {run_dir}/")

    log_path = os.path.join(run_dir, 'training_log.txt')
    logger.save(log_path)
    print(f"  Log: {log_path}")
    logger.close()

    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
