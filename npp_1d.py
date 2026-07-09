import sys
import io
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import copy
import argparse
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    p = argparse.ArgumentParser(
        description='1D NP+P PINN — equilibrium EDL')
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--zeta', type=float, default=1.0)
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='soap')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=30000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-interior', type=int, default=300)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--weighting', choices=['none', 'gradnorm'], default='none')
    p.add_argument('--gn-update-freq', type=int, default=1000)
    p.add_argument('--gn-momentum', type=float, default=0.9)
    p.add_argument('--single-network', action='store_true',
                   help='Use single shared MLP instead of independent networks')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--checkpoint-freq', type=int, default=5000)
    return p.parse_args()


def solve_reference(epsilon, zeta, n_eval=2000):
    def ode(x, y):
        return [y[1], 2.0 * np.sinh(y[0]) / epsilon**2]
    def bc(ya, yb):
        return [ya[0] - zeta, yb[0] - 0.0]

    n_mesh = 500
    x_mesh = np.linspace(0, 1, n_mesh)
    y_init = np.zeros((2, n_mesh))
    y_init[0] = zeta * (1.0 - x_mesh)

    sol = solve_bvp(ode, bc, x_mesh, y_init, tol=1e-8, max_nodes=50000)
    if not sol.success:
        y_init[0] = zeta * np.exp(-x_mesh / max(epsilon, 0.01))
        sol = solve_bvp(ode, bc, x_mesh, y_init, tol=1e-8, max_nodes=50000)
    if not sol.success:
        print(f"WARNING: BVP solver did not converge (eps={epsilon}, zeta={zeta})")

    x_eval = np.linspace(0, 1, n_eval)
    phi_ref = sol.sol(x_eval)[0]
    cp_ref = np.exp(-phi_ref)
    cm_ref = np.exp(phi_ref)
    return x_eval, phi_ref, cp_ref, cm_ref


class MLP(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_out))
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
    def __init__(self, n_in=1, n_out=3, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_out))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(2.0 * x - 1.0)


def make_collocation(n_interior, epsilon, device):
    n_half = n_interior // 2
    x_uniform = np.linspace(0, 1, n_half + 2)[1:-1]
    bl_width = min(5.0 * epsilon, 0.5)
    x_bl = bl_width * (1.0 - np.cos(np.linspace(0, np.pi/2, n_half)))
    x_all = np.unique(np.concatenate([x_uniform, x_bl]))
    x_all = np.sort(x_all)
    x_all = x_all[(x_all > 1e-6) & (x_all < 1.0 - 1e-6)]
    x_t = torch.tensor(x_all, dtype=torch.float32, device=device).reshape(-1, 1)
    return x_t


def main():
    args = parse_args()
    logger = TeeLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"epsilon={args.epsilon}, zeta={args.zeta}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, epochs={args.epochs}")
    print(f"Network: {args.layers}×{args.hidden}, Weighting: {args.weighting}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_ref_np, phi_ref_np, cp_ref_np, cm_ref_np = solve_reference(
        args.epsilon, args.zeta)
    print(f"Reference: phi=[{phi_ref_np.min():.4f}, {phi_ref_np.max():.4f}], "
          f"c+=[{cp_ref_np.min():.4f}, {cp_ref_np.max():.4f}], "
          f"c-=[{cm_ref_np.min():.4f}, {cm_ref_np.max():.4f}]")

    phi_interp = interp1d(x_ref_np, phi_ref_np, kind='cubic')
    cp_interp = interp1d(x_ref_np, cp_ref_np, kind='cubic')
    cm_interp = interp1d(x_ref_np, cm_ref_np, kind='cubic')

    net_names = ['cp', 'cm', 'phi']
    if args.single_network:
        shared = SingleNet(1, 3, args.hidden, args.layers).to(device)
        class _View:
            def __init__(self, net, idx): self.net, self.idx = net, idx
            def __call__(self, x): return self.net(x)[:, self.idx:self.idx+1]
        nets = {'cp': _View(shared, 0), 'cm': _View(shared, 1), 'phi': _View(shared, 2)}
        all_params = list(shared.parameters())
        print(f'  SingleNet: {sum(p.numel() for p in all_params):,} params')
    else:
        nets = {
            'cp':  MLP(1, 1, args.hidden, args.layers).to(device),
            'cm':  MLP(1, 1, args.hidden, args.layers).to(device),
            'phi': MLP(1, 1, args.hidden, args.layers).to(device),
        }
        all_params = []
        for name in net_names:
            all_params.extend(list(nets[name].parameters()))
    total_params = sum(p.numel() for p in all_params)
    print(f"Networks: cp(1), cm(1), phi(1) — {total_params:,} params total")

    if args.optimizer == 'soap':
        try:
            from soap import SOAP
            optimizer = SOAP(all_params, lr=args.lr, betas=(0.99, 0.999),
                             precondition_frequency=2, precondition_1d=False,
                             weight_decay=0.0)
        except ImportError:
            print("SOAP not found, falling back to Adam")
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
            'weight_hist': weight_hist, 'loss_history': loss_history, 'L2_history': L2_history,
        }, path)

    x_int = make_collocation(args.n_interior, args.epsilon, device)
    x_bc0 = torch.zeros(1, 1, device=device)
    x_bc1 = torch.ones(1, 1, device=device)
    print(f"Collocation: {x_int.shape[0]} interior points")

    zeta = args.zeta
    eps = args.epsilon
    bc_cp_0 = np.exp(-zeta)
    bc_cm_0 = np.exp(zeta)
    bc_phi_0 = zeta
    bc_cp_1 = 1.0
    bc_cm_1 = 1.0
    bc_phi_1 = 0.0

    PDE_NAMES = ['R_NP_p', 'R_NP_m', 'R_P']
    BC_NAMES = ['BC_phi', 'BC_cp', 'BC_cm']
    ALL_LOSS_NAMES = PDE_NAMES + BC_NAMES

    loss_history = {k: [] for k in ALL_LOSS_NAMES + ['total']}
    L2_history = []
    best_L2_avg = float('inf')
    best_epoch = 0
    best_state = {}
    best_pde_total = float('inf')
    best_pde_epoch = 0
    best_pde_state = {}
    gn_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    weight_hist = {n: [] for n in ALL_LOSS_NAMES}
    backward_times = []
    eval_times = []

    x_eval = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    phi_ref_eval = phi_interp(x_eval.cpu().numpy().flatten())
    cp_ref_eval = cp_interp(x_eval.cpu().numpy().flatten())
    cm_ref_eval = cm_interp(x_eval.cpu().numpy().flatten())
    phi_ref_eval_t = torch.tensor(phi_ref_eval, dtype=torch.float32, device=device).reshape(-1,1)
    cp_ref_eval_t = torch.tensor(cp_ref_eval, dtype=torch.float32, device=device).reshape(-1,1)
    cm_ref_eval_t = torch.tensor(cm_ref_eval, dtype=torch.float32, device=device).reshape(-1,1)

    t0 = time.time()
    LOG_EVERY = 500
    EVAL_EVERY = args.eval_every

    tag = (f"npp_eps{args.epsilon}_z{args.zeta}_s{args.seed}"
           f"_{args.weighting}_{args.optimizer}")
    if args.single_network:
        tag += "_sn"
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _load_state(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_state, best_L2_avg, best_epoch = ckpt['best_state'], ckpt['best_L2_avg'], ckpt['best_epoch']
        best_pde_state, best_pde_total, best_pde_epoch = ckpt['best_pde_state'], ckpt['best_pde_total'], ckpt['best_pde_epoch']
        gn_weights = ckpt['gn_weights']
        weight_hist, loss_history = ckpt['weight_hist'], ckpt['loss_history']
        L2_history = ckpt['L2_history']
        print(f"  Resumed at epoch {start_epoch}, best L2={best_L2_avg:.4e} @ ep{best_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()

        x = x_int.clone().requires_grad_(True)
        cp_pred = nets['cp'](x)
        cm_pred = nets['cm'](x)
        phi_pred = nets['phi'](x)

        ones = torch.ones_like(cp_pred)
        cp_x = torch.autograd.grad(cp_pred, x, ones, create_graph=True)[0]
        cm_x = torch.autograd.grad(cm_pred, x, ones, create_graph=True)[0]
        phi_x = torch.autograd.grad(phi_pred, x, ones, create_graph=True)[0]
        cp_xx = torch.autograd.grad(cp_x, x, ones, create_graph=True)[0]
        cm_xx = torch.autograd.grad(cm_x, x, ones, create_graph=True)[0]
        phi_xx = torch.autograd.grad(phi_x, x, ones, create_graph=True)[0]

        r_np_p = cp_xx + cp_x * phi_x + cp_pred * phi_xx
        r_np_m = cm_xx - cm_x * phi_x - cm_pred * phi_xx
        r_p = eps**2 * phi_xx + (cp_pred - cm_pred)

        L_np_p = r_np_p.pow(2).mean()
        L_np_m = r_np_m.pow(2).mean()
        L_p = r_p.pow(2).mean()
        pde_total = L_np_p + L_np_m + L_p

        bc_losses = {
            'BC_phi': (nets['phi'](x_bc0) - bc_phi_0).pow(2).mean() +
                      (nets['phi'](x_bc1) - bc_phi_1).pow(2).mean(),
            'BC_cp':  (nets['cp'](x_bc0) - bc_cp_0).pow(2).mean() +
                      (nets['cp'](x_bc1) - bc_cp_1).pow(2).mean(),
            'BC_cm':  (nets['cm'](x_bc0) - bc_cm_0).pow(2).mean() +
                      (nets['cm'](x_bc1) - bc_cm_1).pow(2).mean(),
        }
        bc_loss = sum(bc_losses.values())

        pde_losses = {'R_NP_p': L_np_p, 'R_NP_m': L_np_m, 'R_P': L_p}

        for k in ALL_LOSS_NAMES:
            if k in pde_losses:
                loss_history[k].append(pde_losses[k].item())
            else:
                loss_history[k].append(bc_losses[k].item())
        loss_history['total'].append((pde_total + bc_loss).item())

        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            _save_checkpoint(os.path.join(run_dir, f"ckpt_ep{epoch}.pt"), epoch, optimizer)

        t_bw = time.time()
        grads = {}
        for pde_name in PDE_NAMES:
            optimizer.zero_grad()
            pde_losses[pde_name].backward(retain_graph=True)
            g = torch.cat([p.grad.detach().clone().flatten()
                           if p.grad is not None
                           else torch.zeros(p.numel(), device=device)
                           for p in all_params])
            grads[pde_name] = g

        for bc_name in BC_NAMES:
            optimizer.zero_grad()
            bc_losses[bc_name].backward(retain_graph=True)
            grads[bc_name] = torch.cat([p.grad.detach().clone().flatten()
                              if p.grad is not None
                              else torch.zeros(p.numel(), device=device)
                              for p in all_params])

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

        optimizer.zero_grad()
        idx = 0
        for p in all_params:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].reshape(p.shape).clone()
            idx += numel
        optimizer.step()
        backward_times.append(time.time() - t_bw)

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            t_ev = time.time()
            with torch.no_grad():
                cp_ev = nets['cp'](x_eval)
                cm_ev = nets['cm'](x_eval)
                phi_ev = nets['phi'](x_eval)
                L2_cp = (torch.norm(cp_ev - cp_ref_eval_t) / torch.norm(cp_ref_eval_t)).item()
                L2_cm = (torch.norm(cm_ev - cm_ref_eval_t) / torch.norm(cm_ref_eval_t)).item()
                L2_phi = (torch.norm(phi_ev - phi_ref_eval_t) / torch.norm(phi_ref_eval_t)).item()
                L2_avg = (L2_cp + L2_cm + L2_phi) / 3.0
            eval_times.append(time.time() - t_ev)

            L2_history.append({
                'epoch': epoch, 'L2_cp': L2_cp, 'L2_cm': L2_cm,
                'L2_phi': L2_phi, 'L2_avg': L2_avg
            })

            if L2_avg < best_L2_avg:
                best_L2_avg = L2_avg
                best_epoch = epoch
                best_state = _save_state()

        pde_val = (pde_total + bc_loss).item()
        if pde_val < best_pde_total:
            best_pde_total = pde_val
            best_pde_epoch = epoch
            best_pde_state = _save_state()

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60.0
            improved = ' ***' if epoch == best_epoch else ''
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"L2={L2_avg:.3e}{improved}")
            print(f"  R_NP+={L_np_p.item():.1e}  R_NP-={L_np_m.item():.1e}  "
                  f"R_P={L_p.item():.1e}  BC={bc_loss.item():.1e}  "
                  f"[{' '.join(f'{n}:{bc_losses[n].item():.1e}' for n in BC_NAMES)}]")
            print(f"  best={best_L2_avg:.3e}@{best_epoch}  "
                  f"(cp={L2_cp:.1e} cm={L2_cm:.1e} phi={L2_phi:.1e})")

    _load_state(best_state)
    x_eval = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    with torch.no_grad():
        cp_eval = nets['cp'](x_eval).cpu().numpy().flatten()
        cm_eval = nets['cm'](x_eval).cpu().numpy().flatten()
        phi_eval = nets['phi'](x_eval).cpu().numpy().flatten()
        x_np = x_eval.cpu().numpy().flatten()

    cp_exact = cp_interp(x_np)
    cm_exact = cm_interp(x_np)
    phi_exact = phi_interp(x_np)

    L2_cp_best = np.sqrt(np.mean((cp_eval - cp_exact)**2)) / (np.sqrt(np.mean(cp_exact**2)) + 1e-30)
    L2_cm_best = np.sqrt(np.mean((cm_eval - cm_exact)**2)) / (np.sqrt(np.mean(cm_exact**2)) + 1e-30)
    L2_phi_best = np.sqrt(np.mean((phi_eval - phi_exact)**2)) / (np.sqrt(np.mean(phi_exact**2)) + 1e-30)
    L2_avg_best = (L2_cp_best + L2_cm_best + L2_phi_best) / 3.0

    _load_state(best_pde_state)
    with torch.no_grad():
        cp_pde = nets['cp'](x_eval).cpu().numpy().flatten()
        cm_pde = nets['cm'](x_eval).cpu().numpy().flatten()
        phi_pde = nets['phi'](x_eval).cpu().numpy().flatten()

    L2_cp_pde = np.sqrt(np.mean((cp_pde - cp_exact)**2)) / (np.sqrt(np.mean(cp_exact**2)) + 1e-30)
    L2_cm_pde = np.sqrt(np.mean((cm_pde - cm_exact)**2)) / (np.sqrt(np.mean(cm_exact**2)) + 1e-30)
    L2_phi_pde = np.sqrt(np.mean((phi_pde - phi_exact)**2)) / (np.sqrt(np.mean(phi_exact**2)) + 1e-30)
    L2_avg_pde = (L2_cp_pde + L2_cm_pde + L2_phi_pde) / 3.0

    if args.checkpoint_freq > 0:
        _save_checkpoint(os.path.join(run_dir, "ckpt_final.pt"), args.epochs, optimizer)

    final_L2 = L2_history[-1]['L2_avg']
    total_time = (time.time() - t0) / 60.0

    print()
    print("=" * 70)
    print(f"RESULTS (eps={args.epsilon}, zeta={args.zeta}, "
          f"{args.weighting}, seed={args.seed}, {args.optimizer})")
    print("=" * 70)
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    print(f"  L2_cp  = {L2_cp_best:.4e}")
    print(f"  L2_cm  = {L2_cm_best:.4e}")
    print(f"  L2_phi = {L2_phi_best:.4e}")
    print(f"  L2_avg = {L2_avg_best:.4e}")
    print(f"  === Best-PDE-loss (epoch {best_pde_epoch}) [practical] ===")
    print(f"  L2_cp  = {L2_cp_pde:.4e}")
    print(f"  L2_cm  = {L2_cm_pde:.4e}")
    print(f"  L2_phi = {L2_phi_pde:.4e}")
    print(f"  L2_avg = {L2_avg_pde:.4e}  (PDE_total={best_pde_total:.4e})")
    print(f"  === Final epoch ({args.epochs}) ===")
    print(f"  L2_avg = {final_L2:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {final_L2 / (L2_avg_best + 1e-30):.2f}x")
    print(f"  PDE-select/Best ratio = {L2_avg_pde / (L2_avg_best + 1e-30):.2f}x")
    print(f"  Total time: {total_time:.1f} min")

    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_L2': {'cp': L2_cp_best, 'cm': L2_cm_best,
                     'phi': L2_phi_best, 'avg': L2_avg_best},
        'pde_select_epoch': best_pde_epoch,
        'pde_select_L2': {'cp': L2_cp_pde, 'cm': L2_cm_pde,
                           'phi': L2_phi_pde, 'avg': L2_avg_pde},
        'final_L2': final_L2,
        'fb_ratio': final_L2 / (L2_avg_best + 1e-30),
        'pde_best_ratio': L2_avg_pde / (L2_avg_best + 1e-30),
        'total_time_min': total_time,
    }
    if args.weighting == 'gradnorm':
        w_final = gn_weights
        results['final_weights'] = {n: w_final[n] for n in ALL_LOSS_NAMES}
        pde_w = [w_final[n] for n in PDE_NAMES]
        results['final_weight_ratio'] = max(pde_w) / (min(pde_w) + 1e-30)
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if args.single_network:
        torch.save(best_state, os.path.join(run_dir, 'best_checkpoint.pt'))
        torch.save(best_pde_state, os.path.join(run_dir, 'pde_best_checkpoint.pt'))
    else:
        for n in net_names:
            torch.save(best_state[n], os.path.join(run_dir, f'best_{n}.pt'))
            torch.save(best_pde_state[n], os.path.join(run_dir, f'pde_best_{n}.pt'))

    SAVE_EVERY = 100
    n_total = len(loss_history['total'])
    save_idx = list(range(0, n_total, SAVE_EVERY))
    if (n_total - 1) not in save_idx:
        save_idx.append(n_total - 1)

    curves = {'epochs': np.array([i + 1 for i in save_idx])}
    for k in ALL_LOSS_NAMES + ['total']:
        curves[k] = np.array([loss_history[k][i] for i in save_idx])
    curves['L2_epochs'] = np.array([h['epoch'] for h in L2_history])
    for field in ['L2_cp', 'L2_cm', 'L2_phi', 'L2_avg']:
        curves[field] = np.array([h[field] for h in L2_history])
    for name in ALL_LOSS_NAMES:
        if weight_hist[name]:
            curves[f'w_{name}'] = np.array([weight_hist[name][i] for i in save_idx])
    np.savez(os.path.join(run_dir, 'loss_curves.npz'), **curves)
    print(f"  Loss curves: {run_dir}/loss_curves.npz ({len(save_idx)} points)")

    try:
        fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
        fig_lc.suptitle(f'NP+P Loss Curves  |  ε={args.epsilon}, {args.optimizer}+{args.weighting}, '
                        f'seed={args.seed}', fontsize=13, fontweight='bold')
        ep = curves['epochs']

        ax = axes_lc[0]
        for k in PDE_NAMES:
            ax.semilogy(ep, curves[k], label=k.replace('R_', ''), alpha=0.8)
        ax.semilogy(ep, curves['total'], 'k-', lw=1.5, alpha=0.5, label='total')
        ax.set_title('PDE Residuals')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes_lc[1]
        for k in BC_NAMES:
            ax.semilogy(ep, curves[k], label=k.replace('BC_', ''), alpha=0.8)
        ax.set_title('BC Losses')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes_lc[2]
        l2_ep = curves['L2_epochs']
        for field, label in [('L2_cp', 'c⁺'), ('L2_cm', 'c⁻'), ('L2_phi', 'φ')]:
            ax.semilogy(l2_ep, curves[field], label=label, alpha=0.8)
        ax.semilogy(l2_ep, curves['L2_avg'], 'k-', lw=2, label='avg')
        ax.axhline(L2_avg_best, color='r', ls='--', lw=1, alpha=0.5,
                   label=f'oracle={L2_avg_best:.1e}')
        ax.set_title('L2 Error')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'loss_curves.png'), dpi=150)
        plt.close()
        print(f"  Loss plot: {run_dir}/loss_curves.png")
    except Exception as e:
        print(f"  Loss plot failed: {e}")

    try:
        _load_state(best_state)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'1D NP+P  |  ε={args.epsilon}, {args.optimizer}+{args.weighting}, '
                     f'seed={args.seed}', fontsize=14, fontweight='bold')

        with torch.no_grad():
            cp_plot = nets['cp'](x_eval).cpu().numpy().flatten()
            cm_plot = nets['cm'](x_eval).cpu().numpy().flatten()
            phi_plot = nets['phi'](x_eval).cpu().numpy().flatten()

        axes[0, 0].plot(x_np, phi_exact, 'k-', label='Exact', lw=2)
        axes[0, 0].plot(x_np, phi_plot, 'r--', label='PINN', lw=1.5)
        axes[0, 0].set_title(f'phi (eps={args.epsilon})')
        axes[0, 0].legend(); axes[0, 0].set_xlabel('x')

        axes[0, 1].plot(x_np, cp_exact, 'k-', label='Exact c+', lw=2)
        axes[0, 1].plot(x_np, cp_plot, 'r--', label='PINN c+', lw=1.5)
        axes[0, 1].plot(x_np, cm_exact, 'b-', label='Exact c-', lw=2)
        axes[0, 1].plot(x_np, cm_plot, 'b--', label='PINN c-', lw=1.5)
        axes[0, 1].set_title('Concentrations')
        axes[0, 1].legend(); axes[0, 1].set_xlabel('x')

        axes[0, 2].plot(x_np, np.abs(phi_plot - phi_exact), 'r-', label='|err phi|')
        axes[0, 2].plot(x_np, np.abs(cp_plot - cp_exact), 'g-', label='|err c+|')
        axes[0, 2].plot(x_np, np.abs(cm_plot - cm_exact), 'b-', label='|err c-|')
        axes[0, 2].set_yscale('log')
        axes[0, 2].set_title('Pointwise error')
        axes[0, 2].legend(); axes[0, 2].set_xlabel('x')

        lh = loss_history
        for key in PDE_NAMES:
            axes[1, 0].semilogy(lh[key], label=key, alpha=0.7)
        for bcn in BC_NAMES:
            axes[1, 0].semilogy(lh[bcn], label=bcn, alpha=0.7)
        axes[1, 0].set_title('Loss history')
        axes[1, 0].legend(); axes[1, 0].set_xlabel('Epoch')

        l2_ep = [h['epoch'] for h in L2_history]
        axes[1, 1].semilogy(l2_ep, [h['L2_cp'] for h in L2_history], label='L2 c+')
        axes[1, 1].semilogy(l2_ep, [h['L2_cm'] for h in L2_history], label='L2 c-')
        axes[1, 1].semilogy(l2_ep, [h['L2_phi'] for h in L2_history], label='L2 phi')
        axes[1, 1].semilogy(l2_ep, [h['L2_avg'] for h in L2_history], 'k-', label='L2 avg', lw=2)
        axes[1, 1].axvline(best_epoch, color='red', ls='--', alpha=0.5,
                           label=f'Best@{best_epoch}')
        axes[1, 1].set_title('L2 error history')
        axes[1, 1].legend(); axes[1, 1].set_xlabel('Epoch')

        axes[1, 2].axis('off')
        summary = (
            f"eps={args.epsilon}, zeta={args.zeta}\n"
            f"Optimizer: {args.optimizer}\n"
            f"Weighting: {args.weighting}\n"
            f"Seed: {args.seed}\n\n"
            f"Best L2_avg: {L2_avg_best:.3e} @ ep {best_epoch}\n"
            f"PDE-select:  {L2_avg_pde:.3e} @ ep {best_pde_epoch}\n"
            f"Final L2:    {final_L2:.3e}\n"
            f"F/B ratio:   {final_L2/(L2_avg_best+1e-30):.1f}x\n"
            f"Time: {total_time:.1f} min"
        )
        axes[1, 2].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                        verticalalignment='center', transform=axes[1, 2].transAxes)

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


if __name__ == '__main__':
    main()
