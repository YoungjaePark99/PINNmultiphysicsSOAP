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
        description='2D NP+P+Stokes PINN — EDL-resolved EOF')
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--zeta', type=float, default=1.0)
    p.add_argument('--Ex', type=float, default=1.0)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--optimizer', choices=['adam', 'soap'], default='soap')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=50000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-interior', type=int, default=3000)
    p.add_argument('--n-boundary', type=int, default=200)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--weighting', choices=['none', 'gradnorm'],
                   default='gradnorm')
    p.add_argument('--gn-update-freq', type=int, default=1000)
    p.add_argument('--gn-momentum', type=float, default=0.9)
    p.add_argument('--single-network', action='store_true',
                   help='Use single shared MLP instead of independent networks')
    p.add_argument('--eval-every', type=int, default=100,
                   help='Evaluate L2 error every N epochs (default: 100)')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint .pt to resume from')
    p.add_argument('--checkpoint-freq', type=int, default=5000,
                   help='Save checkpoint every N epochs (0=off)')
    return p.parse_args()

def solve_reference(epsilon, zeta, Ex, mu, n_eval=500):
    def ode(y, u):
        return [u[1], 2.0 * np.sinh(u[0]) / epsilon**2]
    def bc(ua, ub):
        return [ua[0] - zeta, ub[0] - zeta]

    y_mesh = np.linspace(0, 1, 500)
    y_init = np.zeros((2, 500))
    y_init[0] = zeta * np.ones(500)
    sol = solve_bvp(ode, bc, y_mesh, y_init, tol=1e-10, max_nodes=50000)
    if not sol.success:
        y_init[0] = zeta * (1.0 - 4*(y_mesh - 0.5)**2)
        sol = solve_bvp(ode, bc, y_mesh, y_init, tol=1e-8, max_nodes=50000)

    y_eval = np.linspace(0, 1, n_eval)
    phi_ref = sol.sol(y_eval)[0]
    cp_ref = np.exp(-phi_ref)
    cm_ref = np.exp(phi_ref)
    u_ref = Ex * epsilon**2 / mu * (phi_ref - zeta)
    return y_eval, phi_ref, cp_ref, cm_ref, u_ref

class MLP(nn.Module):
    def __init__(self, n_in=2, n_out=1, n_hidden=128, n_layers=5):
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

    def forward(self, xy):
        return self.net(2.0 * xy - 1.0)

class SingleNet2D(nn.Module):
    def __init__(self, n_in=2, n_out=6, n_hidden=128, n_layers=5):
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

    def forward(self, xy):
        return self.net(2.0 * xy - 1.0)

def make_collocation(n_interior, n_boundary, epsilon, device):
    n_half = n_interior // 2
    xy_uniform = np.random.rand(n_half, 2) * 0.998 + 0.001
    bl = min(5.0 * epsilon, 0.3)
    xy_bl = np.random.rand(n_half, 2)
    xy_bl[:, 0] = xy_bl[:, 0] * 0.998 + 0.001
    n_q = n_half // 2
    xy_bl[:n_q, 1] = np.random.rand(n_q) * bl + 0.001
    xy_bl[n_q:, 1] = 1.0 - np.random.rand(n_half - n_q) * bl - 0.001

    x_int = torch.tensor(np.vstack([xy_uniform, xy_bl]),
                          dtype=torch.float32, device=device)

    nb = n_boundary
    bc_wall = np.vstack([
        np.column_stack([np.linspace(0, 1, nb), np.zeros(nb)]),
        np.column_stack([np.linspace(0, 1, nb), np.ones(nb)]),
    ])
    bc_side = np.vstack([
        np.column_stack([np.zeros(nb), np.linspace(0, 1, nb)]),
        np.column_stack([np.ones(nb), np.linspace(0, 1, nb)]),
    ])
    bc_wall_t = torch.tensor(bc_wall, dtype=torch.float32, device=device)
    bc_side_t = torch.tensor(bc_side, dtype=torch.float32, device=device)
    return x_int, bc_wall_t, bc_side_t

def laplacian_and_grads(f, xy):
    ones = torch.ones_like(f)
    g = torch.autograd.grad(f, xy, ones, create_graph=True)[0]
    f_x, f_y = g[:, 0:1], g[:, 1:2]
    g_xx = torch.autograd.grad(f_x, xy, ones, create_graph=True)[0]
    g_yy = torch.autograd.grad(f_y, xy, ones, create_graph=True)[0]
    return g_xx[:, 0:1] + g_yy[:, 1:2], f_x, f_y

def main():
    args = parse_args()
    logger = TeeLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*70}")
    print(f"2D NP+P+Stokes — EDL-Resolved EOF (4-network architecture)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"epsilon={args.epsilon}, zeta={args.zeta}, Ex={args.Ex}, mu={args.mu}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}, epochs={args.epochs}")
    print(f"Network: {args.layers}×{args.hidden}, Weighting: {args.weighting}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eps = args.epsilon
    zeta = args.zeta
    Ex = args.Ex
    mu = args.mu

    y_ref, phi_ref, cp_ref, cm_ref, u_ref = solve_reference(eps, zeta, Ex, mu)
    phi_interp = interp1d(y_ref, phi_ref, kind='cubic')
    cp_interp = interp1d(y_ref, cp_ref, kind='cubic')
    cm_interp = interp1d(y_ref, cm_ref, kind='cubic')
    u_interp = interp1d(y_ref, u_ref, kind='cubic')
    print(f"Reference: phi=[{phi_ref.min():.4f}, {phi_ref.max():.4f}], "
          f"u_max={u_ref.max():.6e}")

    net_phi = MLP(2, 1, args.hidden, args.layers).to(device)
    net_cp  = MLP(2, 1, args.hidden, args.layers).to(device)
    net_cm  = MLP(2, 1, args.hidden, args.layers).to(device)
    net_flow = MLP(2, 3, args.hidden, args.layers).to(device)

    if args.single_network:
        shared = SingleNet2D(2, 6, args.hidden, args.layers).to(device)
        class _View:
            def __init__(self, net, slc): self.net, self.slc = net, slc
            def __call__(self, x): return self.net(x)[:, self.slc]
        nets = {'phi': _View(shared, slice(0,1)), 'cp': _View(shared, slice(1,2)),
                'cm': _View(shared, slice(2,3)), 'flow': _View(shared, slice(3,6))}
        net_names = list(nets.keys())
        all_params = list(shared.parameters())
        total_params = sum(p.numel() for p in all_params)
        print(f'  SingleNet2D: {total_params:,} params')
    else:
        nets = {'phi': net_phi, 'cp': net_cp, 'cm': net_cm, 'flow': net_flow}
        net_names = list(nets.keys())
        all_params = []
        for n in net_names:
            all_params.extend(list(nets[n].parameters()))
        total_params = sum(p.numel() for p in all_params)
    print(f"Networks: phi(1), cp(1), cm(1), flow(3) — {total_params:,} params total")

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
            'weight_hist': weight_hist, 'loss_history': loss_history, 'L2_history': L2_history, 'neg_count': neg_count,
        }, path)

    x_int, bc_wall, bc_side = make_collocation(
        args.n_interior, args.n_boundary, eps, device)
    print(f"Collocation: {x_int.shape[0]} int, {bc_wall.shape[0]} wall, {bc_side.shape[0]} side")

    # Precompute BC reference values (fixed unless resampled)
    y_wall = bc_wall[:, 1:2].cpu().numpy().flatten()
    y_side = bc_side[:, 1:2].cpu().numpy().flatten()
    phi_ref_s = torch.tensor(phi_interp(y_side), dtype=torch.float32, device=device).reshape(-1, 1)
    cp_ref_s = torch.tensor(cp_interp(y_side), dtype=torch.float32, device=device).reshape(-1, 1)
    cm_ref_s = torch.tensor(cm_interp(y_side), dtype=torch.float32, device=device).reshape(-1, 1)
    u_ref_s = torch.tensor(u_interp(y_side), dtype=torch.float32, device=device).reshape(-1, 1)

    nx_ev, ny_ev = 50, 200
    x_ev = np.linspace(0, 1, nx_ev)
    y_ev = np.linspace(0, 1, ny_ev)
    xx_ev, yy_ev = np.meshgrid(x_ev, y_ev)
    xy_ev_t = torch.tensor(
        np.column_stack([xx_ev.ravel(), yy_ev.ravel()]),
        dtype=torch.float32, device=device)
    phi_exact_ev = phi_interp(yy_ev.ravel())
    cp_exact_ev = cp_interp(yy_ev.ravel())
    cm_exact_ev = cm_interp(yy_ev.ravel())
    u_exact_ev = u_interp(yy_ev.ravel())

    PDE_NAMES = ['R_Poisson', 'R_NP_p', 'R_NP_m', 'R_Stokes_x', 'R_Stokes_y', 'R_cont']
    BC_NAMES = ['BC_phi', 'BC_cp', 'BC_cm', 'BC_u', 'BC_v', 'BC_p']
    ALL_LOSS_NAMES = PDE_NAMES + BC_NAMES

    loss_history = {k: [] for k in ALL_LOSS_NAMES + ['total']}
    L2_history = []
    best_L2_avg = float('inf')
    best_epoch = 0
    best_state = None
    best_pde_total = float('inf')
    best_pde_epoch = 0
    best_pde_state = None
    L2_avg = L2_phi = L2_cp = L2_cm = L2_u = float('inf')
    gn_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    weight_hist = {n: [] for n in ALL_LOSS_NAMES}
    backward_times = []
    eval_times = []

    t0 = time.time()
    LOG_EVERY = 500
    EVAL_EVERY = args.eval_every

    tag = (f"npps2d_eps{eps}_Ex{Ex}_s{args.seed}"
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
        L2_history, neg_count = ckpt['L2_history'], ckpt['neg_count']
        print(f"  Resumed at epoch {start_epoch}, best L2={best_L2_avg:.4e} @ ep{best_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()

        # Resample collocation if requested

        xy = x_int.clone().requires_grad_(True)

        phi = nets['phi'](xy)
        cp = nets['cp'](xy)
        cm = nets['cm'](xy)
        flow_out = nets['flow'](xy)
        u_vel = flow_out[:, 0:1]
        v_vel = flow_out[:, 1:2]
        p_pres = flow_out[:, 2:3]

        ones = torch.ones_like(phi)

        lap_phi, phi_x, phi_y = laplacian_and_grads(phi, xy)
        lap_cp, cp_x, cp_y = laplacian_and_grads(cp, xy)
        lap_cm, cm_x, cm_y = laplacian_and_grads(cm, xy)
        lap_u, u_x, u_y = laplacian_and_grads(u_vel, xy)
        lap_v, v_x, v_y = laplacian_and_grads(v_vel, xy)

        p_grad = torch.autograd.grad(p_pres, xy, ones, create_graph=True)[0]
        p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]

        r_poisson = eps**2 * lap_phi + (cp - cm)
        r_np_p = lap_cp + cp_x * phi_x + cp_y * phi_y + cp * lap_phi
        r_np_m = lap_cm - cm_x * phi_x - cm_y * phi_y - cm * lap_phi
        r_stokes_x = -p_x + mu * lap_u + Ex * (cp - cm)
        r_stokes_y = -p_y + mu * lap_v
        r_cont = u_x + v_y

        pde_losses = {
            'R_Poisson': r_poisson.pow(2).mean(),
            'R_NP_p': r_np_p.pow(2).mean(),
            'R_NP_m': r_np_m.pow(2).mean(),
            'R_Stokes_x': r_stokes_x.pow(2).mean(),
            'R_Stokes_y': r_stokes_y.pow(2).mean(),
            'R_cont': r_cont.pow(2).mean(),
        }

        # Walls: phi=zeta, c+=exp(-zeta), c-=exp(zeta), u=0, v=0
        flow_wall = nets['flow'](bc_wall)
        flow_side = nets['flow'](bc_side)

        # Per-field BC losses
        bc_losses = {
            'BC_phi': (nets['phi'](bc_wall) - zeta).pow(2).mean() +
                      (nets['phi'](bc_side) - phi_ref_s).pow(2).mean(),
            'BC_cp':  (nets['cp'](bc_wall) - np.exp(-zeta)).pow(2).mean() +
                      (nets['cp'](bc_side) - cp_ref_s).pow(2).mean(),
            'BC_cm':  (nets['cm'](bc_wall) - np.exp(zeta)).pow(2).mean() +
                      (nets['cm'](bc_side) - cm_ref_s).pow(2).mean(),
            'BC_u':   flow_wall[:, 0:1].pow(2).mean() +
                      (flow_side[:, 0:1] - u_ref_s).pow(2).mean(),
            'BC_v':   flow_wall[:, 1:2].pow(2).mean() +
                      flow_side[:, 1:2].pow(2).mean(),
            'BC_p':   flow_side[:, 2:3].pow(2).mean(),
        }
        bc_loss = sum(bc_losses.values())
        for bcn in BC_NAMES:
            pde_losses[bcn] = bc_losses[bcn]

        pde_total = sum(pde_losses[k] for k in PDE_NAMES)

        for k in ALL_LOSS_NAMES:
            loss_history[k].append(pde_losses[k].item())
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

        # Per-BC gradients
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

        # Apply
        optimizer.zero_grad()
        idx = 0
        for p in all_params:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].reshape(p.shape).clone()
            idx += numel
        backward_times.append(time.time() - t_bw)
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            t_ev = time.time()
            with torch.no_grad():
                phi_pred = nets['phi'](xy_ev_t).cpu().numpy().flatten()
                cp_pred = nets['cp'](xy_ev_t).cpu().numpy().flatten()
                cm_pred = nets['cm'](xy_ev_t).cpu().numpy().flatten()
                flow_pred = nets['flow'](xy_ev_t).cpu().numpy()
                u_pred = flow_pred[:, 0]

                L2_phi = np.sqrt(np.mean((phi_pred - phi_exact_ev)**2)) / \
                         (np.sqrt(np.mean(phi_exact_ev**2)) + 1e-30)
                L2_cp = np.sqrt(np.mean((cp_pred - cp_exact_ev)**2)) / \
                        (np.sqrt(np.mean(cp_exact_ev**2)) + 1e-30)
                L2_cm = np.sqrt(np.mean((cm_pred - cm_exact_ev)**2)) / \
                        (np.sqrt(np.mean(cm_exact_ev**2)) + 1e-30)
                L2_u = np.sqrt(np.mean((u_pred - u_exact_ev)**2)) / \
                       (np.sqrt(np.mean(u_exact_ev**2)) + 1e-30)
                L2_avg = (L2_phi + L2_cp + L2_cm + L2_u) / 4.0

            eval_times.append(time.time() - t_ev)
            L2_history.append({'epoch': epoch, 'L2_avg': L2_avg,
                               'L2_phi': L2_phi, 'L2_cp': L2_cp,
                               'L2_cm': L2_cm, 'L2_u': L2_u})

            if L2_avg < best_L2_avg:
                best_L2_avg = L2_avg
                best_epoch = epoch
                best_state = _save_state()

        total_loss_val = (pde_total + bc_loss).item()
        if total_loss_val < best_pde_total:
            best_pde_total = total_loss_val
            best_pde_epoch = epoch
            best_pde_state = _save_state()

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60
            improved = ' ***' if epoch == best_epoch else ''
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"L2={L2_avg:.3e}{improved}")
            pde_str = '  '.join(f'{k.replace("R_","")}={pde_losses[k].item():.1e}'
                                for k in PDE_NAMES)
            print(f"  {pde_str}  BC={bc_loss.item():.1e}  [{' '.join(f'{n[3:]}:{bc_losses[n].item():.1e}' for n in BC_NAMES)}]")
            print(f"  best={best_L2_avg:.3e}@{best_epoch}  "
                  f"(phi={L2_phi:.1e} cp={L2_cp:.1e} cm={L2_cm:.1e} u={L2_u:.1e})")

    _load_state(best_state)
    with torch.no_grad():
        phi_b = nets['phi'](xy_ev_t).cpu().numpy().flatten()
        cp_b = nets['cp'](xy_ev_t).cpu().numpy().flatten()
        cm_b = nets['cm'](xy_ev_t).cpu().numpy().flatten()
        u_b = nets['flow'](xy_ev_t).cpu().numpy()[:, 0]

        L2s = {}
        for name, pred, exact in [('phi', phi_b, phi_exact_ev),
                                   ('cp', cp_b, cp_exact_ev),
                                   ('cm', cm_b, cm_exact_ev),
                                   ('u', u_b, u_exact_ev)]:
            L2s[name] = np.sqrt(np.mean((pred - exact)**2)) / \
                        (np.sqrt(np.mean(exact**2)) + 1e-30)
        L2s['avg'] = np.mean([L2s[k] for k in ['phi', 'cp', 'cm', 'u']])

    _load_state(best_pde_state)
    with torch.no_grad():
        phi_p = nets['phi'](xy_ev_t).cpu().numpy().flatten()
        cp_p = nets['cp'](xy_ev_t).cpu().numpy().flatten()
        cm_p = nets['cm'](xy_ev_t).cpu().numpy().flatten()
        u_p = nets['flow'](xy_ev_t).cpu().numpy()[:, 0]

        L2s_pde = {}
        for name, pred, exact in [('phi', phi_p, phi_exact_ev),
                                   ('cp', cp_p, cp_exact_ev),
                                   ('cm', cm_p, cm_exact_ev),
                                   ('u', u_p, u_exact_ev)]:
            L2s_pde[name] = np.sqrt(np.mean((pred - exact)**2)) / \
                            (np.sqrt(np.mean(exact**2)) + 1e-30)
        L2s_pde['avg'] = np.mean([L2s_pde[k] for k in ['phi', 'cp', 'cm', 'u']])
    if args.checkpoint_freq > 0:
        _save_checkpoint(os.path.join(run_dir, "ckpt_final.pt"), args.epochs, optimizer)

    final_L2 = L2_history[-1]['L2_avg']
    total_time = (time.time() - t0) / 60.0

    print()
    print("=" * 70)
    print(f"RESULTS (eps={eps}, Ex={Ex}, {args.weighting}, "
          f"seed={args.seed}, {args.optimizer})")
    print("=" * 70)
    print(f"  === Best-ever (epoch {best_epoch}) [oracle] ===")
    for k in ['phi', 'cp', 'cm', 'u']:
        print(f"  L2_{k:>3} = {L2s[k]:.4e}")
    print(f"  L2_avg = {L2s['avg']:.4e}")
    print(f"  === Best-PDE (epoch {best_pde_epoch}) [oracle-free] ===")
    print(f"  PDE_total = {best_pde_total:.4e}")
    for k in ['phi', 'cp', 'cm', 'u']:
        print(f"  L2_{k:>3} = {L2s_pde[k]:.4e}")
    print(f"  L2_avg = {L2s_pde['avg']:.4e}")
    print(f"  === Final epoch ({args.epochs}) ===")
    print(f"  L2_avg = {final_L2:.4e}")
    print(f"  === Gap ===")
    print(f"  Final/Best ratio = {final_L2 / (L2s['avg'] + 1e-30):.2f}x")
    print(f"  PDE-select/Best ratio = {L2s_pde['avg'] / (L2s['avg'] + 1e-30):.2f}x")
    print(f"  Total time: {total_time:.1f} min")

    results = {
        'args': vars(args),
        'best_epoch': best_epoch, 'best_L2': L2s,
        'pde_select_epoch': best_pde_epoch,
        'pde_select_total': best_pde_total,
        'pde_select_L2': L2s_pde,
        'final_L2': final_L2,
        'fb_ratio': final_L2 / (L2s['avg'] + 1e-30),
        'pde_best_ratio': L2s_pde['avg'] / (L2s['avg'] + 1e-30),
        'total_time_min': total_time,
    }
    if args.weighting == 'gradnorm':
        w_final = gn_weights if args.weighting == 'gradnorm' else gn_weights
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
    # L2 curves (already at EVAL_EVERY intervals, separate epoch array)
    curves['L2_epochs'] = np.array([h['epoch'] for h in L2_history])
    for field in ['L2_phi', 'L2_cp', 'L2_cm', 'L2_u', 'L2_avg']:
        curves[field] = np.array([h[field] for h in L2_history])
    for name in ALL_LOSS_NAMES:
        if weight_hist[name]:
            curves[f'w_{name}'] = np.array([weight_hist[name][i] for i in save_idx])
    np.savez(os.path.join(run_dir, 'loss_curves.npz'), **curves)
    print(f"  Loss curves: {run_dir}/loss_curves.npz ({len(save_idx)} points)")

    try:
        fig_lc, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
        fig_lc.suptitle(f'2D NPPS Loss Curves  |  ε={eps}, {args.optimizer}+{args.weighting}, '
                        f'seed={args.seed}', fontsize=13, fontweight='bold')
        ep = curves['epochs']

        ax = axes_lc[0]
        for k in PDE_NAMES:
            ax.semilogy(ep, curves[k], label=k.replace('R_', ''), alpha=0.8)
        ax.semilogy(ep, curves['total'], 'k-', lw=1.5, alpha=0.5, label='total')
        ax.set_title('PDE Residuals')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes_lc[1]
        for k in BC_NAMES:
            ax.semilogy(ep, curves[k], label=k.replace('BC_', ''), alpha=0.8)
        ax.set_title('BC Losses')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes_lc[2]
        l2_ep = curves['L2_epochs']
        for field, label in [('L2_phi', 'φ'), ('L2_cp', 'c⁺'), ('L2_cm', 'c⁻'), ('L2_u', 'u')]:
            ax.semilogy(l2_ep, curves[field], label=label, alpha=0.8)
        ax.semilogy(l2_ep, curves['L2_avg'], 'k-', lw=2, label='avg')
        ax.axhline(L2s['avg'], color='r', ls='--', lw=1, alpha=0.5,
                   label=f'oracle={L2s["avg"]:.1e}')
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
        with torch.no_grad():
            phi_plot = nets['phi'](xy_ev_t).cpu().numpy().reshape(ny_ev, nx_ev)
            cp_plot = nets['cp'](xy_ev_t).cpu().numpy().reshape(ny_ev, nx_ev)
            cm_plot = nets['cm'](xy_ev_t).cpu().numpy().reshape(ny_ev, nx_ev)
            flow_plot = nets['flow'](xy_ev_t).cpu().numpy()
            u_plot = flow_plot[:, 0].reshape(ny_ev, nx_ev)
            v_plot = flow_plot[:, 1].reshape(ny_ev, nx_ev)
            p_plot = flow_plot[:, 2].reshape(ny_ev, nx_ev)

            phi_ex_2d = phi_exact_ev.reshape(ny_ev, nx_ev)
            cp_ex_2d = cp_exact_ev.reshape(ny_ev, nx_ev)
            cm_ex_2d = cm_exact_ev.reshape(ny_ev, nx_ev)
            u_ex_2d = u_exact_ev.reshape(ny_ev, nx_ev)
            v_ex_2d = np.zeros_like(u_ex_2d)
            p_ex_2d = np.zeros_like(u_ex_2d)

        fig = plt.figure(figsize=(28, 13))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.8],
                              hspace=0.35, wspace=0.45)
        fig.suptitle(f'2D NP+P+Stokes (EDL-resolved)  |  ε={eps}, Ex={Ex}, '
                     f'{args.optimizer}+{args.weighting}',
                     fontsize=15, fontweight='bold')

        fields = [
            ('φ', phi_plot, 'RdBu_r'),
            ('c⁺', cp_plot, 'YlOrRd'),
            ('c⁻', cm_plot, 'YlOrRd'),
            ('u', u_plot, 'viridis'),
            ('v', v_plot, 'RdBu_r'),
            ('p', p_plot, 'RdBu_r'),
        ]
        for j, (label, data, cmap) in enumerate(fields):
            ax = fig.add_subplot(gs[0, j])
            im = ax.pcolormesh(xx_ev, yy_ev, data, cmap=cmap, shading='auto')
            ax.set_title(f'{label} (PINN)', fontsize=11)
            plt.colorbar(im, ax=ax, format='%.2e')
            ax.set_aspect('equal')

        errors = [
            ('|φ err|', phi_plot, phi_ex_2d),
            ('|c⁺ err|', cp_plot, cp_ex_2d),
            ('|c⁻ err|', cm_plot, cm_ex_2d),
            ('|u err|', u_plot, u_ex_2d),
            ('|v err|', v_plot, v_ex_2d),
            ('|p err|', p_plot, p_ex_2d),
        ]
        for j, (label, pred, exact) in enumerate(errors):
            ax = fig.add_subplot(gs[1, j])
            err = np.abs(pred - exact)
            im = ax.pcolormesh(xx_ev, yy_ev, err, cmap='inferno', shading='auto')
            ax.set_title(f'{label}', fontsize=11)
            plt.colorbar(im, ax=ax, format='%.1e')
            ax.set_aspect('equal')

        ax_l2 = fig.add_subplot(gs[2, :])
        ep_arr = [h['epoch'] for h in L2_history]
        ax_l2.semilogy(ep_arr, [h['L2_avg'] for h in L2_history], 'k-', lw=0.8,
                        alpha=0.7, label='L2_avg')
        ax_l2.axhline(L2s['avg'], color='r', ls='--', lw=1.5,
                       label=f'Oracle best = {L2s["avg"]:.2e} (ep {best_epoch})')
        ax_l2.axhline(L2s_pde['avg'], color='b', ls='--', lw=1.5,
                       label=f'PDE best = {L2s_pde["avg"]:.2e} (ep {best_pde_epoch})')
        ax_l2.set_title('L2_avg history', fontsize=12)
        ax_l2.set_xlabel('Epoch'); ax_l2.set_ylabel('L2 relative error')
        ax_l2.legend(fontsize=10); ax_l2.grid(True, alpha=0.3)

        plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150,
                    bbox_inches='tight')
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
