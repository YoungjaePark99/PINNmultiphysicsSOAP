"""
Multi-band Phonon BTE (DOM) — General N_B x N_s formulation
=============================================================
1D Steady-State, BGK, Discrete Ordinates (Gauss-Legendre), MMS

PDE count = N_B (bands) x N_s (directions)
Energy conservation (R_cons) is logged as diagnostic but NOT in loss
  (it is identically sum_d w_d R_{b,d}, i.e. linearly dependent on BTE residuals).

Physics (GiftBTE / Li et al.):
  For band b and ordinate d:
    v_{g,b} mu_d de_{b,d}/dx = [e_b^0(T_L) - e_{b,d}] / tau_b + f_{b,d}

  Lattice temperature (1/tau weighted):
    T_L = sum_b (1/tau_b) sum_d w_d e_{b,d}  /  sum_b C_b/tau_b

  Kn-based sweep:  tau_b = Kn * alpha_b / v_{g,b}

Usage:
  python phonon_bte_dom.py --n-bands 2 --n-dirs 2 --Kn 1.0    # 4 PDEs
  python phonon_bte_dom.py --n-bands 4 --n-dirs 4 --Kn 0.01   # 16 PDEs
  python phonon_bte_dom.py --n-bands 3 --n-dirs 2 --Kn 0.001  # 6 PDEs
"""

import sys, io, argparse, copy, os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PI = math.pi


# ═══════════════════════════════════════════════════════════════
# TeeLogger
# ═══════════════════════════════════════════════════════════════
class TeeLogger:
    def __init__(self):
        self._orig = sys.stdout; self._buf = io.StringIO(); sys.stdout = self
    def write(self, msg):
        self._orig.write(msg); self._buf.write(msg)
    def flush(self):
        self._orig.flush(); self._buf.flush()
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f: f.write(self._buf.getvalue())
    def close(self):
        sys.stdout = self._orig; self._buf.close()


# ═══════════════════════════════════════════════════════════════
# Arguments
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description='Phonon BTE DOM (general N_B x N_s)')

    # -- Physics --
    p.add_argument('--n-bands', type=int, default=2,
                   help='Number of frequency bands N_B')
    p.add_argument('--n-dirs', type=int, default=2,
                   help='Number of discrete ordinates N_s (Gauss-Legendre)')
    p.add_argument('--Kn', type=float, default=1.0,
                   help='Knudsen number (sweep param)')
    p.add_argument('--vg-base', type=float, default=1.0,
                   help='Base group velocity (band 0)')
    p.add_argument('--vg-spread', type=float, default=0.4,
                   help='Group velocity spread: vg_b = vg_base*(1 - spread*b/(B-1))')
    p.add_argument('--alpha-spread', type=float, default=1.0,
                   help='Relaxation time ratio spread: alpha_b = 1 + spread*b/(B-1)')

    # -- Training --
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--N-domain', type=int, default=200)

    # -- Architecture --
    p.add_argument('--n-hidden', type=int, default=4)
    p.add_argument('--n-neurons', type=int, default=64)

    # -- Optimizer --
    p.add_argument('--optimizer', choices=['adam', 'soap', 'gd'], default='adam')
    p.add_argument('--soap-beta1', type=float, default=0.99)
    p.add_argument('--soap-beta2', type=float, default=0.999)
    p.add_argument('--soap-precond-freq', type=int, default=2)

    # -- Surgery / Weighting --
    p.add_argument('--surgery', choices=['none', 'pcgrad'], default='none')
    p.add_argument('--weighting', choices=['none', 'lra', 'gradnorm'], default='none')
    p.add_argument('--lra-alpha', type=float, default=0.1)
    p.add_argument('--gn-update-freq', type=int, default=1000)
    p.add_argument('--gn-momentum', type=float, default=0.9)

    # -- Network mode --
    p.add_argument('--single-network', action='store_true')
    p.add_argument('--resample-every', type=int, default=0)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--checkpoint-freq', type=int, default=5000)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Band / direction generation
# ═══════════════════════════════════════════════════════════════
def build_bands(args):
    """Generate band properties: vg_b, C_b, alpha_b, tau_b."""
    NB = args.n_bands
    bands = []
    for b in range(NB):
        frac = b / max(NB - 1, 1)
        vg = args.vg_base * (1.0 - args.vg_spread * frac)
        C  = 1.0   # uniform heat capacity
        alpha = 1.0 + args.alpha_spread * frac
        tau = args.Kn * alpha / vg
        bands.append({'vg': vg, 'C': C, 'alpha': alpha, 'tau': tau})
    return bands


def build_dirs(args):
    """Gauss-Legendre quadrature on [-1, 1] with N_s points (even only)."""
    assert args.n_dirs % 2 == 0, (
        f"n_dirs must be even (got {args.n_dirs}): odd N_s produces mu=0 node "
        f"which degenerates the streaming term and breaks inflow BC logic.")
    mu, w = np.polynomial.legendre.leggauss(args.n_dirs)
    dirs = [{'mu': float(mu[d]), 'w': float(w[d])} for d in range(args.n_dirs)]
    return dirs


def field_name(b, d):
    return f'e{b}d{d}'


# ═══════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════
class SubNet(nn.Module):
    def __init__(self, n_hidden, n_neurons):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(2.0 * x - 1.0)


class MultiNet(nn.Module):
    def __init__(self, n_hidden, n_neurons, n_out):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(2.0 * x - 1.0)


# ═══════════════════════════════════════════════════════════════
# MMS exact solutions — auto-generated
# ═══════════════════════════════════════════════════════════════
def build_exact_fns(NB, NS):
    """Generate exact solution functions for each (b, d) pair.
    e_{b,d}(x) = A*sin(n*pi*x) + B*cos(m*pi*x) + offset
    Varied to ensure non-trivial coupling."""
    fns = {}
    for b in range(NB):
        for d in range(NS):
            idx = b * NS + d
            # Vary frequency, amplitude, phase across fields
            n_freq = 1 + (idx % 3)
            amp = 0.8 + 0.2 * (idx % 4)
            sign = 1.0 if (idx % 2 == 0) else -1.0
            offset = 2.0 + 0.5 * b
            phase = idx * 0.3

            n_f = n_freq
            a_val = sign * amp
            o_val = offset

            # closure capture
            def _make_fn(nf, av, ov, ph):
                def fn(x):
                    return av * torch.sin(nf * PI * x + ph) + ov
                return fn
            fns[field_name(b, d)] = _make_fn(n_f, a_val, o_val, phase)
    return fns


# ═══════════════════════════════════════════════════════════════
# Source computation
# ═══════════════════════════════════════════════════════════════
def compute_sources(x, bands, dirs, exact_fns, field_names):
    """MMS sources: f_{b,d} = streaming - collision, using exact solutions."""
    NB = len(bands); NS = len(dirs)

    # Evaluate exact fields and derivatives
    x_ad = x.detach().requires_grad_(True)
    vals = {}; derivs = {}
    for fn_name in field_names:
        v = exact_fns[fn_name](x_ad)
        dv = torch.autograd.grad(v, x_ad, torch.ones_like(v),
                                  create_graph=False)[0]
        vals[fn_name] = v.detach()
        derivs[fn_name] = dv.detach()

    # Lattice temperature (1/tau weighted)
    numer = torch.zeros_like(x)
    denom = 0.0
    for b in range(NB):
        tau_b = bands[b]['tau']
        C_b = bands[b]['C']
        e_sum = sum(dirs[d]['w'] * vals[field_name(b, d)] for d in range(NS))
        numer = numer + e_sum / tau_b
        denom += C_b / tau_b
    TL = numer / denom

    # Sources
    srcs = {}
    f_cons_sum = torch.zeros_like(x)
    for b in range(NB):
        tau_b = bands[b]['tau']; vg_b = bands[b]['vg']; C_b = bands[b]['C']
        eeq_b = C_b * TL / 2.0
        for d in range(NS):
            fn = field_name(b, d)
            mu_d = dirs[d]['mu']; w_d = dirs[d]['w']
            streaming = vg_b * mu_d * derivs[fn]
            collision = (eeq_b - vals[fn]) / tau_b
            srcs[fn] = (streaming - collision).detach()
            f_cons_sum = f_cons_sum + w_d * srcs[fn]

    # f_cons = sum w_d * f_{b,d}  (but from streaming part directly)
    f_cons = torch.zeros_like(x)
    for b in range(NB):
        vg_b = bands[b]['vg']
        for d in range(NS):
            mu_d = dirs[d]['mu']; w_d = dirs[d]['w']
            f_cons = f_cons + w_d * vg_b * mu_d * derivs[field_name(b, d)]
    f_cons = f_cons.detach()

    return srcs, f_cons


# ═══════════════════════════════════════════════════════════════
# Residuals
# ═══════════════════════════════════════════════════════════════
def compute_residuals(nets, x, srcs, f_cons, bands, dirs, field_names):
    NB = len(bands); NS = len(dirs)

    # Forward + derivatives
    vals = {}; derivs = {}
    for fn in field_names:
        v = nets[fn](x)
        dv = torch.autograd.grad(v, x, torch.ones_like(v),
                                  create_graph=True, retain_graph=True)[0]
        vals[fn] = v; derivs[fn] = dv

    # Lattice temperature
    numer = torch.zeros_like(x)
    denom = 0.0
    for b in range(NB):
        tau_b = bands[b]['tau']; C_b = bands[b]['C']
        e_sum = sum(dirs[d]['w'] * vals[field_name(b, d)] for d in range(NS))
        numer = numer + e_sum / tau_b
        denom += C_b / tau_b
    TL = numer / denom

    # Per-mode residuals
    losses = {}
    for b in range(NB):
        tau_b = bands[b]['tau']; vg_b = bands[b]['vg']; C_b = bands[b]['C']
        eeq_b = C_b * TL / 2.0
        for d in range(NS):
            fn = field_name(b, d)
            mu_d = dirs[d]['mu']
            R = vg_b * mu_d * derivs[fn] - (eeq_b - vals[fn]) / tau_b - srcs[fn]
            losses[f'R_{fn}'] = torch.mean(R**2)

    # Energy conservation
    div_q = torch.zeros_like(x)
    for b in range(NB):
        vg_b = bands[b]['vg']
        for d in range(NS):
            mu_d = dirs[d]['mu']; w_d = dirs[d]['w']
            div_q = div_q + w_d * vg_b * mu_d * derivs[field_name(b, d)]
    losses['R_cons'] = torch.mean((div_q - f_cons)**2)

    return losses


def compute_bc_loss(nets, exact_fns, field_names, dirs, NS, device):
    """Inflow-only BCs: mu_d > 0 -> Dirichlet at x=0, mu_d < 0 -> at x=1.
    1st-order advection only admits BC on the inflow boundary."""
    x0 = torch.zeros(1, 1, device=device)
    x1 = torch.ones(1, 1, device=device)
    bc = {}
    for i, fn in enumerate(field_names):
        mu_d = dirs[i % NS]['mu']
        xb = x0 if mu_d > 0 else x1   # inflow side only
        bc[f'BC_{fn}'] = ((nets[fn](xb) - exact_fns[fn](xb))**2).squeeze()
    return bc


# ═══════════════════════════════════════════════════════════════
# Gradient utilities
# ═══════════════════════════════════════════════════════════════
def get_grad_vec(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([g.reshape(-1) if g is not None
                      else torch.zeros_like(p).reshape(-1)
                      for g, p in zip(grads, params)])

def cosine_sim(g1, g2):
    n1, n2 = g1.norm(), g2.norm()
    return (g1 @ g2 / (n1 * n2)).item() if n1 > 1e-30 and n2 > 1e-30 else 0.0

def pcgrad_project(ga, gb):
    dot = ga @ gb
    if dot < 0: return ga - (dot / (gb @ gb + 1e-30)) * gb, True
    return ga, False


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════
def train(args):
    logger = TeeLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bands = build_bands(args)
    dirs  = build_dirs(args)
    NB, NS = args.n_bands, args.n_dirs
    N_fields = NB * NS
    N_pdes = N_fields  # independent BTE residuals only

    field_names = [field_name(b, d) for b in range(NB) for d in range(NS)]
    PDE_NAMES = [f'R_{fn}' for fn in field_names]   # loss terms
    BC_NAMES  = [f'BC_{fn}' for fn in field_names]
    ALL_LOSS_NAMES = PDE_NAMES + BC_NAMES
    # R_cons is linearly dependent (= sum w_d R_{b,d}), logged as diagnostic only
    DIAG_NAMES = ['R_cons']

    print(f"Device: {device}")
    print(f"=== Phonon BTE DOM: {NB} bands x {NS} dirs = {N_fields} fields, "
          f"{N_pdes} PDEs (R_cons as diagnostic) ===")
    print(f"  Kn={args.Kn}")
    for b, band in enumerate(bands):
        print(f"  Band {b}: vg={band['vg']:.3f}, C={band['C']:.2f}, "
              f"alpha={band['alpha']:.3f}, tau={band['tau']:.4e}")
    for d, dr in enumerate(dirs):
        print(f"  Dir {d}: mu={dr['mu']:+.4f}, w={dr['w']:.4f}")
    print(f"  Optimizer: {args.optimizer}, Weighting: {args.weighting}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # -- Exact solutions --
    exact_fns = build_exact_fns(NB, NS)

    # -- Networks --
    if args.single_network:
        shared = MultiNet(args.n_hidden, args.n_neurons, N_fields).to(device)
        class _V:
            def __init__(self, net, idx): self.net, self.idx = net, idx
            def __call__(self, x): return self.net(x)[:, self.idx:self.idx+1]
        nets = {fn: _V(shared, i) for i, fn in enumerate(field_names)}
        all_params = list(shared.parameters())
        print(f'  SingleNet: {sum(p.numel() for p in all_params):,} params, '
              f'{N_fields} outputs')
    else:
        nets = {fn: SubNet(args.n_hidden, args.n_neurons).to(device)
                for fn in field_names}
        all_params = []
        for fn in field_names: all_params.extend(list(nets[fn].parameters()))
        pp = sum(p.numel() for p in all_params)
        print(f'  IndepNets: {pp:,} params ({pp//N_fields:,} x {N_fields})')

    # -- Optimizer --
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(all_params, lr=args.lr)
    elif args.optimizer == 'soap':
        try:
            from soap import SOAP
            optimizer = SOAP(all_params, lr=args.lr,
                             betas=(args.soap_beta1, args.soap_beta2),
                             weight_decay=0.0,
                             precondition_frequency=args.soap_precond_freq,
                             precondition_1d=False)
        except ImportError:
            print("SOAP not found -> Adam"); optimizer = torch.optim.Adam(all_params, lr=args.lr)
            args.optimizer = 'adam'
    elif args.optimizer == 'gd':
        optimizer = torch.optim.SGD(all_params, lr=args.lr, momentum=0.0)

    # -- State --
    def _save_state():
        if args.single_network:
            return {'_model': copy.deepcopy(shared.state_dict())}
        return {fn: copy.deepcopy(nets[fn].state_dict()) for fn in field_names}
    def _load_state(s):
        if args.single_network: shared.load_state_dict(s['_model'])
        else:
            for fn in field_names: nets[fn].load_state_dict(s[fn])

    # -- Collocation --
    def _make_coll():
        if args.resample_every > 0:
            pts = torch.rand(args.N_domain, 1, device=device)*0.98 + 0.01
        else:
            pts = torch.linspace(0.01, 0.99, args.N_domain, device=device).reshape(-1,1)
        pts.requires_grad_(True); return pts

    x_int = _make_coll()
    srcs, f_cons = compute_sources(x_int, bands, dirs, exact_fns, field_names)

    # -- Monitoring --
    MONITOR_PAIRS = []
    if NB >= 2:
        MONITOR_PAIRS.append((f'R_{field_name(0,0)}', f'R_{field_name(1,0)}'))
    if NB >= 2 and NS >= 2:
        MONITOR_PAIRS.append((f'R_{field_name(0,0)}', f'R_{field_name(NB-1,NS-1)}'))

    loss_hist = {k: [] for k in ALL_LOSS_NAMES + DIAG_NAMES + ['total']}
    cosine_hist = {f'{a}-{b}': [] for a, b in MONITOR_PAIRS}
    neg_count   = {f'{a}-{b}': 0  for a, b in MONITOR_PAIRS}

    lra_weights = {n: 1.0 for n in ALL_LOSS_NAMES}
    lra_weight_hist = {n: [] for n in ALL_LOSS_NAMES}
    gn_weights  = {n: 1.0 for n in ALL_LOSS_NAMES}

    best_L2_avg = float('inf'); best_epoch = -1; best_state = None
    best_pde_total = float('inf'); best_pde_epoch = -1; best_pde_state = None
    L2_history = []

    x_eval_mon = torch.linspace(0, 1, 200, device=device).reshape(-1,1)
    with torch.no_grad():
        exact_mon = {fn: exact_fns[fn](x_eval_mon) for fn in field_names}

    t0 = time.time(); LOG_EVERY = 500

    tag = (f"phonon_Kn{args.Kn:.4g}_{NB}b{NS}d_s{args.seed}_"
           f"{args.surgery}_{args.weighting}_{args.optimizer}")
    if args.single_network: tag += '_sn'
    run_dir = os.path.join("runs", f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _load_state(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_state = ckpt.get('best_state'); best_L2_avg = ckpt.get('best_L2_avg', float('inf'))
        best_epoch = ckpt.get('best_epoch', -1)
        best_pde_state = ckpt.get('best_pde_state')
        best_pde_total = ckpt.get('best_pde_total', float('inf'))
        best_pde_epoch = ckpt.get('best_pde_epoch', -1)
        loss_hist = ckpt.get('loss_hist', loss_hist)
        L2_history = ckpt.get('L2_history', L2_history)
        neg_count = ckpt.get('neg_count', neg_count)
        print(f"  Resumed at epoch {start_epoch}")

    # ═══════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()

        if (args.resample_every > 0 and epoch > 1
                and (epoch-1) % args.resample_every == 0):
            x_int = _make_coll()
            srcs, f_cons = compute_sources(x_int, bands, dirs, exact_fns, field_names)

        pde_losses = compute_residuals(nets, x_int, srcs, f_cons, bands, dirs, field_names)
        bc_losses  = compute_bc_loss(nets, exact_fns, field_names, dirs, NS, device)

        t_back = time.time()

        # Determine if this epoch needs expensive per-loss gradients
        need_per_loss = (
            args.surgery == 'pcgrad' or
            args.weighting == 'lra' or
            (args.weighting == 'gradnorm' and
             (epoch % args.gn_update_freq == 1 or args.gn_update_freq == 1))
        )

        if need_per_loss:
            # === SLOW PATH: 16 backward passes (surgery/LRA/GN update) ===
            grads = {}
            for n in PDE_NAMES:
                grads[n] = get_grad_vec(pde_losses[n], all_params)
            for n in BC_NAMES:
                grads[n] = get_grad_vec(bc_losses[n], all_params)

            # Cosine monitoring (only when grads available)
            for a, b in MONITOR_PAIRS:
                k = f'{a}-{b}'
                cv = cosine_sim(grads[a], grads[b])
                cosine_hist[k].append(cv)
                if cv < 0: neg_count[k] += 1

            if args.surgery == 'pcgrad':
                pn = list(PDE_NAMES)
                for i in range(len(pn)):
                    for j in range(i+1, len(pn)):
                        grads[pn[i]], _ = pcgrad_project(grads[pn[i]], grads[pn[j]])
                        grads[pn[j]], _ = pcgrad_project(grads[pn[j]], grads[pn[i]])

            if args.weighting == 'lra':
                mg = {n: grads[n].abs().max().item() for n in ALL_LOSS_NAMES}
                mm = np.mean(list(mg.values()))
                for n in ALL_LOSS_NAMES:
                    lh = mm/mg[n] if mg[n] > 1e-30 else 1.0
                    lra_weights[n] = (1-args.lra_alpha)*lra_weights[n] + args.lra_alpha*lh
                for n in ALL_LOSS_NAMES: lra_weight_hist[n].append(lra_weights[n])
                g_total = sum(lra_weights[n]*grads[n] for n in ALL_LOSS_NAMES)
            elif args.weighting == 'gradnorm':
                l2n = {n: grads[n].norm().item() for n in ALL_LOSS_NAMES}
                mn = np.mean(list(l2n.values()))
                for n in ALL_LOSS_NAMES:
                    gh = mn/l2n[n] if l2n[n] > 1e-30 else 1.0
                    gn_weights[n] = args.gn_momentum*gn_weights[n] + (1-args.gn_momentum)*gh
                for n in ALL_LOSS_NAMES: lra_weight_hist[n].append(gn_weights[n])
                g_total = sum(gn_weights[n]*grads[n] for n in ALL_LOSS_NAMES)
            else:
                g_total = sum(grads[n] for n in ALL_LOSS_NAMES)

            idx = 0
            for p in all_params:
                nu = p.numel()
                p.grad = g_total[idx:idx+nu].reshape(p.shape).clone()
                idx += nu

        else:
            # === FAST PATH: single backward pass ===
            if args.weighting == 'gradnorm':
                weighted_loss = sum(gn_weights[n] * pde_losses[n] for n in PDE_NAMES) + \
                                sum(gn_weights[n] * bc_losses[n] for n in BC_NAMES)
                for n in ALL_LOSS_NAMES: lra_weight_hist[n].append(gn_weights[n])
            else:
                weighted_loss = sum(pde_losses[n] for n in PDE_NAMES) + \
                                sum(bc_losses[n] for n in BC_NAMES)
            weighted_loss.backward()

        optimizer.step()

        # L2 eval
        with torch.no_grad():
            l2f = {}
            for fn in field_names:
                pr = nets[fn](x_eval_mon); ref = exact_mon[fn]
                l2f[fn] = (torch.norm(pr-ref)/torch.norm(ref)).item()
            l2_avg = np.mean(list(l2f.values()))
        L2_history.append({**l2f, 'avg': l2_avg, 'epoch': epoch})

        if l2_avg < best_L2_avg:
            best_L2_avg = l2_avg; best_epoch = epoch; best_state = _save_state()

        pde_sum = sum(pde_losses[n].item() for n in PDE_NAMES)
        bc_sum  = sum(bc_losses[n].item() for n in BC_NAMES)
        total_loss = pde_sum + bc_sum   # R_cons excluded (dependent)
        for n in PDE_NAMES: loss_hist[n].append(pde_losses[n].item())
        for n in BC_NAMES:  loss_hist[n].append(bc_losses[n].item())
        loss_hist['R_cons'].append(pde_losses['R_cons'].item())  # diagnostic only
        loss_hist['total'].append(total_loss)

        if total_loss < best_pde_total:
            best_pde_total = total_loss; best_pde_epoch = epoch
            best_pde_state = _save_state()

        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            torch.save({'epoch': epoch, 'model': _save_state(),
                        'optimizer': optimizer.state_dict(),
                        'best_state': best_state, 'best_L2_avg': best_L2_avg,
                        'best_epoch': best_epoch,
                        'best_pde_state': best_pde_state,
                        'best_pde_total': best_pde_total,
                        'best_pde_epoch': best_pde_epoch,
                        'loss_hist': loss_hist, 'L2_history': L2_history,
                        'neg_count': neg_count},
                       os.path.join(run_dir, f"ckpt_ep{epoch}.pt"))

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == args.epochs:
            elapsed = (time.time() - t0) / 60
            cons_v = pde_losses['R_cons'].item()
            bte_sum = sum(pde_losses[f'R_{fn}'].item() for fn in field_names)
            print(f"[{epoch:>6}/{args.epochs}] ({elapsed:5.1f}min) "
                  f"Total={total_loss:.3E}  BTE={bte_sum:.2E}  "
                  f"cons={cons_v:.1E}  BC={bc_sum:.1E}")
            print(f"        L2_avg={l2_avg:.3e}  "
                  f"(best={best_L2_avg:.3e} @ ep {best_epoch})")

    # ═══════════════════════════════════════════════════════════
    # Final eval
    # ═══════════════════════════════════════════════════════════
    _load_state(best_state)
    x_eval = torch.linspace(0, 1, 500, device=device).reshape(-1,1)
    with torch.no_grad():
        exacts_ev = {fn: exact_fns[fn](x_eval) for fn in field_names}
        l2 = {fn: (torch.norm(nets[fn](x_eval)-exacts_ev[fn]) /
                    torch.norm(exacts_ev[fn])).item() for fn in field_names}
        l2_avg_best = np.mean(list(l2.values()))

    _load_state(best_pde_state)
    with torch.no_grad():
        l2_pde = {fn: (torch.norm(nets[fn](x_eval)-exacts_ev[fn]) /
                        torch.norm(exacts_ev[fn])).item() for fn in field_names}
        l2_avg_pde = np.mean(list(l2_pde.values()))

    final_l2 = L2_history[-1]['avg']
    ratio = final_l2 / (best_L2_avg + 1e-30)

    # Lattice temperature
    _load_state(best_state)
    with torch.no_grad():
        numer_ex = torch.zeros_like(x_eval); numer_pr = torch.zeros_like(x_eval)
        denom_val = 0.0
        for b in range(NB):
            tau_b = bands[b]['tau']; C_b = bands[b]['C']
            for d in range(NS):
                fn = field_name(b, d); w_d = dirs[d]['w']
                numer_ex += w_d * exact_fns[fn](x_eval) / tau_b
                numer_pr += w_d * nets[fn](x_eval) / tau_b
            denom_val += C_b / tau_b
        TL_ex = numer_ex / denom_val; TL_pr = numer_pr / denom_val
        l2_TL = (torch.norm(TL_pr - TL_ex) / torch.norm(TL_ex)).item()

    print("\n" + "="*70)
    print(f"RESULTS (Kn={args.Kn}, {NB}b x {NS}d = {N_pdes} PDEs, "
          f"{args.optimizer}+{args.weighting})")
    print("="*70)
    print(f"  Oracle (ep {best_epoch}): L2_avg={l2_avg_best:.4e}, L2_TL={l2_TL:.4e}")
    print(f"  PDE-best (ep {best_pde_epoch}): L2_avg={l2_avg_pde:.4e}")
    print(f"  Final: L2_avg={final_l2:.4e}, ratio={ratio:.2f}x")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    for a, b in MONITOR_PAIRS:
        k = f'{a}-{b}'
        print(f"  Neg: {k}={100*neg_count[k]/args.epochs:.1f}%")

    # Save
    results = {'L2_best': l2, 'L2_avg_best': l2_avg_best, 'L2_TL_best': l2_TL,
               'best_epoch': best_epoch,
               'L2_pde_best': l2_pde, 'L2_avg_pde_best': l2_avg_pde,
               'best_pde_epoch': best_pde_epoch, 'best_pde_total': best_pde_total,
               'L2_avg_final': final_l2, 'ratio': ratio,
               'neg_count': neg_count}
    config = {'Kn': args.Kn, 'n_bands': NB, 'n_dirs': NS,
              'n_pdes': N_pdes, 'n_fields': N_fields,
              'bands': [{'vg': b['vg'], 'C': b['C'], 'alpha': b['alpha'],
                         'tau': b['tau']} for b in bands],
              'dirs': [{'mu': d['mu'], 'w': d['w']} for d in dirs],
              'optimizer': args.optimizer, 'surgery': args.surgery,
              'weighting': args.weighting, 'seed': args.seed,
              'epochs': args.epochs, 'lr': args.lr,
              'N_domain': args.N_domain,
              'single_network': args.single_network}
    if args.weighting in ('gradnorm','lra'):
        w = gn_weights if args.weighting == 'gradnorm' else lra_weights
        results['final_weights'] = {n: w[n] for n in ALL_LOSS_NAMES}
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)

    # Curves
    SE = 100; nt = len(loss_hist['total'])
    si = list(range(0, nt, SE))
    if (nt-1) not in si: si.append(nt-1)
    curves = {'epochs': np.array([i+1 for i in si])}
    curves['total'] = np.array([loss_hist['total'][i] for i in si])
    curves['R_cons'] = np.array([loss_hist['R_cons'][i] for i in si])
    for fn in field_names:
        curves[f'L2_{fn}'] = np.array([L2_history[i][fn] for i in si])
    curves['L2_avg'] = np.array([L2_history[i]['avg'] for i in si])
    np.savez(os.path.join(run_dir, 'loss_curves.npz'), **curves)

    # Plot
    _load_state(best_state)
    try:
        n_plot = min(N_fields, 4)
        fig, axes = plt.subplots(2, max(n_plot, 2), figsize=(5*max(n_plot,2), 9))
        fig.suptitle(f"Phonon BTE DOM | Kn={args.Kn} | {NB}b x {NS}d = {N_pdes} PDEs | "
                     f"{args.optimizer}+{args.weighting}", fontsize=13, fontweight='bold')
        xx = x_eval.cpu().numpy().ravel()
        with torch.no_grad():
            for col in range(n_plot):
                fn = field_names[col]
                ax = axes[0, col]
                ax.plot(xx, exacts_ev[fn].cpu().numpy().ravel(), 'k-', lw=2, label='Exact')
                ax.plot(xx, nets[fn](x_eval).cpu().numpy().ravel(), 'r--', lw=1.5,
                        label=f'PINN ({l2[fn]:.1e})')
                ax.set_title(fn); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Bottom: TL, L2, total loss, pointwise err
        ax = axes[1, 0]
        ax.plot(xx, TL_ex.cpu().numpy().ravel(), 'k-', lw=2, label='Exact')
        ax.plot(xx, TL_pr.cpu().numpy().ravel(), 'r--', lw=1.5,
                label=f'PINN ({l2_TL:.1e})')
        ax.set_title('$T_L$'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ep = curves['epochs']
        ax = axes[1, 1]
        ax.semilogy(ep, curves['L2_avg'], 'k-', lw=1.5)
        ax.axhline(l2_avg_best, color='r', ls='--', lw=1, label=f'best={l2_avg_best:.2e}')
        ax.set_title('L2_avg'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        if n_plot >= 3:
            ax = axes[1, 2]
            ax.semilogy(ep, curves['total'], 'k-', lw=1)
            ax.semilogy(ep, curves['R_cons'], 'r-', lw=1, label='cons')
            ax.set_title('Loss'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        if n_plot >= 4:
            ax = axes[1, 3]
            with torch.no_grad():
                for fn in field_names[:4]:
                    err = (nets[fn](x_eval)-exacts_ev[fn]).abs().cpu().numpy().ravel()
                    ax.semilogy(xx, err, lw=1, label=fn)
            ax.set_title('Error'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'summary.png'), dpi=150)
        plt.close()
        print(f"  Plot: {run_dir}/summary.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"  Saved to: {run_dir}/")
    logger.save(os.path.join(run_dir, 'training_log.txt'))
    logger.close()
    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
