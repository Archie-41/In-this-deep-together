"""
Full 3D Gross-Pitaevskii simulation for a Rubidium-87 Bose-Einstein condensate.

Stages modeled:
1) Compute trapped 3D ground state via imaginary-time evolution
2) Imprint a gray soliton along x
3) Evolve in real time with trap ON
4) Switch OFF all traps and evolve interacting 12 ms TOF
5) Output 2D column density integrated along z

Numerics:
- Split-step Fourier method (spectral) in 3D
- SI units throughout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


HBAR = 1.054_571_817e-34  # J*s
PI = np.pi


@dataclass
class SimParams:
    # Atomic properties (Rb-87)
    mass: float = 1.443e-25  # kg
    a_s: float = 5.3e-9  # m

    # Trap frequencies (rad/s)
    omega_x: float = 2.0 * PI * 9.1
    omega_y: float = 2.0 * PI * 94.5
    omega_z: float = 2.0 * PI * 153.0

    # Atom number
    n_atoms: float = 2.4e5

    # Spatial grid in x-y-z
    nx: int = 192
    ny: int = 96
    nz: int = 96
    x_max: float = 2.2e-4  # m; domain is [-x_max, x_max)
    y_max: float = 4.2e-5  # m; domain is [-y_max, y_max)
    z_max: float = 3.8e-5  # m; domain is [-z_max, z_max)

    # Time stepping
    dt_real: float = 4.0e-6  # s
    dt_imag: float = 2.5e-6  # s
    n_imag_steps: int = 900
    imag_tol: float = 6.0e-8
    imag_check_every: int = 30

    # Soliton imprint
    delta_phi: float = 1.7 * PI

    # Real-time schedule
    t_trap_before_release: float = 6.0e-3  # s
    t_extra_trap: float = 2.0e-3  # s
    t_tof: float = 12.0e-3  # s


def make_grids(
    p: SimParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    lx = 2.0 * p.x_max
    ly = 2.0 * p.y_max
    lz = 2.0 * p.z_max

    dx = lx / p.nx
    dy = ly / p.ny
    dz = lz / p.nz

    x = np.linspace(-p.x_max, p.x_max - dx, p.nx)
    y = np.linspace(-p.y_max, p.y_max - dy, p.ny)
    z = np.linspace(-p.z_max, p.z_max - dz, p.nz)

    kx = 2.0 * PI * np.fft.fftfreq(p.nx, d=dx)
    ky = 2.0 * PI * np.fft.fftfreq(p.ny, d=dy)
    kz = 2.0 * PI * np.fft.fftfreq(p.nz, d=dz)

    kx2 = kx[:, None, None] ** 2
    ky2 = ky[None, :, None] ** 2
    kz2 = kz[None, None, :] ** 2
    ek = (HBAR**2) * (kx2 + ky2 + kz2) / (2.0 * p.mass)

    return x, y, z, dx, dy, dz, ek


def g_3d(p: SimParams) -> float:
    return 4.0 * PI * (HBAR**2) * p.a_s / p.mass


def trap_potential(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    p: SimParams,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return np.zeros((x.size, y.size, z.size), dtype=np.float64)

    xx = x[:, None, None]
    yy = y[None, :, None]
    zz = z[None, None, :]

    vx = 0.5 * p.mass * (p.omega_x**2) * (xx**2)
    vy = 0.5 * p.mass * (p.omega_y**2) * (yy**2)
    vz = 0.5 * p.mass * (p.omega_z**2) * (zz**2)
    return vx + vy + vz


def normalize_to_atoms(psi: np.ndarray, dx: float, dy: float, dz: float, n_atoms: float) -> np.ndarray:
    current_n = np.sum(np.abs(psi) ** 2) * dx * dy * dz
    if current_n <= 0.0:
        raise ValueError("Wavefunction norm vanished during evolution.")
    return psi * np.sqrt(n_atoms / current_n)


def tf_chemical_potential_3d_harmonic(
    n_atoms: float,
    g3d: float,
    mass: float,
    omega_x: float,
    omega_y: float,
    omega_z: float,
) -> float:
    # 3D TF normalization in harmonic trap:
    # N = (8*pi*2^(3/2)/15) * mu^(5/2) / (g3d * m^(3/2) * omega_x*omega_y*omega_z)
    pref = (15.0 * g3d * n_atoms * (mass**1.5) * omega_x * omega_y * omega_z) / (16.0 * PI * np.sqrt(2.0))
    return pref ** (2.0 / 5.0)


def initial_thomas_fermi_state(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    p: SimParams,
    g3d_val: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    mu_tf = tf_chemical_potential_3d_harmonic(
        p.n_atoms,
        g3d_val,
        p.mass,
        p.omega_x,
        p.omega_y,
        p.omega_z,
    )

    v = trap_potential(x, y, z, p, enabled=True)
    n_tf = np.maximum((mu_tf - v) / g3d_val, 0.0)
    if np.all(n_tf <= 0.0):
        raise ValueError("Thomas-Fermi initialization failed: zero density everywhere.")

    psi_tf = np.sqrt(n_tf).astype(np.complex128)
    psi_tf = normalize_to_atoms(psi_tf, dx, dy, dz, p.n_atoms)

    r_tf_x = np.sqrt(2.0 * mu_tf / (p.mass * p.omega_x**2))
    r_tf_y = np.sqrt(2.0 * mu_tf / (p.mass * p.omega_y**2))
    r_tf_z = np.sqrt(2.0 * mu_tf / (p.mass * p.omega_z**2))

    return psi_tf, mu_tf, r_tf_x, r_tf_y, r_tf_z


def ground_state_imaginary_time(
    psi0: np.ndarray,
    v_trap: np.ndarray,
    ek: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    p: SimParams,
    g3d_val: float,
) -> np.ndarray:
    psi = psi0.copy()
    exp_k = np.exp(-ek * p.dt_imag / HBAR)

    check_every = max(1, int(p.imag_check_every))
    psi_prev = psi.copy()

    for step in range(p.n_imag_steps):
        n = np.abs(psi) ** 2
        psi *= np.exp(-(v_trap + g3d_val * n) * (p.dt_imag / (2.0 * HBAR)))

        psi_k = np.fft.fftn(psi)
        psi_k *= exp_k
        psi = np.fft.ifftn(psi_k)

        n = np.abs(psi) ** 2
        psi *= np.exp(-(v_trap + g3d_val * n) * (p.dt_imag / (2.0 * HBAR)))
        psi = normalize_to_atoms(psi, dx, dy, dz, p.n_atoms)

        if (step + 1) % check_every == 0:
            rel_change = np.linalg.norm(psi - psi_prev) / np.linalg.norm(psi)
            if rel_change < p.imag_tol:
                print(
                    "Imaginary-time convergence reached at "
                    f"step {step + 1} (relative change {rel_change:.2e})."
                )
                break
            psi_prev = psi.copy()

    return psi


def imprint_gray_soliton(
    psi: np.ndarray,
    x: np.ndarray,
    delta_phi: float,
    xi_center: float,
    dx: float,
    dy: float,
    dz: float,
    n_atoms: float,
) -> Tuple[np.ndarray, float]:
    # Gray-soliton relation: Delta phi = 2 arccos(v/c).
    v_over_c = np.cos(0.5 * delta_phi)
    beta = np.sqrt(max(1.0 - v_over_c**2, 1e-10))

    width = xi_center / beta
    width = max(width, 2.0 * dx)

    xx = x[:, None, None]
    s = xx / width

    phase = 0.5 * delta_phi * np.tanh(s)
    amp = np.sqrt(v_over_c**2 + beta**2 * np.tanh(s) ** 2)

    psi_s = psi * amp * np.exp(1j * phase)
    psi_s = normalize_to_atoms(psi_s, dx, dy, dz, n_atoms)
    return psi_s, width


def evolve_real_time_with_snapshots(
    psi0: np.ndarray,
    v: np.ndarray,
    ek: np.ndarray,
    g3d_val: float,
    dt: float,
    t_final: float,
    snapshot_times: Iterable[float],
) -> Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    psi = psi0.copy()
    exp_k = np.exp(-1j * ek * dt / HBAR)

    snapshots_ncol: Dict[float, np.ndarray] = {}
    snapshots_phase_xy: Dict[float, np.ndarray] = {}

    targets = np.array(sorted(snapshot_times), dtype=float)
    target_idx = 0

    z_mid = psi.shape[2] // 2

    t = 0.0
    n_steps = int(np.round(t_final / dt))

    for step in range(n_steps + 1):
        if target_idx < len(targets) and t >= targets[target_idx] - 0.5 * dt:
            n3d = np.abs(psi) ** 2
            snapshots_ncol[targets[target_idx]] = n3d.sum(axis=2)
            snapshots_phase_xy[targets[target_idx]] = np.angle(psi[:, :, z_mid])
            target_idx += 1

        if step == n_steps:
            break

        n = np.abs(psi) ** 2
        psi *= np.exp(-1j * (v + g3d_val * n) * (dt / (2.0 * HBAR)))

        psi_k = np.fft.fftn(psi)
        psi_k *= exp_k
        psi = np.fft.ifftn(psi_k)

        n = np.abs(psi) ** 2
        psi *= np.exp(-1j * (v + g3d_val * n) * (dt / (2.0 * HBAR)))

        t += dt

    return psi, snapshots_ncol, snapshots_phase_xy


def unwrap_phase_2d_local(phase_2d: np.ndarray) -> np.ndarray:
    return np.unwrap(np.unwrap(phase_2d, axis=0), axis=1)


def diagnostics(
    p: SimParams,
    dx: float,
    dy: float,
    dz: float,
    psi_gs: np.ndarray,
    g3d_val: float,
    mu_tf: float,
    r_tf_x: float,
    r_tf_y: float,
    r_tf_z: float,
) -> float:
    n0 = np.max(np.abs(psi_gs) ** 2)
    mu_center = g3d_val * n0
    xi = HBAR / np.sqrt(2.0 * p.mass * mu_center)

    kx_max = PI / dx
    ky_max = PI / dy
    kz_max = PI / dz
    omega_k_max = HBAR * (kx_max**2 + ky_max**2 + kz_max**2) / (2.0 * p.mass)
    kinetic_phase_step = omega_k_max * p.dt_real

    points_per_xi_x = xi / dx
    points_per_xi_y = xi / dy
    points_per_xi_z = xi / dz

    print("=== Numerical/physical diagnostics (3D) ===")
    print(f"N = {p.n_atoms:.3e}")
    print(f"g_3D = {g3d_val:.3e} J*m^3")
    print(f"TF chemical potential (analytic) = {mu_tf:.3e} J")
    print(
        f"TF radii: R_TF,x = {r_tf_x*1e6:.2f} um, "
        f"R_TF,y = {r_tf_y*1e6:.2f} um, R_TF,z = {r_tf_z*1e6:.2f} um"
    )
    print(f"dx = {dx*1e6:.3f} um, dy = {dy*1e6:.3f} um, dz = {dz*1e6:.3f} um")
    print(f"Center healing length xi_0 ~ {xi*1e6:.3f} um")
    print(
        f"xi_0/dx = {points_per_xi_x:.2f}, "
        f"xi_0/dy = {points_per_xi_y:.2f}, xi_0/dz = {points_per_xi_z:.2f}"
    )
    print(f"Max kinetic phase per step ~ {kinetic_phase_step:.3f} rad")
    print("==========================================")

    if min(points_per_xi_x, points_per_xi_y, points_per_xi_z) < 2.0:
        print("Warning: healing length is under-resolved in at least one direction.")
    elif min(points_per_xi_x, points_per_xi_y, points_per_xi_z) < 3.5:
        print("Warning: healing length is only moderately resolved; finer grid improves accuracy.")

    if kinetic_phase_step > 2.0:
        raise RuntimeError(
            "Time step too large for current spectral cutoff (max kinetic phase > 2.0 rad). "
            "Reduce dt_real or use a coarser grid."
        )
    if kinetic_phase_step > 1.2:
        print("Warning: kinetic phase step is relatively large; reducing dt_real improves fidelity.")

    return xi


def build_tf_column_density(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mu_tf: float,
    p: SimParams,
    g3d_val: float,
    dz: float,
) -> np.ndarray:
    v = trap_potential(x, y, z, p, enabled=True)
    n_tf_3d = np.maximum((mu_tf - v) / g3d_val, 0.0)
    return n_tf_3d.sum(axis=2) * dz


def plot_density_images(
    x: np.ndarray,
    y: np.ndarray,
    snapshots_ncol: Dict[float, np.ndarray],
    dz: float,
    title_prefix: str,
    cmap: str = "magma",
) -> None:
    times = sorted(snapshots_ncol.keys())
    n_cols = len(times)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.3 * n_cols, 4.5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]

    for ax, t in zip(axes, times):
        n_col = snapshots_ncol[t] * dz
        im = ax.imshow(
            n_col.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(f"{title_prefix}\nt = {t*1e3:.1f} ms")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, label=r"$n_{col}(x,y)$ (atoms/m$^2$)")

    plt.show()


def plot_delta_density(
    x: np.ndarray,
    y: np.ndarray,
    n_col: np.ndarray,
    n_tf_col: np.ndarray,
    title: str,
) -> None:
    delta_n = n_col - n_tf_col
    vmax = np.max(np.abs(delta_n))
    if vmax <= 0.0:
        vmax = 1.0

    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
    fig, ax = plt.subplots(figsize=(6.7, 4.9), constrained_layout=True)
    im = ax.imshow(
        delta_n.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax, label=r"$\delta n = n_{col} - n_{TF,col}$ (atoms/m$^2$)")
    plt.show()


def plot_phase_images(
    x: np.ndarray,
    y: np.ndarray,
    snapshots_phase: Dict[float, np.ndarray],
    title_prefix: str,
) -> None:
    times = sorted(snapshots_phase.keys())
    n_cols = len(times)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.3 * n_cols, 4.5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]

    for ax, t in zip(axes, times):
        ph = unwrap_phase_2d_local(snapshots_phase[t])
        im = ax.imshow(
            ph.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="twilight",
        )
        ax.set_title(f"{title_prefix}\nt = {t*1e3:.1f} ms")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, label="Phase (rad)")

    plt.show()


def main() -> None:
    p = SimParams()

    x, y, z, dx, dy, dz, ek = make_grids(p)
    g3d_val = g_3d(p)

    print("Preparing 3D Thomas-Fermi trapped profile...")
    psi_init, mu_tf, r_tf_x, r_tf_y, r_tf_z = initial_thomas_fermi_state(
        x,
        y,
        z,
        dx,
        dy,
        dz,
        p,
        g3d_val,
    )

    print("Computing trapped 3D ground state via imaginary-time evolution...")
    v_trap = trap_potential(x, y, z, p, enabled=True)
    psi_gs = ground_state_imaginary_time(psi_init, v_trap, ek, dx, dy, dz, p, g3d_val)
    xi_center = diagnostics(p, dx, dy, dz, psi_gs, g3d_val, mu_tf, r_tf_x, r_tf_y, r_tf_z)

    print("Imprinting smooth gray-soliton phase profile along x...")
    psi_soliton, imprint_width = imprint_gray_soliton(
        psi_gs,
        x,
        p.delta_phi,
        xi_center,
        dx,
        dy,
        dz,
        p.n_atoms,
    )
    print(f"Imprint width set by center healing length: {imprint_width*1e6:.3f} um")

    trap_snap_times = [0.0, 2.0e-3, 4.0e-3, p.t_trap_before_release]
    print("Evolving 3D dynamics with trap ON...")
    psi_before_release, trap_ncol, trap_phase = evolve_real_time_with_snapshots(
        psi_soliton,
        v_trap,
        ek,
        g3d_val,
        p.dt_real,
        p.t_trap_before_release,
        snapshot_times=trap_snap_times,
    )

    if p.t_extra_trap > 0.0:
        print("Optional additional in-trap evolution before release...")
        psi_before_release, _, _ = evolve_real_time_with_snapshots(
            psi_before_release,
            v_trap,
            ek,
            g3d_val,
            p.dt_real,
            p.t_extra_trap,
            snapshot_times=[],
        )

    tof_snap_times = [0.0, 4.0e-3, 8.0e-3, p.t_tof]
    print("Switching OFF all traps and evolving interacting TOF...")
    v_off = trap_potential(x, y, z, p, enabled=False)
    _, tof_ncol, tof_phase = evolve_real_time_with_snapshots(
        psi_before_release,
        v_off,
        ek,
        g3d_val,
        p.dt_real,
        p.t_tof,
        snapshot_times=tof_snap_times,
    )

    print("Plotting 2D z-integrated column densities...")
    plot_density_images(x, y, trap_ncol, dz, title_prefix="In-trap column density")
    plot_density_images(x, y, tof_ncol, dz, title_prefix="TOF column density")

    n_tf_col = build_tf_column_density(x, y, z, mu_tf, p, g3d_val, dz)
    plot_delta_density(
        x,
        y,
        trap_ncol[p.t_trap_before_release] * dz,
        n_tf_col,
        title=r"$\delta n(x,y)$ at release: in-trap column density - TF",
    )
    plot_delta_density(
        x,
        y,
        tof_ncol[p.t_tof] * dz,
        n_tf_col,
        title=r"$\delta n(x,y)$ after 12 ms TOF: column density - TF",
    )

    phase_subset_trap = {
        trap_snap_times[0]: trap_phase[trap_snap_times[0]],
        trap_snap_times[-1]: trap_phase[trap_snap_times[-1]],
    }
    phase_subset_tof = {
        tof_snap_times[0]: tof_phase[tof_snap_times[0]],
        tof_snap_times[-1]: tof_phase[tof_snap_times[-1]],
    }

    plot_phase_images(x, y, phase_subset_trap, title_prefix="Phase in trap (z=0 slice)")
    plot_phase_images(x, y, phase_subset_tof, title_prefix="Phase in TOF (z=0 slice)")


if __name__ == "__main__":
    main()
