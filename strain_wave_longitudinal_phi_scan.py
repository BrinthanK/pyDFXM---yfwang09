import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import forward_model as fwd


def load_profile_file(path):
    path = Path(path)
    try:
        profile = np.loadtxt(path, dtype=float)
        if profile.ndim == 2 and profile.shape[1] >= 2:
            return profile[:, 0], profile[:, 1]
    except Exception:
        pass

    profile = np.genfromtxt(path, delimiter=",", names=True)
    names = profile.dtype.names
    return np.asarray(profile[names[0]], dtype=float), np.asarray(profile[names[1]], dtype=float)


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    return matrix / norms[:, None]


def build_forward_dict(Ug, hkl, two_theta_deg, mu_deg, omega_deg, pixel_size_nm, beam_thickness_um, npixels, nrays):
    forward_dict = fwd.default_forward_dict()
    Ug_norm = normalize_rows(Ug)
    forward_dict["Ug"] = Ug_norm
    forward_dict["x_c"] = Ug_norm[0]
    forward_dict["y_c"] = Ug_norm[1]
    forward_dict["hkl"] = np.asarray(hkl, dtype=float)
    forward_dict["two_theta"] = float(two_theta_deg)
    forward_dict["theta"] = np.deg2rad(two_theta_deg / 2.0)
    forward_dict["mu"] = np.deg2rad(mu_deg)
    forward_dict["omega"] = np.deg2rad(omega_deg)
    forward_dict["psize"] = pixel_size_nm * 1e-9
    forward_dict["zl_rms"] = beam_thickness_um * 1e-6 / 2.35
    forward_dict["Npixels"] = list(npixels)
    forward_dict["Nrays"] = int(nrays)
    return forward_dict


def make_longitudinal_profile_Fg(x_profile, eps_profile, propagation_direction_g, face_point_g=None, anchor_to_sample_face=True):
    n_g = np.asarray(propagation_direction_g, dtype=float)
    n_g = n_g / np.linalg.norm(n_g)
    projector = np.outer(n_g, n_g)
    if face_point_g is not None:
        face_point_g = np.asarray(face_point_g, dtype=float)
        face_offset = float(np.dot(n_g, face_point_g))
    else:
        face_offset = None

    def profile_Fg(xg, yg, zg):
        s_raw = n_g[0] * xg + n_g[1] * yg + n_g[2] * zg
        if face_offset is not None:
            s = s_raw - face_offset
        elif anchor_to_sample_face:
            s = s_raw - np.min(s_raw)
        else:
            s = s_raw

        eps = np.interp(s, x_profile, eps_profile, left=0.0, right=0.0)
        shape = xg.shape + (3, 3)
        Fg = np.broadcast_to(np.eye(3), shape).copy()
        Fg += eps[..., None, None] * projector
        return Fg

    return profile_Fg


def identity_Fg(xg, yg, zg):
    shape = xg.shape + (3, 3)
    Fg = np.zeros(shape, dtype=float)
    Fg[..., 0, 0] = 1.0
    Fg[..., 1, 1] = 1.0
    Fg[..., 2, 2] = 1.0
    return Fg


def plot_input_profile(x_profile_m, eps_profile, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_profile_m * 1e6, eps_profile, lw=2)
    ax.set_xlabel("distance from sample face along propagation direction (um)")
    ax.set_ylabel("strain")
    ax.set_title("Applied longitudinal strain profile")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


save_dir = Path("data")
save_dir.mkdir(parents=True, exist_ok=True)

strain_profile_file = Path("strain_profile_100ps_100nm.csv")
save_file = save_dir / "strain_wave_longitudinal_phi_scan.pkl"
res_fn_file = save_dir / "Res_qi_strain_wave_longitudinal.npz"
profile_plot_file = save_dir / "strain_wave_longitudinal_input_profile.png"

Ug = np.array(
    [
        [1, 1, 1],
        [1, 0, -1],
        [-1, 2, -1],
    ],
    dtype=float,
)
hkl = np.array([-1, 1, -1], dtype=float)
propagation_direction_g = np.array([1, 1, 1], dtype=float)

two_theta_deg = 35.04
mu_deg = 1.95
omega_deg = 0.0
pixel_size_nm = 40.0
beam_thickness_um = 1.7 * 2.35 / 6
npixels = (200, 5, 50)
nrays = 5_000_000
phi_scan_deg = (-0.03, 0.03, 21)
wave_start_depth_um = 0.0
strain_face_point_g = None

x_profile_raw, eps_profile_raw = load_profile_file(strain_profile_file)
x_profile_m = np.asarray(x_profile_raw, dtype=float) * 1e-9
eps_profile = np.asarray(eps_profile_raw, dtype=float)
order = np.argsort(x_profile_m)
x_profile_m = x_profile_m[order]
eps_profile = eps_profile[order]
x_profile_m_applied = x_profile_m + wave_start_depth_um * 1e-6

print(f"Loaded {x_profile_m.size} strain-profile samples from {strain_profile_file}")
print(f"Applied depth range: {x_profile_m_applied.min() * 1e6:.6f} to {x_profile_m_applied.max() * 1e6:.6f} um")
print(f"Strain range: {eps_profile.min():.6e} to {eps_profile.max():.6e}")
print(f"Propagation direction in grain frame: {propagation_direction_g}")
print(f"Reference output file: {save_file}")

plot_input_profile(x_profile_m_applied, eps_profile, profile_plot_file)
print(f"Saved input profile plot to {profile_plot_file}")

forward_dict = build_forward_dict(
    Ug=Ug,
    hkl=hkl,
    two_theta_deg=two_theta_deg,
    mu_deg=mu_deg,
    omega_deg=omega_deg,
    pixel_size_nm=pixel_size_nm,
    beam_thickness_um=beam_thickness_um,
    npixels=npixels,
    nrays=nrays,
)

Fg_func = make_longitudinal_profile_Fg(
    x_profile_m_applied,
    eps_profile,
    propagation_direction_g,
    face_point_g=strain_face_point_g,
    anchor_to_sample_face=True,
)

phi_values_deg = np.linspace(phi_scan_deg[0], phi_scan_deg[1], int(phi_scan_deg[2]))
phi_values_rad = np.deg2rad(phi_values_deg)

im_stack = []
im_ref_stack = []
rulers_last = None

for idx, phi_rad in enumerate(phi_values_rad):
    print(f"Running phi step {idx + 1}/{len(phi_values_rad)} at {np.rad2deg(phi_rad):+.6f} deg")
    forward_dict["phi"] = float(phi_rad)
    model = fwd.DFXM_forward(forward_dict, load_res_fn=str(res_fn_file), verbose=True)
    im, _, rulers = model.forward(Fg_func)
    im_ref, _, _ = model.forward(identity_Fg)
    im_stack.append(im)
    im_ref_stack.append(im_ref)
    rulers_last = rulers

result = {
    "phi_values_deg": phi_values_deg,
    "phi_values_rad": phi_values_rad,
    "im_stack": np.stack(im_stack),
    "im_ref_stack": np.stack(im_ref_stack),
    "rulers": rulers_last,
    "x_profile_m": x_profile_m_applied,
    "eps_profile": eps_profile,
    "Ug": normalize_rows(Ug),
    "hkl": hkl,
    "propagation_direction_g": propagation_direction_g / np.linalg.norm(propagation_direction_g),
    "wave_start_depth_um": wave_start_depth_um,
    "forward_dict": forward_dict,
}

with save_file.open("wb") as f:
    pickle.dump(result, f)

print(f"Saved phi scan to {save_file}")
