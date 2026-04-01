import numpy as np
import matplotlib.pyplot as plt
import forward_model as fwd
import os
import pickle


forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = 35.04
forward_dict['hkl'] = [-1, 1, -1]
# forward_dict['x_c'] = [1, 0, 0]
# forward_dict['y_c'] = [0, 1, 0]
forward_dict['Ug'] = np.array(
    [
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
        [1/np.sqrt(2), 0, -1/np.sqrt(2)],
        [-1/np.sqrt(6), 2/np.sqrt(6), -1/np.sqrt(6)],
    ],
    dtype=float,
)
forward_dict['mu'] = np.deg2rad(1.95)
forward_dict['psize'] = 40e-9
forward_dict['zl_rms'] = 1.7e-6 / 2.35
forward_dict['zl_truncation_sigma'] = 2.0
# forward_dict['Npixels'] = [250, 200, 200]

d = forward_dict.copy()
if type(d['Npixels']) is int:
    Nx = Ny = Nz = d['Npixels']
else:
    Nx, Ny, Nz = d['Npixels']
Nsub = 2
NNx, NNy, NNz = Nsub*Nx, Nsub*Ny, Nsub*Nz

psize = d['psize']
zl_rms = d['zl_rms']
zl_truncation_sigma = d.get('zl_truncation_sigma')
theta_0 = np.deg2rad(d['two_theta']/2)
TwoDeltaTheta = d['TwoDeltaTheta']
theta = theta_0 + TwoDeltaTheta/2

yl_start = -psize*Ny/2 + psize/(2*Nsub)
yl_step = psize/Nsub
xl_start = (-psize*Nx/2 + psize/(2*Nsub))/np.tan(2*theta)
xl_step = psize/Nsub/np.tan(2*theta)
beam_half_width_sigma = 3 if zl_truncation_sigma is None else float(zl_truncation_sigma)
zl_span = 2 * beam_half_width_sigma * zl_rms
zl_start = -0.5 * zl_span
zl_step = zl_span/(NNz-1)

yl = yl_start + np.arange(NNy)*yl_step
zl = zl_start + np.arange(NNz)*zl_step
xl0 = xl_start + np.arange(NNx)*xl_step

ZL, YL, XL0 = np.meshgrid(zl, yl, xl0)
XL = XL0 + ZL/np.tan(2*theta)
RL = np.stack([XL, YL, ZL], axis=-1)

mu = forward_dict['mu']

# Visualize gauge volume for this mu
M = np.array([
    [np.cos(mu), 0, np.sin(mu)],
    [0, 1, 0],
    [-np.sin(mu), 0, np.cos(mu)],
])
RS = np.einsum('ij,...j->...i', M, RL)

nskip = 5
xs = RS[::nskip, ::nskip, ::nskip, 0].flatten() * 1e6
zs = RS[::nskip, ::nskip, ::nskip, 2].flatten() * 1e6

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# ax.scatter(xs, zs, s=0.05, alpha=0.3, c='C0', rasterized=True)
# ax.set_xlabel(r'$x_s$ ($\mu$m)')
# ax.set_ylabel(r'$z_s$ ($\mu$m)')
# ax.set_title(r'Gauge volume in sample coordinates (xz projection) for $\mu = %.2f°$' % np.rad2deg(mu))
# ax.set_aspect('equal')
# ax.grid(True, alpha=0.3)
# fig.tight_layout()
# plt.savefig('strain_wave_gauge_volume.png', dpi=150, bbox_inches='tight')
# print('Saved: strain_wave_gauge_volume.png')

# Print extent
xs_all = RS[..., 0].flatten() * 1e6
zs_all = RS[..., 2].flatten() * 1e6
# print('\n' + '='*70)
# print(f'Gauge volume extent in sample coordinates for mu = {np.rad2deg(mu):.2f}°')
# print('='*70)
# print(f'xs range: {xs_all.min():+.3f} to {xs_all.max():+.3f} μm')
# print(f'zs range: {zs_all.min():+.3f} to {zs_all.max():+.3f} μm')
# print('='*70)


# =====================================================================
# Part 2: Forward model integration with a spatially-varying Fg
# =====================================================================

# print('\n' + '#'*70)
# print('# Part 2: Forward model with spatially-varying strain')
# print('#'*70)

# Use a smaller grid for speed
forward_dict['Npixels'] = [600, 10, 200]
forward_dict['Nrays'] = 5000000

Ug = forward_dict['Ug']
strain_profile_file = 'strain_profile_100ps.csv'
strain_profile = np.genfromtxt(strain_profile_file, delimiter=',', names=True)
profile_xs_m = strain_profile['distance_nm'] * 1e-9
profile_exx = strain_profile['strain'] 

if np.any(np.diff(profile_xs_m) < 0):
    sort_idx = np.argsort(profile_xs_m)
    profile_xs_m = profile_xs_m[sort_idx]
    profile_exx = profile_exx[sort_idx]

profile_xs_min = profile_xs_m[0]
profile_xs_max = profile_xs_m[-1]

def Fg_func(xg, yg, zg):
    # Convert grain-frame coordinates to sample-frame coordinates.
    rg = np.stack([xg, yg, zg], axis=-1)
    rs = np.einsum('ij,...j->...i', Ug, rg)
    xs = rs[..., 0]

    # Interpolate epsilon_xx^sample(xs) from the supplied profile.
    eps_xx_sample = np.interp(xs, profile_xs_m, profile_exx, left=0.0, right=0.0)

    eps_sample = np.zeros(xg.shape + (3, 3))
    eps_sample[..., 0, 0] = eps_xx_sample

    # Transform the sample strain tensor to the grain/unit-cell frame.
    eps_grain = np.einsum('ji,...jk,kl->...il', Ug, eps_sample, Ug)

    Fg = np.zeros(xg.shape + (3, 3))
    Fg[..., :, :] = np.eye(3)
    Fg += eps_grain
    return Fg

# print(f'Loaded strain profile: {strain_profile_file}')
# print(f'x_s range in profile: {profile_xs_min*1e9:.3f} to {profile_xs_max*1e9:.3f} nm')

# # plot the imported strain profile
# plt.plot(profile_xs_m * 1e9, profile_exx)
# plt.xlabel(r'$x_s$ (nm)')
# plt.ylabel(r'$\epsilon_{xx}$')
# plt.show()

# Compute resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
res_fn_file = os.path.join(datapath, 'Res_qi_strain_wave.npz')

# Set up forward model with the chosen mu
forward_dict['mu'] = mu

# set up a phi scan

phi_start = -0.03
phi_stop = 0.03
phi_start_rad = np.deg2rad(phi_start)
phi_stop_rad = np.deg2rad(phi_stop)
phi_step = 21

phi_values = np.linspace(phi_start_rad, phi_stop_rad, phi_step)


img_stack = []

for phi in phi_values:

    forward_dict['phi'] = phi
    print(f'phi = {np.rad2deg(phi)} deg')

    model = fwd.DFXM_forward(forward_dict, load_res_fn=res_fn_file, verbose=True)
    im, qi_field, rulers = model.forward(Fg_func)
    img_stack.append(im)
# save the image stack as dictonary with phi_values as key
img_stack_dict = {phi: im for phi, im in zip(phi_values, img_stack)}
with open('/Users/brinthan/Library/CloudStorage/GoogleDrive-brinthan@stanford.edu/My Drive/Phonon_manuscript/codes/simulations/pyDFXM/data/strain_wave_phi_scan_100ps.pkl', 'wb') as f:
    pickle.dump(img_stack_dict, f)



# print(f'mu = {np.rad2deg(mu):6.2f} deg  |  total intensity = {im.sum():.4f}  |  '
#       f'max = {im.max():.6f}  |  shape = {im.shape}')

# # Plot the forward model image with detector axes in microns
# det_x_um = rulers[0] * 1e6
# det_y_um = rulers[1] * 1e6

# fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# cax = ax.imshow(
#     im.T,
#     origin='lower',
#     cmap='viridis',
#     extent=[det_x_um.min(), det_x_um.max(), det_y_um.min(), det_y_um.max()],
#     aspect='auto',
# )
# ax.set_title(r'Forward model image with spatially-varying strain' + '\n' +
#              r'$\mu = %.2f°$, $\Sigma I = %.2f$' % (np.rad2deg(mu), im.sum()))
# ax.set_xlabel(r"$x'$ ($\mu$m)")
# ax.set_ylabel(r"$y'$ ($\mu$m)")
# fig.colorbar(cax, ax=ax, shrink=0.8, label='Intensity (a.u.)')
# fig.tight_layout()
# plt.savefig('strain_wave_forward.png', dpi=150, bbox_inches='tight')
# print('Saved: strain_wave_forward.png')

# # plt.show()
