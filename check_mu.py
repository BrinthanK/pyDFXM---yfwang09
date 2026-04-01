import numpy as np
import matplotlib.pyplot as plt
import forward_model as fwd

# =====================================================================
# Setup: diamond (004) parameters
# =====================================================================
forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = 48.16
forward_dict['hkl'] = [0, 0, 1]
forward_dict['x_c'] = [1, 0, 0]
forward_dict['y_c'] = [0, 1, 0]
forward_dict['Ug'] = np.eye(3)

# =====================================================================
# Part 1: Gauge volume visualization for different mu
# =====================================================================
d = forward_dict.copy()
if type(d['Npixels']) is int:
    Nx = Ny = Nz = d['Npixels']
else:
    Nx, Ny, Nz = d['Npixels']
Nsub = 2
NNx, NNy, NNz = Nsub*Nx, Nsub*Ny, Nsub*Nz

psize = d['psize']
zl_rms = d['zl_rms']
theta_0 = np.deg2rad(d['two_theta']/2)
TwoDeltaTheta = d['TwoDeltaTheta']
theta = theta_0 + TwoDeltaTheta/2

yl_start = -psize*Ny/2 + psize/(2*Nsub)
yl_step = psize/Nsub
xl_start = (-psize*Nx/2 + psize/(2*Nsub))/np.tan(2*theta)
xl_step = psize/Nsub/np.tan(2*theta)
zl_start = -0.5*zl_rms*6
zl_step = zl_rms*6/(NNz-1)

yl = yl_start + np.arange(NNy)*yl_step
zl = zl_start + np.arange(NNz)*zl_step
xl0 = xl_start + np.arange(NNx)*xl_step

ZL, YL, XL0 = np.meshgrid(zl, yl, xl0)
XL = XL0 + ZL/np.tan(2*theta)
RL = np.stack([XL, YL, ZL], axis=-1)

mu_values_deg = [0, 10, 24.08, 45, 70]
nskip = 5

fig, axes = plt.subplots(1, len(mu_values_deg), figsize=(4*len(mu_values_deg), 4),
                         sharey=True)

for ax, mu_deg in zip(axes, mu_values_deg):
    mu = np.deg2rad(mu_deg)
    M = np.array([
        [np.cos(mu), 0, np.sin(mu)],
        [0, 1, 0],
        [-np.sin(mu), 0, np.cos(mu)],
    ])
    RS = np.einsum('ij,...j->...i', M, RL)

    xs = RS[::nskip, ::nskip, ::nskip, 0].flatten() * 1e6
    zs = RS[::nskip, ::nskip, ::nskip, 2].flatten() * 1e6

    ax.scatter(xs, zs, s=0.05, alpha=0.3, c='C0', rasterized=True)
    ax.set_xlabel(r'$x_s$ ($\mu$m)')
    ax.set_title(r'$\mu = %g°$' % mu_deg)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(r'$z_s$ ($\mu$m)')
fig.suptitle('Gauge volume in sample coordinates (xz projection) for different $\\mu$',
             fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig('check_mu_gauge_volume.png', dpi=150, bbox_inches='tight')
print('Saved: check_mu_gauge_volume.png')

# Print extents
print('\n' + '='*70)
print('Gauge volume extent in sample coordinates for different mu')
print('='*70)
print(f'{"mu (deg)":>10} {"xs range (um)":>20} {"zs range (um)":>20}')
print('-'*55)
for mu_deg in mu_values_deg:
    mu = np.deg2rad(mu_deg)
    M = np.array([
        [np.cos(mu), 0, np.sin(mu)],
        [0, 1, 0],
        [-np.sin(mu), 0, np.cos(mu)],
    ])
    RS = np.einsum('ij,...j->...i', M, RL)
    xs_all = RS[..., 0].flatten() * 1e6
    zs_all = RS[..., 2].flatten() * 1e6
    print(f'{mu_deg:>10.1f} {xs_all.min():>+10.3f} to {xs_all.max():>+8.3f}'
          f'   {zs_all.min():>+10.3f} to {zs_all.max():>+8.3f}')
print('='*70)


# =====================================================================
# Part 2: Forward model integration with a spatially-varying Fg
# =====================================================================

print('\n' + '#'*70)
print('# Part 2: Forward model with spatially-varying strain')
print('#'*70)

# Use a smaller grid for speed
forward_dict['Npixels'] = [20, 18, 16]
forward_dict['Nrays'] = 5000000

# Strain field: Gaussian bump of eps_zz centered at (x0, 0, 0) in grain coords.
# When mu rotates the gauge volume, different amounts of the bump fall inside,
# producing visibly different images.
x0_bump = 1.5e-6   # center of bump along xg (1.5 um offset)
sigma_bump = 0.5e-6 # width of bump (0.5 um)
eps_peak = 2e-3     # peak strain

def Fg_func(xg, yg, zg):
    bump = eps_peak * np.exp(-0.5*((xg - x0_bump)/sigma_bump)**2)
    Fg = np.zeros(xg.shape + (3, 3))
    Fg[..., 0, 0] = 1.0
    Fg[..., 1, 1] = 1.0
    Fg[..., 2, 2] = 1.0 + bump  # strain along hkl direction
    return Fg

# Compute resolution function once (same optics for all mu)
import os
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
res_fn_file = os.path.join(datapath, 'Res_qi_check_mu.npz')

# Build model with mu=0 just to get the resolution function
forward_dict['mu'] = 0.0
model_tmp = fwd.DFXM_forward(forward_dict, load_res_fn=res_fn_file, verbose=True)
Res_qi = model_tmp.Res_qi

mu_test_deg = [0, 24.08, 45, 70]
images = {}
total_intensities = {}

for mu_deg in mu_test_deg:
    mu_rad = np.deg2rad(mu_deg)
    forward_dict['mu'] = mu_rad
    forward_dict['phi'] = 0.0

    model = fwd.DFXM_forward(forward_dict, load_res_fn=res_fn_file)
    im, qi_field, rulers = model.forward(Fg_func, Res_qi=Res_qi)

    images[mu_deg] = im
    total_intensities[mu_deg] = im.sum()
    print(f'mu = {mu_deg:6.1f} deg  |  total intensity = {im.sum():.4f}  |  '
          f'max = {im.max():.6f}  |  shape = {im.shape}')

# Plot the forward model images
fig, axes = plt.subplots(1, len(mu_test_deg), figsize=(5*len(mu_test_deg), 4))

for ax, mu_deg in zip(axes, mu_test_deg):
    im = images[mu_deg]
    vmax = max(v.max() for v in images.values())
    cax = ax.imshow(im, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
    ax.set_title(r'$\mu = %g°$' % mu_deg + '\n' +
                 r'$\Sigma I = %.2f$' % total_intensities[mu_deg])
    ax.set_xlabel('pixel y')
    ax.set_ylabel('pixel x')
    fig.colorbar(cax, ax=ax, shrink=0.8)

fig.suptitle('Forward model image with spatially-varying strain for different $\\mu$',
             fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig('check_mu_forward.png', dpi=150, bbox_inches='tight')
print('\nSaved: check_mu_forward.png')

# Also show a difference plot relative to mu=0
if len(mu_test_deg) > 1:
    fig2, axes2 = plt.subplots(1, len(mu_test_deg)-1, figsize=(5*(len(mu_test_deg)-1), 4))
    if len(mu_test_deg) == 2:
        axes2 = [axes2]
    im_ref = images[mu_test_deg[0]]
    for ax, mu_deg in zip(axes2, mu_test_deg[1:]):
        diff = images[mu_deg] - im_ref
        vlim = max(abs(diff.min()), abs(diff.max()))
        if vlim == 0:
            vlim = 1.0
        cax = ax.imshow(diff, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        ax.set_title(r'$\mu = %g° - \mu = %g°$' % (mu_deg, mu_test_deg[0]))
        ax.set_xlabel('pixel y')
        ax.set_ylabel('pixel x')
        fig2.colorbar(cax, ax=ax, shrink=0.8)

    fig2.suptitle('Difference in forward model image relative to $\\mu = 0°$',
                  fontsize=13, y=1.02)
    fig2.tight_layout()
    plt.savefig('check_mu_forward_diff.png', dpi=150, bbox_inches='tight')
    print('Saved: check_mu_forward_diff.png')

plt.show()
