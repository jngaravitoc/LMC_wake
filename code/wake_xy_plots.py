import numpy as np
import matplotlib.pyplot as plt



path_mw=''


N_halo = 37500000

for i in range(len()):
snap = 
mw_pos, mw_vel = halo_particles_CM(path_mw, N_halo, LMC=False)
halo_pos, halo_vel = halo_particles_CM(path, N_halo, LMC=True)

lmc_orbit = np.loadtxt('../../LMC-MW/code/LMC_orbit/orbits/LMC6_40Mb0_ic11_orbit.txt')
x_orbit = lmc_orbit[:114,6] - lmc_orbit[:114,0]
y_orbit = lmc_orbit[:114,7] - lmc_orbit[:114,1]
z_orbit = lmc_orbit[:114,8] - lmc_orbit[:114,2]


nbinsx=50
nbinsy=50
x_bins = np.linspace(-300, 300, nbinsx-1)
y_bins = np.linspace(-300, 300, nbinsy-1)

wake_grid = wake(halo_pos, mw_pos, nbinsx, nbinsy, 10, 20)

title(r'$\Delta \rho$')
plt.contourf(x_bins, y_bins, wake_grid.T, cmap='inferno')
plt.plot(y_orbit[:100], z_orbit[:100], lw=2, c='k')
plt.savefig('wake_xy_proyection_{:0>3d}.png'.format(i))
