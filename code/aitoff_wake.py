"""
Code to make mock beta and velocity dispersion
observations.

input:
------

snap
Coordinates of where to observe.

output:
-------

Anisotropy profile and sigmas.

"""


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pygadgetreader import *
from astropy.coordinates import SkyCoord
from astropy import units as u
import octopus
import sys
#import beta

def re_center(pos, cm):
    """
    Re center a halo to its center of mass.
    """
    pos_n = np.copy(pos)
    for i in range(3):
        pos_n[:,i] = pos[:,i] - cm[i]
    return pos_n


def truncate(pos, r_cut):
    """
    Truncates the halo radii.
    """
    pos_cm, v = octopus.CM(pos, pos)
    pos_centered = re_center(pos, pos_cm)
    r = (pos_centered[:,0]**2 + pos_centered[:,1]**2 + pos_centered[:,2]**2)**0.5
    index_cut = np.where(r<r_cut)
    return(pos[index_cut])

def halo_particles_CM(path, N_halo, LMC=True):
    """
    returns the MW DM halo particles positions and velocities.
    In cartessian coordinates relative to the disk COM.
    """
    ## Reading snapshots

    ## Disk

    pos_disk = readsnap(path, 'pos', 'disk')
    vel_disk = readsnap(path, 'vel', 'disk')
    pot_disk = readsnap(path, 'pot', 'disk')

    ## Halo

    pos_DM = readsnap(path, 'pos', 'dm')
    vel_DM = readsnap(path, 'vel', 'dm')
    ids_DM = readsnap(path, 'pid', 'dm')

    # MW LMC particles
    if LMC == True:
        MW_pos, MW_vel, LMC_pos, LMC_vel = octopus.MW_LMC_particles(pos_DM, vel_DM, ids_DM, N_halo)
        del LMC_pos, LMC_vel
    elif LMC == False:
        MW_pos = pos_DM
        MW_vel = vel_DM
    # center_halo
    r_cm, v_cm = octopus.CM_disk_potential(pos_disk, vel_disk,pot_disk)
    print('Center of mass at \n', r_cm)
    print('Velocity of the center of mass \n', v_cm)

    MW_pos_cm = re_center(MW_pos, r_cm)
    MW_vel_cm = re_center(MW_vel, v_cm)

    MW_disk_pos_cm = re_center(pos_disk, r_cm)
    MW_disk_vel_cm = re_center(vel_disk, v_cm)
    r_disk = (MW_disk_pos_cm[:,0]**2 + MW_disk_pos_cm[:,1]**2\
            + MW_disk_pos_cm[:,2]**2)**0.5
    index = np.where(r_disk<30)[0]

    plt.scatter(MW_disk_pos_cm[index,0], MW_disk_pos_cm[index,1], s=0.1)
    plt.savefig('disk_alignment.png')
    plt.show()

    del pos_disk, vel_disk, pot_disk, pos_DM, vel_DM, ids_DM, MW_pos,\
        MW_vel, r_cm, v_cm

    return MW_pos_cm, MW_vel_cm

def observations(pos, vel):
    """
    Make mock observations in galactic coordinates.
    uses Astropy SkyCoord module.
    l and b coordinates are computed from cartesian coordinates.
    l : [-180, 180]
    b : [-90, 90]

    Parameters:
    -----------
    pos : 3d-numpy array.
    vel : 3d-numpy array.
    lmin : float.
           Minimum latitute of the observation in degrees.
    lmax : float.
           Maximum latitute of the observation in degrees.
    bmin : float.
           Minimum longitude of the observation in degrees.
    bmax : float.
           Maximum longitude of the observation in degrees.
    rmax : float.
           Maximum radius to compute the anisotropy and dispersion
           profiles.
    n_bins_r : int
           Number of bins to do the radial measurements.

    """

    ## transforming to galactic coordinates.

    c_gal = SkyCoord(x=pos[:,0]*u.kpc, y=pos[:,1]*u.kpc,\
                     z=pos[:,2]*u.kpc,\
                     representation='cartesian',\
                     frame='galactocentric')

    c_gal.representation = 'spherical'


    ## to degrees and range of l.

    l_degrees = c_gal.lon.wrap_at(180 * u.deg).radian
    #l_degrees = c_gal.l.radian
    b_degrees = c_gal.lat.radian

    ## Selecting the region of observation.


    return l_degrees, b_degrees

def wake(pos, pos_init, nbinsx, nbinsy, xmin, xmax):
    """
    Returns a 2d histogram of the over densities on the MW halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the overdensities are going to be computed.
    """

    y_bins = np.linspace(-200, 200, nbinsx)
    z_bins = np.linspace(-200, 200, nbinsy)

    density_grid = np.zeros((nbinsx-1, nbinsy-1))

    for i in range(len(y_bins)-1):
        for j in range(len(z_bins)-1):
                #print(y_bins[i+1], y_bins[i], z_bins[j+1], z_bins[j])
                index = np.where((pos[:,1]<y_bins[i+1]) &\
                                  (pos[:,1]>y_bins[i]) &\
                                  (pos[:,2]>z_bins[j]) &\
                                  (pos[:,2]<z_bins[j+1]) &\
                                  (pos[:,0]<xmax) &\
                                  (pos[:,0]>xmin))[0]

                index_init = np.where(( pos_init[:,1]<y_bins[i+1]) &\
                                       (pos_init[:,1]>y_bins[i]) &\
                                       (pos_init[:,2]>z_bins[j]) &\
                                       (pos_init[:,2]<z_bins[j+1]) &\
                                       (pos_init[:,0]<xmax) &\
                                       (pos_init[:,0]>xmin))[0]
                n_grid = len(index)
                n_grid_init = len(index_init)
                #print(n_grid, n_grid_init)
                density_grid[i][j] = (n_grid/n_grid_init) - 1
    return density_grid


def wake_aitoff(l_lmc, b_lmc, l_mw, b_mw, lbins, bbins, rmin, rmax, \
                pos_lmc, pos_mw):
    """
    Returns a 2d histogram of the over densities on the MW halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the overdensities are going to be computed.
    """

    d_b_rads = np.linspace(-np.pi/2., np.pi/2., bbins)
    d_l_rads = np.linspace(-np.pi, np.pi, lbins)
    r_lmc = (pos_lmc[:,0]**2 + pos_lmc[:,1]**2 + pos_lmc[:,2]**2)**0.5
    r_mw = (pos_mw[:,0]**2 + pos_mw[:,1]**2 + pos_mw[:,2]**2)**0.5
    density_grid = np.zeros((lbins-1, bbins-1))

    for i in range(len(d_l_rads)-1):
        for j in range(len(d_b_rads)-1):
                #print(y_bins[i+1], y_bins[i], z_bins[j+1], z_bins[j])
                index = np.where((l_lmc<d_l_rads[i+1]) &\
                                  (l_lmc>d_l_rads[i]) &\
                                  (b_lmc>d_b_rads[j]) &\
                                  (b_lmc<d_b_rads[j+1]) &\
                                  (r_lmc<rmax) &\
                                  (r_lmc>rmin))[0]

                index_init = np.where((l_mw<d_l_rads[i+1]) &\
                                  (l_mw>d_l_rads[i]) &\
                                  (b_mw>d_b_rads[j]) &\
                                  (b_mw<d_b_rads[j+1]) &\
                                  (r_mw<rmax) &\
                                  (r_mw>rmin))[0]

                n_grid = len(index)
                n_grid_init = len(index_init)
                density_grid[i][j] = (n_grid/n_grid_init) - 1
    return density_grid


def vel_grid(pos, pos_init, vel, vel_init, nbinsx, nbinsy, xmin, xmax):
    """
    Returns a 2d histogram of the over densities on the MW halo.
    In Cartessian coordinates, xmin and xmax determine the range in
    x-coordinates in which the overdensities are going to be computed.
    """

    y_bins = np.linspace(-200, 200, nbinsx)
    z_bins = np.linspace(-200, 200, nbinsy)

    beta_grid = np.zeros((nbinsx-1, nbinsy-1))
    v_r_grid = np.zeros((nbinsx-1, nbinsy-1))
    v_t_grid = np.zeros((nbinsx-1, nbinsy-1))

    for i in range(len(y_bins)-1):
        for j in range(len(z_bins)-1):
                #print(y_bins[i+1], y_bins[i], z_bins[j+1], z_bins[j])
                index = np.where((pos[:,1]<y_bins[i+1]) &\
                                  (pos[:,1]>y_bins[i]) &\
                                  (pos[:,2]>z_bins[j]) &\
                                  (pos[:,2]<z_bins[j+1]) &\
                                  (pos[:,0]<xmax) &\
                                  (pos[:,0]>xmin))[0]

                index_init = np.where((pos_init[:,1]<y_bins[i+1]) &\
                                       (pos_init[:,1]>y_bins[i]) &\
                                       (pos_init[:,2]>z_bins[j]) &\
                                       (pos_init[:,2]<z_bins[j+1]) &\
                                       (pos_init[:,0]<xmax) &\
                                       (pos_init[:,0]>xmin))[0]
                #vr_mean, vt_mean = beta.velocities_means(pos[index], vel[index])
                beta_p = beta.beta(pos[index], vel[index])
                #vr_mean_init, vt_mean_init = beta.velocities_means(pos_init[index_init],vel_init[index_init])
                #_grid_init = len(index_init)
                #print(n_grid, n_grid_init)
                #print(vr_mean, vr_mean_init)
                #v_r_grid[i][j] = vr_mean#vr_mean_init)
                #v_t_grid[i][j] = vt_mean#vt_mean_init)
                beta_grid[i][j] = beta_p
    #return v_r_grid, v_t_grid
    return beta_grid

if __name__ == "__main__":


    N_halo = 37500000
    #path_mw='/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b1/simulations/MW2_40M/b1/MW2_40M_b1_new_vir_042'
    path_mw='/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b1/simulations/MW2_40M/b0/MW2_40M_vir_022'
    mw_pos, mw_vel = halo_particles_CM(path_mw, N_halo, LMC=False)
    #fig_name = sys.argv[8]
    for j in range(100, 101, 1):
        #path = '/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b1/simulations/LMCMW40M/MWLMC6_b1/MWLMC6_40M_new_b1_{:0>3d}'.format(j)
        path ='/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b1/simulations/LMCMW40M/MWLMC6/MW_11LMC6_40M_b0_{:0>3d}'.format(j)

        ## Comparison with isotropic MW:

        halo_pos, halo_vel = halo_particles_CM(path, N_halo, LMC=True)

        lbins = 60
        bbins = 30

        ## data to plot the orbit of the LMC!
        lmc_orbit = np.loadtxt('../../LMC-MW/code/LMC_orbit/orbits/LMC6_40Mb0_ic11_orbit.txt')
        #r_orbit_past = ((lmc_orbit[:114,6]-lmc_orbit[:114,0])**2 +(lmc_orbit[:114,7]\
         #           -lmc_orbit[:114,1])**2 + (lmc_orbit[:114,8]\
         #           -lmc_orbit[:114,2])**2)**0.5


        #r_orbit_fut = ((lmc_orbit[114:,6]-lmc_orbit[114:,0])**2 +(lmc_orbit[114:,7]\
        #                -lmc_orbit[114:,1])**2 + (lmc_orbit[114:,8]\
        #                -lmc_orbit[114:,2])**2)**0.5


        #r_lmc_orbit = np.array([lmc_orbit[:,6]-lmc_orbit[:,0],\
        #                        lmc_orbit[:,7]-lmc_orbit[:,1],\
        #                        lmc_orbit[:,8]-lmc_orbit[:,2]]).T

        nbinsx=30
        nbinsy=30

        ## Halo positions in galactocentric coordinates
        #l_halo, b_halo = observations(halo_pos, halo_vel)
        #l_mw, b_mw = observations(mw_pos, mw_vel)
        for i in range(20, 30, 10):
            print(i, i+10)
            #l_bins = np.linspace(-np.pi, np.pi, lbins-1)
            #b_bins = np.linspace(-np.pi/2., np.pi/2., bbins-1)
            #l_grid, b_grid = np.meshgrid(l_bins, b_bins)
            #wake_grid = wake_aitoff(l_halo, b_halo, l_mw, b_mw, lbins, \
            #                        bbins, i, i+10, halo_pos, mw_pos) #

            x_bins = np.linspace(-300, 300, nbinsx-1)
            y_bins = np.linspace(-300, 300, nbinsy-1)
            #l_grid, b_grid = np.meshgrid(x_bins, y_bins)
            #vr_grid, vt_grid = vel_grid(halo_pos, mw_pos, halo_vel, mw_vel, nbinsx, nbinsy, i, i+10) #
            beta_grid = vel_grid(halo_pos, mw_pos, halo_vel, mw_vel, nbinsx, nbinsy, i, i+10) #



            fig = plt.figure(figsize=(8,7))
            #ax = fig.add_subplot(111, projection="aitoff")
            ax = fig.add_subplot(111)
            #
            #plt.title("$LMC particles$")
            #y_bins = np.linspace(-300, 300, nbinsx-1)
            #z_bins = np.linspace(-300, 300, nbinsy-1)
            #y_grid, z_grid = np.meshgrid(y_bins, z_bins)
            #ax.xaxis.set_ticks([-np.pi, -np.pi/2., 0, np.pi/2., np.pi])
            #ax.xaxis.set_ticks([-np.pi, -3*np.pi/4., -np.pi/2., -np.pi/4.\
            #                    ,0, np.pi/4. ,np.pi/2., 3*np.pi/4., np.pi],\
            #                    minor=True)

            #ax.yaxis.set_ticks([ -np.pi/2.1, 0, np.pi/2.1])

            #ax.yaxis.set_ticks([ -np.pi/2.1, -np.pi/4., 0, np.pi/4.,\
            #                     np.pi/2.1], minor=True)

            #ax.tick_params(axis='x', colors='m')
            color_plot = ax.pcolormesh(x_bins, y_bins, ((beta_grid).T),\
                                       cmap="viridis")
            #plt.contour(y_bins, z_bins, wake_grid.T)
            #plt.axis([y_bins.min(), y_bins.max(), z_bins.min(), z_bins.max()])
            plt.colorbar(color_plot)
            #plt.plot(r_lmc_orbit[:j,1], r_lmc_orbit[:j,2], c='r', lw=2)
            #plt.scatter(r_lmc_orbit[j,1], r_lmc_orbit[j,2], marker='*', s=80, c='orange')
            #ax.scatter(l_orbit_past[orbit_cut_past],\
            #           b_orbit_past[orbit_cut_past], c='r')

            #ax.scatter(l_orbit_fut[orbit_cut_fut],\
            #           b_orbit_fut[orbit_cut_fut], c='m')

            #x.scatter(l_orbit_past, b_orbit_past, c='r', s=6)
            #ax.scatter(l_sgr, b_sgr, c='m',s=0.01)
            #ax.scatter(l_lamost.value, b_lamost.value, c='orange',s=10, marker='*')

            #ax.scatter(l_orbit_fut, b_orbit_fut, c='m', s=6)
            #ax.text(-np.pi/1.5, np.pi/6., '$5$', fontsize=15, color='white')
            #ax.text(-np.pi/3., np.pi/6., '$6$', fontsize=15, color='white')
            #ax.text(np.pi/3., np.pi/6., '$7$', fontsize=15, color='white')
            #ax.text(np.pi/1.5, np.pi/6., '$8$', fontsize=15, color='white')
            #ax.text(-np.pi/1.5, -np.pi/6., '$1$', fontsize=15, color='white')
            #ax.text(-np.pi/3., -np.pi/6., '$2$', fontsize=15, color='white')
            #ax.text(np.pi/3, -np.pi/6., '$3$', fontsize=15, color='white')
            #ax.text(np.pi/1.5, -np.pi/6., '$4$', fontsize=15, color='white')
            #ax.yaxis.set_ticks([ -np.pi/2.1, 0, np.pi/2.1])
            #ax.yaxis.set_ticks([ -np.pi/2.1, 0, np.pi/2.1])
            #ax.yaxis.set_ticks([ -np.pi/2.1, 0, np.pi/2.1])
            #ax.grid(which='major')
            plt.title(r'$\beta$')
            plt.xlabel('$y[Kpc]$', fontsize=25)
            plt.ylabel('$z[Kpc]$', fontsize=25)
            plt.savefig('beta_yz_b0_{}_snap_{}.png'.format(i, j), bbox_inches='tight', dpi=300)
            #plt.savefig('beta_aitoff_LMC6_ic11_{}_2515_30-40kpc.png'.format(i), bbox_inches='tight', dpi=300)

