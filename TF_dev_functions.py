
import numpy as np
from numpy import pi
import optical_admittance_package as oap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import permittivities as perm


c = 299792458
hplanck = 6.626e-34
hbar = hplanck/2/pi
q = 1.602e-19
kb = 1.38e-23
eps0 = 8.854e-12
mu0 = 1.257e-6



def set_K(Ep,N_K=160,mat='gaas'):
    """
    Set the K values to account for, with a given energy vector.
    
    Parameters:
    Ep: Vector including photon energies
    N_K: Number of desired K values for each energy
    mat: String denoting the material whose propagating modes K should span.
    Has to be one of the options of get_permittivities.
    
    Returns:
    K: Matrix including in-plane wave numbers (in 1/m)
    Kmax: Vector including the maximum values of K
    """

    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    matlist = [mat]
    eps_mat = get_permittivities(matlist,Ep)
    
    # Set the maximum K number for each photon energy
    Nmax = np.squeeze(np.sqrt(eps_mat).real)
    Kmax = 1.05*Nmax*k0
    
    K = np.zeros((Ep.size,N_K))
    for i in range(Ep.size):
        K[i] = np.linspace(0, Kmax[i], N_K)

    return K, Kmax



def get_permittivities(epslist,Ep):
    """
    Help function to return the permittivities for a layer structure for given photon energies
    
    Parameters:
    epslist: String array including the materials of subsequent layers (see below for options)
    Ep: Vector including photon energies (in eV)
    
    Returns:
    eps: Matrix including permittivities for each energy and each layer
    """

    eps_gaas = perm.permittivity_gaas_palik(Ep)
    eps_algaas = perm.permittivity_algaas_palik(Ep)
    eps_au = perm.permittivity_au_palik(Ep)
    eps_gainp = perm.permittivity_gainp(Ep)
    eps_ag = perm.permittivity_ag_palik(Ep)
    eps_mgf2 = perm.permittivity_mgf2(Ep)
    eps_zns = perm.permittivity_zns(Ep)
    eps_air = np.ones(Ep.size)
    
    eps = np.zeros((len(epslist),Ep.size), dtype=complex)
    for i, epsi in enumerate(epslist):
        if epsi=='gaas':
            eps[i] = eps_gaas
        elif epsi=='gaas_lossless':
            eps[i] = np.abs(eps_gaas)+1e-6*1j
        elif epsi=='algaas':
            eps[i] = eps_algaas
        elif epsi=='au':
            eps[i] = eps_au
        elif epsi=='gainp':
            eps[i] = eps_gainp
        elif epsi=='ingap':
            eps[i] = eps_gainp
        elif epsi=='ag':
            eps[i] = eps_ag
        elif epsi=='mgf2':
            eps[i] = eps_mgf2
        elif epsi=='zns':
            eps[i] = eps_zns
        elif epsi=='air':
            eps[i] = eps_air
        elif epsi=='lossy_air':
            eps[i] = eps_air+1j*1e-3
        else:
            print('Warning: unknown material. eps set to zero.')
    eps = eps.T
    return eps



def set_source_coordinates(L,ind):
    """
    Help function to spread 60 source points evenly throughout a chosen layer
    
    Parameters:
    L: Vector including layer thicknesses (in m)
    ind: Index of the desired source layer (index of the first layer is 0)
    
    Returns:
    z0: Vector of source coordinates (in m)
    """
    
    Nz0 = 60
    zBoundaries = np.cumsum(L)
    z0start = zBoundaries[ind-1]-1e-9
    z0end = zBoundaries[ind]+1e-9
    z0 = np.linspace(z0start,z0end,Nz0)
    return z0



def set_source_coordinates_N(L,ind,N):
    """
    Help function to spread a given number of source points evenly throughout a chosen layer
    
    Parameters:
    L: Vector including layer thicknesses (in m)
    ind: Index of the desired source layer (index of the first layer is 0)
    N: Number of source points to be included.
    
    Returns:
    z0: Vector of source coordinates (in m)
    """
    Nz0 = N
    zBoundaries = np.cumsum(L)
    z0start = zBoundaries[ind-1]-1e-9
    z0end = zBoundaries[ind]+1e-9
    z0 = np.linspace(z0start,z0end,Nz0)
    return z0



def find_z_given_layer(z,L,ind):
    """
    Help function to find the coordinates of a given layer    
    
    Parameters:
    z: Vector including all the coordinates that constitute the geometry (in m)
    L: Vector including layer thicknesses (in m)
    ind: Index of the layer of interest
    
    Returns:
    zIndices: Indices of z that correspond to the layer of interest
    """

    zBoundaries = np.cumsum(L)
    if ind==0:
        zStart=0
    else:
        zStart = zBoundaries[ind-1]
    zEnd = zBoundaries[ind]
    zIndices = np.nonzero((z>=zStart)&(z<=zEnd))
    
    return zIndices



def plot_recombination_single_E(L,N,Ep,K,rad_TE,rad_TM):
    """
    Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Photon energy (in eV)
    K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E
    in optical_admittance_package_final
    rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E
    in optical_admittance_package_final
    """

    z = oap.distribute_z_uneven(L,N)
    theta = oap.propagation_angles_gaas(Ep,K)
    
    zplot, Tplot = np.meshgrid(z*1e6, theta)
    
    rad_TE_tweak = np.zeros(rad_TE.shape)
    indexit = np.nonzero(rad_TE.real>0)
    rad_TE_tweak[indexit] = rad_TE[indexit].real/np.max(rad_TE.real)
    indexit = np.nonzero(rad_TE.real<0)
    rad_TE_tweak[indexit] = rad_TE[indexit].real/(-np.min(rad_TE.real))
    
    rad_TM_tweak = np.zeros(rad_TM.shape)
    indexit = np.nonzero(rad_TM.real>0)
    rad_TM_tweak[indexit] = rad_TM[indexit].real/np.max(rad_TM.real)
    indexit = np.nonzero(rad_TM.real<0)
    rad_TM_tweak[indexit] = rad_TM[indexit].real/(-np.min(rad_TM.real))

    rad_tot_tweak = np.zeros(rad_TE.shape)
    indexit = np.nonzero((rad_TE.real+rad_TM.real)>0)
    rad_tot_tweak[indexit] = (rad_TE[indexit].real+rad_TM[indexit].real)/np.max(rad_TE.real+rad_TM.real)
    indexit = np.nonzero((rad_TE.real+rad_TM.real)<0)
    rad_tot_tweak[indexit] = (rad_TE[indexit].real+rad_TM[indexit].real)/(-np.min(rad_TE.real+rad_TM.real))
    
    cmappi = colormap_SCpaper()
    
    plt.figure(figsize=(10,5),facecolor=(1,1,1))
    plt.subplot(121)
    plt.pcolormesh(zplot, Tplot, rad_TE_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    #plt.text(0.2,80,'(a)',color="white")
    plt.text(0,10,'Max:'+"%.3g" % np.max(rad_TE.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(rad_TE.real),color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot, Tplot, rad_TM_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    #plt.text(0.2,80,'(b)',color="white")
    plt.text(0,10,'Max:'+"%.3g" % np.max(rad_TM.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(rad_TM.real),color="white")
    plt.colorbar()
    plt.tight_layout()
    
    plt.figure()
    plt.pcolormesh(zplot, Tplot, rad_tot_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    plt.text(0,10,'Max:'+"%.3g" % np.max(rad_TE.real+rad_TM.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(rad_TE.real+rad_TM.real),color="white")
    plt.colorbar()
    plt.show()



def plot_radiance_single_E(L,N,Ep,K,P_TE,P_TM):
    """
    Function to plot radiances for a single energy as a function of position and propagation angle.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Photon energy (in eV)
    K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    P_TE: Radiance for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    P_TM: Radiance for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    """

    z = oap.distribute_z_uneven(L,N)
    theta = oap.propagation_angles_gaas(Ep,K)
    
    zplot, Tplot = np.meshgrid(z*1e6, theta)

    P_TE_tweak = np.zeros(P_TE.shape)
    indexit = np.nonzero(P_TE.real>0)
    P_TE_tweak[indexit] = P_TE[indexit].real/np.max(P_TE.real)
    indexit = np.nonzero(P_TE.real<0)
    P_TE_tweak[indexit] = P_TE[indexit].real/(-np.min(P_TE.real))
    
    P_TM_tweak = np.zeros(P_TM.shape)
    indexit = np.nonzero(P_TM.real>0)
    P_TM_tweak[indexit] = P_TM[indexit].real/np.max(P_TM.real)
    indexit = np.nonzero(P_TM.real<0)
    P_TM_tweak[indexit] = P_TM[indexit].real/(-np.min(P_TM.real))
    
    P_tot_tweak = np.zeros(P_TE.shape)
    indexit = np.nonzero((P_TE.real+P_TM.real)>0)
    P_tot_tweak[indexit] = (P_TE[indexit].real+P_TM[indexit].real)/np.max(P_TE.real+P_TM.real)
    indexit = np.nonzero((P_TE.real+P_TM.real)<0)
    P_tot_tweak[indexit] = (P_TE[indexit].real+P_TM[indexit].real)/(-np.min(P_TE.real+P_TM.real))
    
    cmappi = colormap_SCpaper()
    
    plt.figure(figsize=(10,5),facecolor=(1,1,1))
    plt.subplot(121)
    plt.pcolormesh(zplot, Tplot, P_TE_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    #plt.text(0.2,80,'(a)',color="white")
    plt.text(0,10,'Max:'+"%.3g" % np.max(P_TE.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(P_TE.real),color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot, Tplot, P_TM_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    #plt.text(0.2,80,'(b)',color="white")
    plt.text(0,10,'Max:'+"%.3g" % np.max(P_TM.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(P_TM.real),color="white")
    plt.colorbar()
    plt.tight_layout()
    
    plt.figure()
    plt.pcolormesh(zplot, Tplot, P_tot_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    plt.text(0,10,'Max:'+"%.3g" % np.max(P_TE.real+P_TM.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(P_TE.real+P_TM.real),color="white")
    plt.colorbar()
    plt.show()



def plot_recombination_single_E_K(L,N,Ep,K,rad_TE,rad_TM):
    """
    Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Photon energy (in eV)
    K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E
    in optical_admittance_package_final
    rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E
    in optical_admittance_package_final
    """

    z = oap.distribute_z_uneven(L,N)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    zplot, Tplot = np.meshgrid(z*1e6, K/k0)
    
    rad_TE_tweak = np.zeros(rad_TE.shape)
    indexit = np.nonzero(rad_TE.real>0)
    rad_TE_tweak[indexit] = rad_TE[indexit].real/np.max(rad_TE.real)
    indexit = np.nonzero(rad_TE.real<0)
    rad_TE_tweak[indexit] = rad_TE[indexit].real/(-np.min(rad_TE.real))
    
    rad_TM_tweak = np.zeros(rad_TM.shape)
    indexit = np.nonzero(rad_TM.real>0)
    rad_TM_tweak[indexit] = rad_TM[indexit].real/np.max(rad_TM.real)
    indexit = np.nonzero(rad_TM.real<0)
    rad_TM_tweak[indexit] = rad_TM[indexit].real/(-np.min(rad_TM.real))
    
    cmappi = colormap_SCpaper()
    
    plt.figure(figsize=(10,5),facecolor=(1,1,1))
    plt.subplot(121)
    plt.pcolormesh(zplot, Tplot, rad_TE_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K/k0), np.max(K/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('K/k0')
    plt.text(0.2,80,'(a)',color="white")
    plt.text(0,0.5,'Max:'+"%.3g" % np.max(rad_TE.real),color="white")
    plt.text(0,0.25,'Min:'+"%.3g" % np.min(rad_TE.real),color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot, Tplot, rad_TM_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K/k0), np.max(K/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('K/k0')
    plt.text(0.2,80,'(b)',color="white")
    plt.text(0,0.5,'Max:'+"%.3g" % np.max(rad_TM.real),color="white")
    plt.text(0,0.25,'Min:'+"%.3g" % np.min(rad_TM.real),color="white")
    plt.colorbar()
    plt.tight_layout()
    
    plt.figure()
    plt.pcolormesh(zplot, Tplot, (rad_TE_tweak+rad_TM_tweak)/2, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K/k0), np.max(K/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('K/k0')
    plt.text(0,0.5,'Max:'+"%.3g" % np.max(rad_TE.real+rad_TM.real),color="white")
    plt.text(0,0.25,'Min:'+"%.3g" % np.min(rad_TE.real+rad_TM.real),color="white")
    plt.colorbar()
    plt.show()



def plot_recombination_energy_spread(L,N,Ep,qte_w,qtm_w):
    """
    Function to plot recombination-generation rate integrated over all directions as a function of position and photon energy.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    """
    
    z = oap.distribute_z_uneven(L,N)
    zplot, Epplot = np.meshgrid(z*1e6, Ep)
    
    qte_w_tweak = np.zeros(qte_w.shape)
    indexit = np.nonzero(qte_w.real>0)
    qte_w_tweak[indexit] = qte_w[indexit].real/np.max(qte_w.real)
    indexit = np.nonzero(qte_w.real<0)
    qte_w_tweak[indexit] = qte_w[indexit].real/(-np.min(qte_w.real))
    
    qtm_w_tweak = np.zeros(qtm_w.shape)
    indexit = np.nonzero(qtm_w.real>0)
    qtm_w_tweak[indexit] = qtm_w[indexit].real/np.max(qtm_w.real)
    indexit = np.nonzero(qtm_w.real<0)
    qtm_w_tweak[indexit] = qtm_w[indexit].real/(-np.min(qtm_w.real))
    
    cmappi = colormap_SCpaper()
    
    plt.figure(figsize=(10,5),facecolor=(1,1,1))
    plt.subplot(121)
    plt.pcolormesh(zplot, Epplot, qte_w_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(Ep), np.max(Ep)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Energy (eV)')
    plt.text(0.2,1.57,'(a)',color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot, Epplot, qtm_w_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(Ep), np.max(Ep)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Energy (eV)')
    plt.text(0.2,1.57,'(b)',color="white")
    plt.colorbar()
    plt.tight_layout()
    plt.show()



def calculate_QE_energy_spread(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    """
    Calculate the quantum efficiency of light transfer between the emitter and the absorber layer.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    indEm: Index of the emitting layer (index of the first layer is 0)
    indAbs: Index of the absorbing layer (index of the first layer is 0)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    
    Returns:
    qe_w_te: Vector of quantum efficiencies for TE for each energy
    qe_w_tm: Vector of quantum efficiencies for TM for each energy
    qe_tot_te: Quantum efficiency for TE integrated over energy
    qe_tot_tm: Quantum efficiency for TM integrated over energy
    """

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Rtot_te = np.trapezoid(qte_w,omega,axis=0)
    Rtot_tm = np.trapezoid(qtm_w,omega,axis=0)
    
    qte_w_int_Em = np.trapezoid(qte_w.T[zIndEm],z[zIndEm],axis=0)
    qte_w_int_Abs = np.trapezoid(qte_w.T[zIndAbs],z[zIndAbs],axis=0)
    qtm_w_int_Em = np.trapezoid(qtm_w.T[zIndEm],z[zIndEm],axis=0)
    qtm_w_int_Abs = np.trapezoid(qtm_w.T[zIndAbs],z[zIndAbs],axis=0)
    Rtot_te_int_Em = np.trapezoid(Rtot_te[zIndEm],z[zIndEm])
    Rtot_te_int_Abs = np.trapezoid(Rtot_te[zIndAbs],z[zIndAbs])
    Rtot_tm_int_Em = np.trapezoid(Rtot_tm[zIndEm],z[zIndEm])
    Rtot_tm_int_Abs = np.trapezoid(Rtot_tm[zIndAbs],z[zIndAbs])
    
    qe_w_te = -qte_w_int_Abs/qte_w_int_Em
    qe_w_tm = -qtm_w_int_Abs/qtm_w_int_Em
    qe_tot_te = -Rtot_te_int_Abs/Rtot_te_int_Em
    qe_tot_tm = -Rtot_tm_int_Abs/Rtot_tm_int_Em
    
    return qe_w_te, qe_w_tm, qe_tot_te, qe_tot_tm



def calculate_PCE_energy_spread(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    """
    Calculate the optical power transfer efficiency between the emitter and absorber layer.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    indEm: Index of the emitting layer (index of the first layer is 0)
    indAbs: Index of the absorbing layer (index of the first layer is 0)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    
    Returns:
    pce_w_te: Vector of power transfer efficiencies for TE for each energy
    pce_w_tm: Vector of power transfer efficiencies for TM for each energy
    pce_tot_te: Power transfer efficiency for TE integrated over energy
    pce_tot_tm: Power transfer efficiency for TM integrated over energy
    """

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    
    RPtot_te = np.trapezoid(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapezoid(Pqtm_w,omega,axis=0)
    
    Pqte_w_int_Em = np.trapezoid(Pqte_w.T[zIndEm],z[zIndEm],axis=0)
    Pqte_w_int_Abs = np.trapezoid(Pqte_w.T[zIndAbs],z[zIndAbs],axis=0)
    Pqtm_w_int_Em = np.trapezoid(Pqtm_w.T[zIndEm],z[zIndEm],axis=0)
    Pqtm_w_int_Abs = np.trapezoid(Pqtm_w.T[zIndAbs],z[zIndAbs],axis=0)
    RPtot_te_int_Em = np.trapezoid(RPtot_te[zIndEm],z[zIndEm])
    RPtot_te_int_Abs = np.trapezoid(RPtot_te[zIndAbs],z[zIndAbs])
    RPtot_tm_int_Em = np.trapezoid(RPtot_tm[zIndEm],z[zIndEm])
    RPtot_tm_int_Abs = np.trapezoid(RPtot_tm[zIndAbs],z[zIndAbs])
    
    pce_w_te = -Pqte_w_int_Abs/Pqte_w_int_Em
    pce_w_tm = -Pqtm_w_int_Abs/Pqtm_w_int_Em
    pce_tot_te = -RPtot_te_int_Abs/RPtot_te_int_Em
    pce_tot_tm = -RPtot_tm_int_Abs/RPtot_tm_int_Em
    
    return pce_w_te, pce_w_tm, pce_tot_te, pce_tot_tm



def calculate_em_abs_powers(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    """
    Function to calculate the total power emitted by a certain layer and the power absorbed by another layer.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    indEm: Index of the emitting layer (index of the first layer is 0)
    indAbs: Index of the absorbing layer (index of the first layer is 0)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    
    Returns:
    RPtot_te_int_Em: Emitted optical power in TE by the emitting layer (W/m^2)
    RPtot_te_int_Abs: Absorbed optical power in TE by the absorbing layer (W/m^2)
    RPtot_tm_int_Em: Emitted optical power in TM by the emitting layer (W/m^2)
    RPtot_tm_int_Abs: Absorbed optical power in TM by the absorbing layer (W/m^2)
    """

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    
    RPtot_te = np.trapezoid(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapezoid(Pqtm_w,omega,axis=0)
    
    RPtot_te_int_Em = np.trapezoid(RPtot_te[zIndEm],z[zIndEm])
    RPtot_te_int_Abs = np.trapezoid(RPtot_te[zIndAbs],z[zIndAbs])
    RPtot_tm_int_Em = np.trapezoid(RPtot_tm[zIndEm],z[zIndEm])
    RPtot_tm_int_Abs = np.trapezoid(RPtot_tm[zIndAbs],z[zIndAbs])
    
    return RPtot_te_int_Em, RPtot_te_int_Abs, RPtot_tm_int_Em, RPtot_tm_int_Abs



def calculate_em_abs_powers_E_integration(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    """
    Function to calculate the total power emitted by a certain layer and the power absorbed by another layer.
    Integrates over E instead of omega just to check.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    indEm: Index of the emitting layer (index of the first layer is 0)
    indAbs: Index of the absorbing layer (index of the first layer is 0)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    
    Returns:
    RPtot_te_int_Em: Emitted optical power in TE by the emitting layer (W/m^2)
    RPtot_te_int_Abs: Absorbed optical power in TE by the absorbing layer (W/m^2)
    RPtot_tm_int_Em: Emitted optical power in TM by the emitting layer (W/m^2)
    RPtot_tm_int_Abs: Absorbed optical power in TM by the absorbing layer (W/m^2)
    """

    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q/hplanck*2*pi
    Pqtm_w = qtm_w*Ep_mat*q/hplanck*2*pi
    
    RPtot_te = np.trapezoid(Pqte_w,Ep*q,axis=0)
    RPtot_tm = np.trapezoid(Pqtm_w,Ep*q,axis=0)
    
    RPtot_te_int_Em = np.trapezoid(RPtot_te[zIndEm],z[zIndEm])
    RPtot_te_int_Abs = np.trapezoid(RPtot_te[zIndAbs],z[zIndAbs])
    RPtot_tm_int_Em = np.trapezoid(RPtot_tm[zIndEm],z[zIndEm])
    RPtot_tm_int_Abs = np.trapezoid(RPtot_tm[zIndAbs],z[zIndAbs])
    
    return RPtot_te_int_Em, RPtot_te_int_Abs, RPtot_tm_int_Em, RPtot_tm_int_Abs



def calculate_em_abs_rates(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    """
    Function to calculate the total rate emitted by a certain layer and the rate absorbed by another layer.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    indEm: Index of the emitting layer (index of the first layer is 0)
    indAbs: Index of the absorbing layer (index of the first layer is 0)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    
    Returns:
    Rtot_te_int_Em: Emission rate in TE by the emitting layer
    Rtot_te_int_Abs: Absorption rate in TE by the absorbing layer
    Rtot_tm_int_Em: Emission rate in TM by the emitting layer
    Rtot_tm_int_Abs: Absorption rate in TM by the absorbing layer
    """

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)

    Rtot_te = np.trapezoid(qte_w,omega,axis=0)
    Rtot_tm = np.trapezoid(qtm_w,omega,axis=0)

    Rtot_te_int_Em = np.trapezoid(Rtot_te[zIndEm],z[zIndEm])
    Rtot_te_int_Abs = np.trapezoid(Rtot_te[zIndAbs],z[zIndAbs])
    Rtot_tm_int_Em = np.trapezoid(Rtot_tm[zIndEm],z[zIndEm])
    Rtot_tm_int_Abs = np.trapezoid(Rtot_tm[zIndAbs],z[zIndAbs])

    return Rtot_te_int_Em, Rtot_te_int_Abs, Rtot_tm_int_Em, Rtot_tm_int_Abs



def calculate_RG_spectra(L,N,indAbs,qte_w,qtm_w):
    """
    Function to calculate the emission (positive) or absorption (negative spectra by integrating over
    a certain layer.

    Parameters:
    L: Vector including the lenghts of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    indAbs: Index of the emittive or absorptive layer
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final

    Returns:
    R_te_Abs: Emission (positive) or absorption (negative) spectrum for TE modes
    R_tm_Abs: Emission (positive) or absorption (negative) spectrum for TM modes
    """

    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)

    R_te_Abs = np.trapezoid(qte_w[:,zIndAbs],z[zIndAbs])
    R_tm_Abs = np.trapezoid(qtm_w[:,zIndAbs],z[zIndAbs])

    return R_te_Abs, R_tm_Abs




def calculate_total_em_abs_powers(L,N,Ep,qte_w,qtm_w):
    """
    Function to calculate the total power emitted by the full structure and the power absorbed by the full structure.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    
    Returns:
    RPtot_te_all_Em: Emitted optical power in TE by the full structure (in W/m^2)
    RPtot_te_all_Abs: Absorbed optical power in TE by the full structure (in W/m^2)
    RPtot_tm_all_Em: Emitted optical power in TM by the full structure (in W/m^2)
    RPtot_tm_all_Abs: Absorbed optical power in TM by the full structure (in W/m^2)
    """

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    
    RPtot_te = np.trapezoid(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapezoid(Pqtm_w,omega,axis=0)
    
    RPtot_te_all_Em = np.trapezoid(RPtot_te*(RPtot_te>0),z)
    RPtot_te_all_Abs = np.trapezoid(RPtot_te*(RPtot_te<0),z)
    RPtot_tm_all_Em = np.trapezoid(RPtot_tm*(RPtot_tm>0),z)
    RPtot_tm_all_Abs = np.trapezoid(RPtot_tm*(RPtot_tm<0),z)
    
    return RPtot_te_all_Em, RPtot_te_all_Abs, RPtot_tm_all_Em, RPtot_tm_all_Abs



def calculate_total_em_abs_rates(L,N,Ep,qte_w,qtm_w):
    """
    Function to calculate the total rate emitted by the full structure and the rate absorbed by the full structure.
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    
    Returns:
    RPtot_te_all_Em: Emitted optical rate in TE by the full structure (should be in 1/m^2, but I should double-check)
    RPtot_te_all_Abs: Absorbed optical rate in TE by the full structure (should be in 1/m^2, but I should double-check)
    RPtot_tm_all_Em: Emitted optical rate in TM by the full structure (should be in 1/m^2, but I should double-check)
    RPtot_tm_all_Abs: Absorbed optical rate in TM by the full structure (should be in 1/m^2, but I should double-check)
    """

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    
    Rtot_te = np.trapezoid(qte_w,omega,axis=0)
    Rtot_tm = np.trapezoid(qtm_w,omega,axis=0)
    
    Rtot_te_all_Em = np.trapezoid(Rtot_te*(Rtot_te>0),z)
    Rtot_te_all_Abs = np.trapezoid(Rtot_te*(Rtot_te<0),z)
    Rtot_tm_all_Em = np.trapezoid(Rtot_tm*(Rtot_tm>0),z)
    Rtot_tm_all_Abs = np.trapezoid(Rtot_tm*(Rtot_tm<0),z)
    
    return Rtot_te_all_Em, Rtot_te_all_Abs, Rtot_tm_all_Em, Rtot_tm_all_Abs



def plot_total_rates_energy_spread(L,N,Ep,qte_w,qtm_w):
    """
    Plot recombination-generatin as a function of position and photon energy
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Vector including photon energies (in eV)
    qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread
    in optical_admittance_package_final
    """

    omega = 2*pi*Ep*q/hplanck
    Rtot_te = np.trapezoid(qte_w,omega,axis=0)
    Rtot_tm = np.trapezoid(qtm_w,omega,axis=0)
    z = oap.distribute_z_uneven(L,N)
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    RPtot_te = np.trapezoid(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapezoid(Pqtm_w,omega,axis=0)    
    
    plt.figure()
    plt.subplot(141)
    plt.plot(z,np.real(Rtot_te))
    plt.subplot(142)
    plt.plot(z,np.real(Rtot_tm))
    plt.subplot(143)
    plt.plot(z,RPtot_te.real)
    plt.subplot(144)
    plt.plot(z,RPtot_tm.real)
    plt.show()



def plot_Efield_single_E(L,N,Ep,K,E1,E2,E3):
    """
    Plot normalized electric field amplitude for a given photon energy as a function of position and propagation angle
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Photon energy (in eV)
    K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    E1: Electric field component E1 returned by the function Efield_single_E in optical_admittance_package_final
    E2: Electric field component E2 returned by the function Efield_single_E in optical_admittance_package_final
    E3: Electric field component E3 returned by the function Efield_single_E in optical_admittance_package_final
    """

    z = oap.distribute_z_uneven(L,N)
    theta = oap.propagation_angles_gaas(Ep,K)
    zplot, Tplot = np.meshgrid(z*1e6, theta)
    
    cmappi = colormap_SCpaper_yel()
    
    plt.figure(figsize=(12,5),facecolor=(1,1,1))
    plt.subplot(131)
    plt.pcolormesh(zplot, Tplot, np.abs(E1)/(np.max(np.abs(E1))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Angle (deg.)')
    plt.subplot(132)
    plt.pcolormesh(zplot, Tplot, np.abs(E2)/(np.max(np.abs(E2))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.subplot(133)
    plt.pcolormesh(zplot, Tplot, np.abs(E3)/(np.max(np.abs(E3))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.colorbar()
    plt.show()



def plot_Efield_single_E_K(L,N,Ep,K,E1,E2,E3):
    """
    Plot normalized electric field amplitude for a given photon energy as a function of position and K
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Photon energy (in eV)
    K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    E1: Electric field component E1 returned by the function Efield_single_E in optical_admittance_package_final
    E2: Electric field component E2 returned by the function Efield_single_E in optical_admittance_package_final
    E3: Electric field component E3 returned by the function Efield_single_E in optical_admittance_package_final
    """

    z = oap.distribute_z_uneven(L,N)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    zplot, Tplot = np.meshgrid(z*1e6, K/k0)
    
    cmappi = colormap_SCpaper_yel()
    
    plt.figure(figsize=(12,5),facecolor=(1,1,1))
    plt.subplot(131)
    plt.pcolormesh(zplot, Tplot, np.abs(E1)/(np.max(np.abs(E1))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K/k0), np.max(K/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('K/k0')
    plt.subplot(132)
    plt.pcolormesh(zplot, Tplot, np.abs(E2)/(np.max(np.abs(E2))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K/k0), np.max(K/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.subplot(133)
    plt.pcolormesh(zplot, Tplot, np.abs(E3)/(np.max(np.abs(E3))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K/k0), np.max(K/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.colorbar()
    plt.show()



def plot_Efield_single_E_unnormalized(L,N,Ep,K,E1,E2,E3):
    """
    Plot electric field amplitude for a given photon energy as a function of position and propagation angle
    
    Parameters:
    L: Vector including the lengths of subsequent layers (in m)
    N: Vector including the number of datapoints in each layer
    Ep: Photon energy (in eV)
    K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    E1: Electric field component E1 returned by the function Efield_single_E in optical_admittance_package_final
    E2: Electric field component E2 returned by the function Efield_single_E in optical_admittance_package_final
    E3: Electric field component E3 returned by the function Efield_single_E in optical_admittance_package_final
    """

    z = oap.distribute_z_uneven(L,N)
    theta = oap.propagation_angles_gaas(Ep,K)
    zplot, Tplot = np.meshgrid(z*1e6, theta)
    
    cmappi = colormap_SCpaper_yel()
    
    plt.figure(figsize=(12,5),facecolor=(1,1,1))
    plt.subplot(131)
    plt.pcolormesh(zplot, Tplot, np.abs(E1), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Angle (deg.)')
    plt.subplot(132)
    plt.pcolormesh(zplot, Tplot, np.abs(E2), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.subplot(133)
    plt.pcolormesh(zplot, Tplot, np.abs(E3), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.colorbar()
    plt.show()



def colormap_SCpaper():
    """
    Help function to create a suitable colormap.
    """
    cdict1 = {'red': ((0.0,0.0,0.0), (0.375,0.0,0.0), (0.5,0.0,0.0), (0.625,1.0,1.0), (1.0,1.0,1.0)),
        'green': ((0.0,1.0,1.0), (0.375,0.5,0.5), (0.5,0.0,0.0), (0.625,0.0,0.0), (1.0,1.0,1.0)),
        'blue': ((0.0,1.0,1.0), (0.375,1.0,1.0), (0.5,0.0,0.0), (0.625,0.0,0.0), (1.0,0.0,0.0))}
    SCpap_cmap = LinearSegmentedColormap('BlueRed1', cdict1)
    
    return SCpap_cmap



def colormap_SCpaper_yel():
    """
    Help function to create a suitable colormap.
    """
    cdict1 = {'red': ((0.0,0.0,0.0), (0.25,1.0,1.0), (1.0,1.0,1.0)),
        'green': ((0.0,0.0,0.0), (0.25,0.0,0.0), (1.0,1.0,1.0)),
        'blue': ((0.0,0.0,0.0), (1.0,0.0,0.0))}
    SCpap_cmap_yel = LinearSegmentedColormap('BlueRed1', cdict1)
    
    return SCpap_cmap_yel














