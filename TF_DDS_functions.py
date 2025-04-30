
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


def set_E_and_K():
    # Set the default photon energies to account for.
    # Outputs:
    #     Ep: Vector including photon energies (in eV)
    #     K: Matrix including in-plane k numbers (in 1/m)
    #     Kmax: Vector including the maximum values of K
    #     N_K: Number of K values for each Ep
    #N_E = 10
    N_E = 15
    Ep = np.concatenate((1.41,np.linspace(1.42,1.48,N_E),1.52,1.56,1.6),axis=None)
    #Ep = np.concatenate((1.39,1.40,1.41,np.linspace(1.42,1.48,N_E),1.52,1.56,1.6),axis=None)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    eps_gaas = perm.permittivity_gaas_palik(Ep)
    eps_algaas = perm.permittivity_algaas_palik(Ep)
    eps_gainp = perm.permittivity_gainp(Ep)
    
    # Set the maximum K number for each photon energy
    Nsemicond = np.sqrt(np.array([eps_algaas, eps_gaas, eps_gainp])).real
    Nmax = Nsemicond.max(0)
    Kmax = 1.0494*Nmax*k0
    #Kmax = 1.5*Nmax*k0
    #N_K = 40
    #N_K = 80
    N_K = 159
    #N_K = 317
    
    K = np.zeros((Ep.size,N_K))
    for i in range(Ep.size):
        K[i] = np.linspace(0, Kmax[i], N_K)

    return Ep, K, Kmax, N_K



def set_E_and_K_opt(N_K):
    # Set the default photon energies to account for.
    # Outputs:
    #     Ep: Vector including photon energies (in eV)
    #     K: Matrix including in-plane k numbers (in 1/m)
    #     Kmax: Vector including the maximum values of K
    N_E = 15
    Ep = np.concatenate((1.41,np.linspace(1.42,1.48,N_E),1.52,1.56,1.6),axis=None)
    #Ep = np.concatenate((1.38,np.linspace(1.39,1.6,N_E),1.62,1.63,1.65),axis=None)
    #Ep = np.concatenate((np.linspace(1.4,1.6,N_E),1.61,1.615,1.62,1.625,1.63,1.635,1.64,1.645,1.65),axis=None)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl

    eps_gaas = perm.permittivity_gaas_palik(Ep)
    eps_algaas = perm.permittivity_algaas_palik(Ep)
    eps_gainp = perm.permittivity_gainp(Ep)

    # Set the maximum K number for each photon energy
    Nsemicond = np.sqrt(np.array([eps_algaas, eps_gaas, eps_gainp])).real
    Nmax = Nsemicond.max(0)
    Kmax = 1.0494*Nmax*k0

    K = np.zeros((Ep.size,N_K))
    for i in range(Ep.size):
        K[i] = np.linspace(0, Kmax[i], N_K)

    return Ep, K, Kmax



def set_E_and_K_mater_opt(N_K,mat,Ep=np.array([])):
    # Set the default photon energies to account for. Set K to only include modes propagating in the material provided
    # Outputs:
    #     Ep: Vector including photon energies (in eV)
    #     K: Matrix including in-plane k numbers (in 1/m)
    #     Kmax: Vector including the maximum values of K
    if not Ep.any():
        N_E = 15
        Ep = np.concatenate((1.41,np.linspace(1.42,1.48,N_E),1.52,1.56,1.6),axis=None)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl

    eps_mat = get_permittivities(mat,Ep)

    # Set the maximum K number for each photon energy
    Nmax = np.squeeze(np.sqrt(eps_mat).real)
    Kmax = 1.05*Nmax*k0

    K = np.zeros((Ep.size,N_K))
    for i in range(Ep.size):
        K[i] = np.linspace(0, Kmax[i], N_K)

    return Ep, K, Kmax



def set_K(Ep,N_K):
    # Set the K values to account for, with a given energy vector.
    # Outputs:
    #   K: Matrix including in-plane wave numbers (in 1/m)
    #   Kmax: Vector including the maximum values of K
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    eps_gaas = perm.permittivity_gaas_palik(Ep)
    eps_algaas = perm.permittivity_algaas_palik(Ep)
    eps_gainp = perm.permittivity_gainp(Ep)
    
    # Set the maximum K number for each photon energy
    Nsemicond = np.sqrt(np.array([eps_algaas, eps_gaas, eps_gainp])).real
    Nmax = Nsemicond.max(0)
    Kmax = 1.0494*Nmax*k0
    
    K = np.zeros((Ep.size,N_K))
    for i in range(Ep.size):
        K[i] = np.linspace(0, Kmax[i], N_K)
    
    return K, Kmax



def get_permittivities(epslist,Ep):
    # Help function to return the permittivities for a layer structure for given photon energies
    # Inputs:
    #     epslist: String array including the materials of subsequent layers (see below for options)
    #     Ep: Vector including photon energies (in eV)
    # Outputs:
    #     eps: Matrix including permittivities for each energy and each layer
    
    eps_gaas = perm.permittivity_gaas_palik(Ep)
    eps_algaas = perm.permittivity_algaas_palik(Ep)
    eps_au = perm.permittivity_au_palik(Ep)
    eps_gainp = perm.permittivity_gainp(Ep)
    eps_ag = perm.permittivity_ag_palik(Ep)
    eps_mgf2 = perm.permittivity_mgf2(Ep)
    eps_zns = perm.permittivity_zns(Ep)
    eps_air = np.ones(Ep.size)
    
    eps = np.zeros((len(epslist),Ep.size), dtype=complex)
    for i in range(len(epslist)):
        if epslist[i]=='gaas':
            eps[i] = eps_gaas
        elif epslist[i]=='gaas_lossless':
            eps[i] = np.abs(eps_gaas)+1e-6*1j
        elif epslist[i]=='algaas':
            eps[i] = eps_algaas
        elif epslist[i]=='au':
            eps[i] = eps_au
        elif epslist[i]=='gainp':
            eps[i] = eps_gainp
        elif epslist[i]=='ingap':
            eps[i] = eps_gainp
        elif epslist[i]=='ag':
            eps[i] = eps_ag
        elif epslist[i]=='mgf2':
            eps[i] = eps_mgf2
        elif epslist[i]=='zns':
            eps[i] = eps_zns
        elif epslist[i]=='air':
            eps[i] = eps_air
        elif epslist[i]=='lossy_air':
            eps[i] = eps_air+1j*1e-3
        else:
            print('Warning: unknown material. eps set to zero.')
    eps = eps.T
    return eps



def set_source_coordinates(L,ind):
    # Help function to spread source points evenly throughout a chosen layer
    # Inputs:
    #     L: Vector including layer thicknesses (in m)
    #     ind: Index of the desired source layer (index of the first layer is 0)
    # Outputs:
    #     z0: Vector of source coordinates (in m)
    Nz0 = 60
    zBoundaries = np.cumsum(L)
    z0start = zBoundaries[ind-1]-1e-9
    z0end = zBoundaries[ind]+1e-9
    z0 = np.linspace(z0start,z0end,Nz0)
    return z0



def set_source_coordinates_N(L,ind,N):
    # Help function to spread source points evenly throughout a chosen layer
    # Inputs:
    #     L: Vector including layer thicknesses (in m)
    #     ind: Index of the desired source layer (index of the first layer is 0)
    # Outputs:
    #     z0: Vector of source coordinates (in m)
    Nz0 = N
    zBoundaries = np.cumsum(L)
    z0start = zBoundaries[ind-1]-1e-9
    z0end = zBoundaries[ind]+1e-9
    z0 = np.linspace(z0start,z0end,Nz0)
    return z0



def find_z_given_layer(z,L,ind):
    # Help function to find the source coordinates of a given layer    
    # Inputs:
    #     z: Vector including all the coordinates that constitute the geometry (in m)
    #     L: Vector including layer thicknesses (in m)
    #     ind: Index of the layer of interest (index of the first layer is 0)
    # Outputs:
    #    zIndices: Indices of z that correspond to the layer of interest
    zBoundaries = np.cumsum(L)
    if ind==0:
        zStart=0
    else:
        zStart = zBoundaries[ind-1]
    zEnd = zBoundaries[ind]
    zIndices = np.nonzero((z>=zStart)&(z<=zEnd))
    
    return zIndices



def solve_optical_properties_single_E_remove(eps,mu,L,N,Ep,kx,z0,dEF,epssubsL=None,epssubsR=None):
    # Solve photon numbers, Poynting vectors and recombination-generation for a
    # single energy Ep as a function of position and K number with source
    # coordinates z0.
    # This also calculates the complex electric field amplitudes (a.u.) of the
    # TE and TM fields based on Eq. (A3) of Phys. Rev. E 98, 063304 (2018),
    # assuming an equally strong source amplitude in each direction at each source point.
    # 
    # Deals with the divergences by ignoring the solutions that diverge.
    # 
    # Inputs:
    #       eps: Vector including the permittivity in each layer
    #       mu: Vector including the permeability in each layer
    #       L: Vector including the lengths of subsequent layers (in m)
    #       N: Vector including the intended number of datapoints in each layer
    #       Ep: Photon energy (in eV)
    #       kx: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #       z0: Vector of source coordinates (in m)
    #       dEF: Quasi-Fermi level separation at the source coordinates (in eV)
    # Outputs:
    #       pup_TE: Rightward-propagating photon number in TE modes as a function of K and z, as in Eq. (5) of Sci. Rep. 7, 11534 (2017).
    #               See also Eq. (8) of Phys. Rev. A 92, 033839 (2015).
    #       pdown_TE: Leftward-propagating photon number in TE modes as a function of K and z, similarly as pup_TE
    #       pup_TM: Rightward-propagating photon number in TM modes as a function of K and z, similarly as pup_TE
    #       pdown_TM: Leftward-propagating photon number in TM modes as a function of K and z, similarly as pup_TE
    #       P_TE: Spectral radiance in TE modes as a function of K and z, based on Eq. (6) of Phys. Rev. A 92, 033839 (2015).
    #       P_TM: Spectral radiance in TM modes as a function of K and z, based on Eq. (6) of Phys. Rev. A 92, 033839 (2015).
    #       rad_TE: Net recombination-generation rate in TE modes as a function of K and z, calculated as a derivative of P_TE.
    #       rad_TM: Net recombination-generation rate in TM modes as a function of K and z, calculated as a derivative of P_TM.
    #       qte_w: Net recombination-generation rate in TE modes as a function of z (rad_TE integrated over K)
    #       qtm_w: Net recombination-generation rate in TM modes as a function of z (rad_TM integrated over K)
    #       E1: Complex amplitude of the TE electric field as a function of K and z
    #       E2: Complex amplitude of the in-plane component of the TM electric field as a function of K and z
    #       E3: Complex amplitude of the normal component of the TM electric field as a function of K and z
    
    T = 300
    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    
    N_cumul = np.cumsum(N)
    
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    eta = 1/(np.exp(q*(Ep-dEF)/kb/T)-1)
    
    indexit = np.nonzero(np.logical_and(z>z0.min(),z<z0.max()))
    index = indexit[0]
    eta_QFED = np.zeros(np.size(z))
    eta_QFED[index] = eta
    
    eps_z = oap.distribute_parameter(eps,N)
    nr_z = np.real(np.sqrt(eps_z))
    if epssubsL == None:
        epssubs_l = eps[0]
        epssubs_r = eps[-1]
        musubs_l = mu[0]
        musubs_r = mu[-1]
    elif epssubsR == None:
        epssubs_l = epssubsL
        epssubs_r = epssubsL
        musubs_l = 1
        musubs_r = 1
    else:
        epssubs_l = epssubsL
        epssubs_r = epssubsR
        musubs_l = 1
        musubs_r = 1
    
    pup_TE = np.zeros((kx.size, z.size), dtype=complex)
    pdown_TE = np.zeros((kx.size, z.size), dtype=complex)
    pup_TM = np.zeros((kx.size, z.size), dtype=complex)
    pdown_TM = np.zeros((kx.size, z.size), dtype=complex)
    P_TE = np.zeros((kx.size, z.size), dtype=complex)
    P_TM = np.zeros((kx.size, z.size), dtype=complex)
    rad_TE = np.zeros((kx.size, z.size), dtype=complex)
    rad_TM = np.zeros((kx.size, z.size), dtype=complex)
    E1 = np.zeros((kx.size, z.size), dtype=complex)
    E2 = np.zeros((kx.size, z.size), dtype=complex)
    E3 = np.zeros((kx.size, z.size), dtype=complex)
    
    failures_TE = []
    failures_TM = []
    
    print("Solving for E =", Ep, "eV...")
    for j in range(kx.size):
        gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = oap.calculate_all_admittances_uneven( \
            eps,mu,L,N,wl,kx[j],epssubs_l,musubs_l,epssubs_r,musubs_r)
        rho_e, rho_m, rho_TE, rho_TM = oap.LDOSes(eps,mu,N,wl,kx[j],gamma_l_TE,gamma_r_TE, \
            gamma_r_TM,gamma_l_TM)
        
        rho_nl_TE = np.zeros((z0.size, z.size), dtype=complex)
        rho_nl_TM = np.zeros((z0.size, z.size), dtype=complex)
        rho_if_TE = np.zeros((z0.size, z.size), dtype=complex)
        rho_if_TM = np.zeros((z0.size, z.size), dtype=complex)
        gee11 = np.zeros((z0.size, z.size), dtype=complex)
        gee22 = np.zeros((z0.size, z.size), dtype=complex)
        gee33 = np.zeros((z0.size, z.size), dtype=complex)
        gee23 = np.zeros((z0.size, z.size), dtype=complex)
        gee32 = np.zeros((z0.size, z.size), dtype=complex)
        for k in range(z0.size):
            gee11[k], gee22[k], gee33[k], gee23[k], gee32[k] = oap.electric_greens_functions(eps,mu,N,wl, \
                kx[j],z,z0[k],gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            gme12, gme13, gme21, gme31 = oap.exchange_greens_functions_me(eps,mu,N,wl,kx[j], \
                z,z0[k],gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            rho_nl_TE[k], rho_nl_TM[k] = oap.NLDOS_TETM_electric_sources(eps,mu,N,wl,kx[j],z,z0[k], \
                gee11[k],gee22[k],gee33[k],gee23[k],gee32[k],gme12,gme13,gme21,gme31)
            rho_if_TE[k], rho_if_TM[k] = oap.IFDOS_TETM_electric_sources(eps,mu,N,wl,z,z0[k], \
                gee11[k],gee22[k],gee23[k],gme12,gme13,gme21)
        # Upward and downward photon numbers calculated using Eq. (5) of Sci.
        # Rep. 7, 11534 (2017).
        pup_TE[j] = 1/rho_TE*np.trapz((rho_nl_TE+rho_if_TE)*eta,z0,axis=0)
        pdown_TE[j] = 1/rho_TE*np.trapz((rho_nl_TE-rho_if_TE)*eta,z0,axis=0)
        pup_TM[j] = 1/rho_TM*np.trapz((rho_nl_TM+rho_if_TM)*eta,z0,axis=0)
        pdown_TM[j] = 1/rho_TM*np.trapz((rho_nl_TM-rho_if_TM)*eta,z0,axis=0)
        
        alphap_TE, alpham_TE, betap_TE, betam_TE = oap.RTE_coefficients_TE(eps,mu,N,wl,kx[j], \
            gamma_l_TE,gamma_r_TE)
        alphap_TM, alpham_TM, betap_TM, betam_TM = oap.RTE_coefficients_TM(eps,mu,N,wl,kx[j], \
            gamma_l_TM,gamma_r_TM)
        
        drho_TE, drho_TM = oap.LDOS_derivatives(eps,mu,N,wl,kx[j],gamma_l_TE,gamma_r_TE,gamma_r_TM, \
            gamma_l_TM)
        
        P_TE[j] = Ep*q*c*rho_TE/2*(pup_TE[j]-pdown_TE[j])/nr_z
        P_TM[j] = Ep*q*c*rho_TM/2*(pup_TM[j]-pdown_TM[j])/nr_z
        
        dpup_TE = -alphap_TE*(pup_TE[j]-eta_QFED)+betap_TE*(pdown_TE[j]-eta_QFED)
        dpup_TM = -alphap_TM*(pup_TM[j]-eta_QFED)+betap_TM*(pdown_TM[j]-eta_QFED)
        dpdown_TE = alpham_TE*(pdown_TE[j]-eta_QFED)-betam_TE*(pup_TE[j]-eta_QFED)
        dpdown_TM = alpham_TM*(pdown_TM[j]-eta_QFED)-betam_TM*(pup_TM[j]-eta_QFED)
        
        rad_TE[j] = hplanck/(2*pi)*omega*c/nr_z*1/2*(drho_TE*(pup_TE[j]-pdown_TE[j])+ \
            rho_TE*(dpup_TE-dpdown_TE))/Ep/q
        rad_TM[j] = hplanck/(2*pi)*omega*c/nr_z*1/2*(drho_TM*(pup_TM[j]-pdown_TM[j])+ \
            rho_TM*(dpup_TM-dpdown_TM))/Ep/q
        
        E1[j] = np.trapz(gee11*1,z0,axis=0)
        E2[j] = np.trapz(gee22*1+gee23*1,z0,axis=0)
        E3[j] = np.trapz(gee32*1+gee33*1,z0,axis=0)
        
        N_ledend = N_cumul[3]
        rad_TE_intpd = np.trapz(rad_TE[j,0:N_ledend],z[0:N_ledend])
        rad_TE_intled = np.trapz(rad_TE[j,N_ledend:],z[N_ledend:])
        rad_TM_intpd = np.trapz(rad_TM[j,0:N_ledend],z[0:N_ledend])
        rad_TM_intled = np.trapz(rad_TM[j,N_ledend:],z[N_ledend:])
        if np.abs(rad_TE_intpd) > 2*np.abs(rad_TE_intled):
            failures_TE.append(j)
        if rad_TE_intled < 0:
            failures_TE.append(j)
        if np.abs(rad_TM_intpd) > 2*np.abs(rad_TM_intled):
            failures_TM.append(j)
        if rad_TM_intled < 0:
            failures_TM.append(j)
    
    print(failures_TE)
    print(failures_TM)
    kx_TE = np.delete(kx,failures_TE,axis=None)
    kx_TM = np.delete(kx,failures_TM,axis=None)
    rad_TE = np.delete(rad_TE,failures_TE,axis=0)
    rad_TM = np.delete(rad_TM,failures_TM,axis=0)
    E1 = np.delete(E1,failures_TE,axis=0)
    E2 = np.delete(E2,failures_TM,axis=0)
    E3 = np.delete(E3,failures_TM,axis=0)
    
    kx_mat_TE = np.tile(kx_TE,(z.size,1)).T
    kx_mat_TM = np.tile(kx_TM,(z.size,1)).T
    qte_w = np.trapz(rad_TE*2*pi*kx_mat_TE,kx_TE,axis=0)
    qtm_w = np.trapz(rad_TM*2*pi*kx_mat_TM,kx_TM,axis=0)
    
    return pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, qte_w, qtm_w, kx_TE, kx_TM



def plot_EM_field_fluctuations_single_E(L,N,Ep,K,E2Kw,H2Kw,uKw):
    # This was used for testing, not currently used for anything I believe.
    # I think the quantities studied here are nonetheless based on Phys. Rev. A 95, 013848 (2017).
    z = oap.distribute_z_uneven(L,N)
    theta = oap.propagation_angles_gaas(Ep,K)
    
    zplot, Tplot = np.meshgrid(z*1e6, theta)
    
    cmappi = colormap_SCpaper_yel()
    
    plt.figure()
    plt.subplot(131)
    plt.pcolormesh(zplot, Tplot, (E2Kw.real/np.max(E2Kw.real)), \
        cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.subplot(132)
    plt.pcolormesh(zplot, Tplot, (H2Kw.real/np.max(H2Kw.real)), \
        cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.subplot(133)
    plt.pcolormesh(zplot, Tplot, (uKw.real/np.max(uKw.real)), \
        cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.colorbar()
    plt.show()
    
    # To think about what else to return or write to file here.



def plot_recombination_single_E(L,N,Ep,K,rad_TE,rad_TM):
    # Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    #     rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    
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
    
    # To think about what to return or write to file here.



def plot_recombination_single_E_remove(L,N,Ep,K_TE,K_TM,rad_TE,rad_TM):
    # Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    #     rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    
    z = oap.distribute_z_uneven(L,N)
    theta_TE = oap.propagation_angles_gaas(Ep,K_TE)
    theta_TM = oap.propagation_angles_gaas(Ep,K_TM)
    
    zplot_TE, Tplot_TE = np.meshgrid(z*1e6, theta_TE)
    zplot_TM, Tplot_TM = np.meshgrid(z*1e6, theta_TM)
    
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
    plt.pcolormesh(zplot_TE, Tplot_TE, rad_TE_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta_TE), np.max(theta_TE)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    #plt.text(0.2,80,'(a)',color="white")
    plt.text(0,10,'Max:'+"%.3g" % np.max(rad_TE.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(rad_TE.real),color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot_TM, Tplot_TM, rad_TM_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta_TM), np.max(theta_TM)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    #plt.text(0.2,80,'(b)',color="white")
    plt.text(0,10,'Max:'+"%.3g" % np.max(rad_TM.real),color="white")
    plt.text(0,5,'Min:'+"%.3g" % np.min(rad_TM.real),color="white")
    plt.colorbar()
    plt.tight_layout()
    plt.show()



def plot_radiance_single_E(L,N,Ep,K,P_TE,P_TM):
    # Function to plot radiances for a single energy as a function of position and propagation angle.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     P_TE: Radiance for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    #     P_TM: Radiance for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    
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
    # Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    #     rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    
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



def plot_recombination_single_E_K_remove(L,N,Ep,K_TE,K_TM,rad_TE,rad_TM):
    # Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    #     rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    
    z = oap.distribute_z_uneven(L,N)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    zplot_TE, Tplot_TE = np.meshgrid(z*1e6, K_TE/k0)
    zplot_TM, Tplot_TM = np.meshgrid(z*1e6, K_TM/k0)
    
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
    plt.pcolormesh(zplot_TE, Tplot_TE, rad_TE_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K_TE/k0), np.max(K_TE/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('K/k0')
    plt.text(0.2,3.3,'(a) TE',color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot_TM, Tplot_TM, rad_TM_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(K_TM/k0), np.max(K_TM/k0)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('K/k0')
    plt.text(0.2,3.3,'(b) TM',color="white")
    plt.colorbar()
    plt.tight_layout()
    plt.show()



def plot_recombination_single_E_unnormalized(L,N,Ep,K,rad_TE,rad_TM):
    # Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E in optical_admittance_package_final
    #     rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E in optical_admittance_package_final
    
    z = oap.distribute_z_uneven(L,N)
    theta = oap.propagation_angles_gaas(Ep,K)
    
    zplot, Tplot = np.meshgrid(z*1e6, theta)
    
    cmappi = colormap_SCpaper()
    
    plt.figure(figsize=(10,5),facecolor=(1,1,1))
    plt.subplot(121)
    plt.pcolormesh(zplot, Tplot, rad_TE.real, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    plt.text(0.2,80,'(a)',color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot, Tplot, rad_TM.real, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    plt.text(0.2,80,'(b)',color="white")
    plt.colorbar()
    plt.tight_layout()
    
    plt.figure()
    plt.pcolormesh(zplot, Tplot, (rad_TE.real+rad_TM.real)/2, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta), np.max(theta)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Propagation angle (deg.)')
    plt.colorbar()
    plt.show()



def plot_recombination_energy_spread(L,N,Ep,qte_w,qtm_w):
    # Function to plot recombination-generation rate integrated over all directions as a function of position and photon energy.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
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
    # Calculate the quantum efficiency of light transfer between the emitter and the absorber layer.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     indEm: Index of the emitting layer (index of the first layer is 0)
    #     indAbs: Index of the absorbing layer (index of the first layer is 0)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     qe_w_te: Vector of quantum efficiencies for TE for each energy
    #     qe_w_tm: Vector of quantum efficiencies for TM for each energy
    #     qe_tot_te: Quantum efficiency for TE integrated over energy
    #     qe_tot_tm: Quantum efficiency for TM integrated over energy
    
    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Rtot_te = np.trapz(qte_w,omega,axis=0)
    Rtot_tm = np.trapz(qtm_w,omega,axis=0)
    
    qte_w_int_Em = np.trapz(qte_w.T[zIndEm],z[zIndEm],axis=0)
    qte_w_int_Abs = np.trapz(qte_w.T[zIndAbs],z[zIndAbs],axis=0)
    qtm_w_int_Em = np.trapz(qtm_w.T[zIndEm],z[zIndEm],axis=0)
    qtm_w_int_Abs = np.trapz(qtm_w.T[zIndAbs],z[zIndAbs],axis=0)
    Rtot_te_int_Em = np.trapz(Rtot_te[zIndEm],z[zIndEm])
    Rtot_te_int_Abs = np.trapz(Rtot_te[zIndAbs],z[zIndAbs])
    Rtot_tm_int_Em = np.trapz(Rtot_tm[zIndEm],z[zIndEm])
    Rtot_tm_int_Abs = np.trapz(Rtot_tm[zIndAbs],z[zIndAbs])
    
    qe_w_te = -qte_w_int_Abs/qte_w_int_Em
    qe_w_tm = -qtm_w_int_Abs/qtm_w_int_Em
    qe_tot_te = -Rtot_te_int_Abs/Rtot_te_int_Em
    qe_tot_tm = -Rtot_tm_int_Abs/Rtot_tm_int_Em
    
    return qe_w_te, qe_w_tm, qe_tot_te, qe_tot_tm



def calculate_PCE_energy_spread(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    # Calculate the optical power transfer efficiency between the emitter and absorber layer.
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     indEm: Index of the emitting layer (index of the first layer is 0)
    #     indAbs: Index of the absorbing layer (index of the first layer is 0)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     pce_w_te: Vector of power transfer efficiencies for TE for each energy
    #     pce_w_tm: Vector of power transfer efficiencies for TM for each energy
    #     pce_tot_te: Power transfer efficiency for TE integrated over energy
    #     pce_tot_tm: Power transfer efficiency for TM integrated over energy
    
    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    
    RPtot_te = np.trapz(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapz(Pqtm_w,omega,axis=0)
    
    Pqte_w_int_Em = np.trapz(Pqte_w.T[zIndEm],z[zIndEm],axis=0)
    Pqte_w_int_Abs = np.trapz(Pqte_w.T[zIndAbs],z[zIndAbs],axis=0)
    Pqtm_w_int_Em = np.trapz(Pqtm_w.T[zIndEm],z[zIndEm],axis=0)
    Pqtm_w_int_Abs = np.trapz(Pqtm_w.T[zIndAbs],z[zIndAbs],axis=0)
    RPtot_te_int_Em = np.trapz(RPtot_te[zIndEm],z[zIndEm])
    RPtot_te_int_Abs = np.trapz(RPtot_te[zIndAbs],z[zIndAbs])
    RPtot_tm_int_Em = np.trapz(RPtot_tm[zIndEm],z[zIndEm])
    RPtot_tm_int_Abs = np.trapz(RPtot_tm[zIndAbs],z[zIndAbs])
    
    pce_w_te = -Pqte_w_int_Abs/Pqte_w_int_Em
    pce_w_tm = -Pqtm_w_int_Abs/Pqtm_w_int_Em
    pce_tot_te = -RPtot_te_int_Abs/RPtot_te_int_Em
    pce_tot_tm = -RPtot_tm_int_Abs/RPtot_tm_int_Em
    
    return pce_w_te, pce_w_tm, pce_tot_te, pce_tot_tm



def calculate_em_abs_powers(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    # Function to calculate the total power emitted by a certain layer and the power absorbed by another layer.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     indEm: Index of the emitting layer (index of the first layer is 0)
    #     indAbs: Index of the absorbing layer (index of the first layer is 0)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     RPtot_te_int_Em: Emitted optical power in TE by the emitting layer (should be in W/m^2, but I should double-check)
    #     RPtot_te_int_Abs: Absorbed optical power in TE by the absorbing layer (should be in W/m^2, but I should double-check)
    #     RPtot_tm_int_Em: Emitted optical power in TM by the emitting layer (should be in W/m^2, but I should double-check)
    #     RPtot_tm_int_Abs: Absorbed optical power in TM by the absorbing layer (should be in W/m^2, but I should double-check)
    
    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    
    RPtot_te = np.trapz(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapz(Pqtm_w,omega,axis=0)
    
    RPtot_te_int_Em = np.trapz(RPtot_te[zIndEm],z[zIndEm])
    RPtot_te_int_Abs = np.trapz(RPtot_te[zIndAbs],z[zIndAbs])
    RPtot_tm_int_Em = np.trapz(RPtot_tm[zIndEm],z[zIndEm])
    RPtot_tm_int_Abs = np.trapz(RPtot_tm[zIndAbs],z[zIndAbs])
    
    return RPtot_te_int_Em, RPtot_te_int_Abs, RPtot_tm_int_Em, RPtot_tm_int_Abs



def calculate_em_abs_powers_E_integration(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    # Function to calculate the total power emitted by a certain layer and the power absorbed by another layer.
    # Integrates over E instead of omega.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     indEm: Index of the emitting layer (index of the first layer is 0)
    #     indAbs: Index of the absorbing layer (index of the first layer is 0)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     RPtot_te_int_Em: Emitted optical power in TE by the emitting layer (should be in W/m^2, but I should double-check)
    #     RPtot_te_int_Abs: Absorbed optical power in TE by the absorbing layer (should be in W/m^2, but I should double-check)
    #     RPtot_tm_int_Em: Emitted optical power in TM by the emitting layer (should be in W/m^2, but I should double-check)
    #     RPtot_tm_int_Abs: Absorbed optical power in TM by the absorbing layer (should be in W/m^2, but I should double-check)
    
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q/hplanck*2*pi
    Pqtm_w = qtm_w*Ep_mat*q/hplanck*2*pi
    
    RPtot_te = np.trapz(Pqte_w,Ep*q,axis=0)
    RPtot_tm = np.trapz(Pqtm_w,Ep*q,axis=0)
    
    RPtot_te_int_Em = np.trapz(RPtot_te[zIndEm],z[zIndEm])
    RPtot_te_int_Abs = np.trapz(RPtot_te[zIndAbs],z[zIndAbs])
    RPtot_tm_int_Em = np.trapz(RPtot_tm[zIndEm],z[zIndEm])
    RPtot_tm_int_Abs = np.trapz(RPtot_tm[zIndAbs],z[zIndAbs])
    
    return RPtot_te_int_Em, RPtot_te_int_Abs, RPtot_tm_int_Em, RPtot_tm_int_Abs



def calculate_em_abs_rates(L,N,Ep,indEm,indAbs,qte_w,qtm_w):
    # Function to calculate the total rate emitted by a certain layer and the rate absorbed by another layer.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     indEm: Index of the emitting layer (index of the first layer is 0)
    #     indAbs: Index of the absorbing layer (index of the first layer is 0)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     Rtot_te_int_Em: Emission rate in TE by the emitting layer
    #     Rtot_te_int_Abs: Absorption rate in TE by the absorbing layer
    #     Rtot_tm_int_Em: Emission rate in TM by the emitting layer
    #     Rtot_tm_int_Abs: Absorption rate in TM by the absorbing layer

    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)
    zIndEm = find_z_given_layer(z,L,indEm)

    Rtot_te = np.trapz(qte_w,omega,axis=0)
    Rtot_tm = np.trapz(qtm_w,omega,axis=0)

    Rtot_te_int_Em = np.trapz(Rtot_te[zIndEm],z[zIndEm])
    Rtot_te_int_Abs = np.trapz(Rtot_te[zIndAbs],z[zIndAbs])
    Rtot_tm_int_Em = np.trapz(Rtot_tm[zIndEm],z[zIndEm])
    Rtot_tm_int_Abs = np.trapz(Rtot_tm[zIndAbs],z[zIndAbs])

    return Rtot_te_int_Em, Rtot_te_int_Abs, Rtot_tm_int_Em, Rtot_tm_int_Abs



def calculate_RG_spectra(L,N,indAbs,qte_w,qtm_w):
    # Description missing

    z = oap.distribute_z_uneven(L,N)
    zIndAbs = find_z_given_layer(z,L,indAbs)

    R_te_Abs = np.trapz(qte_w[:,zIndAbs],z[zIndAbs])
    R_tm_Abs = np.trapz(qtm_w[:,zIndAbs],z[zIndAbs])

    return R_te_Abs, R_tm_Abs




def calculate_total_em_abs_powers(L,N,Ep,qte_w,qtm_w):
    # Function to calculate the total power emitted by the full structure and the power absorbed by the full structure.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     RPtot_te_all_Em: Emitted optical power in TE by the full structure (should be in W/m^2, but I should double-check)
    #     RPtot_te_all_Abs: Absorbed optical power in TE by the full structure (should be in W/m^2, but I should double-check)
    #     RPtot_tm_all_Em: Emitted optical power in TM by the full structure (should be in W/m^2, but I should double-check)
    #     RPtot_tm_all_Abs: Absorbed optical power in TM by the full structure (should be in W/m^2, but I should double-check)
    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    
    RPtot_te = np.trapz(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapz(Pqtm_w,omega,axis=0)
    
    RPtot_te_all_Em = np.trapz(RPtot_te*(RPtot_te>0),z)
    RPtot_te_all_Abs = np.trapz(RPtot_te*(RPtot_te<0),z)
    RPtot_tm_all_Em = np.trapz(RPtot_tm*(RPtot_tm>0),z)
    RPtot_tm_all_Abs = np.trapz(RPtot_tm*(RPtot_tm<0),z)
    
    return RPtot_te_all_Em, RPtot_te_all_Abs, RPtot_tm_all_Em, RPtot_tm_all_Abs



def calculate_total_em_abs_rates(L,N,Ep,qte_w,qtm_w):
    # Function to calculate the total rate emitted by the full structure and the rate absorbed by the full structure.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
    # Outputs:
    #     RPtot_te_all_Em: Emitted optical rate in TE by the full structure (should be in 1/m^2, but I should double-check)
    #     RPtot_te_all_Abs: Absorbed optical rate in TE by the full structure (should be in 1/m^2, but I should double-check)
    #     RPtot_tm_all_Em: Emitted optical rate in TM by the full structure (should be in 1/m^2, but I should double-check)
    #     RPtot_tm_all_Abs: Absorbed optical rate in TM by the full structure (should be in 1/m^2, but I should double-check)
    omega = 2*pi*Ep*q/hplanck
    z = oap.distribute_z_uneven(L,N)
    
    Rtot_te = np.trapz(qte_w,omega,axis=0)
    Rtot_tm = np.trapz(qtm_w,omega,axis=0)
    
    Rtot_te_all_Em = np.trapz(Rtot_te*(Rtot_te>0),z)
    Rtot_te_all_Abs = np.trapz(Rtot_te*(Rtot_te<0),z)
    Rtot_tm_all_Em = np.trapz(Rtot_tm*(Rtot_tm>0),z)
    Rtot_tm_all_Abs = np.trapz(Rtot_tm*(Rtot_tm<0),z)
    
    return Rtot_te_all_Em, Rtot_te_all_Abs, Rtot_tm_all_Em, Rtot_tm_all_Abs



def plot_total_rates_energy_spread(L,N,Ep,qte_w,qtm_w):
    # Plot recombination-generatin as a function of position and photon energy
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Vector including photon energies (in eV)
    #     qte_w: Recombination-generation rate for TE calculated with solve_recombination_energy_spread in optical_admittance_package_final
    #     qtm_w: Recombination-generation rate for TM calculated with solve_recombination_energy_spread in optical_admittance_package_final
        
    omega = 2*pi*Ep*q/hplanck
    Rtot_te = np.trapz(qte_w,omega,axis=0)
    Rtot_tm = np.trapz(qtm_w,omega,axis=0)
    z = oap.distribute_z_uneven(L,N)
    Ep_mat = np.tile(Ep,(z.size,1)).T
    Pqte_w = qte_w*Ep_mat*q
    Pqtm_w = qtm_w*Ep_mat*q
    RPtot_te = np.trapz(Pqte_w,omega,axis=0)
    RPtot_tm = np.trapz(Pqtm_w,omega,axis=0)    
    
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
    # Plot normalized electric field amplitude for a given photon energy as a function of position and propagation angle
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     E1: Electric field component E1 returned by the function Efield_single_E in optical_admittance_package_final
    #     E2: Electric field component E2 returned by the function Efield_single_E in optical_admittance_package_final
    #     E3: Electric field component E3 returned by the function Efield_single_E in optical_admittance_package_final
    
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
    # Plot normalized electric field amplitude for a given photon energy as a function of position and K
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     E1: Electric field component E1 returned by the function Efield_single_E in optical_admittance_package_final
    #     E2: Electric field component E2 returned by the function Efield_single_E in optical_admittance_package_final
    #     E3: Electric field component E3 returned by the function Efield_single_E in optical_admittance_package_final
   
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
    # Plot electric field amplitude for a given photon energy as a function of position and propagation angle
    # Inputs:
        #       L: Vector including the lengths of subsequent layers (in m)
        #       N: Vector including the intended number of datapoints in each layer
        #       Ep: Photon energy (in eV)
        #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     E1: Electric field component E1 returned by the function Efield_single_E in optical_admittance_package_final
    #     E2: Electric field component E2 returned by the function Efield_single_E in optical_admittance_package_final
    #     E3: Electric field component E3 returned by the function Efield_single_E in optical_admittance_package_final
    
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
    # Help function to create a suitable colormap.
    cdict1 = {'red': ((0.0,0.0,0.0), (0.375,0.0,0.0), (0.5,0.0,0.0), (0.625,1.0,1.0), (1.0,1.0,1.0)),
        'green': ((0.0,1.0,1.0), (0.375,0.5,0.5), (0.5,0.0,0.0), (0.625,0.0,0.0), (1.0,1.0,1.0)),
        'blue': ((0.0,1.0,1.0), (0.375,1.0,1.0), (0.5,0.0,0.0), (0.625,0.0,0.0), (1.0,0.0,0.0))}
    SCpap_cmap = LinearSegmentedColormap('BlueRed1', cdict1)
    
    return SCpap_cmap



def colormap_SCpaper_yel():
    # Help function to create a suitable colormap.
    cdict1 = {'red': ((0.0,0.0,0.0), (0.25,1.0,1.0), (1.0,1.0,1.0)),
        'green': ((0.0,0.0,0.0), (0.25,0.0,0.0), (1.0,1.0,1.0)),
        'blue': ((0.0,0.0,0.0), (1.0,0.0,0.0))}
    SCpap_cmap_yel = LinearSegmentedColormap('BlueRed1', cdict1)
    
    return SCpap_cmap_yel














