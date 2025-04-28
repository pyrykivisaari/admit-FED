
import numpy as np
from numpy import pi
import sys
sys.path.append('/u/05/pkivisaa/unix/tutkimus/optical_modeling/python3/optical_admittance_package_final')
import optical_admittance_package as oap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.append('/u/05/pkivisaa/unix/tutkimus/optical_modeling/python3/permittivities')
import permittivities as perm


c = 299792458
hplanck = 6.626e-34
hbar = hplanck/2/pi
q = 1.602e-19
kb = 1.38e-23
eps0 = 8.854e-12
mu0 = 1.257e-6



def set_E_and_K_vac_opt(N_K,Ep=np.array([])):
    # Set the default photon energies to account for. Set K to only include modes propagating in vacuum (or air)
    # Outputs:
    #     Ep: Vector including photon energies (in eV)
    #     K: Matrix including in-plane k numbers (in 1/m)
    #     Kmax: Vector including the maximum values of K
    if not Ep.any():
        N_E = 15
        Ep = np.concatenate((1.41,np.linspace(1.42,1.48,N_E),1.52,1.56,1.6),axis=None)
    #Ep = np.concatenate((1.38,np.linspace(1.39,1.6,N_E),1.62,1.63,1.65),axis=None)
    #Ep = np.concatenate((np.linspace(1.4,1.6,N_E),1.61,1.615,1.62,1.625,1.63,1.635,1.64,1.645,1.65),axis=None)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl

    # Set the maximum K number for each photon energy
    Kmax = 1.05*k0

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



def set_source_coordinates(L,ind,Nz0):
    # Help function to spread source points evenly throughout a chosen layer
    # Inputs:
    #       L: Vector including layer thicknesses (in m)
    #       ind: Index of the desired source layer (index of the first layer is 0)
    # 	Nz0: Desired number of source points
    # Outputs:
    #       z0: Vector of source coordinates (in m)
    zBoundaries = np.cumsum(L)
    z0start = zBoundaries[ind-1]-1e-9
    z0end = zBoundaries[ind]+1e-9
    z0 = np.linspace(z0start,z0end,Nz0)
    return z0



def get_permittivities(epslist,Ep):
    # Help function to return the permittivities for a layer structure for given photon energies
    # Inputs:
    #     epslist: String array including the materials of subsequent layers (see below for options)
    #     Ep: Vector including photon energies (in eV)
    # Outputs:
    #     eps: Matrix including permittivities for each energy and each layer
    
    eps = np.zeros((len(epslist),Ep.size), dtype=complex)
    for i in range(len(epslist)):
        if epslist[i]=='gaas':
            eps_gaas = perm.permittivity_gaas_palik_sturge(Ep)
            eps[i] = eps_gaas
        elif epslist[i]=='algaas':
            eps_algaas = perm.permittivity_algaas_palik(Ep)
            eps[i] = eps_algaas
        elif epslist[i]=='algaas097':
            eps_algaas = perm.permittivity_algaas097_Papatryfonos(Ep)
            eps[i] = eps_algaas
        elif epslist[i]=='algaas219':
            eps_algaas = perm.permittivity_algaas219_Papatryfonos(Ep)
            eps[i] = eps_algaas
        elif epslist[i]=='algaas342':
            eps_algaas = perm.permittivity_algaas342_Papatryfonos(Ep)
            eps[i] = eps_algaas
        elif epslist[i]=='au':
            eps_au = perm.permittivity_au_palik(Ep)
            eps[i] = eps_au
        elif epslist[i]=='gainp' or epslist[i]=='ingap':
            eps_gainp = perm.permittivity_gainp(Ep)
            eps_gainp_a = perm.permittivity_gainp_adachi(Ep)
            eps[i] = eps_gainp_a
        #elif epslist[i]=='ingap':
        #    eps_gainp = perm.permittivity_gainp(Ep)
        #    eps[i] = eps_gainp
        elif epslist[i]=='ag':
            eps_ag = perm.permittivity_ag_palik(Ep)
            eps_ag_jiang = perm.permittivity_ag_jiang(Ep)
            eps[i] = eps_ag_jiang
        elif epslist[i]=='mgf2':
            eps_mgf2 = perm.permittivity_mgf2(Ep)
            eps[i] = eps_mgf2
        elif epslist[i]=='zns':
            eps_zns = perm.permittivity_zns(Ep)
            eps[i] = eps_zns
        elif epslist[i]=='air':
            #eps_air = np.ones(Ep.size)*0.9975+1e-1*1j
            #eps_air = np.ones(Ep.size)+1e-3*1j
            eps_air = np.ones(Ep.size)
            eps[i] = eps_air
        elif epslist[i]=='gaas_p':
            eps_gaas_p = perm.permittivity_gaas_casey_p(Ep)
            eps[i] = eps_gaas_p
        elif epslist[i]=='gaas_n':
            eps_gaas_n = perm.permittivity_gaas_casey_n(Ep)
            eps[i] = eps_gaas_n
        elif epslist[i]=='zn':
            eps_zn_we = perm.permittivity_zn_werner_exp(Ep)
            eps_zn_wd = perm.permittivity_zn_werner_dft(Ep)
            eps_zn_q = perm.permittivity_zn_querry(Ep)
            eps[i] = eps_zn_we
        elif epslist[i]=='auzn':
            eps_auzn_23 = perm.permittivity_auzn_23p4(Ep)
            eps_auzn_12 = perm.permittivity_auzn_12p5(Ep)
            eps[i] = eps_auzn_23
        elif epslist[i]=='sin':
            eps_si3n4 = perm.permittivity_si3n4_palik(Ep)
            eps_sinx_k = perm.permittivity_sinx_kischkat(Ep)
            eps_sinx_l = perm.permittivity_sinx_lpn(Ep)
            #eps_sin = 4*np.ones(Ep.size)
            eps_sin = perm.permittivity_sin(Ep)
            eps[i] = eps_sinx_k
        elif epslist[i]=='alas':
            eps_alas = perm.permittivity_alas_ioffe(Ep)
            eps[i] = eps_alas
        elif epslist[i]=='al2o3':
            eps_al2o3 = perm.permittivity_al2o3_palik_n0k0(Ep)
            eps[i] = eps_al2o3
        else:
            print('Warning: unknown material. eps set to zero.')
    eps = eps.T
    return eps



def solve_optical_properties_single_E(eps,mu,L,N,Ep,kx,z0,dEF,epssubsL=None,epssubsR=None):
    # Solve photon numbers, Poynting vectors and recombination-generation for a
    # single energy Ep as a function of position and K number with source
    # coordinates z0.
    # This also calculates the complex electric field amplitudes (a.u.) of the
    # TE and TM fields based on Eq. (A3) of Phys. Rev. E 98, 063304 (2018),
    # assuming an equally strong source amplitude in each direction at each source point.
    # 
    # Deals with the divergences by flagging the solutions that diverge.
    # For solutions with a negative rate in the emitting layer, it also flags the solution.
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
        
        N_ledend = N_cumul[2]
        rad_TE_intled = np.trapz(rad_TE[j,0:N_ledend],z[0:N_ledend])
        rad_TE_intabove = np.trapz(rad_TE[j,N_ledend:],z[N_ledend:])
        rad_TM_intled = np.trapz(rad_TM[j,0:N_ledend],z[0:N_ledend])
        rad_TM_intabove = np.trapz(rad_TM[j,N_ledend:],z[N_ledend:])
        if np.abs(rad_TE_intabove) > np.abs(rad_TE_intled):
            failures_TE.append(j)
        if rad_TE_intled < 0:
            failures_TE.append(j)
        if np.abs(rad_TM_intabove) > np.abs(rad_TM_intled):
            failures_TM.append(j)
     
    print(failures_TE)
    print(failures_TM)
    
    kx_mat = np.tile(kx,(z.size,1)).T
    qte_w = np.trapz(rad_TE*2*pi*kx_mat,kx,axis=0)
    qtm_w = np.trapz(rad_TM*2*pi*kx_mat,kx,axis=0)
    
    return pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, qte_w, qtm_w, E1, E2, E3



def solve_optical_properties_single_E_zero(eps,mu,L,N,Ep,kx,z0,dEF,epssubsL=None,epssubsR=None):
    # Solve photon numbers, Poynting vectors and recombination-generation for a
    # single energy Ep as a function of position and K number with source
    # coordinates z0.
    # This also calculates the complex electric field amplitudes (a.u.) of the
    # TE and TM fields based on Eq. (A3) of Phys. Rev. E 98, 063304 (2018),
    # assuming an equally strong source amplitude in each direction at each source point.
    # 
    # Deals with the divergences by zeroing the quantities to the right from the emitting layer for solutions that diverge.
    # For solutions with a negative rate in the emitting layer, zeroes the whole solution
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
        
        N_ledend = N_cumul[2]
        rad_TE_intled = np.trapz(rad_TE[j,0:N_ledend],z[0:N_ledend])
        rad_TE_intabove = np.trapz(rad_TE[j,N_ledend:],z[N_ledend:])
        rad_TM_intled = np.trapz(rad_TM[j,0:N_ledend],z[0:N_ledend])
        rad_TM_intabove = np.trapz(rad_TM[j,N_ledend:],z[N_ledend:])
        if np.abs(rad_TE_intabove) > np.abs(rad_TE_intled):
            rad_TE[j,N_ledend:] = 0*rad_TE[j,N_ledend:]
            E1[j,N_ledend:] = 0*E1[j,N_ledend:]
            failures_TE.append(j)
        if rad_TE_intled < 0:
            rad_TE[j] = 0*rad_TE[j]
            E1[j] = 0*E1[j]
            failures_TE.append(j)
        if np.abs(rad_TM_intabove) > np.abs(rad_TM_intled):
            rad_TM[j,N_ledend:] = 0*rad_TM[j,N_ledend:]
            E2[j,N_ledend:] = 0*E2[j,N_ledend:]
            E3[j,N_ledend:] = 0*E3[j,N_ledend:]
            failures_TM.append(j)
     
    print(failures_TE)
    print(failures_TM)
    
    kx_mat = np.tile(kx,(z.size,1)).T
    qte_w = np.trapz(rad_TE*2*pi*kx_mat,kx,axis=0)
    qtm_w = np.trapz(rad_TM*2*pi*kx_mat,kx,axis=0)
    
    return pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, qte_w, qtm_w, E1, E2, E3



def solve_optical_properties_single_E_remove(eps,mu,L,N,Ep,kx,z0,dEF,epssubsL=None,epssubsR=None,i_ledend=2):
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
        
        N_ledend = N_cumul[i_ledend]
        rad_TE_intled = np.trapz(rad_TE[j,0:N_ledend],z[0:N_ledend])
        rad_TE_intabove = np.trapz(rad_TE[j,N_ledend:],z[N_ledend:])
        rad_TM_intled = np.trapz(rad_TM[j,0:N_ledend],z[0:N_ledend])
        rad_TM_intabove = np.trapz(rad_TM[j,N_ledend:],z[N_ledend:])
        if np.abs(rad_TE_intabove) > np.abs(rad_TE_intled):
            failures_TE.append(j)
        elif rad_TE_intled < 0:
            failures_TE.append(j)
        if np.abs(rad_TM_intabove) > np.abs(rad_TM_intled):
            failures_TM.append(j)
    
    print(failures_TE)
    print(failures_TM)
    kx_TE = np.delete(kx,failures_TE,axis=None)
    kx_TM = np.delete(kx,failures_TM,axis=None)
    rad_TE = np.delete(rad_TE,failures_TE,axis=0)
    rad_TM = np.delete(rad_TM,failures_TM,axis=0)
    P_TE = np.delete(P_TE,failures_TE,axis=0)
    P_TM = np.delete(P_TM,failures_TM,axis=0)
    E1 = np.delete(E1,failures_TE,axis=0)
    E2 = np.delete(E2,failures_TM,axis=0)
    E3 = np.delete(E3,failures_TM,axis=0)
    
    kx_mat_TE = np.tile(kx_TE,(z.size,1)).T
    kx_mat_TM = np.tile(kx_TM,(z.size,1)).T
    qte_w = np.trapz(rad_TE*2*pi*kx_mat_TE,kx_TE,axis=0)
    qtm_w = np.trapz(rad_TM*2*pi*kx_mat_TM,kx_TM,axis=0)
    
    return pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, qte_w, qtm_w, E1, E2, E3, kx_TE, kx_TM



def solve_recombination_energy_spread(L,N,eps,mu,Ep,Kmax,N_K,z0,dEF,epssubsL=np.array([]),epssubsR=np.array([])):
    # Solve recombination-generation as a function of position and photon
    # energy Ep with source coordinates z0.
    # See the function solve_optical_properties_single_E above for equations.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
    #       N: Vector including the intended number of datapoints in each layer
    #       eps: Vector including the permittivity in each layer
    #       mu: Vector including the permeability in each layer
    #       Ep: Vector including photon energies (in eV)
    #       Kmax: Maximum in-plane k number to be used (in 1/m)
    #       N_K: Desired number of in-plane k number values to perform the calculation for
    #       z0: Vector of source coordinates (in m)
    #       dEF: Quasi-Fermi level separation at the source coordinates (in eV)
    # Outputs:
    #       qte_w: Net recombination-generation rate in TE modes as a function of Ep and z (rad_TE integrated over K)
    #       qtm_w: Net recombination-generation rate in TM modes as a function of Ep and z (rad_TM integrated over K)
    
    z = oap.distribute_z_uneven(L,N)
    
    qte_w = np.zeros((Ep.size, z.size), dtype=complex)
    qtm_w = np.zeros((Ep.size, z.size), dtype=complex)
    
    for i in range(Ep.size):
        K = np.linspace(0, Kmax[i], N_K)
        
        if epssubsL.any():
            if epssubsR.any():
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w[i], qtm_w[i], E1, E2, E3 = solve_optical_properties_single_E(\
                    eps[i],mu[i],L,N,Ep[i],K,z0,dEF,epssubsL[i],epssubsR[i])
            else:
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w[i], qtm_w[i], E1, E2, E3 = solve_optical_properties_single_E(\
                    eps[i],mu[i],L,N,Ep[i],K,z0,dEF,epssubsL[i])
        else:
            pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                qte_w[i], qtm_w[i], E1, E2, E3 = solve_optical_properties_single_E(\
                eps[i],mu[i],L,N,Ep[i],K,z0,dEF)

    return qte_w, qtm_w



def solve_recombination_energy_spread_remove(L,N,eps,mu,Ep,Kmax,N_K,z0,dEF,epssubsL=np.array([]),epssubsR=np.array([]),i_ledend=2):
    # Solve recombination-generation as a function of position and photon
    # energy Ep with source coordinates z0.
    # See the function solve_optical_properties_single_E_remove above for equations.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
    #       N: Vector including the intended number of datapoints in each layer
    #       eps: Vector including the permittivity in each layer
    #       mu: Vector including the permeability in each layer
    #       Ep: Vector including photon energies (in eV)
    #       Kmax: Maximum in-plane k number to be used (in 1/m)
    #       N_K: Desired number of in-plane k number values to perform the calculation for
    #       z0: Vector of source coordinates (in m)
    #       dEF: Quasi-Fermi level separation at the source coordinates (in eV)
    # Outputs:
    #       qte_w: Net recombination-generation rate in TE modes as a function of Ep and z (rad_TE integrated over K)
    #       qtm_w: Net recombination-generation rate in TM modes as a function of Ep and z (rad_TM integrated over K)
    
    z = oap.distribute_z_uneven(L,N)
    
    qte_w = np.zeros((Ep.size, z.size), dtype=complex)
    qtm_w = np.zeros((Ep.size, z.size), dtype=complex)
    
    for i in range(Ep.size):
        K = np.linspace(0, Kmax[i], N_K)
        
        if epssubsL.any():
            if epssubsR.any():
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w[i], qtm_w[i], E1, E2, E3, kx_te, kx_tm = solve_optical_properties_single_E_remove(\
                    eps[i],mu[i],L,N,Ep[i],K,z0,dEF,epssubsL=epssubsL[i],epssubsR=epssubsR[i],i_ledend=i_ledend)
            else:
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w[i], qtm_w[i], E1, E2, E3, kx_te, kx_tm = solve_optical_properties_single_E_remove(\
                    eps[i],mu[i],L,N,Ep[i],K,z0,dEF,epssubsL=epssubsL[i],i_ledend=i_ledend)
        else:
            pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                qte_w[i], qtm_w[i], E1, E2, E3, kx_te, kx_tm = solve_optical_properties_single_E_remove(\
                eps[i],mu[i],L,N,Ep[i],K,z0,dEF,i_ledend=i_ledend)
    
    return qte_w, qtm_w



def calculate_radiances_energy_spread_remove(L,N,eps,mu,Ep,Kmax,N_K,z0,dEF,epssubsL=np.array([]),epssubsR=np.array([]),i_ledend=2):
    # Solve radiances as a function of position and photon
    # energy Ep with source coordinates z0.
    # See the function solve_optical_properties_single_E above for equations.
    # Inputs:
    #     L: Vector including the lengths of subsequent layers (in m)
    #     N: Vector including the intended number of datapoints in each layer
    #    eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     Ep: Vector including photon energies (in eV)
    #     Kmax: Maximum in-plane k number to be used (in 1/m)
    #    N_K: Desired number of in-plane k number values to perform the calculation for
    #     z0: Vector of source coordinates (in m)
    #     dEF: Quasi-Fermi level separation at the source coordinates (in eV)
    # Outputs:
    #     PTE_w: Spectral radiance in TE modes as a function of Ep and z (P_TE integrated over K)
    #     PTM_w: Spectral radiance in TM modes as a function of Ep and z (P_TM integrated over K)

    z = oap.distribute_z_uneven(L,N)

    PTE_w = np.zeros((Ep.size, z.size), dtype=complex)
    PTM_w = np.zeros((Ep.size, z.size), dtype=complex)

    for i, E in enumerate(Ep):
        K = np.linspace(0, Kmax[i], N_K)

        if epssubsL.any():
            if epssubsR.any():
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w, qtm_w, E1, E2, E3, kx_TE, kx_TM = solve_optical_properties_single_E_remove(\
                    eps[i],mu[i],L,N,E,K,z0,dEF,epssubsL=epssubsL[i],epssubsR=epssubsR[i],i_ledend=i_ledend)
            else:
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w, qtm_w, E1, E2, E3, kx_TE, kx_TM = solve_optical_properties_single_E_remove(\
                    eps[i],mu[i],L,N,E,K,z0,dEF,epssubsL=epssubsL[i],i_ledend=i_ledend)
        else:
            pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                qte_w, qtm_w, E1, E2, E3, kx_TE, kx_TM = solve_optical_properties_single_E_remove(\
                eps[i],mu[i],L,N,E,K,z0,dEF,i_ledend=i_ledend)

        K_mat_TE = np.tile(kx_TE,(z.size,1)).T
        K_mat_TM = np.tile(kx_TM,(z.size,1)).T
        PTE_w[i] = np.trapz(P_TE*2*pi*K_mat_TE,kx_TE,axis=0)
        PTM_w[i] = np.trapz(P_TM*2*pi*K_mat_TM,kx_TM,axis=0)

    return PTE_w, PTM_w



def plot_recombination_single_E_remove(L,N,Ep,K_TE,K_TM,rad_TE,rad_TM):
    # Function to plot recombination-generation rate for a single energy as a function of position and propagation angle.
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
    #       N: Vector including the intended number of datapoints in each layer
    #       Ep: Photon energy (in eV)
    #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #       rad_TE: Recombination-generation rate for TE calculated with solve_optical_properties_single_E_remove above
    #       rad_TM: Recombination-generation rate for TM calculated with solve_optical_properties_single_E_remove above
    
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
    plt.ylabel('Energy (eV)')
    plt.text(0.2,80,'(a)',color="white")
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(zplot_TM, Tplot_TM, rad_TM_tweak, cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta_TM), np.max(theta_TM)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Energy (eV)')
    plt.text(0.2,80,'(b)',color="white")
    plt.colorbar()
    plt.tight_layout()
    plt.show()



def plot_Efield_single_E_remove(L,N,Ep,K_TE,K_TM,E1,E2,E3):
    # Plot normalized electric field amplitude for a given photon energy as a function of position and propagation angle
    # Inputs:
    #       L: Vector including the lengths of subsequent layers (in m)
    #       N: Vector including the intended number of datapoints in each layer
    #       Ep: Photon energy (in eV)
    #       K: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #       E1: Electric field component E1 returned by the function solve_optical_properties_single_E_remove above
    #       E2: Electric field component E2 returned by the function solve_optical_properties_single_E_remove above
    #       E3: Electric field component E3 returned by the function solve_optical_properties_single_E_remove above
    
    z = oap.distribute_z_uneven(L,N)
    theta_TE = oap.propagation_angles_gaas(Ep,K_TE)
    theta_TM = oap.propagation_angles_gaas(Ep,K_TM)
    zplot_TE, Tplot_TE = np.meshgrid(z*1e6, theta_TE)
    zplot_TM, Tplot_TM = np.meshgrid(z*1e6, theta_TM)
    
    cmappi = colormap_SCpaper_yel()
    
    plt.figure(figsize=(12,5),facecolor=(1,1,1))
    plt.subplot(131)
    plt.pcolormesh(zplot_TE, Tplot_TE, np.abs(E1)/(np.max(np.abs(E1))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta_TE), np.max(theta_TE)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.ylabel('Angle (deg.)')
    plt.subplot(132)
    plt.pcolormesh(zplot_TM, Tplot_TM, np.abs(E2)/(np.max(np.abs(E2))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta_TM), np.max(theta_TM)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.subplot(133)
    plt.pcolormesh(zplot_TM, Tplot_TM, np.abs(E3)/(np.max(np.abs(E3))), cmap=cmappi, shading='gouraud')
    plt.axis([np.min(z)*1e6, np.max(z)*1e6, np.min(theta_TM), np.max(theta_TM)])
    plt.xlabel(r'Position ($\mu$m)')
    plt.colorbar()
    plt.show()



def colormap_SCpaper():
    # Help function to create a suitable colormap
    cdict1 = {'red': ((0.0,0.0,0.0), (0.375,0.0,0.0), (0.5,0.0,0.0), (0.625,1.0,1.0), (1.0,1.0,1.0)),
        'green': ((0.0,1.0,1.0), (0.375,0.5,0.5), (0.5,0.0,0.0), (0.625,0.0,0.0), (1.0,1.0,1.0)),
        'blue': ((0.0,1.0,1.0), (0.375,1.0,1.0), (0.5,0.0,0.0), (0.625,0.0,0.0), (1.0,0.0,0.0))}
    SCpap_cmap = LinearSegmentedColormap('BlueRed1', cdict1)
    
    return SCpap_cmap



def colormap_SCpaper_yel():
    # Help function to create a suitable colormap
    cdict1 = {'red': ((0.0,0.0,0.0), (0.25,1.0,1.0), (1.0,1.0,1.0)),
        'green': ((0.0,0.0,0.0), (0.25,0.0,0.0), (1.0,1.0,1.0)),
        'blue': ((0.0,0.0,0.0), (1.0,0.0,0.0))}
    SCpap_cmap_yel = LinearSegmentedColormap('BlueRed1', cdict1)
    
    return SCpap_cmap_yel


