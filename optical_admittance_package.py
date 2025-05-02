

import numpy as np
from numpy import pi
from scipy import integrate
import sys
sys.path.append('/u/05/pkivisaa/unix/tutkimus/optical_modeling/python3/permittivities')
import permittivities as perm


c = 299792458 
hplanck = 6.626e-34
hbar = hplanck/2/pi
q = 1.602e-19
kb = 1.38e-23
eps0 = 8.854e-12
mu0 = 1.257e-6



def distribute_z_uneven(L,N):
    # Create a vector of z coordinates.
    # Inputs:
    #     L: Vector including the lengths of subsequent layers (in m)
    #     N: Vector including the intended number of datapoints in each layer
    # Outputs:
    #     z: Vector including the z coordinates (in m)
    
    z = np.zeros(np.sum(N))
    last = 0
    position = 0
    
    for k in range(L.size):
        zk = np.linspace(last, last+L[k], N[k])
        z[position:(position+N[k])] = zk
        
        last = last + L[k]
        position += N[k]
    
    return z



def distribute_parameter(param,N):
    # Create a vector with a chosen material parameter that corresponds to the
    # z coordinates of a structure.
    # Inputs:
    #     param: Vector including the material parameter of subsequent layers
    #     N: Vector including the intended number of datapoints in each layer
    # Outputs:
    #     param_z: Vector including the material parameter at each z coordinate of the structure
    
    param_z = np.zeros(np.sum(N), dtype=complex)
    position = 0
    
    for j in range(N.size):
        param_z[position:(position+N[j])] = param[j]
        position += N[j]
    
    return param_z



def calculate_all_admittances_uneven(eps,mu,L,N,wl,kx,epssubs_l,musubs_l,epssubs_r,musubs_r):
    # Calculate the leftward- and rightward optical admittances for TE and TM satisfying Eqs. (11)-(16)
    # of Phys. Rev. E 98, 063304 (2018) for a single photon energy and propagation direction.
    # Inputs:
    #    eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     L: Vector including the lengths of subsequent layers (in m)
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #    epssubs_l: Permittivity at the left end of the structure
    #     musubs_l: Permeability at the left end of the structure
    #     epssubs_r: Permittivity at the right end of the structure
    #     musubs_r: Permeability at the right end of the structure
    # Outputs:
    #     gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    #     gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #     gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    #    gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    
    eps_r = eps[::-1]
    mu_r = mu[::-1]
    L_r = L[::-1]
    N_r = N[::-1]
    z = distribute_z_uneven(L,N)
    z_r = distribute_z_uneven(L_r,N_r)
    
    gamma_l_TE = admittance_layer_structure_TE(eps,mu,N,wl,kx,z,epssubs_l,musubs_l,-1)
    gamma_r_TE = admittance_layer_structure_TE(eps_r,mu_r,N_r,wl,kx,z_r,epssubs_r,musubs_r,-1)
    gamma_r_TE = gamma_r_TE[::-1] # This is important, since gamma_r was calculated by flipping the structure
    
    gamma_l_TM = admittance_layer_structure_TM(eps,mu,N,wl,kx,z,epssubs_l,musubs_l,-1)
    gamma_r_TM = admittance_layer_structure_TM(eps_r,mu_r,N_r,wl,kx,z_r,epssubs_r,musubs_r,-1)
    gamma_r_TM = gamma_r_TM[::-1] # This is important, since gamma_r was calculated by flipping the structure
    
    return gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM



def admittance_layer_structure_TE(eps,mu,N,wl,kx,z,epssubs,musubs,sgn):
    # Calculate the TE optical admittance satisfying Eq. (19) of Phys. Rev. E 98, 063304 (2018) for a single
    # photon energy and propagation direction.
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     z: Vector including the z coordinates (in m)
    #     epssubs: Permittivity at the end of the structure with the outgoing wave
    #     musubs: Permeabilit at the end of the structure with the outgoing wave
    #     sgn: Just a relic from debugging method. 1 or -1 depending on whether the time dependence is
    #        written as omega*t or -omega*t. Here we use -omegat*t, and this should be -1.
    # Outputs:
    #     gamma: Optical admittance for TE
    k0 = 2*pi/wl
    eta_s = sgn*np.sqrt(epssubs*musubs-kx**2/k0**2)/musubs
    gamma = np.zeros(z.size, dtype=complex)
    position = 0
    
    for j in range(N.size):
        if j == 0:
            init = eta_s
        else:
            init = gammazj[-1]
        zj = z[position:(position+N[j])] # Note that zj is a view of z, not a copy as in matlab
        kz = np.sqrt(eps[j]*mu[j]*k0**2-kx**2)
        kappa = -np.sqrt(eps[j]*mu[j]-kx**2/k0**2)/mu[j]
        gammazj = -kappa*np.tanh(1j*kz*(zj-zj[0])+np.arctanh(-init/kappa+1e-15*1j))
        gamma[position:(position+N[j])] = gammazj
        position += N[j]
    
    return gamma



def admittance_layer_structure_TM(eps,mu,N,wl,kx,z,epssubs,musubs,sgn):
    # Calculate the TM optical admittance satisfying Eq. (19) of Phys. Rev. E 98, 063304 (2018) for a single
    # photon energy and propagation direction.
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     z: Vector including the z coordinates (in m)
    #     epssubs: Permittivity at the end of the structure with the outgoing wave
    #     musubs: Permeabilit at the end of the structure with the outgoing wave
    #     sgn: Just a relic from debugging method. 1 or -1 depending on whether the time dependence is
    #        written as omega*t or -omega*t. Here we use -omegat*t, and this should be -1.
    # Outputs:
    #     gamma: Optical admittance for TM
    
    k0 = 2*pi/wl
    eta_s = sgn*np.sqrt(epssubs*musubs-kx**2/k0**2)/epssubs
    gamma = np.zeros(z.size, dtype=complex)
    position = 0
    
    for j in range(N.size):
        if j == 0:
            init = eta_s
        else:
            init = gammazj[-1]
        zj = z[position:(position+N[j])] # Note that zj is a view of z, not a copy as in matlab
        kz = np.sqrt(eps[j]*mu[j]*k0**2-kx**2)
        kappa = -np.sqrt(eps[j]*mu[j]-kx**2/k0**2)/eps[j]
        gammazj = -kappa*np.tanh(1j*kz*(zj-zj[0])+np.arctanh(-init/kappa+1e-15*1j))
        gamma[position:(position+N[j])] = gammazj
        position += N[j]
    
    return gamma



def LDOSes(eps,mu,N,wl,kx,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM):
    # Calculate local densities of states for a single photon energy and propagation direction
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #     gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    #     gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    #    gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    # Outputs:
    #     rho_e: Local density of states of Eq. (28) of Phys. Rev. E 98, 063304 (2018)
    #     rho_TE: Local density of states of Eq. (S5) of Sci. Rep. 7, 11534 (2017)
    #     rho_m: Local density of states of Eq. (29) of Phys. Rev. E 98, 063304 (2018)
    #     rho_TM: Local density of states of Eq. (S6) of Sci. Rep. 7, 11534 (2017)
    
    k0 = 2*pi/wl
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    
    firstterm_e = -1/(4*pi**3*c)*(1/(gamma_r_TE+gamma_l_TE)).real
    secondterm_e = -1/(4*pi**3*c)*(gamma_r_TM*gamma_l_TM/(gamma_r_TM+gamma_l_TM)).real
    thirdterm_e = -1/(4*pi**3*c)*((kx/k0)**2/((np.abs(eps_z))**2*(gamma_r_TM+gamma_l_TM))).real
    
    firstterm_m = -1/(4*pi**3*c)*(1/(gamma_r_TM+gamma_l_TM)).real
    secondterm_m = -1/(4*pi**3*c)*(gamma_r_TE*gamma_l_TE/(gamma_r_TE+gamma_l_TE)).real
    thirdterm_m = -1/(4*pi**3*c)*((kx/k0)**2/((np.abs(mu_z))**2*(gamma_r_TE+gamma_l_TE))).real
    
    rho_e = firstterm_e+secondterm_e+thirdterm_e
    rho_TE = np.abs(eps_z)*firstterm_e+np.abs(mu_z)*secondterm_m+np.abs(mu_z)*thirdterm_m
    rho_m = firstterm_m+secondterm_m+thirdterm_m
    rho_TM = np.abs(mu_z)*firstterm_m+np.abs(eps_z)*secondterm_e+np.abs(eps_z)*thirdterm_e
    
    return rho_e, rho_m, rho_TE, rho_TM



def electric_greens_functions(eps,mu,N,wl,kx,z,z0,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM):
    # Calculate the electric Green's function for a single source point following Section III of
    # the suppl. mat. of Sci. Rep. 7, 11534 (2017) and Appendix A of Phys. Rev. E 98, 063304 (2018)
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     z: Vector including the z coordinates (in m)
    #     z0: Source coordinate (in m)
    #     gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #     gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    #     gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    #    gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    # Outputs:
    #     gee11: Green's dyadic of Eq. (A4) in Phys. Rev. E 98, 063304 (2018)
    #     gee22: Green's dyadic of Eq. (A5) in Phys. Rev. E 98, 063304 (2018)
    #     gee33: Green's dyadic of Eq. (A8) in Phys. Rev. E 98, 063304 (2018)
    #     gee23: Green's dyadic of Eq. (A7) in Phys. Rev. E 98, 063304 (2018)
    #     gee32: Green's dyadic of Eq. (A6) in Phys. Rev. E 98, 063304 (2018)
    
    indexit = np.nonzero(z>z0)
    index = indexit[0][0]
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    k0 = 2*pi/wl
    
    Ufactor_gee11 = np.zeros(z.size, dtype=complex)
    Ufactor_gee22 = np.zeros(z.size, dtype=complex)
    Ufactor_gee33 = np.zeros(z.size, dtype=complex)
    Ufactor_gee23 = np.zeros(z.size, dtype=complex)
    Ufactor_gee32 = np.zeros(z.size, dtype=complex)
    
    if not index==0:
        seq_dec = np.arange(index+1)[::-1]
        Ufactor_gee11[seq_dec] = np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(mu_z[seq_dec]*gamma_r_TE[seq_dec], \
            z[seq_dec])),axis=None))
        Ufactor_gee22[seq_dec] = gamma_r_TM[index]*gamma_l_TM[seq_dec]*np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid( \
            eps_z[seq_dec]*gamma_l_TM[seq_dec],z[seq_dec])),axis=None))
        Ufactor_gee33[seq_dec] = np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq_dec]*gamma_l_TM[seq_dec], \
            z[seq_dec])),axis=None))
        Ufactor_gee23[seq_dec] = gamma_l_TM[seq_dec]*np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq_dec]* \
            gamma_r_TM[seq_dec],z[seq_dec])),axis=None))
        Ufactor_gee32[seq_dec] = -gamma_r_TM[index]*np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq_dec]* \
            gamma_l_TM[seq_dec],z[seq_dec])),axis=None))
    
    if not index==(z.size-1):
        seq = np.linspace(index,(z.size-1),(z.size-index),dtype=int)
        Ufactor_gee11[seq] = np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(mu_z[seq]*gamma_l_TE[seq],z[seq])),axis=None))
        Ufactor_gee22[seq] = gamma_l_TM[index]*gamma_r_TM[seq]*np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid( \
            eps_z[seq]*gamma_r_TM[seq],z[seq])),axis=None))
        Ufactor_gee33[seq] = np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq]*gamma_r_TM[seq],z[seq])),axis=None))
        Ufactor_gee23[seq] = -gamma_r_TM[seq]*np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq]*gamma_l_TM[seq], \
            z[seq])),axis=None))
        Ufactor_gee32[seq] = gamma_l_TM[index]*np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq]*gamma_r_TM[seq], \
            z[seq])),axis=None))
    
    gee11 = -1j/k0*1/(gamma_r_TE+gamma_l_TE)*Ufactor_gee11
    gee22 = -1j/k0*1/(gamma_l_TM[index]+gamma_r_TM[index])*Ufactor_gee22
    gee33 = -1j*(kx**2/k0**3)*1/(eps_z[index]*eps_z)*1/(gamma_r_TM[index]+gamma_l_TM[index])*Ufactor_gee33
    gee23 = 1j*(kx/k0**2)*1/(eps_z[index])*1/(gamma_l_TM+gamma_r_TM)*Ufactor_gee23
    gee32 = -kx/k0**2*1/eps_z*1j/(gamma_r_TM[index]+gamma_l_TM[index])*Ufactor_gee32
    
    return gee11, gee22, gee33, gee23, gee32



def exchange_greens_functions_me(eps,mu,N,wl,kx,z,z0,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM):
    # Calculate the exchange Green's function for a single source point following Section III of
    # the suppl. mat. of Sci. Rep. 7, 11534 (2017) and Appendix A of Phys. Rev. E 98, 063304 (2018)
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     z: Vector including the z coordinates (in m)
    #     z0: Source coordinate (in m)
    #     gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #     gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    #     gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    #    gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    # Outputs:
    #     gme12: Green's dyadic of Eq. (A11) in Phys. Rev. E 98, 063304 (2018)
    #     gme13: Green's dyadic of Eq. (A12) in Phys. Rev. E 98, 063304 (2018)
    #     gme21: Green's dyadic of Eq. (A9) in Phys. Rev. E 98, 063304 (2018)
    #     gme31: Green's dyadic of Eq. (A10) in Phys. Rev. E 98, 063304 (2018)
    
    indexit = np.nonzero(z>z0)
    index = indexit[0][0]
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    k0 = 2*pi/wl
    
    Ufactor_me12 = np.zeros(z.size, dtype=complex)
    Ufactor_me13 = np.zeros(z.size, dtype=complex)
    Ufactor_me21 = np.zeros(z.size, dtype=complex)
    Ufactor_me31 = np.zeros(z.size, dtype=complex)
    
    if not index==0:
        seq_dec = np.arange(index+1)[::-1]
        Ufactor_me12[seq_dec] = -gamma_r_TM[index]*np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq_dec]* \
            gamma_l_TM[seq_dec],z[seq_dec])),axis=None))
        Ufactor_me13[seq_dec] = np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq_dec]*gamma_l_TM[seq_dec], \
            z[seq_dec])),axis=None))
        Ufactor_me21[seq_dec] = gamma_l_TE[seq_dec]*np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(mu_z[seq_dec]* \
            gamma_r_TE[seq_dec],z[seq_dec])),axis=None))
        Ufactor_me31[seq_dec] = np.exp(1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(mu_z[seq_dec]*gamma_r_TE[seq_dec], \
            z[seq_dec])),axis=None))
    
    if not index==(z.size-1):
        seq = np.linspace(index,(z.size-1),(z.size-index),dtype=int)
        Ufactor_me12[seq] = gamma_l_TM[index]*np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq]* \
            gamma_r_TM[seq],z[seq])),axis=None))
        Ufactor_me13[seq] = np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(eps_z[seq]*gamma_r_TM[seq], \
            z[seq])),axis=None))
        Ufactor_me21[seq] = -gamma_r_TE[seq]*np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(mu_z[seq]* \
            gamma_l_TE[seq],z[seq])),axis=None))
        Ufactor_me31[seq] = np.exp(-1j*k0*np.concatenate((0,integrate.cumulative_trapezoid(mu_z[seq]*gamma_l_TE[seq],
            z[seq])),axis=None))
    
    gme12 = 1/k0*1/(gamma_r_TM[index]+gamma_l_TM[index])*Ufactor_me12
    gme13 = kx/k0**2*1/eps_z[index]*1/(gamma_r_TM[index]+gamma_l_TM[index])*Ufactor_me13
    gme21 = 1/k0*1/(gamma_l_TE+gamma_r_TE)*Ufactor_me21
    gme31 = -1/mu_z*kx/k0**2*1/(gamma_r_TE+gamma_l_TE)*Ufactor_me31
    
    return gme12, gme13, gme21, gme31



def NLDOS_TETM_electric_sources(eps,mu,N,wl,kx,z,z0,gee11,gee22,gee33,gee23,gee32,gme12,gme13,gme21,gme31):
    # Calculate the nonlocal densities of states of Eqs. (S1) and (S2) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    # (excluding magnetic sources)
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     z: Vector including the z coordinates (in m)
    #     z0: Source coordinate (in m)
    #     gee11, gee22, gee33, gee23, gee32, gme12, gme13, gme21, gme31: Green's dyadics
    # Outputs:
    #     rho_nl_TE: Nonlocal density of states for TE, Eq. (S1) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    #     rho_nl_TM: Nonlocal density of states for TM, Eq. (S1) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    
    k0 = 2*pi/wl
    indexit = np.nonzero(z>z0)
    index = indexit[0][0]
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    if kx.size>1:
        eps_z_mat = np.tile(eps_z, (kx.size, 1))
        mu_z_mat = np.tile(mu_z, (kx.size, 1))
    else:
        eps_z_mat = eps_z
        mu_z_mat = mu_z
    
    rho_nl_TE = k0**3/(4*pi**3*c)*(np.abs(eps_z_mat)*(np.imag(eps_z[index])*(np.abs(gee11))**2)+ \
        np.abs(mu_z_mat)*(np.imag(eps_z[index])*(np.abs(gme21))**2+np.imag(eps_z[index])* \
        (np.abs(gme31))**2))
    rho_nl_TM = k0**3/(4*pi**3*c)*(np.abs(mu_z_mat)*(np.imag(eps_z[index])*(np.abs(gme12))**2+ \
        np.imag(eps_z[index])*(np.abs(gme13))**2)+np.abs(eps_z_mat)*(np.imag(eps_z[index])* \
        (np.abs(gee22))**2+np.imag(eps_z[index])*(np.abs(gee23))**2+np.imag(eps_z[index])* \
        (np.abs(gee32))**2+np.imag(eps_z[index])*(np.abs(gee33))**2))
    
    return rho_nl_TE, rho_nl_TM



def IFDOS_TETM_electric_sources(eps,mu,N,wl,z,z0,gee11,gee22,gee23,gme12,gme13,gme21):
    # Calculate the interference densities of states of Eqs. (S3) and (S4) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    # (excluding magnetic sources)
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     z: Vector including the z coordinates (in m)
    #     z0: Source coordinate (in m)
    #     gee11, gee22, gee23, gme12, gme13, gme21: Green's dyadics
    # Outputs:
    #     rho_if_TE: Interference density of states for TE, Eq. (S3) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    #     rho_if_TM: Interference density of states for TM, Eq. (S4) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    
    k0 = 2*pi/wl
    indexit = np.nonzero(z>z0)
    index = indexit[0][0]
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    nr_z = np.real(np.sqrt(eps_z*mu_z))
    gme21c = np.conj(gme21)
    gme12c = np.conj(gme12)
    gme13c = np.conj(gme13)
    
    rho_if_TE = -k0**3*nr_z/(2*pi**3*c)*np.imag(np.imag(eps_z[index])*gee11*(gme21c))
    rho_if_TM = k0**3*nr_z/(2*pi**3*c)*np.imag(np.imag(eps_z[index])*gee22*(gme12c)+ \
        np.imag(eps_z[index])*gee23*(gme13c))
    
    return rho_if_TE, rho_if_TM



def NLDOS_electric_sources(eps,N,wl,z,z0,gee11,gee22,gee33,gee23,gee32,gme12,gme13,gme21,gme31):
    # Calculate the nonlocal densities of states of Eqs. (30) and (31) of Appendix A of Phys. Rev. E 98, 063304 (2018).
    # (excluding magnetic sources)
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     z: Vector including the z coordinates (in m)
    #     z0: Source coordinate (in m)
    #     gee11, gee22, gee33, gee23, gee32, gme12, gme13, gme21, gme31: Green's dyadics
    # Outputs:
    #     rho_nl_e: Electric nonlocal density of states, Eq. (30) of Appendix A of Phys. Rev. E 98, 063304 (2018)
    #     rho_nl_m: Magnetic nonlocal density of states, Eq. (31) of Appendix A of Phys. Rev. E 98, 063304 (2018)
    
    k0 = 2*pi/wl
    indexit = np.nonzero(z>z0)
    index = indexit[0][0]
    eps_z = distribute_parameter(eps,N)
    
    rho_nl_e = k0**3/(4*pi**3*c)*(np.imag(eps_z[index])*((np.abs(gee11))**2+(np.abs(gee22))**2+ \
        (np.abs(gee23))**2+(np.abs(gee32))**2+(np.abs(gee33))**2))
    rho_nl_m = k0**3/(4*pi**3*c)*(np.imag(eps_z[index])*((np.abs(gme12))**2+(np.abs(gme13))**2+ \
        (np.abs(gme21))**2+(np.abs(gme31))**2))
    
    return rho_nl_e, rho_nl_m



def RTE_coefficients_TE(eps,mu,N,wl,kx,gamma_l_TE,gamma_r_TE):
    # Calculate the coefficients for the interference-aware radiative transfer model for TE modes
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #     gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    # Outputs:
    #     alphap: The alpha+ coefficient of Eq. (S13), supplementary material of Sci. Rep. 7, 11534 (2017)
    #     alpham: The alpha- coefficient of Eq. (S13), supplementary material of Sci. Rep. 7, 11534 (2017)
    #     betap: The beta+ coefficient of Eq. (S14), supplementary material of Sci. Rep. 7, 11534 (2017)
    #     betam: The beta- coefficient of Eq. (S14), supplementary material of Sci. Rep. 7, 11534 (2017)
    
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    k0 = 2*pi/wl
    kr = np.real(np.sqrt(eps_z*mu_z)*k0)
    k = np.sqrt(eps_z*mu_z)*k0
    kz = np.sqrt(k**2-kx**2)
    
    rho_TE_norm = np.imag(-1j*(np.abs(eps_z)/(gamma_r_TE+gamma_l_TE)+ \
        np.abs(mu_z)*gamma_l_TE*gamma_r_TE/(gamma_l_TE+gamma_r_TE)+ \
        1/np.abs(mu_z)*(kx/k0)**2/(gamma_r_TE+gamma_l_TE)))
    
    stigma_c = np.imag(kz*(np.abs(k)**2+np.abs(kz)**2+kx**2)**2/(4*k0*kr**2*np.abs(kz)**2)+ \
        (k**2+kz**2)/(k0*kz))+1j*np.real(kz*(np.abs(k)**2-np.abs(kz)**2+kx**2)**2/ \
        (4*k0*kr**2*np.abs(kz)**2)-(k**2+kz**2)/(k0*kz))
    stigma_e_TE = 2*1j*eps_z+stigma_c*kz/(k0*mu_z)
    stigma_m_TE = 2*1j*mu_z+stigma_c*k0*mu_z/kz
    stigma_ex_TE = 1/(2*np.real(np.sqrt(eps_z*mu_z)))*(np.abs(eps_z)*mu_z-np.abs(mu_z)* \
        eps_z+2*np.real(mu_z)*kx**2/(k0**2*np.abs(mu_z)))
    
    gee11 = -1j/k0/(gamma_r_TE+gamma_l_TE)
    gmm22 = -1j/k0*gamma_l_TE*gamma_r_TE/(gamma_l_TE+gamma_r_TE)
    gmm33 = -kx**2/(k0**3)/(mu_z**2)*1j/(gamma_r_TE+gamma_l_TE)
    
    NLterms_alpha = (stigma_e_TE+np.imag(eps_z))*gee11+(stigma_m_TE+np.imag(mu_z))*gmm22+ \
        np.imag(mu_z)*mu_z**2/(np.abs(mu_z)**2)*gmm33
    NLterms_beta = (stigma_e_TE-np.imag(eps_z))*gee11+(stigma_m_TE-np.imag(mu_z))*gmm22- \
        np.imag(mu_z)*mu_z**2/(np.abs(mu_z)**2)*gmm33
    
    gdiff_TE = 1/k0*(gamma_l_TE-gamma_r_TE)/(gamma_r_TE+gamma_l_TE)
    PMterm_alpha = stigma_ex_TE*gdiff_TE
    stigma_ex_TE_c = np.conj(stigma_ex_TE)
    PMterm_beta = stigma_ex_TE_c*gdiff_TE
    
    alphap = kr*k0/rho_TE_norm*np.imag(NLterms_alpha+PMterm_alpha)
    alpham = kr*k0/rho_TE_norm*np.imag(NLterms_alpha-PMterm_alpha)
    betap = kr*k0/rho_TE_norm*np.imag(NLterms_beta+PMterm_beta)
    betam = kr*k0/rho_TE_norm*np.imag(NLterms_beta-PMterm_beta)
    
    return alphap, alpham, betap, betam



def RTE_coefficients_TM(eps,mu,N,wl,kx,gamma_l_TM,gamma_r_TM):
    # Calculate the coefficients for the interference-aware radiative transfer model for TM modes
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    #     gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    # Outputs:
    #     alphap: The alpha+ coefficient of Eq. (S15), supplementary material of Sci. Rep. 7, 11534 (2017)
    #     alpham: The alpha- coefficient of Eq. (S15), supplementary material of Sci. Rep. 7, 11534 (2017)
    #     betap: The beta+ coefficient of Eq. (S16), supplementary material of Sci. Rep. 7, 11534 (2017)
    #     betam: The beta- coefficient of Eq. (S16), supplementary material of Sci. Rep. 7, 11534 (2017)
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    k0 = 2*pi/wl
    kr = np.real(np.sqrt(eps_z*mu_z)*k0)
    k = np.sqrt(eps_z*mu_z)*k0
    kz = np.sqrt(k**2-kx**2)
    
    rho_TM_norm = np.imag(-1j*(np.abs(mu_z)/(gamma_r_TM+gamma_l_TM)+ \
        np.abs(eps_z)*gamma_r_TM*gamma_l_TM/(gamma_r_TM+gamma_l_TM)+ \
        (kx/k0)**2/np.abs(eps_z)*1/(gamma_r_TM+gamma_l_TM)))
    
    stigma_c = np.imag(kz*(np.abs(k)**2+np.abs(kz)**2+kx**2)**2/(4*k0*kr**2*np.abs(kz)**2)+ \
        (k**2+kz**2)/(k0*kz))+1j*np.real(kz*(np.abs(k)**2-np.abs(kz)**2+kx**2)**2/ \
        (4*k0*kr**2*np.abs(kz)**2)-(k**2+kz**2)/(k0*kz))
    stigma_e_TM = 2*1j*eps_z+stigma_c*k0*eps_z/kz
    stigma_m_TM = 2*1j*mu_z+stigma_c*kz/(k0*eps_z)
    stigma_ex_TM = -1/(2*np.real(np.sqrt(eps_z*mu_z)))*(np.abs(mu_z)*eps_z-np.abs(eps_z)* \
        mu_z+2*np.real(eps_z)*kx**2/(k0**2*np.abs(eps_z)))
    
    gmm11 = -1j/k0/(gamma_r_TM+gamma_l_TM)
    gee22 = -1j/k0*gamma_l_TM*gamma_r_TM/(gamma_l_TM+gamma_r_TM)
    gee33 = -1j*kx**2/(k0**3)/(eps_z**2)/(gamma_r_TM+gamma_l_TM)
    
    NLterms_alpha = (stigma_m_TM+np.imag(mu_z))*gmm11+(stigma_e_TM+np.imag(eps_z))*gee22+ \
        np.imag(eps_z)*eps_z**2/(np.abs(eps_z)**2)*gee33
    NLterms_beta = (stigma_m_TM-np.imag(mu_z))*gmm11+(stigma_e_TM-np.imag(eps_z))*gee22- \
        np.imag(eps_z)*eps_z**2/(np.abs(eps_z)**2)*gee33
    
    gdiff_TM = 1/k0*(gamma_r_TM-gamma_l_TM)/(gamma_r_TM+gamma_l_TM)
    PMterm_alpha = stigma_ex_TM*gdiff_TM
    stigma_ex_TM_conj = np.conj(stigma_ex_TM)
    PMterm_beta = stigma_ex_TM_conj*gdiff_TM
    
    alphap = kr*k0/rho_TM_norm*np.imag(NLterms_alpha+PMterm_alpha)
    alpham = kr*k0/rho_TM_norm*np.imag(NLterms_alpha-PMterm_alpha)
    betap = kr*k0/rho_TM_norm*np.imag(NLterms_beta+PMterm_beta)
    betam = kr*k0/rho_TM_norm*np.imag(NLterms_beta-PMterm_beta)
    
    return alphap, alpham, betap, betam



def LDOS_derivatives(eps,mu,N,wl,kx,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM):
    # Calculate the derivatives of the local densities of states
    # Inputs:
    #     eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     N: Vector including the intended number of datapoints in each layer
    #     wl: Wavelength of light in vacuum (in m)
    #     kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #     gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #     gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    #     gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    #     gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    # Outputs:
    #     drho_TE: Derivative of the LDOS for TE, Eq. (S7) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    #     drho_TM: Derivative of the LDOS for TM, Eq. (S8) of the supplementary material of Sci. Rep. 7, 11534 (2017)
    
    k0 = 2*pi/wl
    Ep = hplanck*c/wl/q
    omega = 2*pi*Ep*q/hplanck # Why the **** am I doing this by first calculating Ep, which is not needed elsewhere in this function?? Totally ******* idiotic.
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    
    k = np.sqrt(eps_z*mu_z)*k0
    kz = np.sqrt(k**2-kx**2)
    
    gme12 = 1/k0/(gamma_r_TM+gamma_l_TM)*1/2*(gamma_l_TM-gamma_r_TM)
    gem12 = -1/(k0*(gamma_l_TE+gamma_r_TE))*1/2*(gamma_l_TE-gamma_r_TE)
    gme21 = 1/k0/(gamma_l_TE+gamma_r_TE)*1/2*(gamma_l_TE-gamma_r_TE)
    gem21 = -1/(k0*(gamma_l_TM+gamma_r_TM))*1/2*(gamma_l_TM-gamma_r_TM)
    
    drho_TE = -omega**2/(4*pi**3*c**3)*np.imag((np.abs(eps_z)*mu_z-np.abs(mu_z)*kz**2/ \
        (k0**2*mu_z)+mu_z*kx**2/(k0**2*np.abs(mu_z)))*(gem12-gme21))
    drho_TM = omega**2/(4*pi**3*c**3)*np.imag((np.abs(mu_z)*eps_z-np.abs(eps_z)*kz**2/ \
        (k0**2*eps_z)+eps_z*kx**2/(k0**2*np.abs(eps_z)))*(gme12-gem21))
    
    return drho_TE, drho_TM



def solve_optical_properties_single_E(eps,mu,L,N,Ep,kx,z0,dEF,T,epssubsL=None,epssubsR=None):
    # Solve photon numbers, Poynting vectors and recombination-generation rates for a
    # single energy Ep and desired K numbers with source coordinates z0
    # Inputs:
    #    eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     L: Vector including the lengths of subsequent layers (in m)
    #     N: Vector including the intended number of datapoints in each layer
    #     Ep: Photon energy (in eV)
    #     kx: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     z0: Vector of source coordinates (in m)
    #     dEF: Quasi-Fermi level separation at the source coordinates (in eV)
    #     T: Temperature in K
    # Outputs:
    #     pup_TE: Rightward-propagating photon number in TE modes as a function of K and z, as in Eq. (5) of Sci. Rep. 7, 11534 (2017).
    #        See also Eq. (8) of Phys. Rev. A 92, 033839 (2015).
    #     pdown_TE: Leftward-propagating photon number in TE modes as a function of K and z, similarly as pup_TE
    #     pup_TM: Rightward-propagating photon number in TM modes as a function of K and z, similarly as pup_TE
    #     pdown_TM: Leftward-propagating photon number in TM modes as a function of K and z, similarly as pup_TE
    #     P_TE: Spectral radiance in TE modes as a function of K and z, based on Eq. (6) of Phys. Rev. A 92, 033839 (2015).
    #     P_TM: Spectral radiance in TM modes as a function of K and z, based on Eq. (6) of Phys. Rev. A 92, 033839 (2015).
    #     rad_TE: Net recombination-generation rate in TE modes as a function of K and z, calculated as a derivative of P_TE.
    #     rad_TM: Net recombination-generation rate in TM modes as a function of K and z, calculated as a derivative of P_TM.
    #     qte_w: Net recombination-generation rate in TE modes as a function of z (rad_TE integrated over K)
    #     qtm_w: Net recombination-generation rate in TM modes as a function of z (rad_TM integrated over K)
    
    omega = 2*pi*Ep*q/hplanck
    z = distribute_z_uneven(L,N)
    
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    eta = 1/(np.exp(q*(Ep-dEF)/kb/T)-1)
    
    indexit = np.nonzero(np.logical_and(z>z0.min(),z<z0.max()))
    index = indexit[0]
    eta_QFED = np.zeros(np.size(z))
    eta_QFED[index] = eta
    
    eps_z = distribute_parameter(eps,N)
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
    
    print("Solving for E =", Ep, "eV...")
    for j in range(kx.size):
        gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = calculate_all_admittances_uneven( \
            eps,mu,L,N,wl,kx[j],epssubs_l,musubs_l,epssubs_r,musubs_r)
        rho_e, rho_m, rho_TE, rho_TM = LDOSes(eps,mu,N,wl,kx[j],gamma_l_TE,gamma_r_TE, \
            gamma_r_TM,gamma_l_TM)
        
        rho_nl_TE = np.zeros((z0.size, z.size), dtype=complex)
        rho_nl_TM = np.zeros((z0.size, z.size), dtype=complex)
        rho_if_TE = np.zeros((z0.size, z.size), dtype=complex)
        rho_if_TM = np.zeros((z0.size, z.size), dtype=complex)
        for k in range(z0.size):
            gee11, gee22, gee33, gee23, gee32 = electric_greens_functions(eps,mu,N,wl, \
                kx[j],z,z0[k],gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            gme12, gme13, gme21, gme31 = exchange_greens_functions_me(eps,mu,N,wl,kx[j], \
                z,z0[k],gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            rho_nl_TE[k], rho_nl_TM[k] = NLDOS_TETM_electric_sources(eps,mu,N,wl,kx[j],z,z0[k], \
                gee11,gee22,gee33,gee23,gee32,gme12,gme13,gme21,gme31)
            rho_if_TE[k], rho_if_TM[k] = IFDOS_TETM_electric_sources(eps,mu,N,wl,z,z0[k], \
                gee11,gee22,gee23,gme12,gme13,gme21)
        # Upward and downward photon numbers calculated using Eq. (5) of Sci.
        # Rep. 7, 11534 (2017).
        pup_TE[j] = 1/rho_TE*np.trapz((rho_nl_TE+rho_if_TE)*eta,z0,axis=0)
        pdown_TE[j] = 1/rho_TE*np.trapz((rho_nl_TE-rho_if_TE)*eta,z0,axis=0)
        pup_TM[j] = 1/rho_TM*np.trapz((rho_nl_TM+rho_if_TM)*eta,z0,axis=0)
        pdown_TM[j] = 1/rho_TM*np.trapz((rho_nl_TM-rho_if_TM)*eta,z0,axis=0)
        
        alphap_TE, alpham_TE, betap_TE, betam_TE = RTE_coefficients_TE(eps,mu,N,wl,kx[j], \
            gamma_l_TE,gamma_r_TE)
        alphap_TM, alpham_TM, betap_TM, betam_TM = RTE_coefficients_TM(eps,mu,N,wl,kx[j], \
            gamma_l_TM,gamma_r_TM)
        
        drho_TE, drho_TM = LDOS_derivatives(eps,mu,N,wl,kx[j],gamma_l_TE,gamma_r_TE,gamma_r_TM, \
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
    
    kx_mat = np.tile(kx,(z.size,1)).T
    qte_w = np.trapz(rad_TE*2*pi*kx_mat,kx,axis=0)
    qtm_w = np.trapz(rad_TM*2*pi*kx_mat,kx,axis=0)
    
    return pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, qte_w, qtm_w



def EM_field_fluctuations_single_E(eps,mu,L,N,Ep,kx,z0,dEF,T):
    # This has been used for testing. Not currently used for anything I believe. Whatever this does, I think this
    # was based on Phys. Rev. A 95, 013848 (2017).
    omega = 2*pi*Ep*q/hplanck
    z = distribute_z_uneven(L,N)
    
    wl = hplanck*c/Ep/q
    eta = 1/(np.exp(q*(Ep-dEF)/kb/T)-1)
    k0 = 2*pi/wl
    
    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    epssubs_l = eps[0]
    epssubs_r = eps[-1]
    musubs_l = mu[0]
    musubs_r = mu[-1]
    
    rho_e = np.zeros((kx.size, z.size), dtype=complex)
    rho_m = np.zeros((kx.size, z.size), dtype=complex)
    rho_tot = np.zeros((kx.size, z.size), dtype=complex)
    p_e = np.zeros((kx.size, z.size), dtype=complex)
    p_m = np.zeros((kx.size, z.size), dtype=complex)
    p_tot = np.zeros((kx.size, z.size), dtype=complex)
    
    for j in range(kx.size):
        gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = calculate_all_admittances_uneven( \
            eps,mu,L,N,wl,kx[j],epssubs_l,musubs_l,epssubs_r,musubs_r)
        rho_e[j], rho_m[j], rho_TE, rho_TM = LDOSes(eps,mu,N,wl,kx[j],gamma_l_TE,gamma_r_TE, \
            gamma_r_TM,gamma_l_TM)
        rho_tot[j] = 0.5*(np.abs(eps_z)*rho_e[j]+np.abs(mu_z)*rho_m[j])
        
        rho_nl_e = np.zeros((z0.size, z.size), dtype=complex)
        rho_nl_m = np.zeros((z0.size, z.size), dtype=complex)
        rho_nl_tot = np.zeros((z0.size, z.size), dtype=complex)
        
        for k in range(z0.size):
            gee11, gee22, gee33, gee23, gee32 = \
                electric_greens_functions(eps,mu,N,wl,kx[j],z,z0[k], \
                gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            gme12, gme13, gme21, gme31 = exchange_greens_functions_me( \
                eps,mu,N,wl,kx[j],z,z0[k],gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            rho_nl_e[k], rho_nl_m[k] = NLDOS_electric_sources(eps,N,wl,z,z0[k],gee11,gee22, \
                gee33,gee23,gee32,gme12,gme13,gme21,gme31)
            rho_nl_tot[k] = 0.5*(np.abs(eps_z)*rho_nl_e[k]+np.abs(mu_z)*rho_nl_m[k])
        p_e[j] = 1/rho_e[j]*np.trapz(rho_nl_e*eta,z0,axis=0)
        p_m[j] = 1/rho_m[j]*np.trapz(rho_nl_m*eta,z0,axis=0)
        p_tot[j] = 1/rho_tot[j]*np.trapz(rho_nl_tot*eta,z0,axis=0)
        
        print("Solved for E =", Ep, "eV and K/k0 =", kx[j]/k0)
    
    E2Kw = hbar*omega/eps0*rho_e*(p_e+0.5)
    H2Kw = hbar*omega/mu0*rho_m*(p_m+0.5)
    uKw = hbar*omega*rho_tot*(p_tot+0.5)
    
    kx_mat = np.tile(kx,(z.size,1)).T
    E2w = np.trapz(E2Kw*2*pi*kx_mat,kx,axis=0)
    H2w = np.trapz(H2Kw*2*pi*kx_mat,kx,axis=0)
    uw = np.trapz(uKw*2*pi*kx_mat,kx,axis=0)
    
    return E2Kw, H2Kw, uKw, E2w, H2w, uw



def Efield_single_E(eps,mu,L,N,Ep,kx,z0,epssubs=None):
    # Calculate the complex electric field amplitudes (a.u.) of the TE and TM fields based on Eq. (A3) of
    # Phys. Rev. E 98, 063304 (2018), assuming an equally strong source amplitude in each direction at each source point.
    # Inputs:
    #    eps: Vector including the permittivity in each layer
    #     mu: Vector including the permeability in each layer
    #     L: Vector including the lengths of subsequent layers (in m)
    #     N: Vector including the intended number of datapoints in each layer
    #     Ep: Photon energy (in eV)
    #     kx: Vector of K values (lateral components of the k vector, essentially propagation directions) (in 1/m)
    #     z0: Vector of source coordinates (in m)
    # Outputs:
    #     E1: Complex amplitude of the TE electric field as a function of K and z
    #     E2: Complex amplitude of the in-plane component of the TM electric field as a function of K and z
    #     E3: Complex amplitude of the normal component of the TM electric field as a function of K and z
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    eps_z = distribute_parameter(eps,N)
    nr_z = np.real(np.sqrt(eps_z))
    
    if epssubs == None:
        epssubs_l = eps[0]
        epssubs_r = eps[-1]
        musubs_l = mu[0]
        musubs_r = mu[-1]
    else:
        epssubs_l = epssubs
        epssubs_r = epssubs
        musubs_l = 1
        musubs_r = 1
    
    z = distribute_z_uneven(L,N)
    
    E1 = np.zeros((kx.size, z.size), dtype=complex)
    E2 = np.zeros((kx.size, z.size), dtype=complex)
    E3 = np.zeros((kx.size, z.size), dtype=complex)
    print("Solving for E =", Ep, "eV...")
    for j in range(kx.size):
        gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = calculate_all_admittances_uneven( \
            eps,mu,L,N,wl,kx[j],epssubs_l,musubs_l,epssubs_r,musubs_r)
        gee11 = np.zeros((z0.size, z.size), dtype=complex)
        gee22 = np.zeros((z0.size, z.size), dtype=complex)
        gee33 = np.zeros((z0.size, z.size), dtype=complex)
        gee23 = np.zeros((z0.size, z.size), dtype=complex)
        gee32 = np.zeros((z0.size, z.size), dtype=complex)
        for k in range(z0.size):
            gee11[k], gee22[k], gee33[k], gee23[k], gee32[k] = \
                electric_greens_functions(eps,mu,N,wl,kx[j],z,z0[k], \
                gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
        E1[j] = np.trapz(gee11*1,z0,axis=0)
        E2[j] = np.trapz(gee22*1+gee23*1,z0,axis=0)
        E3[j] = np.trapz(gee32*1+gee33*1,z0,axis=0)
        
    
    return E1, E2, E3



def solve_recombination_energy_spread(L,N,eps,mu,Ep,Kmax,N_K,z0,dEF,epssubsL=np.array([]),epssubsR=np.array([])):
    # Solve recombination-generation as a function of position and photon
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
    #     qte_w: Net recombination-generation rate in TE modes as a function of Ep and z (rad_TE integrated over K)
    #     qtm_w: Net recombination-generation rate in TM modes as a function of Ep and z (rad_TM integrated over K)
    
    z = distribute_z_uneven(L,N)
    
    qte_w = np.zeros((Ep.size, z.size), dtype=complex)
    qtm_w = np.zeros((Ep.size, z.size), dtype=complex)
    
    for i in range(Ep.size):
        K = np.linspace(0, Kmax[i], N_K)
        
        if epssubsL.any():
            if epssubsR.any():
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w[i], qtm_w[i] = solve_optical_properties_single_E(\
                    eps[i],mu[i],L,N,Ep[i],K,z0,dEF,epssubsL[i],epssubsR[i])
            else:
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w[i], qtm_w[i] = solve_optical_properties_single_E(\
                    eps[i],mu[i],L,N,Ep[i],K,z0,dEF,epssubsL[i])
        else:
            pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                qte_w[i], qtm_w[i] = solve_optical_properties_single_E(\
                eps[i],mu[i],L,N,Ep[i],K,z0,dEF)
    
    return qte_w, qtm_w



def calculate_radiances_energy_spread(L,N,eps,mu,Ep,Kmax,N_K,z0,dEF,epssubsL=np.array([]),epssubsR=np.array([])):
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

    z = distribute_z_uneven(L,N)

    PTE_w = np.zeros((Ep.size, z.size), dtype=complex)
    PTM_w = np.zeros((Ep.size, z.size), dtype=complex)

    for i, E in enumerate(Ep):
        K = np.linspace(0, Kmax[i], N_K)

        if epssubsL.any():
            if epssubsR.any():
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w, qtm_w = solve_optical_properties_single_E(\
                    eps[i],mu[i],L,N,E,K,z0,dEF,epssubsL[i],epssubsR[i])
            else:
                pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                    qte_w, qtm_w = solve_optical_properties_single_E(\
                    eps[i],mu[i],L,N,E,K,z0,dEF,epssubsL[i])
        else:
            pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
                qte_w, qtm_w = solve_optical_properties_single_E(\
                eps[i],mu[i],L,N,E,K,z0,dEF)

        K_mat = np.tile(K,(z.size,1)).T
        PTE_w[i] = np.trapz(P_TE*2*pi*K_mat,K,axis=0)
        PTM_w[i] = np.trapz(P_TM*2*pi*K_mat,K,axis=0)

    return PTE_w, PTM_w




def propagation_angles(Ep,K,eps,mu):
    # Help function to calculate the propagation angles corresponding to given in-plane k numbers
    # in the material specified by eps and mu for the single photon energy Ep 
    # Inputs:
    #     Ep: Photon energy (in eV)
    #     K: Vector including the in-plane k numbers (in 1/m)
    #     eps: Permittivity of the material corresponding to energy Ep
    #     mu: Permeability of the material corresponding to energy Ep
    # Outputs:
    #     theta: Vector including the propagation angles corresponding to K (in degrees)
    
    nr = np.real(np.sqrt(eps*mu))
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    # There should be an if-else or try-catch here. Now this gives a warning for K>nr*k0.
    # But it doesn't give an error.
    thetarad = np.arcsin(K/nr/k0)
    theta = thetarad/pi*180
    theta[np.nonzero(np.isnan(theta))]=90
    
    return theta



def propagation_angles_gaas(Ep,K):
    # Help function to calculate the propagation angles corresponding to given in-plane k numbers
    # in GaAs for the single photon energy Ep. Forces the angle to be real
    # Inputs:
    #     Ep: Photon energy (in eV)
    #     K: Vector including the in-plane k numbers (in 1/m)
    # Outputs:
    #     theta: Vector including the propagation angles corresponding to K (in degrees)
    eps = perm.permittivity_gaas_palik(Ep)
    mu = 1
    nr = np.real(np.sqrt(eps*mu))
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    # There should be an if-else or try-catch here. Now this gives a warning for K>nr*k0.
    # But it doesn't give an error.
    thetarad = np.arcsin(K/nr/k0)
    theta = thetarad/pi*180
    theta[np.nonzero(np.isnan(theta))]=90
    
    return theta



def propagation_angles_gaas_complex(Ep,K):
    # Help function to calculate the propagation angles corresponding to given in-plane k numbers
    # in GaAs for the single photon energy Ep. This allows the angle to be complex.
    # Inputs:
    #     Ep: Photon energy (in eV)
    #     K: Vector including the in-plane k numbers (in 1/m)
    # Outputs:
    #     theta: Vector including the propagation angles corresponding to K (in degrees)
    eps = perm.permittivity_gaas_palik(Ep)
    mu = 1
    nr = np.sqrt(eps*mu)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    # There should be an if-else or try-catch here. Now this gives a warning for K>nr*k0.
    # But it doesn't give an error.
    thetarad = np.arcsin(K/nr/k0)
    theta = thetarad/pi*180
    theta[np.nonzero(np.isnan(theta))]=90
    
    return theta



def Ks_gaas(Ep,theta):
    # Help function to calculate the Ks corresponding to given angles
    # in GaAs for the single photon energy Ep. Forces K to be real.
    # Inputs:
    #     Ep: Photon energy (in eV)
    #     theta: Vector including the propagation angles corresponding to K (in degrees)
    # Outputs:
    #     K: Vector including the in-plane k numbers (in 1/m)
    eps = perm.permittivity_gaas_palik(Ep)
    mu = 1
    nr = np.real(np.sqrt(eps*mu))
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    thetarad = theta/180*pi
    K = nr*k0*np.sin(thetarad)
    
    return K



def Ks_gaas_complex(Ep,theta):
    # Help function to calculate the Ks corresponding to given angles
    # in GaAs for the single photon energy Ep. Allows K to be complex.
    # Inputs:
    #     Ep: Photon energy (in eV)
    #     theta: Vector including the propagation angles corresponding to K (in degrees)
    # Outputs:
    #     K: Vector including the in-plane k numbers (in 1/m)
    eps = perm.permittivity_gaas_palik(Ep)
    mu = 1
    nr = np.sqrt(eps*mu)
    wl = hplanck*c/Ep/q
    k0 = 2*pi/wl
    
    thetarad = theta/180*pi
    K = nr*k0*np.sin(thetarad)
    
    return K



def reflectance_transmission_kx(eps,mu,N,wl,kx,z,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM):
    # Function to calculate the amplitude reflection coefficients for rightward and leftward
    # TE and TM modes
    # Inputs:
    #   eps: Permittivity of the material corresponding to wavelength wl
    #   mu : Permeability of the material corresponding to wavelength wl
    #   L: Vector including the lengths of subsequent layers (in m)
    #   wl: Wavelength of light in vacuum (in m)
    #   kx: Lateral component of the k vector, essentially propagation direction (in 1/m)
    #   z: Vector including the z coordinates (in m)
    #   gamma_l_TE: Optical admittance for TE for leftward-propagating modes
    #   gamma_r_TE: Optical admittance for TE for rightward-propagating modes
    #   gamma_r_TM: Optical admittance for TM for rightward-propagating modes
    #   gamma_l_TM: Optical admittance for TM for leftward-propagating modes
    # Outputs:
    #   r_l_TE: Amplitude reflection coefficient for leftward TE modes as a function of z
    #   r_r_TE: Amplitude reflection coefficient for rightward TE modes as a function of z
    #   r_r_TM: Amplitude reflection coefficient for rightward TM modes as a function of z
    #   r_l_TM: Amplitude reflection coefficient for leftward TM modes as a function of z

    eps_z = distribute_parameter(eps,N)
    mu_z = distribute_parameter(mu,N)
    k0 = 2*pi/wl

    eta_i_TE = -np.sqrt(eps_z*mu_z-kx**2/k0**2)/mu_z
    eta_i_TM = -np.sqrt(eps_z*mu_z-kx**2/k0**2)/eps_z

    r_l_TE = (eta_i_TE-gamma_l_TE)/(eta_i_TE+gamma_l_TE)
    r_r_TE = (eta_i_TE-gamma_r_TE)/(eta_i_TE+gamma_r_TE)
    r_l_TM = (eta_i_TM-gamma_l_TM)/(eta_i_TM+gamma_l_TM)
    r_r_TM = (eta_i_TM-gamma_r_TM)/(eta_i_TM+gamma_r_TM)

    return r_l_TE, r_r_TE, r_r_TM, r_l_TM

















