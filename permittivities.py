
import numpy as np
from numpy import pi

hplanck = 6.626e-34
c = 299792458
q = 1.6022e-19

def permittivity_gaas_palik(Evec):
    # Palik: "Handbook of Optical Constants of Solids," edited by Edward D.
    # Palik, Academic Press, 1998 (p429-443)
    
    data = np.loadtxt('nk_data/gaas_nk_palik.txt')
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    if Evec.size>1:
        # Add an infinitesimal imaginary part to the zeros due to convergence issues
        indices = np.nonzero(Evec<1.371)
        eps_imag[indices] = eps_imag[indices]+1e-6
    
    return eps_real+eps_imag*1j

def permittivity_algaas_palik(Evec):
    # Palik: Handbook of Optical Constants of Solids", edited by Edward D. Palik,
    # Academic Press, 1985 (p434-443)
    # For Al0.3GaAs, Palik and Aspnes are the same but Adachi has the same
    # problem after the bandgap as for the GaAs Pyry first used.
    
    data = np.loadtxt("nk_data/algaas_nk_palik.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    # Add an infinitesimal imaginary part to the zeros due to convergence issues
    indices = np.nonzero(Evec<1.8)
    #eps_imag[indices] = eps_imag[indices]+1e-6
    
    return eps_real+eps_imag*1j

def permittivity_algaas097_Papatryfonos(Evec):
    # K. Papatryfonos, T. Angelova, A. Brimont, B. Reid, S. Guldin, P. R. Smith,
    # M. Tang, K. Li, A. J. Seeds, H. Liu, D. R. Selviah. Refractive indices of
    # MBE-grown AlxGa1-xAs ternary alloys in the transparent wavelength region,
    # AIP Adv. 11, 025327 (2021) (Numerical data kindly provided by Konstantinos Papatryfonos)
    
    data = np.loadtxt("nk_data/Papatryfonos-AlGaAs097_nk.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_algaas219_Papatryfonos(Evec):
    # K. Papatryfonos, T. Angelova, A. Brimont, B. Reid, S. Guldin, P. R. Smith,
    # M. Tang, K. Li, A. J. Seeds, H. Liu, D. R. Selviah. Refractive indices of
    # MBE-grown AlxGa1-xAs ternary alloys in the transparent wavelength region,
    # AIP Adv. 11, 025327 (2021) (Numerical data kindly provided by Konstantinos Papatryfonos)
    
    data = np.loadtxt("nk_data/Papatryfonos-AlGaAs219_nk.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_algaas342_Papatryfonos(Evec):
    # K. Papatryfonos, T. Angelova, A. Brimont, B. Reid, S. Guldin, P. R. Smith,
    # M. Tang, K. Li, A. J. Seeds, H. Liu, D. R. Selviah. Refractive indices of
    # MBE-grown AlxGa1-xAs ternary alloys in the transparent wavelength region,
    # AIP Adv. 11, 025327 (2021) (Numerical data kindly provided by Konstantinos Papatryfonos)
    
    data = np.loadtxt("nk_data/Papatryfonos-AlGaAs342_nk.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_alas_ioffe(Evec):
    # Unknown reference in Ioffe database, "http://www.ioffe.ru/SVA/NSM/nk/" in the section
    # AluminumCompounds scroll to Aluminum Gallium Arsenide Al1Ga0As to get
    # "http://www.ioffe.ru/SVA/NSM/nk/AluminumCompounds/Gif/algaas10.gif"

    data = np.loadtxt("nk_data/alas_nk_ioffe.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_gainp(Evec):
    # Schubert, Gottschalch, Herzinger, Yao, Snyder, Woollam, J. Appl. Phys.
    # 77, 3416 (1995), taken from refractiveindex.info
    
    data = np.loadtxt("nk_data/gainp_nk.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    # add an infinitesimal imaginary part to the zeros due to convergence issues
    indices = np.nonzero(Evec<1.85)
    eps_imag[indices] = eps_imag[indices]+1e-6
    
    return eps_real+eps_imag*1j

def permittivity_mgf2(Evec):
    # L. V. Rodriguez-de Marcos, J. I. Larruquert, J. A. Mendez, J. A. Aznarez. Self-consistent optical 
    # constants of MgF2, LaF3, and CeF3 films, Opt. Mater. Express 7, 989-1006 (2017)
    # From refractiveindex.info
    
    n_real = np.loadtxt("nk_data/mgf2_realn.txt")
    n_imag = np.loadtxt("nk_data/mgf2_imagn.txt")
    
    Ephot = hplanck*c/n_real[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = n_real[::-1,1]
    nimag = n_imag[::-1,1]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_zns(Evec):
    # S. Ozaki and S. Adachi. Optical constants of cubic ZnS, Jpn. J. Appl. Phys. 32, 5008-5013 (1993)
    # From refractiveindex.info
    
    n_real = np.loadtxt("nk_data/zns_realn.txt")
    n_imag = np.loadtxt("nk_data/zns_imagn.txt")
    
    Ephot = hplanck*c/n_real[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = n_real[::-1,1]
    nimag = n_imag[::-1,1]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    eps = eps_real + 1j*1/(1+np.exp(25*(-Evec+3.5)))*eps_imag+1j*1e-6
    
    return eps

def permittivity_au_palik(Evec):
    # Palik: "Handbook of Optical Constants of Solids", edited by Edward D.
    # Palik, Academic Press, 1998 (p286-295)
    
    data = np.loadtxt("nk_data/au_nk_palik.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_ag_palik(Evec):
    # Palik: "Handbook of Optical Constants of Solids", edited by Edward D.
    # Palik, Academic Press, 1998 (p350-357)
    
    data = np.loadtxt("nk_data/ag_nk_palik.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_ag_jiang(Evec):
    # Y. Jiang et al., Realistic Silver Optical Constants for Plasmonics,
    # Sci. Rep. 6, 30605 (2016).
    # https://refractiveindex.info/?shelf=main&book=Ag&page=Jiang
    
    data = np.loadtxt("nk_data/ag_nk_jiang.txt")
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j

def permittivity_au(Evec):
    # Calculated with Drude's model, so that eps =
    # 1-omage_p^2/(omega(omega+i*omega_t), where omega_p is the plasma
    # frequency and omega_t is the damping frequency.
    
    # Drude model presented in e.g. Zeman and Schatz, J. Phys. Chem. 91, 634
    # (1987).
    
    # Parameters also from Zeman's and Schatz's paper..
    
    hbar = hplanck/2/pi
    
    omega_p = 8.89/hbar
    omega_t = 0.07088/hbar
    
    omega = Evec/hbar
    
    eps = 1-omega_p**2/(omega*(omega+1j*omega_t))
    
    return eps

def permittivity_sin(Evec):
    # Vogt, Development of physical models for the simulation of optical
    # properties of solar cell modules. PhD Thesis (2015)
    
    n_real = np.loadtxt("nk_data/sin_realn.txt")
    n_imag = np.loadtxt("nk_data/sin_imagn.txt")
    
    Ephot = hplanck*c/n_real[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = n_real[::-1,1]
    nimag = n_imag[::-1,1]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j+1j*1e-6

def permittivity_si3n4_palik(Evec):
    # Palik: "Handbook of Optical Constants of Solids", edited by Edward D.
    # Palik, Academic Press, 1998 (p286-295)

    data = np.loadtxt('nk_data/si3n4_nk_palik.txt')
    
    Ephot = hplanck*c/data[:,0]/1e-6/q
    
    Ephot = Ephot[::-1]
    nreal = data[::-1,1]
    nimag = data[::-1,2]
    
    eps_data = (nreal+1j*nimag)**2
    
    eps_real = np.interp(Evec, Ephot, eps_data.real)
    eps_imag = np.interp(Evec, Ephot, eps_data.imag)
    
    return eps_real+eps_imag*1j



