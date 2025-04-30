
import optical_admittance_package as oap

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

N_2 = np.array([2.51+0.0029j, 2.51+0.094j, 2.51+0.0029j, 2.51+0.094j, 2.51+0.0029j, 2.51+0.094j,
	2.51+0.0029j, 2.51+0.094j, 2.51+0.0029j, 2.51+0.094j,2.51+0.0029j, 0.013+3.119j, 1+1e-15*1j])
eps_2 = N_2**2
mu_2 = np.ones(eps_2.size, dtype=np.int16)
L_2 = np.array([700, 3, 10, 3, 10, 3, 10, 3, 10, 3, 600, 20, 1500])
L_2 = L_2*1e-9
N_2 = np.array([700, 50, 200, 50, 200, 50, 200, 50, 200, 50, 2000, 1500, 1500], dtype=int)

q = 1.602e-19
kb = 1.38e-23
Ep = 2.786
h = 6.62607e-34
c = 299792458

wl = h*c/Ep/q
k0 = 2*pi/wl
omega = k0*c
epssubs_r = 1+1e-15*1j
epssubs_l = eps_2[0]
musubs = 1
kx = np.linspace(0.05, 3.5*k0, 60)

z_2 = oap.distribute_z_uneven(L_2,N_2)
z0_12 = 701.5e-9
z0_52 = 752.5e-9
eps_z_2 = oap.distribute_parameter(eps_2,N_2)

rho_e_2 = np.zeros((kx.size, z_2.size), dtype=complex)
rho_m_2 = np.zeros((kx.size, z_2.size), dtype=complex)
rho_TE_2 = np.zeros((kx.size, z_2.size), dtype=complex)
rho_TM_2 = np.zeros((kx.size, z_2.size), dtype=complex)
gee11_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gee22_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gee33_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gee23_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gee32_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gme12_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gme13_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gme21_12 = np.zeros((kx.size, z_2.size), dtype=complex)
gme31_12 = np.zeros((kx.size, z_2.size), dtype=complex)

for k in range(kx.size):
	gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = oap.calculate_all_admittances_uneven(eps_2,mu_2,L_2,N_2,wl,kx[k], \
		epssubs_l,musubs,epssubs_r,musubs)
	rho_e_2[k], rho_m_2[k], rho_TE_2[k], rho_TM_2[k] = oap.LDOSes(eps_2,mu_2,N_2,wl,kx[k],gamma_l_TE,gamma_r_TE, \
		gamma_r_TM,gamma_l_TM)
	gee11_12[k], gee22_12[k], gee33_12[k], gee23_12[k], gee32_12[k] = oap.electric_greens_functions(eps_2,mu_2, \
		N_2,wl,kx[k],z_2,z0_12,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
	gme12_12[k], gme13_12[k], gme21_12[k], gme31_12[k] = oap.exchange_greens_functions_me(eps_2,mu_2,N_2,wl,kx[k], \
		z_2,z0_12,gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
	print('Solved for kx=', kx[k]/k0, '\n')

rho_l_2 = rho_e_2/2+rho_m_2/2
rho_nl_TE_12, rho_nl_TM_12 = oap.NLDOS_TETM_electric_sources(eps_2,mu_2,N_2,wl,kx,z_2,z0_12,gee11_12,gee22_12,gee33_12,\
	gee23_12,gee32_12,gme12_12,gme13_12,gme21_12,gme31_12)
rho_if_TE_12, rho_if_TM_12 = oap.IFDOS_TETM_electric_sources(eps_2,mu_2,N_2,wl,z_2,z0_12,gee11_12,gee22_12,gee23_12, \
	gme12_12,gme13_12,gme21_12)

zplot, kxplot = np.meshgrid(z_2, kx/k0)
plt.pcolormesh(zplot, kxplot, np.log10(rho_TM_2.real))
plt.colorbar()
plt.show()

#plt.pcolormesh(zplot, kxplot, np.log10(np.abs(gee11_12)))
#plt.colorbar()
#plt.show()

#plt.pcolormesh(zplot, kxplot, np.log10(np.abs(gee32_12)))
#plt.colorbar()
#plt.show()

#plt.pcolormesh(zplot, kxplot, (np.abs(gme31_12)))
#plt.colorbar()
#plt.show()

#plt.pcolormesh(zplot, kxplot, (np.abs(rho_if_TE_12)))
#plt.colorbar()
#plt.show()

plt.plot(z_2,np.abs(rho_nl_TE_12[0]))
plt.show()

