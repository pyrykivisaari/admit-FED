
import optical_admittance_package as oap

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

n_r = np.array([2.51+0.0029j, 2.51+0.094j, 2.51+0.0029j, 2.51+0.094j, 2.51+0.0029j, 2.51+0.094j,
	2.51+0.0029j, 2.51+0.094j, 2.51+0.0029j, 2.51+0.094j,2.51+0.0029j, 0.013+3.119j, 1+1e-15*1j])
eps = n_r**2 # Permittivity of the layers
mu = np.ones(eps.size, dtype=np.int16)
L = np.array([700, 3, 10, 3, 10, 3, 10, 3, 10, 3, 600, 20, 1500]) # Length of the layers in nm
L = L*1e-9
N = np.array([700, 50, 200, 50, 200, 50, 200, 50, 200, 50, 2000, 1500, 1500], dtype=int) # No. datapoints in the layers

# Required constants
q = 1.602e-19
Ep = 2.786
h = 6.62607e-34
c = 299792458

wl = h*c/Ep/q
k0 = 2*pi/wl
omega = k0*c
epssubs_r = 1+1e-15*1j # Permittivity to the right of the domain
epssubs_l = eps[0] # Permittivity to the left of the domain
musubs = 1
kx = np.linspace(0.05, 3.5*k0, 60) # In-plane wave numbers used in the calculation

z = oap.distribute_z_uneven(L,N)
eps_z = oap.distribute_parameter(eps,N)

rho_e = np.zeros((kx.size, z.size), dtype=complex)
rho_m = np.zeros((kx.size, z.size), dtype=complex)
rho_TE = np.zeros((kx.size, z.size), dtype=complex)
rho_TM = np.zeros((kx.size, z.size), dtype=complex)

for k in range(kx.size):
	gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = oap.calculate_all_admittances_uneven(eps,mu,L,N,wl,kx[k], \
		epssubs_l,musubs,epssubs_r,musubs)
	rho_e[k], rho_m[k], rho_TE[k], rho_TM[k] = oap.LDOSes(eps,mu,N,wl,kx[k],gamma_l_TE,gamma_r_TE, \
		gamma_r_TM,gamma_l_TM)
	print('Solved for kx=', kx[k]/k0)

rho_l = rho_e/2+rho_m/2

zplot, kxplot = np.meshgrid(z, kx/k0)
plt.figure()
plt.pcolormesh(zplot*1e6, kxplot, np.log10(rho_TE.real),cmap='coolwarm',shading='gouraud')
plt.colorbar()
plt.xlabel(r'Position ($\mu$m)')
plt.ylabel('In-plane K number / k$_0$')
plt.title('Local DOS for TE modes')
plt.figure()
plt.pcolormesh(zplot*1e6, kxplot, np.log10(rho_TM.real),cmap='Spectral',shading='gouraud')
plt.colorbar()
plt.xlabel(r'Position ($\mu$m)')
plt.ylabel('In-plane K number / k$_0$')
plt.title('Local DOS for TM modes')
plt.show()


