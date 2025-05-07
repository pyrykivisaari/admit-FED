
import optical_admittance_package as oap
import numpy as np
from numpy import pi
import TF_DDS_functions as tf
import matplotlib.pyplot as plt

c = 299792458 
hplanck = 6.626e-34
q = 1.602e-19
Ep = np.concatenate((1.41,np.linspace(1.42,1.48,15),1.52,1.56,1.6),axis=None)
kx, Kmax = tf.set_K(Ep=Ep,N_K=100,mat='algaas')
wl = hplanck*c/Ep/q
k0 = 2*pi/wl

L_gaas = np.array([10,20,30,40,50])

ind = 4 # Index of the Ep value for which the reflectivity is calculated

R_r_TE = np.zeros((L_gaas.size,kx[ind].size))
R_r_TM = np.zeros((L_gaas.size,kx[ind].size))

for i, Len in enumerate(L_gaas):
    L = np.array([100,400, Len, 200])
    L = L*1e-9
    N = np.array([100,400, Len, 1000])
    Nc = np.cumsum(N)
    layers = ['gaas','algaas','gaas','ag']
    eps = tf.get_permittivities(layers,Ep)
    # Bottom layer should be lossless GaAs.
    eps[:,0] = 3.65**2*np.ones(eps[:,0].size)
    mu = np.ones(eps.shape, dtype=np.int16)
    z = oap.distribute_z_uneven(L,N)
    for k in range(kx[ind].size):
        gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = oap.calculate_all_admittances_uneven(eps[ind],mu[ind],L,N,wl[ind], \
            kx[ind,k],eps[ind,0],1,eps[ind,-1],1)
        r_l_TE, r_r_TE, r_r_TM, r_l_TM = oap.reflectance_transmission_kx(eps[ind],mu[ind],N,wl[ind],kx[ind,k],z, \
            gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
        R_r_TE[i,k] = np.abs(r_r_TE[Nc[0]-1])**2
        R_r_TM[i,k] = np.abs(r_r_TM[Nc[0]-1])**2
    print('Solved for L_gaas=', Len, ' nm')

plt.figure(figsize=(7.5,7.5))
plt.subplot(2,1,1)
for i, Len in enumerate(L_gaas):
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(eps[ind,0]))/k0[ind])/pi*180,R_r_TE[i],label=str(Len)+' nm GaAs')
plt.axis([0, 80, 0.75, 1])
plt.suptitle('Reflectivity with photon energy ' +  str("%.4f" % Ep[ind]) + ' eV')
plt.xlabel('Angle (deg.)')
plt.ylabel('Reflectivity (TE)')
plt.legend(frameon=False)
plt.subplot(2,1,2)
for i, Len in enumerate(L_gaas):
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(eps[ind,0]))/k0[ind])/pi*180,R_r_TM[i],label=str(Len)+' nm GaAs')
plt.axis([0, 80, 0.75, 1])
plt.xlabel('Angle (deg.)')
plt.ylabel('Reflectivity (TM)')
plt.legend(frameon=False)

plt.show()

