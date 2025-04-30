

import optical_admittance_package as oap
import numpy as np
from numpy import pi
import TF_DDS_functions as tf
import matplotlib.pyplot as plt

c = 299792458 
hplanck = 6.626e-34
q = 1.602e-19
Ep, kx, Kmax = tf.set_E_and_K_mater_opt(100,['algaas'])
wl = hplanck*c/Ep/q
k0 = 2*pi/wl

L_gaas = np.array([10,20,30,40,50])

ind = 4

R_r_TE = np.zeros((L_gaas.size,kx[ind].size))
R_r_TM = np.zeros((L_gaas.size,kx[ind].size))
R_r_0 = np.zeros(L_gaas.shape)

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
        if k==0:
            R_r_0 = np.abs(r_r_TM[Nc[0]-100])**2
    print('Solved for E_al2o3=', Len, ' nm')

plt.figure()
plt.subplot(2,1,1)
for i, Len in enumerate(L_gaas):
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(eps[ind,0]))/k0[ind])/pi*180,R_r_TE[i],label=str(Len)+' nm Al2O3, TE')
plt.axis([0, 90, 0.9, 1])
plt.title('Reflection at Ep=' +  str("%.4f" % Ep[ind]) + ' eV')
plt.xlabel('Angle (deg.)')
plt.ylabel('Reflectivity')
plt.legend()
plt.subplot(2,1,2)
for i, Len in enumerate(L_gaas):
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(eps[ind,0]))/k0[ind])/pi*180,R_r_TM[i],'--',label=str(Len)+' nm Al2O3, TM')
plt.axis([0, 90, 0.75, 1])
plt.title('Reflection at Ep=' +  str("%.4f" % Ep[ind]) + ' eV')
plt.xlabel('Angle (deg.)')
plt.ylabel('Reflectivity')
plt.legend()

plt.figure()
plt.subplot(2,1,1)
for i, Len in enumerate(L_gaas):
    plt.plot(kx[ind],R_r_TE[i],label=str(Len)+' nm Al2O3, TE')
plt.axis([0, np.max(kx[ind]), 0.9, 1])
plt.title('Reflection at Ep=' +  str("%.4f" % Ep[ind]) + ' eV')
plt.xlabel('Angle (deg.)')
plt.ylabel('Reflectivity')
plt.legend()
plt.subplot(2,1,2)
for i, Len in enumerate(L_gaas):
    plt.plot(kx[ind],R_r_TM[i],'--',label=str(Len)+' nm Al2O3, TM')
plt.axis([0, np.max(kx[ind]), 0.75, 1])
plt.title('Reflection at Ep=' +  str("%.4f" % Ep[ind]) + ' eV')
plt.xlabel('Angle (deg.)')
plt.ylabel('Reflectivity')
plt.legend()

R_TE_ave = (1-np.mean(R_r_TE,1))/2
R_TM_ave = (1-np.mean(R_r_TM,1))/2
R_ave = (1-np.mean(R_r_TE,1))/2/2+(1-np.mean(R_r_TM,1))/2/2
plt.figure()
plt.plot(L_gaas,R_TE_ave,label='TE')
plt.plot(L_gaas,R_TM_ave,label='TM')
plt.plot(L_gaas,R_ave,label='Both')
plt.legend()

plt.figure()
plt.plot(L_gaas,(1-R_r_TE[:,0])/2)
plt.plot(L_gaas,(1-R_r_TM[:,0])/2)
plt.plot(L_gaas,((1-R_r_TE[:,0])+(1-R_r_TM[:,0]))/4)
plt.show()

if False:
    R_r_TE = np.zeros(kx.shape)
    R_r_TM = np.zeros(kx.shape)
    R_r_TEs = np.zeros(kx.shape)
    R_r_TMs = np.zeros(kx.shape)
    R_r_0 = np.zeros(wl.shape)
    R_r_0s = np.zeros(wl.shape)
    for i in range(Ep.size):
        for k in range(kx[i].size):
            gamma_r_TE, gamma_l_TE, gamma_r_TM, gamma_l_TM = oap.calculate_all_admittances_uneven(eps[i],mu[i],L,N,wl[i], \
                kx[i,k],eps[i,0],1,eps[i,-1],1)
            gamma_r_TEs, gamma_l_TEs, gamma_r_TMs, gamma_l_TMs = oap.calculate_all_admittances_uneven(epss[i],mus[i],Ls,Ns,wl[i], \
                kx[i,k],epss[i,0],1,epss[i,-1],1)
            r_l_TE, r_r_TE, r_r_TM, r_l_TM = oap.reflectance_transmission_kx(eps[i],mu[i],N,wl[i],kx[i,k],z, \
                gamma_l_TE,gamma_r_TE,gamma_r_TM,gamma_l_TM)
            r_l_TEs, r_r_TEs, r_r_TMs, r_l_TMs = oap.reflectance_transmission_kx(epss[i],mus[i],Ns,wl[i],kx[i,k],zs, \
                gamma_l_TEs,gamma_r_TEs,gamma_r_TMs,gamma_l_TMs)
            R_r_TE[i,k] = np.abs(r_r_TE[Nc[0]-1])**2
            R_r_TM[i,k] = np.abs(r_r_TM[Nc[0]-1])**2
            R_r_TEs[i,k] = np.abs(r_r_TEs[Nsc[0]-1])**2
            R_r_TMs[i,k] = np.abs(r_r_TMs[Nsc[0]-1])**2
            if k==0:
                R_r_0[i] = np.abs(r_r_TM[Nc[0]-100])**2
                R_r_0s[i] = np.abs(r_r_TEs[Nsc[0]-100])**2
        print('Solved for Ep=', Ep[i], ' eV')
    ind = 4
    
    plt.figure()
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(eps[ind,0]))/k0[ind])/pi*180,R_r_TE[ind],label='100 nm Al2O3, TE')
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(eps[ind,0]))/k0[ind])/pi*180,R_r_TM[ind],'--',label='100 nm AL2O3, TM')
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(epss[ind,0]))/k0[ind])/pi*180,R_r_TEs[ind],'-.',label='200 nm Al2O3, TE')
    plt.plot(np.arcsin(kx[ind]/np.abs(np.sqrt(epss[ind,0]))/k0[ind])/pi*180,R_r_TMs[ind],':',label='200 nm Al2O3, TM')
    plt.axis([0, 90, 0.7, 1])
    plt.title('Reflection at Ep=' +  str("%.4f" % Ep[ind]) + ' eV')
    plt.xlabel('Angle (deg.)')
    plt.ylabel('Reflectivity')
    plt.legend()
    
    plt.figure()
    plt.plot(wl*1e9,(1-np.mean(R_r_TE,1))/2/2+(1-np.mean(R_r_TM,1))/2/2,label='100 nm Al2O3, R coeffs.')
    #plt.plot(wl*1e9,(1-np.mean(R_r_TM,1))/2,label='AuZn, TM')
    plt.plot(wl*1e9,(1-np.mean(R_r_TEs,1))/2/2+(1-np.mean(R_r_TMs,1))/2/2,'--',label='200 nm Al2O3, R coeffs.')
    #plt.plot(wl*1e9,(1-np.mean(R_r_TMs,1))/2,'--',label='SiN, TM')
    plt.title('Reflection losses averaged over all angles')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('1-R')
    plt.legend()
    plt.show()


