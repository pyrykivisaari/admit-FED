
import TF_DDS_functions as tf
import optical_admittance_package as oap
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import permittivities as perm

Lled = 10
Lpd = 50

cavity2 = {}
cavity2['d'] = 120

cavity2['L'] = np.array([(cavity2['d']-Lled-Lpd)/4,Lpd,(cavity2['d']-Lled-Lpd)/4, \
    (cavity2['d']-Lled-Lpd)/4,Lled,(cavity2['d']-Lled-Lpd)/4])
cavity2['L'] = cavity2['L']*1e-9
cavity2['N'] = np.array([round((cavity2['d']-Lled-Lpd)/4*2),2*Lpd,round((cavity2['d']-Lled-Lpd)/4*2), \
    round((cavity2['d']-Lled-Lpd)/4*2),2*Lled,round((cavity2['d']-Lled-Lpd)/4*2)])
cavity2['layers'] = ['algaas','gaas','algaas','algaas','gaas','algaas']
Ep = np.concatenate((1.41,np.linspace(1.42,1.48,15),1.52,1.56,1.6),axis=None)
N_K = 80
K, Kmax = tf.set_K(Ep=Ep,N_K=80)
cavity2['eps'] = tf.get_permittivities(cavity2['layers'],Ep)

eps_Ag = perm.permittivity_ag_palik(Ep)
eps_Ag = 100*eps_Ag

cavity2['mu'] = np.ones(cavity2['eps'].shape)
cavity2['z0'] = tf.set_source_coordinates(cavity2['L'],4)

dEF = 1.3
T = 300

cavity2['z'] = oap.distribute_z_uneven(cavity2['L'],cavity2['N'])
cavity2['Nc'] = np.cumsum(cavity2['N'])
cavity2['qte_w'] = np.zeros((Ep.size, cavity2['z'].size), dtype=complex)
cavity2['qtm_w'] = np.zeros((Ep.size, cavity2['z'].size), dtype=complex)
cavity2['qte_wK'] = np.zeros((Ep.size, K[0].size), dtype=complex)
cavity2['qtm_wK'] = np.zeros((Ep.size, K[0].size), dtype=complex)
cavity2['P_TE_wK'] = np.zeros((Ep.size, K[0].size), dtype=complex)
cavity2['P_TM_wK'] = np.zeros((Ep.size, K[0].size), dtype=complex)
cavity2['qte_wKint'] = np.zeros((Ep.size, K[0].size), dtype=complex)
cavity2['qtm_wKint'] = np.zeros((Ep.size, K[0].size), dtype=complex)
for i, E in enumerate(Ep):
    pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
        cavity2['qte_w'][i], cavity2['qtm_w'][i] = oap.solve_optical_properties_single_E(\
        cavity2['eps'][i],cavity2['mu'][i],cavity2['L'],cavity2['N'],E,K[i],cavity2['z0'],dEF,T,eps_Ag[i])
    cavity2['qte_wK'][i] = rad_TE[:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))]
    cavity2['qtm_wK'][i] = rad_TM[:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))]
    cavity2['P_TE_wK'][i] = P_TE[:,int(round((cavity2['Nc'][2]+cavity2['Nc'][3])/2))]
    cavity2['P_TM_wK'][i] = P_TM[:,int(round((cavity2['Nc'][2]+cavity2['Nc'][3])/2))]
    cavity2['qte_wKint'][i] = np.trapezoid(rad_TE[:,cavity2['Nc'][3]:cavity2['Nc'][4]],cavity2['z'][cavity2['Nc'][3]:cavity2['Nc'][4]],axis=1)/10e-9
    cavity2['qtm_wKint'][i] = np.trapezoid(rad_TM[:,cavity2['Nc'][3]:cavity2['Nc'][4]],cavity2['z'][cavity2['Nc'][3]:cavity2['Nc'][4]],axis=1)/10e-9

theta = oap.propagation_angles_gaas(Ep[0],K[0])
Epplot, Tplot = np.meshgrid(Ep, theta)

hplanck = 6.626e-34
c = 299792458
q = 1.602e-19
wl = hplanck*c/Ep/q
k0 = 2*pi/wl

plt.figure(figsize=(5.4,2.5))
plt.pcolormesh(Epplot, Tplot, np.transpose(cavity2['qte_wK'].real+cavity2['qtm_wK'].real)*np.cos(Tplot/180*pi)*np.sin(Tplot/180*pi), cmap='gnuplot', shading='gouraud')
plt.ylabel('Angle (deg.)')
plt.xlabel('Photon energy (eV)')
plt.xlim(1.41,1.55)
plt.tight_layout()
plt.colorbar()

plt.figure(figsize=(5.4,2.5))
plt.pcolormesh(Epplot, Tplot, np.transpose(cavity2['qte_wKint'].real+cavity2['qtm_wKint'].real)*np.cos(Tplot/180*pi)*np.sin(Tplot/180*pi), cmap='gnuplot', shading='gouraud')
plt.ylabel('Angle (deg.)')
plt.xlabel('Photon energy (eV)')
plt.xlim(1.41,1.55)
plt.tight_layout()
plt.colorbar()

plt.figure(figsize=(5.4,2.5))
plt.pcolormesh(Epplot, Tplot, np.transpose(-cavity2['P_TE_wK'].real-cavity2['P_TM_wK'].real), cmap='gnuplot', shading='gouraud')
plt.xlabel('Photon energy (eV')
plt.ylabel('Angle (deg.)')
plt.tight_layout()
plt.colorbar()

cavity2['spectr'] = np.real(cavity2['qte_w'][:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))])+ \
    np.real(cavity2['qtm_w'][:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))])

cavity2['Emean'] = np.sum(cavity2['spectr']*Ep)/np.sum(cavity2['spectr'])

plt.figure(figsize=(4.625,2.55),facecolor=(1,1,1))
plt.plot(Ep,cavity2['spectr']*1e-18,label=r'$d_{cav}=240$ nm')
plt.xlim([np.min(Ep), np.max(Ep)])
plt.xlabel('Photon energy (eV)')
plt.ylabel(r'Emission spectrum (10$^{18}$ m$^{-3}$)')
plt.legend(frameon=False)
plt.tight_layout()

cavity2['spectrNorm'] = (np.real(cavity2['qte_w'][:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))])+ \
    np.real(cavity2['qtm_w'][:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))]))/ \
    np.max(np.real(cavity2['qte_w'][:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))])+ \
    np.real(cavity2['qtm_w'][:,int(round((cavity2['Nc'][3]+cavity2['Nc'][4])/2))]))

plt.figure(figsize=(4.425,2.5),facecolor=(1,1,1))
plt.plot(Ep,cavity2['spectrNorm'],label='240 nm')
plt.xlim([np.min(Ep), np.max(Ep)])
plt.xlabel('Photon energy (eV)')
plt.ylabel('Emission spectrum (a.u.)')
plt.legend()
plt.tight_layout()

cavity2['spectrum_te_int'] = np.trapezoid(cavity2['qte_w'][:,cavity2['Nc'][3]:cavity2['Nc'][4]],\
        cavity2['z'][cavity2['Nc'][3]:cavity2['Nc'][4]],axis=1)/10e-9
cavity2['spectrum_tm_int'] = np.trapezoid(cavity2['qtm_w'][:,cavity2['Nc'][3]:cavity2['Nc'][4]],\
        cavity2['z'][cavity2['Nc'][3]:cavity2['Nc'][4]],axis=1)/10e-9

plt.figure(figsize=(4.625,2.55),facecolor=(1,1,1))
plt.plot(Ep,(cavity2['spectrum_te_int']+cavity2['spectrum_tm_int'])*1e-18,label=r'$d_{cav}=265$ nm')
plt.xlim([np.min(Ep), np.max(Ep)])
plt.xlabel('Photon energy (eV)')
plt.ylabel(r'Emission spectrum (10$^{18}$ m$^{-3}$)')
plt.legend(frameon=False)
plt.tight_layout()

plt.show()

