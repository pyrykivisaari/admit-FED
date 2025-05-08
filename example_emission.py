
import TF_DDS_functions as tf
import optical_admittance_package as oap
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import permittivities as perm

q = 1.602e-19
Lled = 300

L = np.array([100,Lled,100])
L = L*1e-9
N = np.array([100,Lled,100])
layers = ['algaas','gaas','algaas']
Ep = np.concatenate((1.41,np.linspace(1.42,1.48,15),1.52,1.56,1.6),axis=None)
K, Kmax = tf.set_K(Ep=Ep,N_K=80)
eps = tf.get_permittivities(layers,Ep)

eps_algaas = perm.permittivity_algaas_palik(Ep)

mu = np.ones(eps.shape)
z0 = tf.set_source_coordinates(L,1)

dEF = 1.3
T = 300

z = oap.distribute_z_uneven(L,N)
Nc = np.cumsum(N)
qte_w = np.zeros((Ep.size, z.size), dtype=complex)
qtm_w = np.zeros((Ep.size, z.size), dtype=complex)
qte_wK = np.zeros((Ep.size, K[0].size), dtype=complex)
qtm_wK = np.zeros((Ep.size, K[0].size), dtype=complex)
P_TE_wK = np.zeros((Ep.size, K[0].size), dtype=complex)
P_TM_wK = np.zeros((Ep.size, K[0].size), dtype=complex)
qte_wKint = np.zeros((Ep.size, K[0].size), dtype=complex)
qtm_wKint = np.zeros((Ep.size, K[0].size), dtype=complex)
for i, E in enumerate(Ep):
    pup_TE, pdown_TE, pup_TM, pdown_TM, P_TE, P_TM, rad_TE, rad_TM, \
        qte_w[i], qtm_w[i] = oap.solve_optical_properties_single_E(\
        eps[i],mu[i],L,N,E,K[i],z0,dEF,T,eps_algaas[i])
    qte_wK[i] = rad_TE[:,int(round((Nc[0]+Nc[1])/2))]
    qtm_wK[i] = rad_TM[:,int(round((Nc[0]+Nc[1])/2))]
    P_TE_wK[i] = P_TE[:,int(round((Nc[0])/2))]/E/q
    P_TM_wK[i] = P_TM[:,int(round((Nc[0])/2))]/E/q
    qte_wKint[i] = np.trapezoid(rad_TE[:,Nc[0]:Nc[1]],z[Nc[0]:Nc[1]],axis=1)
    qtm_wKint[i] = np.trapezoid(rad_TM[:,Nc[0]:Nc[1]],z[Nc[0]:Nc[1]],axis=1)

RPtot_te_int_Em, RPtot_te_int_Abs, RPtot_tm_int_Em, RPtot_tm_int_Abs = \
    tf.calculate_total_em_abs_powers(L,N,Ep,qte_w,qtm_w)
print("Emission power: ", (RPtot_te_int_Em+RPtot_tm_int_Em)*1e-4, "W/cm2")

theta = oap.propagation_angles_gaas(Ep[0],K[0])
Epplot, Tplot = np.meshgrid(Ep, theta)

hplanck = 6.626e-34
c = 299792458
wl = hplanck*c/Ep/q
k0 = 2*pi/wl

plt.figure(figsize=(4.5,2.5))
plt.pcolormesh(Epplot, Tplot, np.transpose(qte_wKint.real+qtm_wKint.real)*np.cos(Tplot/180*pi), cmap='gnuplot', shading='gouraud')
plt.ylabel('Angle (deg.)')
plt.xlabel('Photon energy (eV)')
plt.xlim(1.41,1.55)
plt.ylim(0,69)
plt.title('Emission rate (a.u.)')
plt.tight_layout()
plt.colorbar()

plt.figure(figsize=(5.4,2.5)) # Radiance leftwards; this is half of total R in Fig. 1, as it should be (other half goes to the right)
plt.pcolormesh(Epplot, Tplot, np.transpose(-P_TE_wK.real-P_TM_wK.real)*np.cos(Tplot/180*pi), cmap='gnuplot', shading='gouraud')
plt.xlabel('Photon energy (eV)')
plt.ylabel('Angle (deg.)')
plt.xlim(1.41,1.55)
plt.ylim(0,69)
plt.title('Leftward radiance (a.u.)')
plt.tight_layout()
plt.colorbar()

spectrum_te_int = np.trapezoid(qte_w[:,Nc[0]:Nc[1]],\
        z[Nc[0]:Nc[1]],axis=1)
spectrum_tm_int = np.trapezoid(qtm_w[:,Nc[0]:Nc[1]],\
        z[Nc[0]:Nc[1]],axis=1)
# Multiply by photon energy to give watts, multiply by another factor to change
# the integration variable from omega to Ep
spectrum_P_E = (spectrum_te_int+spectrum_tm_int)*(Ep*q)*(2*pi*q/hplanck)

if False:
    print("Debug: The output from this should be equal to the emission power printed above")
    kaka = np.trapezoid(spectrum_P_E,Ep)
    print(kaka*1e-4)

plt.figure(figsize=(4.625,2.55),facecolor=(1,1,1))
plt.plot(Ep,spectrum_P_E.real*1e-4)
plt.xlim([np.min(Ep), np.max(Ep)])
plt.xlabel('Photon energy (eV)')
plt.ylabel(r'Emission power (W/cm$^2$/eV)')
plt.title('Emission spectrum')
plt.tight_layout()

plt.show()

