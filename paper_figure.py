from huberutils.utils import my_qconv
from huberutils.cbfTools import HuberCBFRun, HuberCBF
from eval_peak import eval_peak, process_dependency
from pyFAI import load
import fabio
from matplotlib.pyplot import *
import numpy as np
import glob
import os
from scipy.ndimage import center_of_mass
import ase
import pandas as pd

def get_voltage(name):
    return float(name.split('/')[0].split('_')[0][1:])

T = "80K"
local_dir = os.getcwd() 
os.chdir(f"/mnt/z/VEGA/CsV3Sb5_strain/2024/07/CsV3Sb5_July24_Kalpha/runs/{T}")

xyz = [140,850,25] # To X -640 was taken. Y maintained Z = 156- Frame number or just the frame number for merged
dxdydz = [20,20,10]
margin = 3
peak_label = '4 -2 0'
head = glob.glob('V[0-9]*/fulltth')
head.sort(key=get_voltage)
ai = load('LaB6_Ka_tth0.poni')
tail = 'CsV3Sb5_strain_01_03_'
voltages = [12,16,20,32,38]

result_bragg = process_dependency(head, tail, xyz, 
                                dxdydz, margin, 
                                extract_parameter=get_voltage, 
                                azimuthal_integrator=ai, 
                                peak_label=peak_label, 
                                fix_sigma=False,
                                npts=52,
                                paramter_include=voltages,
                                plot=False)

peak_label = '3_-3.5_-0.5'  # This is (0.5,3,-0.5)
xyz = [729-641,1088,156-68]
dxdydz = [10, 10, 10]
margin = 2
tail = 'CsV3Sb5_strain_01_05_'
result_SL1 = process_dependency(head, tail, xyz, 
                                dxdydz, margin, 
                                extract_parameter=get_voltage, 
                                azimuthal_integrator=ai, 
                                peak_label=peak_label, 
                                fix_sigma=0.0708,
                                npts=52,
                                paramter_include=voltages,
                                plot=False)

# Track SL peaks 2.5 -3 -0.5 (UB a = strain direction)

voltages = [12,16,20,32,38]
peak_label = '2.5_-3_-0.5'  # (0.5, 2.5, -0.5)
xyz = [712-641,925,156-43]
dxdydz = [10, 10, 10]
margin=1
tail = 'CsV3Sb5_strain_01_05_'
result_SL2 = process_dependency(head, tail, xyz, 
                                dxdydz, margin, 
                                extract_parameter=get_voltage, 
                                azimuthal_integrator=ai, 
                                peak_label=peak_label, 
                                fix_sigma=0.0708,
                                npts=52,
                                paramter_include=voltages,
                                plot=False)

"""# Track SL peaks 3 -2.5 -0.25 (UB a = strain direction)

voltages = [12,16,20,32,38]
peak_label = '3_-2.5_-0.25'  # (-0.5, 3, -0.25)
xyz = [866-641,826,156-100]
dxdydz = [15, 15, 7]
margin=2
tail = 'CsV3Sb5_strain_01_05_'
tail_esp = 'CsV3Sb5_strain_01_08_'
result_SL3 = process_dependency(head, tail, xyz, 
                                dxdydz, margin, 
                                extract_parameter=get_voltage, 
                                azimuthal_integrator=ai, 
                                peak_label=peak_label, 
                                fix_sigma=0.0708,
                                npts=52,
                                plot=False)

# Track SL peaks 2.5 -3 -0.25 (UB a = strain direction)

voltages = [12,16,20,32,38]
peak_label = '2.5_-3_-0.25'  # (0.5, 2.5, -0.25)
xyz = [934-641,672,156-57]
dxdydz = [15, 15, 7]
margin=2
tail = 'CsV3Sb5_strain_01_05_'
tail_esp = 'CsV3Sb5_strain_01_08_'
result_SL4 = process_dependency(head, tail, xyz, 
                                dxdydz, margin, 
                                extract_parameter=get_voltage, 
                                azimuthal_integrator=ai, 
                                peak_label=peak_label, 
                                fix_sigma=0.0708,
                                npts=52,
                                plot=False)"""

# create figure for paper
from lmfit.models import StepModel, ConstantModel, LinearModel

strain = np.array([result_bragg['fit_results'][_][0].values['d'] for _ in sorted(result_bragg['fit_results'].keys())])
strain = (strain - strain[0])/strain[0] *100

close('all')
from matplotlib.style import use
use('paper')

rcParams["font.serif"] = ["Times New Roman"]
rcParams["font.family"] = "serif"
rcParams['mathtext.fontset'] = "stix"
rcParams["lines.markersize"] = 2


fig, axs = subplots(2, 3, figsize=(17.8/2.54, 10/2.54), constrained_layout=True)
for i in [0,1]:
    for j in [0,1]:
        axs[i,j].set_ylabel('Intensity (cts/s)')

voltages = [12., 16., 20., 32., 38.]

colors = ['k', 'b', 'r', 'g', 'm']
m = result_bragg['fit_results'][voltages[0]][0]
norm = 10
x = m.userkws['x']
_x = np.linspace(x.min(), x.max(), 500)
dat = m.eval_components(x=_x)
axs[0,2].errorbar(x, m.data/norm, np.sqrt(m.data)/norm, fmt='ko', label=r'-0.03%')
axs[0,2].plot(_x, m.eval(x=_x)/norm, color='b', marker='none')
axs[0,2].fill_between(_x, dat['pvoigt']/norm, color='r', alpha=0.2, ec='none')
axs[0,2].fill_between(_x, dat['p0_']/norm, color='g', alpha=0.2, ec='none')
axs[0,2].set_xlim(m.values['center']-0.3, m.values['center']+0.5)
axs[0,2].set_yticks([0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4])
axs[0,2].set_yticklabels([0, 2, 4, 6, 8, 10, 12])
axs[0,2].set_ylabel('Intensity ($10^4$ cts/s)')
axs[0,2].set_xlabel(r"$2\theta$ (deg)")
cen = m.values['center']
a = m.values['height']
fwhm = m.values['fwhm']/2
axs[0,2].plot([cen-fwhm,cen+fwhm],[a/norm/2,a/norm/2], marker='none', c='k', lw=5, alpha=0.5)
strain_label = ['-0.03', '-0.07', '-0.09', '-0.14', '-0.19']
yshift = 0.4e4
i = 1
for v, c, e in zip(voltages[1:], colors[1:], strain_label[1:]):
    m = result_bragg['fit_results'][v][0]
    x = m.userkws['x']
    _x = np.linspace(x.min(), x.max(), 500)
    dat = m.eval_components(x=_x)
    axs[0,2].errorbar(x, i*yshift+m.data/norm, np.sqrt(m.data)/norm, color=c, fmt='o', label=e+'%')
    axs[0,2].plot(_x, i*yshift+m.eval(x=_x)/norm, marker='none', color=c)
    i += 1
axs[0,2].text(0.05, 0.8, r'$(4,\bar2,0)$', transform=axs[0,2].transAxes)
axs[0,2].text(29.98, 0.7e4, r'MoK$_{\alpha1}$', fontsize=7)
axs[0,2].text(30.17, 0.4e4, r'MoK$_{\alpha2}$', fontsize=7)
leg = axs[0,2].legend(frameon=False, alignment="right", 
                      borderpad=0, fontsize=7, reverse=1,
                      labelspacing=0.25, handletextpad=0.3)


for v, c, e in zip(voltages, colors, strain_label):
    m = result_SL1['fit_results'][v][0]
    m1 = result_SL1['fit_results'][v][1]
    _lnorm = m1.data.max()/m.data.max()
    x = m.userkws['x']
    _x = np.linspace(x.min(), x.max(), 500)
    dat = m.eval_components(x=_x)
    axs[0,1].errorbar(x, m.data*_lnorm/norm, np.sqrt(m.data)*_lnorm/norm, color=c, fmt='o', label=e+'%')
    axs[0,1].plot(_x, m.eval(x=_x)*_lnorm/norm, marker='none', color=c)
axs[0,1].text(0.95, 0.8, r'(3,-3.5,-0.5)', ha='right' ,transform=axs[0,1].transAxes)
cen = m.values['center']
axs[0,1].plot([cen-fwhm,cen+fwhm],[0,0], marker='none', c='k', lw=5, alpha=0.5)

for v, c, e in zip(voltages, colors, strain_label):
    m = result_SL2['fit_results'][v][0]
    m1 = result_SL2['fit_results'][v][1]
    _lnorm = m1.data.max()/m.data.max()
    x = m.userkws['x']
    _x = np.linspace(x.min(), x.max(), 500)
    dat = m.eval_components(x=_x)
    axs[1,1].errorbar(x, m.data*_lnorm/norm, np.sqrt(m.data)*_lnorm/norm, color=c, fmt='o', label=e+'%')
    axs[1,1].plot(_x, m.eval(x=_x)*_lnorm/norm, marker='none', color=c)
axs[1,1].text(0.95, 0.8, r'(2.5,-3,-0.5)', ha='right' ,transform=axs[1,1].transAxes)
cen=24.125
axs[1,1].plot([cen-fwhm,cen+fwhm],[0,0], marker='none', c='k', lw=5, alpha=0.5)

for v, c, e in zip(voltages, colors, strain_label):
    m = result_SL1['fit_results'][v][1]
    x = m.userkws['x']
    _x = np.linspace(x.min(), x.max(), 500)
    dat = m.eval_components(x=_x)
    axs[0,0].errorbar(x, m.data/norm, np.sqrt(m.data)/norm, color=c, fmt='o', label=e+'%')
    axs[0,0].plot(_x, m.eval(x=_x)/norm, marker='none', color=c)
axs[0,0].text(0.95, 0.8, r'(3,-3.5,-0.5)', ha='right' ,transform=axs[0,0].transAxes)

for v, c, e in zip(voltages, colors, strain_label):
    m = result_SL2['fit_results'][v][1]
    x = m.userkws['x']
    _x = np.linspace(x.min(), x.max(), 500)
    dat = m.eval_components(x=_x)
    axs[1,0].errorbar(x, m.data/norm, np.sqrt(m.data)/norm, color=c, fmt='o', label=e+'%')
    axs[1,0].plot(_x, m.eval(x=_x)/norm, marker='none', color=c)
axs[1,0].text(0.95, 0.8, r'(2.5,-3,-0.5)', ha='right' ,transform=axs[1,0].transAxes)
lkws = dict(loc=2,frameon=False, alignment="right", 
                borderpad=0, fontsize=7, reverse=1,
                labelspacing=0.25, handletextpad=0.3)
axs[0,0].legend(**lkws)
axs[0,1].legend(**lkws)
lkws.update(reverse=False)
axs[1,0].legend(**lkws)
axs[1,1].legend(**lkws)

SL1 = np.array(result_SL1['integtrated_intesity'])/norm
SL2 = np.array(result_SL2['integtrated_intesity'])/norm

smodel = StepModel(form='atan') + ConstantModel()

#SL1 = np.array([[result_SL1['fit_results'][_][1].params['amplitude'].value, result_SL1['fit_results'][_][1].params['amplitude'].stderr] for _ in sorted(result_SL1['fit_results'].keys())])
#SL2 = np.array([[result_SL2['fit_results'][_][1].params['amplitude'].value, result_SL2['fit_results'][_][1].params['amplitude'].stderr] for _ in sorted(result_SL2['fit_results'].keys())])
resSL1 = smodel.fit(x=strain, data=SL1[:,1])
smodel1 = StepModel(form='atan')
params = smodel1.make_params()
params['center'].set(value=-0.05)
params['amplitude'].set(value=80)
params['sigma'].set(value=0.03)

#params['sigma'].set(value=-0.03)
#params['amplitude'].set(value=-100)
resSL2 = smodel1.fit(x=strain, data=SL2[:,1], params=params)
_x = np.linspace(strain.min(), strain.max(), 500)

# If one uses exclusion then this will only plot the indicated voltage points instead of all of them
axs[1,2].errorbar(strain, *SL1.T[1:], fmt='co', label=r'(3,-3.5,-0.5)')
axs[1,2].plot(_x, resSL1.eval(x=_x), ls='--', marker='none', c='c')
axs[1,2].errorbar(strain, *SL2.T[1:], fmt='yo', label=r'(2.5,-3,-0.5)')
axs[1,2].plot(_x, resSL2.eval(x=_x), ls='--', marker='none', c='y')
axs[1,2].invert_xaxis()
axs[1,2].set_xlim(0.01, -0.2)
axs[1,2].set_xlabel(r"$\delta\varepsilon_a$ (%)")
axs[1,2].set_ylabel(r"Integrated Intensity (cts/s)")
axs[1,2].legend(**lkws)

axs[0,0].set_xlabel(r"$\theta$ (deg)")
axs[1,0].set_xlabel(r"$\theta$ (deg)")

axs[0,1].set_xlabel(r"$2\theta$ (deg)")
axs[1,1].set_xlabel(r"$2\theta$ (deg)")

for a, l in zip(axs.flat, 'abcdef'):
    a.text(-0.1,1,f'({l})', transform=a.transAxes, fontsize=10)

os.chdir(local_dir)
savefig('fig_kleines_delta.pdf')





