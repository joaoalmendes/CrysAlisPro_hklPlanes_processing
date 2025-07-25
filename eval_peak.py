import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from huberutils.cbfTools import HuberCBF
from lmfit.models import GaussianModel, PseudoVoigtModel, LinearModel
import fabio
import os
import glob


def eval_peak(
    frame_list, xyz, dxdydz, theta=None, peak_threshold=4, recenter_peak=True
):
    # try to load the files:
    x, y, z = xyz
    dx, dy, dz = dxdydz
    frames = [HuberCBF(_) for _ in frame_list[slice(z - dz, z + dz + 1)]]
    data = np.stack(
        [_.data[x - dx : x + dx + 1, y - dy : y + dy + 1] for _ in frames], axis=0
    )

    # if theta is not given, try to get it from the frames
    if not theta:
        theta = [(_.th + _.th_increment / 2) for _ in frames]
    # create a theta scan:
    Ith = data.sum(axis=(1, 2))

    # first check if there is a peak in th
    left = Ith[:2].mean()
    right = Ith[-2:].mean()
    bkg = np.linspace(left, right, len(Ith))
    Ith_nobkg = Ith - bkg
    Ith_nobkg[Ith_nobkg < 0] = 0
    if Ith_nobkg.max() > (peak_threshold * Ith.std()):
        is_peak = True
        z_local = np.argmax(Ith_nobkg)
        z = z - dz + z_local
        _x, _y = center_of_mass(data[z_local, :, :])
        xyz_new = np.array([int(_) for _ in [x - dx + _x, y - dy + _y, z]])
    else:
        is_peak = False
        z = z
        xyz_new = np.asarray(xyz)
    myslice = (
        slice(xyz_new[0] - dxdydz[0], xyz_new[0] + dxdydz[0] + 1),
        slice(xyz_new[1] - dxdydz[1], xyz_new[1] + dxdydz[1] + 1),
        slice(xyz_new[2] - dxdydz[2], xyz_new[2] + dxdydz[2] + 1),
    )
    I = Ith_nobkg.sum()
    if recenter_peak:
        if is_peak:
            if np.linalg.norm(xyz_new - np.asarray(xyz)) >= 1:
                print("recenter")
                return eval_peak(
                    frame_list, xyz_new, dxdydz, theta, peak_threshold, recenter_peak
                )
    return {
        "is_peak": is_peak,
        "xyz": xyz_new,
        "dxdydz": np.asarray(dxdydz),
        "data": data,
        "theta": np.asarray(theta),
        "Ith": Ith,
        "Ith_nobkg": Ith_nobkg,
        "bkg": bkg,
        "I_integrated": I,
        "frames": frames,
        "slice": myslice,
        "peak": 1.0 * data[dxdydz[2], :, :],
    }

def my_fit3(x,y, start=None, sigma=None):
    Kbwls = [
        0.713607,
    ]
    Kb_main = 0.70931715
    pv = PseudoVoigtModel()
    lm = LinearModel()
    model = pv + lm
    for i, wl in enumerate(Kbwls):
        model = model + PseudoVoigtModel(prefix=f"p{i}_")
    params = model.make_params()
    params.update(lm.guess(y,x))
    params.add("d", expr=f"{Kb_main}/(2*sin(center/2*pi/180))")
    params.add("q", expr=f"4*pi/{Kb_main}*sin(center/2*pi/180)")
    params.add("I12", value=0.42, vary=False)
    for i, wl in enumerate(Kbwls):
        params[f"p{i}_center"].set(expr=f"2*(asin({wl}/2/d))*180/pi")
        params[f"p{i}_sigma"].set(expr="sigma")
        params[f"p{i}_fraction"].set(expr="fraction")
        params[f"p{i}_amplitude"].set(expr="I12*amplitude")

    params.update(PseudoVoigtModel().guess(y - lm.eval(x=x, params=lm.guess(y,x)), x))
        #params.add("strech")
    if sigma:
        params["sigma"].set(value=sigma, vary=False)
    params['p0_sigma'].set(expr="sigma")
    #params['p2_sigma'].set(expr="sigma*strech")
    #display(params)
    if start:
        params = start
    res = model.fit(y, params=params, x=x)
    return res

def process_dependency(head, 
                       tail, 
                       xyz, 
                       dxdydz, 
                       margin, 
                       extract_parameter, 
                       azimuthal_integrator,
                       fix_sigma=False,
                       npts=None,
                       paramter_include = [],
                       peak_label='', 
                       verbose=True,
                       plot=False):

    def _densify(x, n=500):
        return np.linspace(x.min(), x.max(), n)

    if plot:
        ifig, iaxs = plt.subplots(3, 4, sharex='all', sharey='all', constrained_layout=True)
        fig, axs =plt.subplots(1,3, figsize=(2*18/2.54, 2*6/2.54), constrained_layout=True)
        fig.text(0.5, 0.99, peak_label, va='top', fontsize=16)
        ifig.text(0.5, 0.99, peak_label, va='top', fontsize=16)
        ifig.show()

    fit_results = {}
    tth_scans = []
    tth_fits =  []
    th_scans = []
    th_fits = []
    integtrated_intesity = []
    
    
    for i, pre in enumerate(head):
        parameter = extract_parameter(pre)
        print(os.path.join(pre, tail))
        if parameter in paramter_include:
            frame_list = sorted(glob.glob(os.path.join(pre, tail+'*.cbf')))
            if len(frame_list) == 0:
                print('nothing')
                continue
            peak_res = eval_peak(frame_list, xyz, dxdydz, recenter_peak=False)
            frames = peak_res["frames"]
            xyz = peak_res["xyz"]
            dxdydz = peak_res["dxdydz"]
            azimuthal_integrator.rot2 = frames[0].th2*np.pi/180
            mask = np.ones(frames[0].data.shape)
            mask[peak_res["slice"][:2]] = 0
            peak = np.stack([_.data for _  in frames[slice(dxdydz[2]-margin,dxdydz[2]+margin+1)]]).sum(axis=0)
            print(xyz, parameter)        
            th = peak_res["theta"]
            Ith = peak_res["Ith_nobkg"]
            Ithe = np.sqrt(Ith)
            Ithe[Ithe==0] = 1
            dI = np.sqrt(peak_res["Ith"].sum() + peak_res["bkg"].sum())      
        
            npts = npts if npts else np.sum((4*dxdydz**2)[:2])**0.5
            tth, I, Ie = azimuthal_integrator.integrate1d(peak, 
                                        npt=npts, 
                                        unit='2th_deg', 
                                        mask=mask, variance=np.sqrt(peak), 
                                        correctSolidAngle=True, error_model='possion',
                                        normalization_factor=1/50)#/(xyz[0]*np.sqrt(2)*margin))
            res = my_fit3(tth, I, sigma=fix_sigma)

            gmod = GaussianModel()
            params = gmod.guess(Ith, th)
            th_res = gmod.fit(Ith, x=th, weights=Ithe, params=params)

            fit_results[parameter] = [res, th_res]
            
            tth_scans.append([parameter] + tth.tolist())
            tth_scans.append([parameter] + I.tolist())
            tth_scans.append([parameter] + Ie.tolist())      
    
            th_scans.append([parameter] + th.tolist())
            th_scans.append([parameter] + Ith.tolist())
            th_scans.append([parameter] + Ithe.tolist())

            th_fits.append([parameter] + _densify(th).tolist())
            th_fits.append([parameter] + th_res.eval(x=_densify(th)).tolist())

            tth_fits.append([parameter] + _densify(tth, 500).tolist())
            tth_fits.append([parameter] + res.eval(x=_densify(tth)).tolist())
            
            integtrated_intesity.append([parameter, peak_res["I_integrated"], dI])
        
        if plot:
            _l = axs[0].errorbar(tth, I,Ie, fmt='o', label=str(parameter))
            axs[0].plot(_densify(tth), res.eval(x=_densify(tth)), color=_l.lines[0].get_color(), ls='-', label=str(parameter))
            axs[0].legend() 
            axs[0].set_xlabel('2Theta')
            axs[1].set_xlabel('Theta')
            axs[1].errorbar(th, Ith, np.sqrt(Ith), fmt='o', color=_l.lines[0].get_color())
            axs[1].plot(_densify(th), th_res.eval(x=_densify(th)), color=_l.lines[0].get_color(), ls='-')
            iaxs.flat[i].imshow(peak_res["data"][slice(dxdydz[2]-margin,dxdydz[2]+margin+1),:,:].sum(axis=0))
            iaxs.flat[i].set_title(f"{parameter} V")

    return {'th_scans' : th_scans,
            'th_fits' : th_fits,
            'tth_scans' : tth_scans,
            'tth_fits' : tth_fits,
            'fit_results' : fit_results,
            'integtrated_intesity' : integtrated_intesity}
