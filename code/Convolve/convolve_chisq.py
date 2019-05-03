import time
import glob
import os.path as p
import multiprocessing as mp
import batman
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from scipy.stats import norm
from astropy.io import ascii, fits
from astropy.table import Table, Column
import matplotlib.pyplot as plt

def read_tess(tess_dir, sector_name, start, end):
    # Read in TESS Sector data
    print("Reading TESS from {}, s:{}, e:{}...".format(sector_name, start, end))
    sector_path = p.join(tess_dir, sector_name)
    sector_files = glob.glob(p.join(sector_path,"*.fits"))
    tess_names = sector_files[start:end]
    return tess_names

def read_batman(batman_dir, batman_suffix):
    # Read in Batman Curves 
    print("Reading Batman transit curves...")
    batman_name = "batmanCurves{}.csv".format(batman_suffix)
    print(batman_name)
    #if sector == 0:
    #    batman_name = "sample_"+batman_name
    batmanCurves = ascii.read(p.join(batman_dir,batman_name), 
                       data_start=1, format='csv')
    times = np.array(batmanCurves['times'])
    curve_names = np.array(batmanCurves.colnames[1:])
    return times, curve_names, batmanCurves
       
def open_tess_fits(tess_fpath, norm=False):
    try:
        with fits.open(tess_fpath, mode="readonly") as hdulist:
            hdr = hdulist[0].header
            tess_time = hdulist[1].data['TIME']
            tess_flux = hdulist[1].data['PDCSAP_FLUX']
        # set NaNs to median
        med = np.nanmedian(tess_flux)
        tess_flux[np.isnan(tess_flux)] = med
        
        if norm:
#             tess_flux[tess_flux > np.median(tess_flux)] = np.median(tess_flux)
            tmin = np.min(tess_flux)
            tmax = np.max(tess_flux)
            tess_flux = (tess_flux - tmin)/(tmax-tmin)

    except Exception as e: 
        print("ERROR reading file: ", tess_fpath, " with error: ", e,flush=True)
        return None, None
    return tess_time, tess_flux
        
def convolve(tess_time, tess_flux, batmanCurves, curve_names, num_keep=10, plot=False):
    conv_start = time.time()
    curves = []
    times = np.zeros(num_keep)
    convs = np.zeros(num_keep)
    print("Starting convolutions...",flush=True)
    for i, curvename in enumerate(curve_names):
        # do convolution
        batman_curve = batmanCurves[curvename]
        conv = np.abs(fftconvolve(1-tess_flux, (1-batman_curve), 'same'))
        ind_max = np.argmax(conv)
        conv_max = conv[ind_max]
        
        # if num_keep, save only the top num_keep curves
        if num_keep < len(curve_names):
            if conv_max > convs[-1]:
                # insert in reverse sorted order
                ind = np.searchsorted(-convs, -conv_max)
                curves = curves[:ind] + [curvename] + curves[ind:-1]
                times = np.insert(times, ind, tess_time[ind_max])[:-1]
                convs = np.insert(convs, ind, conv_max)[:-1]
        else:
            curves.append(curvename)
            times[i] = tess_time[ind_max]
            convs[i] = conv_max
        if plot:
            plt.plot(tess_time, conv, label=curvename)

    conv_time = time.time() - conv_start
    print("Convolved {} curves in {:.3} s".format(len(curve_names), conv_time),flush=True)
    return curves, times, convs

    
def tbconvolve(tess_dir, batman_dir, batman_suffix, sector, start, end, output_dir, num_keep=10, norm_tess=False, write=True, writechunk=10, verbosity=0):
    """
    
    Parameters
    ----------
    tess_dir(str): directory to TESS data
    batman_dir (str): directory to model data
    batman_suffix(str): suffix to append to barmanCurves file (e.g. _small)
    sector (int): sector to pull data from
    start (int): file to start at
    end (int): file to end at
    output_dir (str): directory to write candidates.csv
    """  
    tconv_start = time.time()
    print("===START TCONVOLVE===",flush=True)
    
    # Handle relative paths
    tess_dir = p.abspath(tess_dir)
    batman_dir = p.abspath(batman_dir)
    output_dir = p.abspath(output_dir)
        
    # Read in TESS Sector data
    sector_name = "Sector{}".format(sector)
    if sector == 0:
        sector_name = "sample_"+sector_name
    tess_names = read_tess(tess_dir, sector_name, start, end)
    ntess = len(tess_names)
    print("Found {} TESS files to process".format(ntess),flush=True)
    if ntess < 1:
        print("No tess curves found, quitting....")
        return None
    
    # Read in Batman Curves 
    times, curve_names, batmanCurves = read_batman(batman_dir, batman_suffix)
    nbatman = len(curve_names)
    print("Found {} Batman curves".format(nbatman),flush=True)
    if ntess < 1:
        print("No batman curves found, quitting....")
        return None

    # Read in Batman Params
    params = pd.read_csv(p.join(batman_dir, "batmanParams{}.csv".format(batman_suffix)))



    #Init dict for saving best batman curves 
    d = {key : [] for key in ['sector','tessFile','curveID','tcorr','correlation', 'chisq']}
    s = 0
    nerr = 0  # count number of failed files
    
    # Do convolution on all tess files
    for tind, tess_fpath in enumerate(tess_names):
        tess_start = time.time()
        tess_fname = p.basename(tess_fpath)
        print("Starting TESS file: {}".format(tess_fname),flush=True)
        
        # Read tess lightcurve
        tess_time, tess_flux = open_tess_fits(tess_fpath, norm_tess)
        if tess_time is None:
            nerr += 1
            continue # skip to next iter if read failed
            
        # Do convolution and keep num_keep best curves
        if num_keep < 1:
            num_keep = len(curve_names)
        curves, times, convs = convolve(tess_time, tess_flux, batmanCurves, curve_names, num_keep)
        
        # Save this TESS curve's best batman curves to dict
        d['sector'].extend([sector_name]*num_keep)
        d['tessFile'].extend([tess_fname]*num_keep)
        d['curveID'].extend(curves)
        d['tcorr'].extend(times)
        d['correlation'].extend(convs)
        d['chisq'].extend(get_chi_sq(tess_time, tess_flux, times, params))

        if write:
            # Make table every writechunk tess curves
            if (tind % writechunk == writechunk-1) or (tind == len(tess_names)-1):
                e = start+tind
                outname = 'candidates_sector{}_s{}_e{}.csv'.format(sector, s, e)
                outpath = p.join(output_dir, outname)
                # Convert to astropy table and write to csv
                candidates = Table(d,names=['sector','tessFile','curveID','tcorr','correlation', 'chisq'])
                ascii.write(candidates, outpath, format='csv', overwrite=True, comment='#', fast_writer=False)
                print("Wrote file {} at {} s".format(outname,time.time()-tess_start),flush=True)
                # reset dicts
#                 d = {key : [] for key in ['sector','tessFile','curveID','tcorr','correlation']}
                s=e+1
    candidates = Table(d,names=['sector','tessFile','curveID','tcorr','correlation'])
    
    # make merged table
    cdf = pd.DataFrame.from_dict(d, columns=['sector','tessFile','curveID','tcorr','correlation'])
    df = pd.merge(cdf, params, on = "curveID", how = "left")
    df.to_csv(p.join(output_dir, "chisq.csv"))
    
    tconv_time = time.time() - tconv_start
    print("Convolved {}/{} tess files with {} curves in {:.3} s".format(ntess-nerr, ntess, nbatman, tconv_time),flush=True)
    print("===END TCONVOLVE===",flush=True)
    return candidates

def get_chi_sq(tess_time, tess_flux, tcorr, params):
    current_fname = ""
    chi_squared = np.zeros(len(params))
    #find the lightcurve minima to calculate the exoplanet period
    arr = tess_flux / np.nanmedian(tess_flux)
    arr[np.isnan(arr)] = np.nanmedian(arr)
    arr[arr==0] = np.nanmedian(arr)
    mu, std = norm.fit(1 / arr)
    peaks, _ = find_peaks(1 / arr, height = mu + 4 * std, distance = 1000)
    p = np.diff(tess_time[peaks])
    #define parameters
    PER = np.mean(p)
    u_type = 'quadratic'
    u_param = [0.1, 0.3]
    t = tess_time - tess_time[0]
    #normalize flux
    outcounts = np.nan_to_num(tess_flux[tess_flux > np.nanmean(tess_flux)])
    mu, sigma = norm.fit(outcounts)
    normalized_fluxes = tess_flux / mu
    normalized_sigma = np.sqrt(tess_flux)/mu
        
    for i, row in params.iterrows():
        #get params for this row
        T0 = tcorr[i]- tess_time[0]
        RP = row["rp"]
        INC = row["i"]
        width = row["width"]

        #calculate reduced chi-squared
        chi_squared[i] = np.nansum(((normalized_fluxes - make_lightcurve(T0, RP, INC, PER, width, u_type, u_param, t)) ** 2 / normalized_sigma ** 2) / 8)

    return chi_squared

def make_lightcurve(T0, RP, INC, PER, width, u_type, u_param, t):
        '''
        r: planet radius
        i: inclination
        per = orbital period
        width: "width parameter" a ** 3/per ** 2
        u_type: type of limb darkening
        u_param: list of parameters for limb darkening
                                    
        assume circular orbit
        '''
        params = batman.TransitParams()
        params.w = 0                  #longitude of periastron (in degrees)
        params.ecc = 0                #eccentricity
        params.t0 = T0                 #time of inferior conjunction
        params.rp = RP                 #planet radius (in units of stellar radii)
        params.inc = INC                #orbital inclination (in degrees)
        params.per = PER               #orbital period
        params.a = (width * PER ** 2) ** (1/3)                  #semi-major axis (in units of stellar radii)
        params.u = u_param            #limb darkening coefficients [u1, u2]
        params.limb_dark = u_type             #limb darkening model
        m = batman.TransitModel(params, t)    #initializes model
        flux = m.light_curve(params)          #calculates light curve
        return flux

    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tess_dir", type=str)
    parser.add_argument("batman_dir", type=str)
    parser.add_argument("sector", type=int)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("output_dir", type=str) 
    parser.add_argument("batman_suffix",type=str,default="")
    parser.add_argument("-v", "--verbosity", default=False, 
                        action="store_true", help="Print console output")
    args = parser.parse_args()
    tbconvolve(args.tess_dir, args.batman_dir, args.batman_suffix, args.sector, args.start, 
          args.end, args.output_dir, num_keep=-1, norm_tess=True, verbosity=args.verbosity)
          
if __name__ == '__main__':
    main()
