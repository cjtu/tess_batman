""" tessbatman.py
This file contains helper functions for the tessbatman pipeline.

It is divided into Batman, TESS, and Convolve functions.
"""
from time import time
import glob
import os.path as p

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stat

import astropy as ast
import astropy.table as tbl
import batman


# Batman Functions
def make_batman_config(rmin, rmax, rstep, wmin, wmax, wstep, suffix=""):
    """
    Write batam param file used to generate batmanCurves.
    """
    suffix = '_test'
    param_names = ['LIGHTCURVE_TABLE','PARAMETERS_TABLE','LOG_R_MIN',
                   'LOG_R_MAX','NUM_R_STEP','LOG_W_MIN','LOG_W_MAX',
                   'NUM_W_STEP']
    params = ['batmanCurves{}.csv'.format(suffix), 
              'batmanParams{}.csv'.format(suffix),
              *map(str, (rmin, rmax, rstep, wmin, wmax, wstep))]
    paramfile = 'param{}.txt'.format(suffix)
    with open(paramfile, 'w') as f:
        f.write('# Batman Parameter File\n')
        for i in range(len(params)):
            f.write(' = '.join([param_names[i], params[i]]) + '\n')
            

def make_lightcurve(t0, r, i, p, width, u_type, u_param, t):
    """
    Generate a batman lightcurve with the given parameters.
    
    Parameters
    ----------
    t0 (num): time of inferior conjunction
    r (num): planet radius (in stellar radii)
    i (num): orbital inclination (in degrees)
    p (num): orbital period
    width (num): width parameter (defined as a**3/p**2)
    u_type (str): limb darkening model
    u_param (list): parameters for limb darkening
    
    t: timesteps that you want the fluxes at
    
    assume circular orbit
    """
    # Init batman model
    params = batman.TransitParams()
    params.rp = r
    params.inc = i
    params.w = 0  # longitude of periastron (degenerate with width)
    params.ecc = 0  # eccentricity (0 for circular orbits)
    params.per = p  # orbital period
    params.t0 = t0
    params.a = (width * p ** 2) ** (1 / 3)  # semi-major axis (stellar radii)
    params.limb_dark = u_type
    params.u = u_param
    model = batman.TransitModel(params, t)
    
    # Generate curve
    flux = model.light_curve(params) # compute light curve
    return flux


def make_batman(paramfile, outdir, norm=False, write=True, verbosity=0):
    """ 
    Make the batman curves.

    """
    # TODO: clean up this import & the batmanParams (use dict)
   # read the parameter file
    if verbosity:
        print("Reading param file",flush=True)
    with open(paramfile, "r") as file: 
        data = file.readlines()
        lc_file = outdir+data[1][19:-1]
        pc_file = outdir+data[2][19:-1]
        r_min = float(data[3].split("=")[1])
        r_max = float(data[4].split("=")[1])
        r_step = float(data[5].split("=")[1])
        w_min = float(data[6].split("=")[1])
        w_max = float(data[7].split("=")[1])
        w_step = float(data[8].split("=")[1])
        

    # set up range of parameters
    if verbosity:
        print("Setting param ranges",flush=True)
    potential_radii = np.logspace(r_min, r_max, r_step)
    potential_widths = np.logspace(w_min, w_max, w_step)
    radii = []
    widths = []
    incs = []
    for r in potential_radii:
        for w in potential_widths:
            a = (w * (100)**2)**(1.0/3.0)
            lim = np.arccos((1 + r)/(a))/(2 * np.pi) * 360
            inc = np.linspace(90, lim, 11)[:-1]  # last inc always fails so exclude
            for i in inc: 
                incs.append(i)
                radii.append(r)
                widths.append(w)
                
    # set up file that will eventually become the curve id file
    batmanParams = tbl.Table([radii, incs, widths], names =('rp', 'i', 'width'))
    u = tbl.Column(['0.1 0.3'] * len(batmanParams))
    ld = tbl.Column(['quadratic'] * len(batmanParams))
    t0 = tbl.Column(np.zeros(len(batmanParams))) # set t0 to 0
    e = tbl.Column(np.zeros(len(batmanParams)))
    w = tbl.Column(np.zeros(len(batmanParams)))
    batmanParams.add_column(u, name='u')
    batmanParams.add_column(ld, name='ld')
    batmanParams.add_column(t0, name='t0')
    batmanParams.add_column(e, name='e')
    batmanParams.add_column(w, name='w')

    # make an ID
    ID = tbl.Column(['curve{}'.format(i) for i in range(len(batmanParams))])
    batmanParams['curveID'] = ID

    # actually generate the curves and add them to the curve file
    print("Generating curves",flush=True)
    start = time()
    t = np.arange(-30, 30, 0.13889)
    batmanDict = {'times': t}
    err = 0 # keep track of errored curves
    for i in range(len(batmanParams)): 
        p = batmanParams[i]
        c = make_lightcurve(0, p['rp'], p['i'], 100, p['width'], p['ld'], 
                            [float(val) for val in p['u'].split()], t)
        name = ID[i]
        if norm:
            cmax = np.max(c)
            cmin = np.min(c)
            c = (c-cmin)/(cmax-cmin) # scale to [0,1]
            c = 1-c # flip
            c = c / np.sum(c) # normalize area under curve to 1
            c = 1-c # flip back
            if np.isnan(c).any() or (sum(c==1) < 5):
                print("Batman {} failed".format(ID[i]))
                err += 1
                continue            
        batmanDict[name] = c
        if verbosity and (i % 100 == 0):
            print("Generated {}/{} curves in {} s".format(i+1-err,len(batmanParams),time()-start),flush=True)
    
    batmanCurves = tbl.Table(batmanDict)
    end = time()
    print("Generated {}/{} curves in {} s".format(len(batmanParams)-err,len(batmanParams),time()-start),flush=True)            
    # Write batman params and curves files
    if write:
        twrite = time()
        print("Writing files",flush=True)
        ast.io.ascii.write(batmanParams, pc_file, format='csv', overwrite=True, comment='#', fast_writer=False)
        ast.io.ascii.write(batmanCurves, lc_file, format='csv', overwrite=True, comment='#', fast_writer=False)
        print("Wrote files in {} s".format(time()-twrite),flush=True)
    return(batmanParams, batmanCurves)


def read_batman(batmancurves_file):
    """
    Return times, cureve name, and batman curves from a batmanCurves file.
    
    Parameters
    ----------
    batmancurves_file (str): Path to a batmanCurves file

    Return
    ------
    times (numpy Array): The times array (x axis) of all batmanCurves
    curve_names (numpy Array): The name of each batmanCurve
    batmanCurves (astropy Table): The table of batmanCurves
    """
    # Read in Batman Curves 
    print("Reading batmanCurves from {}...".format(batmancurves_file))
    batmanCurves = ast.io.ascii.read(batmancurves_file, data_start=1, format='csv')
    times = np.array(batmanCurves['times'])
    curve_names = np.array(batmanCurves.colnames[1:])
    return times, curve_names, batmanCurves


# TESS Functions
def read_tess(tess_dir, sector_name, start=0, end=None):
    """
    Return list of tess .fits files in tess_dir from [start:end]. Default
    to all fits files in directory if start and end are not specified.

    Parameters
    ----------
    tess_dir (str): path to tess data directory
    sector_name (str): name of sector subdirectory (e.g. Sector1)
    start (int): (Optional) Index of file in directory to start at
    end (int): (Optional) Index of file to end at
    
    Return
    ------
    tess_names (list): List of file paths to tess .fits data
    """
    print("Reading TESS from {}, s:{}, e:{}...".format(sector_name, start, end))
    sector_path = p.join(tess_dir, sector_name)
    sector_files = glob.glob(p.join(sector_path,"*.fits"))
    tess_names = sector_files[start:end]
    return tess_names

       
def open_tess_fits(tess_fpath, norm=False):
    try:
        with ast.io.fits.open(tess_fpath, mode="readonly") as hdulist:
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
        
    
# Convolve Fucntions
def convolve(tess_time, tess_flux, batmanCurves, curve_names, num_keep=10, plot=False):
    conv_start = time()
    curves = []
    times = np.zeros(num_keep)
    convs = np.zeros(num_keep)
    print("Starting convolutions...",flush=True)
    for i, curvename in enumerate(curve_names):
        # do convolution
        batman_curve = batmanCurves[curvename]
        conv = np.abs(sig.fftconvolve(1-tess_flux, (1-batman_curve), 'same'))
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

    conv_time = time() - conv_start
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
    tconv_start = time()
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
    batmanCurves_file = p.join(batman_dir,"batmanCurves"+batman_suffix)
    times, curve_names, batmanCurves = read_batman(batmanCurves_file)
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
        tess_start = time()
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
                candidates = tbl.Table(d,names=['sector','tessFile','curveID','tcorr','correlation', 'chisq'])
                ast.io.ascii.write(candidates, outpath, format='csv', overwrite=True, comment='#', fast_writer=False)
                print("Wrote file {} at {} s".format(outname,time()-tess_start),flush=True)
                # reset dicts
#                 d = {key : [] for key in ['sector','tessFile','curveID','tcorr','correlation']}
                s=e+1
    candidates = tbl.Table(d,names=['sector','tessFile','curveID','tcorr','correlation'])
    
    # make merged table
    cdf = pd.DataFrame.from_dict(d, columns=['sector','tessFile','curveID','tcorr','correlation'])
    df = pd.merge(cdf, params, on = "curveID", how = "left")
    df.to_csv(p.join(output_dir, "chisq.csv"))
    
    tconv_time = time() - tconv_start
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
    mu, std = stat.norm.fit(1 / arr)
    peaks, _ = sig.find_peaks(1 / arr, height = mu + 4 * std, distance = 1000)
    p = np.diff(tess_time[peaks])
    #define parameters
    PER = np.mean(p)
    u_type = 'quadratic'
    u_param = [0.1, 0.3]
    t = tess_time - tess_time[0]
    #normalize flux
    outcounts = np.nan_to_num(tess_flux[tess_flux > np.nanmean(tess_flux)])
    mu, sigma = stat.norm.fit(outcounts)
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
