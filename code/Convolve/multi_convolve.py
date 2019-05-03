import time
import glob
import os.path as p
import numpy as np
from scipy.signal import fftconvolve
from astropy.io import ascii, fits
from astropy.table import Table, Column

def convolve(

def tconvolve(tess_dir, batman_dir, sector, start, end, output_dir, verbosity=0):
    """
    
    Parameters
    ----------
    tess_dir(str): directory to TESS data
    batman_dir (str): directory to model data
    sector (int): sector to pull data from
    start (int): file to start at
    end (int): file to end at
    output_dir (str): directory to write candidates.csv
    """
    tconv_start = time.time()
    print("===START TCONVOLVE===")
    
    # Handle relative paths
    tess_dir = p.abspath(tess_dir)
    batman_dir = p.abspath(batman_dir)
    output_dir = p.abspath(output_dir)
    
    # Read in TESS Sector data
    print("Reading TESS data...")
    sector_name = "Sector{}".format(sector)
    if sector == 0:
        sector_name = "sample_"+sector_name
    sector_path = p.join(tess_dir, sector_name)
    sector_files = glob.glob(p.join(sector_path,"*.fits"))
    tess_names = sector_files[start:end]
    ntess = len(tess_names)
    print("Found {} TESS files to process".format(ntess))

    # Read in Batman Curves 
    print("Reading Batman transit curves...")
    batmanCurves = ascii.read(p.join(batman_dir,"batmanCurves_small.csv"), 
                       data_start=1, format='csv')
    times = np.array(batmanCurves['times'])
    curve_names = np.array(batmanCurves.colnames[1:])
    nbatman = len(curve_names)
    print("Found {} Batman curves".format(nbatman))
    
    nerr = 0  # count number of failed files
    # Do convolution on all tess files
    for tess_fpath in tess_names:
        tess_fname = p.basename(tess_fpath)
        print("Starting TESS file: {}".format(tess_fname))
        tess_start = time.time()

        try:
            with fits.open(tess_fpath, mode="readonly") as hdulist:
                hdr = hdulist[0].header
                tess_time = hdulist[1].data['TIME']
                tess_flux = hdulist[1].data['PDCSAP_FLUX']
        except Exception as e: 
            print("ERROR reading file: ", tess_fpath, " with error: ", e)
            nerr += 1
            continue  # skip to next loop iter
        
        # clean tess lightcurve of nans
        med = np.nanmedian(tess_flux)
        tess_flux[np.isnan(tess_flux)] = med
        
        tmean = np.mean(tess_flux)
        tstd = np.std(tess_flux)
        tess_flux = (tess_flux - tmean)/tstd
        
        # Do convolution on each batman curve
        peak_times = np.zeros(nbatman)
        peak_convs = np.zeros(nbatman)
        conv_start = time.time()
        print("Starting convolutions...")
        chunks = [curve_names[i::chunk for i in range(chunk)]]
        pool = mp.Pool(processes=nprocs)
        curvenames, peak_convs, peak_times = zip(*pool.map(convolve, chunks))
        
        d = {}
        for curve,conv in p.imap(convolve, range(len(curve_names)):
            d[curve] = np.argmax(conv)

        for i, curvename in enumerate(curve_names):
            # run convolution
            # new way
            batman_curve = batmanCurves[curvename]
            conv = np.abs(fftconvolve(tess_flux, batman_curve, 'same'))
            ind_max = np.argmax(conv)
            peak_times[i] = tess_time[ind_max]
            peak_convs[i] = conv[ind_max]                             

        conv_time = time.time() - conv_start
        print("Convolved {} curves in {:.3} s".format(nbatman, conv_time))
        
        # Keep 10 best curves
        idxs = peak_convs.argsort()[-10:]
        convs = peak_convs[idxs]
        times = peak_times[idxs]
        curves = curve_names[idxs]
        nfitcurves = len(curves)
        print("Found: {} fitting curves".format(nfitcurves))
        
        # Make table
        if nfitcurves > 0:
            outname = 'candidates_s{}_b{}_e{}_{}.csv'.format(sector,start,end,tess_fname)
            outpath = p.join(output_dir, outname)
            print("Writing table: {}".format(outname))
            candidates = Table()
            candidates.add_column(Column([sector_name]*nfitcurves), name="sector")
            candidates.add_column(Column([tess_fname]*nfitcurves), name="tessFile")
            candidates.add_column(Column(curves), name="curveID")
            candidates.add_column(Column(times), name="tcorr")
            candidates.add_column(Column(convs), name="correlation")

            ascii.write(candidates, outpath, format='csv', overwrite=True, comment='#')
            tess_time = time.time() - tess_start
        else:
            print("No curves found for {}, Skipping write...".format(tess_fname))
        print("Finished TESS file in {:.3} s".format(tess_time))
    
    tconv_time = time.time() - tconv_start
    print("Convolved {}/{} tess files with {} curves in {:.3} s".format(ntess-nerr, ntess, nbatman, tconv_time))
    print("===END TCONVOLVE===")
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tess_dir", type=str)
    parser.add_argument("batman_dir", type=str)
    parser.add_argument("sector", type=int)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-v", "--verbosity", default=False, 
                        action="store_true", help="Print console output")
    args = parser.parse_args()
    tconvolve(args.tess_dir, args.batman_dir, args.sector, args.start, 
          args.end, args.output_dir, args.verbosity)
          
if __name__ == '__main__':
    main()
