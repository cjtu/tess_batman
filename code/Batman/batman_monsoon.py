import batman
from astropy.io import ascii
from astropy.table import Table, Column
import numpy as np
from time import time


def make_lightcurve(r, i, width, u_type, u_param, t):
    '''
    r: planet radius (in stellar radii)
    i: inclination (in degrees)
    width: "width parameter" a**3/p**2
    u_type: type of limb darkening
    u_param: list of parameters for limb darkening
    
    t: timesteps that you want the fluxes at
    
    assume circular orbit
    '''
    params = batman.TransitParams()
    params.rp = r                 #planet radius (in units of stellar radii)
    params.inc = i                #orbital inclination (in degrees)
    params.w = 0                  #longitude of periastron (in degrees)
    params.ecc = 0                #eccentricity
    params.per = 100                #orbital period
    params.t0 = 0                 #time of inferior conjunction
    params.a = (width * 100**2)**(1/3)                  #semi-major axis (in units of stellar radii)
    params.u = u_param            #limb darkening coefficients [u1, u2]
    params.limb_dark = u_type             #limb darkening model
    m = batman.TransitModel(params, t)    #initializes model
    flux = m.light_curve(params)          #calculates light curve
    return flux




#param = input('Enter name of the parameter file: ')


def make_batman(paramfile, outdir, norm=False, write=True, verbosity=0):
    ''' Make the batman curves.

    PARAMETER RANGES:
    r: 0.2 - 0.001
        16 intervals (logspaced)
    i: from 90 to (np.arccos((1 + planet radius)/(semimajor axis*(1-ecc)))/(2 * np.pi) * 360) (all in units of stellar radii)
        10 intervals
    w: 1 - 10**5
        16 intervals (logspaced)
    Limb darkening: 
    '''
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
        
# this was a sanity check
# print(lc_file, pc_file, r_min, r_max, r_step, w_min, w_max, w_step)

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
    batmanParams = Table([radii, incs, widths], names =('rp', 'i', 'width'))
    u = Column(['0.1 0.3'] * len(batmanParams))
    ld = Column(['quadratic'] * len(batmanParams))
    t0 = Column(np.zeros(len(batmanParams))) # set t0 to 0
    e = Column(np.zeros(len(batmanParams)))
    w = Column(np.zeros(len(batmanParams)))
    batmanParams.add_column(u, name='u')
    batmanParams.add_column(ld, name='ld')
    batmanParams.add_column(t0, name='t0')
    batmanParams.add_column(e, name='e')
    batmanParams.add_column(w, name='w')

# make an ID
    ID = Column(['curve{}'.format(i) for i in range(len(batmanParams))])
    batmanParams['curveID'] = ID

# actually generate the curves and add them to the curve file
    print("Generating curves",flush=True)
    start = time()
    t = np.arange(-30, 30, 0.13889)
    batmanDict = {'times': t}
    err = 0 # keep track of errored curves
    for i in range(len(batmanParams)): 
        p = batmanParams[i]
        c = make_lightcurve(p['rp'], p['i'], p['width'], p['ld'], 
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
    
    batmanCurves = Table(batmanDict)
    end = time()
    print("Generated {}/{} curves in {} s".format(len(batmanParams)-err,len(batmanParams),time()-start),flush=True)            
    # Write batman params and curves files
    if write:
        twrite = time()
        print("Writing files",flush=True)
        ascii.write(batmanParams, pc_file, format='csv', overwrite=True, comment='#', fast_writer=False)
        ascii.write(batmanCurves, lc_file, format='csv', overwrite=True, comment='#', fast_writer=False)
        print("Wrote files in {} s".format(time()-twrite),flush=True)
    return(batmanParams, batmanCurves)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=str)
    parser.add_argument("outdir", type=str)
    args = parser.parse_args()
    make_batman(args.params, args.outdir)


if __name__ == '__main__':
    main()
