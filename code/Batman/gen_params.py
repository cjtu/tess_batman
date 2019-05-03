suffix = '_test'
param_names = ['LIGHTCURVE_TABLE','PARAMETERS_TABLE','LOG_R_MIN','LOG_R_MAX','NUM_R_STEP',
                 'LOG_W_MIN','LOG_W_MAX','NUM_W_STEP']
params = ['batmanCurves{}.csv'.format(suffix), 'batmanParams{}.csv'.format(suffix),
                  '-1', '-1', '1', '-3.9', '-2.5', '30']
paramfile = 'param{}.txt'.format(suffix)
with open(paramfile, 'w') as f:
    f.write('# Batman Parameter File\n')
    for i in range(len(params)):
        f.write(' = '.join([param_names[i], params[i]]) + '\n')
