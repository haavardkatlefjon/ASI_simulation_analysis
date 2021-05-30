import numpy as np
import pandas as pd
import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import flatspin.data as fsd
from scipy.optimize import curve_fit
import analysisHelpers as tools

sweepConfig = { #sikter på (0.3, 3)
    'dr':       np.arange(0.1, 2.01, 0.2),
    'dtheta':   np.arange(1, 21.01, 2),
}

corrConfig = {
    'N_points_avg':   1,
    'neighbor_dist':  10,#np.inf,
}

def startPFCparamsweep(sim_ds):
    print("Starting sweep")

    if len(sim_ds.index.index) > 1:
        run_index = -1
    else:
        run_index = None

    iteration = 1

    corrLengths = np.zeros((len(sweepConfig['dr']), len(sweepConfig['dtheta'])))

    for i, dr in enumerate(sweepConfig['dr']):
        for j, dtheta in enumerate(sweepConfig['dtheta']):
            print("\nIteration {}/{} \t (dr={}, dtheta={})".format(iteration, len(sweepConfig['dr'])*len(sweepConfig['dtheta']), dr, dtheta))

            corrConfig['dr']     = dr
            corrConfig['dtheta'] = dtheta

            r_k, C, _, _, _ = tools.getAvgCorrFunction(sim_ds, corrConfig, run_index=run_index)

            # Compute correlation length by curve fitting with exp(-r/zeta)
            p0 = r_k[round(0.5*len(r_k))]
            bounds = (0,1000)
            popt, pcov = curve_fit(tools.expfunc, r_k, C, bounds=bounds, p0=p0)
            corrLengths[i,j] = popt[0]
            print("Curve fit bounds ({},{}). Init guess {}".format(bounds[0], bounds[1], p0))
            print("Corr length {} \n".format(round(corrLengths[i,j],2)))

            iteration += 1

    print("\n\n")

    filename = "paramSweep_b1000" + os.path.basename(sim_ds.basepath)
    np.savetxt(filename + "_corrLengths.csv", corrLengths)
    np.savetxt(filename + "_sweepConfig_dr.csv", sweepConfig['dr'])
    np.savetxt(filename + "_sweepConfig_dtheta.csv", sweepConfig['dtheta'])
    print("Saved to file {}".format(filename))




def computeLatticeCorrelation(sim_ds):
    r, C_avg = tools.getLatticeCorrelationFunction(sim_ds)
    data = np.vstack((r, C_avg)).T

    # Compute correlation length by curve fitting with exp(-r/zeta)
    p0 = r[round(0.5*len(r))]
    bounds = (0,100)
    popt, pcov = curve_fit(tools.expfunc, r, C_avg, bounds=bounds, p0=p0)
    zeta = popt[0]
    print("Curve fit bounds ({},{}). Init guess {}".format(bounds[0], bounds[1], p0))
    print("Corr length {} \n".format(round(zeta,2)))


    out_directory = os.path.join('data_for_thesis', 'PCF_grid_sweeps')
    filename = os.path.join(out_directory, "latticeCorrFunc_" + os.path.basename(sim_ds.basepath))
    filename += "_zeta_" + str(round(zeta,2))

    np.savetxt(filename + ".csv", data)
    print("Saved to file {}".format(filename))


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Arguments for analysis')

    # Add the arguments
    my_parser.add_argument("path",
                            type=str,
                            help='the directory containing files from a flatspin run or an analysis number for reuse of existing analysis')
    my_parser.add_argument('-m',
                           '--method',
                           metavar='method',
                           type=str,
                           help='Method (lattice)')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    # Check the directory exists
    if not os.path.isdir(args.path):
        print('The path specified does not exist')
        sys.exit(1)

    else:
        print("Loading flatspin dataset")

        if args.path.endswith('/'):
            args.path = args.path[:-1]

        # Read flatspin sweep data
        sim_ds = fsd.Dataset.read(args.path)

        if args.method == 'lattice':
            print('Lattice method')
            computeLatticeCorrelation(sim_ds)

        else:
            startPFCparamsweep(sim_ds)
