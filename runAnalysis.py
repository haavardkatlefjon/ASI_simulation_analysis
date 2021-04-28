import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib.pyplot as plt
import flatspin.data as fsd
from scipy.optimize import curve_fit

import analysisHelpers as tools

"""""""""""""""""""""""""""""""""""""""   CORR CONFIG   """""""""""""""""""""""""""""""""""""""
corrConfig = {
    'dr':             0.2,    # units of lattice spacing
    'dtheta':         90/70,  # degrees
    'N_points_avg':   1,      # number of timeframes used to make thermal avg
    'neighbor_dist':  8,      # units of lattice spacing. Distance within correlation should be checked
}
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# def tempsweep(input_path, all_runs):
def tempsweep(sweep_ds):
    """ Compute correlation lengths for all runs within a sweep directory """
    # Arrays for storing data
    sweep_length   = len(sweep_ds.index.index)
    corrFunctions  = []
    r_k            = []
    spinConfigs    = []
    corrLengths    = np.zeros(sweep_length)
    corrLengthsVar = np.zeros(sweep_length)
    corrSums       = np.zeros(sweep_length)

    # Loop through runs in sweep
    for i in sweep_ds.index.index:
        print("Run {}/{}, temp={}".format(i, len(sweep_ds.index.index), sweep_ds.index.iloc[i]['temp']))

        # Get 1d correlation function
        r_k, C, corrSums[i], _, spinConfiguration = tools.getAvgCorrFunction(sweep_ds, i, corrConfig)

        # Store C in array for later use (plotting)
        corrFunctions.append(C)
        spinConfigs.append(spinConfiguration)

        # Compute correlation length by curve fitting with exp(-r/zeta)
        popt, pcov = curve_fit(tools.expfunc, r_k, C, bounds=(0,100), p0=10)
        corrLengths[i] = popt[0]
        corrLengthsVar[i] = np.sqrt(np.diag(pcov))

    return np.array(corrFunctions), r_k, corrLengths, corrLengthsVar, corrSums, np.array(spinConfigs)




def main_analysis(sweep_ds, out_directory = 'analysis_output', createPlots=True):
    # Print start info
    startInfo = "Starting temp sweep analysis from flatspin sweep {} ({} runs) \n".format(sweep_ds.basepath, len(sweep_ds.index.index))
    print("-".join(['' for i in range(round(1.1*len(startInfo)))]))
    print(startInfo)
    tools.printConfig(corrConfig)
    print("\nflatspin runs")
    print(sweep_ds.index)
    print("-".join(['' for i in range(round(1.1*len(startInfo)))]))

    # Get correlation lengths
    corrFunctions, r_k, corrLengths, corrLengthsVar, corrSums, spinConfigs = tempsweep(sweep_ds)

    # Extract end temperatures in simulations
    temps = np.array([tools.getEndTemp(sweep_ds.index.iloc[i]['temp']) for i in sweep_ds.index.index])

    # Compute magnetic susceptibilities using the fluctuation-dissipation theorem
    susceptibilities = tools.flucDissSusceptibility(temps, corrSums)

    # Find critical temperature, T_c
    T_c, C_curie = tools.getCriticalTemp(temps, susceptibilities)

    # determine critical parameter
    A, nu = tools.getCriticalExponent(temps, corrLengths, T_c)

    # filename for storing output files
    analysisID    = tools.getAnalysisId(out_directory)
    runName       = tools.getRunName(sweep_ds.basepath, temps)
    filenameBase  = os.path.join(out_directory, str(analysisID) + "_" + runName)
    tempSweepResults, parameterResults = tools.processResults(corrConfig, temps, corrLengths, corrLengthsVar, corrSums, susceptibilities, T_c, C_curie, A, nu, writeToFile=True, filenameBase=filenameBase, printResults=True, input_path=sweep_ds.basepath)

    if createPlots:
        # Analysis plots
        tools.plotAnalysis(sweep_ds, filenameBase, temps, r_k, corrFunctions, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, saveFile=True, directory=out_directory)

        # Spin config plots
        tools.plotASEs(sweep_ds, filenameBase, spinConfigs, temps, saveFile=True, directory=out_directory)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Arguments for analysis')

    # Add the arguments
    my_parser.add_argument('-o',
                           '--path',
                           metavar='path',
                           type=str,
                           help='the directory containing files from a flatspin run')
    my_parser.add_argument('-i',
                           '--index',
                           metavar='index',
                           type=str,
                           help='Specify if just a subset of a simulation should be used')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    # Check the directory exists
    if not os.path.isdir(args.path):
        print('The path specified does not exist')
        sys.exit(1)

    # Read flatspin sweep data
    sweep_ds = fsd.Dataset.read(args.path)

    # Slice run of -i argument specified
    if args.index != None:
        try:
            index = args.index.split(':')
            if index[0] == '':
                sweep_ds.index = sweep_ds.index.iloc[:int(index[1]), :]
            elif index[1] == '':
                sweep_ds.index = sweep_ds.index.iloc[int(index[0]):, :]
            else:
                sweep_ds.index = sweep_ds.index.iloc[int(index[0]):int(index[0]), :]
            sweep_ds.index.reset_index(inplace=True)
        except:
            print("Invalid index. Should be Python list slicing format start:end")
            sys.exit(1)


    if 'temp' in sweep_ds.index.columns:
        main_analysis(sweep_ds)
    else:
        print("Not a temp sweep simulation")
        sys.exit(1)
