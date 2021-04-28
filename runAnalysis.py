import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
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


def tempsweep(input_path, all_runs):
    """ Compute correlation lengths for all runs within a sweep directory """
    # Arrays for storing data
    sweep_length   = len(all_runs)
    corrFunctions  = []
    r_k            = []
    spinConfigs    = []
    E_dips         = []
    corrLengths    = np.zeros(sweep_length)
    corrLengthsVar = np.zeros(sweep_length)
    corrSums       = np.zeros(sweep_length)

    # Loop through runs in sweep
    for i, run in enumerate(all_runs):
        print("Run {}, temp={}".format(i, all_runs[i]['temp']))

        # Load flatspin run data
        fs_data = np.load(input_path + '/' + run['outdir'], allow_pickle=True)

        # Store E_dip and timesteps
        energies = fs_data['energy']
        E_dips.append([E[1] for E in energies])

        # Get 1d correlation function
        r_k, C, corrSums[i], _, spinConfiguration = tools.getAvgCorrFunction(fs_data, corrConfig)

        # Store C in array for later use (plotting)
        corrFunctions.append(C)
        spinConfigs.append(spinConfiguration)

        # Compute correlation length by curve fitting with exp(-r/zeta)
        popt, pcov = curve_fit(tools.expfunc, r_k, C, bounds=(0,100), p0=10)
        corrLengths[i] = popt[0]
        corrLengthsVar[i] = np.sqrt(np.diag(pcov))

    timesteps = np.array([E[0] for E in energies])

    return np.array(corrFunctions), r_k, corrLengths, corrLengthsVar, corrSums, np.array(spinConfigs), np.array(E_dips), np.array(timesteps)





def main_analysis(input_path, all_runs, out_directory = 'analysis_output/', createPlots=True):
    # Print start info
    startInfo = "Starting temp sweep analysis from flatspin sweep {} ({} runs)".format(input_path, len(all_runs))
    print("-".join(['' for i in range(round(1.1*len(startInfo)))]))
    print(startInfo)
    tools.printConfig(corrConfig)
    print("-".join(['' for i in range(round(1.1*len(startInfo)))]))

    """ """
    temps = np.array([tools.getEndTemp(all_runs[i]['temp']) for i in range(len(all_runs))])
    analysisID    = tools.getAnalysisId(out_directory)
    print("analysisID", analysisID)
    runName       = tools.getRunName(input_path, temps)
    print("runName", runName)
    filenameBase  = out_directory + str(analysisID) + "_" + runName
    print("filenameBase")
    print(filenameBase)
    """ """

    # Get correlation lengths
    corrFunctions, r_k, corrLengths, corrLengthsVar, corrSums, spinConfigs, E_dips, timesteps = tempsweep(input_path, all_runs)

    # Extract end temperatures in simulations
    temps = np.array([tools.getEndTemp(all_runs[i]['temp']) for i in range(len(all_runs))])

    # Compute magnetic susceptibilities using the fluctuation-dissipation theorem
    susceptibilities = tools.flucDissSusceptibility(temps, corrSums)

    # Find critical temperature, T_c
    T_c, C_curie = tools.getCriticalTemp(temps, susceptibilities)

    # determine critical parameter
    A, nu = tools.getCriticalExponent(temps, corrLengths, T_c)

    # filename for storing output files
    analysisID    = tools.getAnalysisId(out_directory)
    runName       = tools.getRunName(input_path, temps)
    filenameBase  = out_directory + str(analysisID) + "_" + runName

    tempSweepResults, parameterResults = tools.processResults(corrConfig, temps, corrLengths, corrLengthsVar, corrSums, susceptibilities, T_c, C_curie, A, nu, writeToFile=True, filenameBase=filenameBase, printResults=True, input_path=input_path)

    if createPlots:
        # Analysis plots
        tools.plotAnalysis(input_path, all_runs, filenameBase, temps, r_k, corrFunctions, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, E_dips, timesteps, saveFile=True, directory=out_directory)

        # Spin config plots
        tools.plotASEs(input_path, all_runs, filenameBase, spinConfigs, temps, saveFile=True, directory=out_directory)


    """
    ---------------------------------------------------------------------------------------------------------------------------
    TO DO
    ------------
    - teste på sweeps
    - notere ned runs som er interessante / gir mening
    - POWER POINT av resultater + diskusjonspunkter
    - Plot temp profile for different coefficients/exponents
    ---------------------------------------------------------------------------------------------------------------------------
    """


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


    # All good
    # Start by reading index.csv into a list of dicts [{'temp': '[4000, 100, 3]', 'outdir': 'SquareSpinIceClosed.000000.npz'}, ...]
    all_runs = tools.readIndex(args.path)

    # Slice run of -i argument specified
    if args.index != None:
        try:
            index = args.index.split(':')
            if index[0] == '':
                all_runs = all_runs[:int(index[1])]
            elif index[1] == '':
                all_runs = all_runs[int(index[0]):]
            else:
                all_runs = all_runs[int(index[0]) : int(index[1])]
        except:
            print("Invalid index. Should be Python list slicing format start:end")
            sys.exit(1)

    if 'temp' in all_runs[0].keys():
        main_analysis(args.path, all_runs)
    else:
        print("Not a temp sweep simulation")
        sys.exit(1)
