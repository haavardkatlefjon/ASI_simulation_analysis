import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib.pyplot as plt
import flatspin.data as fsd
from scipy.optimize import curve_fit

try:
    import analysisHelpers as tools
except Exception as e:
    import ASI_simulation_analysis.analysisHelpers as tools

"""""""""""""""""""""""""""""""""""""""   CORR CONFIG   """""""""""""""""""""""""""""""""""""""
corrConfig = {
    'dr':             0.5,     # units of lattice spacing
    'dtheta':         90/70,   # degrees
    'N_points_avg':   1,       # number of timeframes used to make thermal avg
    'neighbor_dist':  np.inf,  # units of lattice spacing. Distance within correlation should be checked
}

#@TODO! NEIGHBOR DIST FUCKUP. ALLOCATE C WHEN np.inf -> fuckings fuck. Finn på noe smart!
#forslag: finn maksimum mulige distanse og bruk det til allokering av C
#if sweep_ds.params['neighbor_distance'] == np.inf:
#    round(np.amax(abs_distances) elns)
# kjør sweep på nyeste I40 alpha 0.06...
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
    for i in range(len(sweep_ds.index.index)):
        print("Run {}/{}, temp={}".format(i+1, len(sweep_ds.index.index), sweep_ds.index.iloc[i]['temp']))

        # Get 1d correlation function
        r_k, C, corrSums[i], _, spinConfiguration = tools.getAvgCorrFunction(sweep_ds, i, corrConfig)

        # Store C in array for later use (plotting)
        corrFunctions.append(C)
        spinConfigs.append(spinConfiguration)

        # Compute correlation length by curve fitting with exp(-r/zeta)
        popt, pcov = curve_fit(tools.expfunc, r_k, C, bounds=(0, 10*r_k[-1]), p0=r_k[round(0.1*len(r_k))])
        corrLengths[i] = popt[0]
        corrLengthsVar[i] = np.sqrt(np.diag(pcov))

        print("Curve fit bounds (0,{}). Init guess {}".format(10*r_k[-1], r_k[round(0.1*len(r_k))]))
        print("Corr length {} \n".format(round(corrLengths[i],2)))

    return np.array(corrFunctions), r_k, corrLengths, corrLengthsVar, corrSums, np.array(spinConfigs)




def main_analysis(sweep_ds, out_directory = 'analysis_output', createPlots=True, returnKey = None):
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
    temps = np.array([tools.getEndTemp(sweep_ds.index.iloc[i]['temp']) for i in range(len(sweep_ds.index.index))])

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
    tempSweepResults, parameterResults = tools.processResults(corrConfig, temps, corrFunctions, corrLengths, corrLengthsVar, corrSums, susceptibilities, T_c, C_curie, A, nu, writeToFile=True, filenameBase=filenameBase, printResults=True, input_path=sweep_ds.basepath)

    if createPlots:
        # Analysis plots
        tools.plotAnalysis(sweep_ds, filenameBase, temps, r_k, corrFunctions, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, saveFile=True, directory=out_directory)

        # Spin config plots
        tools.plotASEs(sweep_ds, filenameBase, spinConfigs, temps, saveFile=True, directory=out_directory)

    if returnKey != None:
        try:
            print("Returning", parameterResults[returnKey])
            return parameterResults[returnKey]
        except KeyError:
            print("Illegal returnKey, must be {}", parameterResults.keys())


def fitnessFunction(sweep_ds):
    """
    Specify returnKey in main_analysis() to get a return value.
    returnKey can be
        'T_c'         critical temperature
        'C_curie'     Curie constant
        'A'           power law coefficient
        'nu'          critical exponent
    """
    return main_analysis(sweep_ds, createPlots=True, returnKey = 'T_c')


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
                sweep_ds.index = sweep_ds.index.iloc[int(index[0]):int(index[1]), :]
            #sweep_ds.index.reset_index(inplace=True)
        except:
            print("Invalid index. Should be Python list slicing format start:end")
            sys.exit(1)


    if 'temp' in sweep_ds.index.columns:
        #main_analysis(sweep_ds)
        fitnessFunction(sweep_ds)
    else:
        print("Not a temp sweep simulation")
        sys.exit(1)
