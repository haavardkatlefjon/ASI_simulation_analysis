import numpy as np
import pandas as pd
import os
import sys
import csv
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


def tempsweep(sweep_ds, temps):
    """ Compute correlation lengths for all runs within a sweep directory """
    # Arrays for storing data
    if len(temps) < len(sweep_ds.index.index):
        assert 'group_id' in sweep_ds.index.columns, 'group_id not a column in dataset'
        print("Several runs per temp")
        groups = np.unique(sweep_ds.index['group_id'].to_numpy())
        print("groups", groups)
    else:
        groups = np.array([0])


    sweep_length   = len(temps)
    corrFunctions  = []
    r_k            = []
    spinConfigs    = []
    corrSums       = np.zeros((sweep_length, len(groups)))
    corrLengths    = np.zeros(sweep_length)
    corrLengthsVar = np.zeros(sweep_length)

    # Loop through runs in sweep
    for i in range(sweep_length):
        print("Run {}/{}, temp={}".format(i+1, sweep_length, temps[i]))

        C_this_temp = []
        for g_id in groups:

            if len(groups) > 1:
                run_index = sweep_ds.index.index[
                                (sweep_ds.index['T_end'] == temps[i]).to_numpy()*
                                (sweep_ds.index['group_id'] == g_id).to_numpy()
                                ].tolist()[0]
            else:
                run_index = i
            # Get 1d correlation function
            r_k, C, corrSums[i,g_id], _, spinConfiguration = tools.getAvgCorrFunction(sweep_ds, corrConfig, run_index)

            C_this_temp.append(C)

        C_this_temp = np.array(C_this_temp)
        C = np.mean(C_this_temp, axis=0)

        # Store C in array for later use (plotting)
        corrFunctions.append(C)
        spinConfigs.append(spinConfiguration)

        # Compute correlation length by curve fitting with exp(-r/zeta)
        p0 = r_k[round(0.5*len(r_k))]
        popt, pcov = curve_fit(tools.expfunc, r_k, C, bounds=(0, 10*r_k[-1]), p0=p0)
        corrLengths[i] = popt[0]
        corrLengthsVar[i] = np.sqrt(np.diag(pcov))
        print("Curve fit bounds (0,{}). Init guess {}".format(10*r_k[-1], p0))
        print("Corr length {} \n".format(round(corrLengths[i],2)))

    corrSumsMean = np.mean(corrSums, axis=1)
    corrSumsStd = np.std(corrSums, axis=1)
    corrSums = np.vstack((corrSumsMean, corrSumsStd)).T

    return np.array(corrFunctions), r_k, corrLengths, corrLengthsVar, corrSums, np.array(spinConfigs)


def main_analysis(sweep_ds, out_directory = 'analysis_output', createPlots=True, returnKey = None):
    # Print start info
    startInfo = "Starting temp sweep analysis from flatspin sweep {} ({} runs) \n".format(sweep_ds.basepath, len(sweep_ds.index.index))
    print("-".join(['' for i in range(round(1.1*len(startInfo)))]))
    print(startInfo)
    tools.printConfig(corrConfig)
    print()

    columns_to_show = []
    if 'T_end' in sweep_ds.index.columns:
        columns_to_show.append('T_end')
    else:
        columns_to_show.append('temp')
    if 'group_id' in sweep_ds.index.columns:
        columns_to_show.append('group_id')
    columns_to_show.append('outdir')

    print(sweep_ds.index[columns_to_show])

    print("-".join(['' for i in range(round(1.1*len(startInfo)))]))

    # Extract end temperatures in simulations
    #temps = np.array([tools.getEndTemp(sweep_ds.index.iloc[i]['temp']) for i in range(len(sweep_ds.index.index))])
    temps = tools.getTemps(sweep_ds)

    # Get correlation lengths
    corrFunctions, r_k, corrLengths, corrLengthsVar, corrSums, spinConfigs = tempsweep(sweep_ds, temps)

    # Compute magnetic susceptibilities using the fluctuation-dissipation theorem
    susceptibilities    = tools.flucDissSusceptibility(temps, corrSums[:,0])
    invSusceptibilitiesStd = tools.invSusceptibilityStd(temps, corrSums)

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
        tools.plotAnalysis(sweep_ds, filenameBase, temps, r_k, corrFunctions, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, invSusceptibilitiesStd=invSusceptibilitiesStd, saveFile=True, directory=out_directory)

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


def main_existing_analysis(path, args, out_directory='analysis_output', createPlots = True):
    data = tools.readData(path, args)

    # Compute magnetic susceptibilities using the fluctuation-dissipation theorem
    susceptibilities    = tools.flucDissSusceptibility(data['temps'], data['corrSumMean'])
    invSusceptibilitiesStd = tools.invSusceptibilityStd(data['temps'], np.vstack((data['corrSumMean'], data['corrSumStd'])).T)

    # Find critical temperature, T_c
    T_c, C_curie = tools.getCriticalTemp(data['temps'], susceptibilities)

    # determine critical parameter
    A, nu = tools.getCriticalExponent(data['temps'], data['corrLengths'], T_c)

    analysisID    = tools.getAnalysisId(out_directory)
    runName       = tools.getRunName(os.path.basename(path), data['temps'])
    filenameBase  = os.path.join(out_directory, str(analysisID) + "_" + runName)
    #tempSweepResults, parameterResults = tools.processResults(corrConfig, temps, corrFunctions, corrLengths, corrLengthsVar, corrSums, susceptibilities, T_c, C_curie, A, nu, writeToFile=True, filenameBase=filenameBase, printResults=True, input_path=sweep_ds.basepath)

    if createPlots:
        # Analysis plots
        tools.plotAnalysisSimplified(filenameBase, data['temps'], data['corrLengths'], data['corrLengthsVar'], susceptibilities, T_c, C_curie, A, nu, invSusceptibilitiesStd=invSusceptibilitiesStd, saveFile=True, directory=out_directory)












""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":

    legg inn vertikal linje for T=T_C i zeta vs T plot.
    Kun bruke T > T_C ?
    Kun bruke T slik at zeta er monotont voksende fra høye mot lave temp?
    Implementer gradvis gauss disorder på rotasjon!

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Arguments for analysis')

    # Add the arguments
    my_parser.add_argument("path",
                            type=str,
                            help='the directory containing files from a flatspin run or an analysis number for reuse of existing analysis')
    my_parser.add_argument('-i',
                           '--index',
                           metavar='index',
                           type=str,
                           help='Specify if just a subset of a simulation should be used')
    my_parser.add_argument('-d',
                           '--drop',
                           metavar='drop',
                           type=str,
                           help='Drop one (int) or more ([list of int]) runs')
    my_parser.add_argument('-t',
                           '--temp',
                           metavar='temp',
                           type=str,
                           help='Temp range to use. Python list slicing format.')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    # Check the directory exists
    if tools.isFloat(args.path):
        path = tools.existingAnalysis(id = int(args.path))
        if path != None:
            print("Existing analysis. Opening {}".format(path))
            main_existing_analysis(path, args)
        else:
            print("Could not find analysis with id {}".format(args.path))
            sys.exit(1)

    elif not os.path.isdir(args.path):
        print('The path specified does not exist')
        sys.exit(1)

    else:
        print("Loading flatspin dataset")

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

        if args.drop != None:
            try:
                if tools.isFloat(args.drop):
                    args.drop = int(args.drop)
                    if args.index != None:
                        if index[0] != '':
                            args.drop += int(index[0])
                    sweep_ds.index.drop(int(args.drop), inplace=True)
            except Exception as e:
                print("Could not drop requested rows")
                print(type(e), e)
                sys.exit(1)


        if 'temp' in sweep_ds.index.columns:
            #main_analysis(sweep_ds)
            fitnessFunction(sweep_ds)
        else:
            print("Not a temp sweep simulation")
            sys.exit(1)
