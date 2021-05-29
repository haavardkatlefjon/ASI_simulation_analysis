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

corrConfig = {
    'N_points_avg':   1,
    'neighbor_dist':  np.inf,
    'dr':             0.3,
    'dtheta':         6,
}

def generateCorrFunctions(simulation_paths, output_directory):
    print("Starting analysis")

    for path in simulation_paths:
        sim_ds = fsd.Dataset.read(path)

        filename = "1d_corrFunc_dr{}-dtheta{}_".format(corrConfig['dr'], corrConfig['dtheta']) + os.path.basename(sim_ds.basepath) + "_corrFunc.csv"

        if os.path.isfile(os.path.join(output_directory, filename)):
            print("File already exist ({}). Continuing to next.".format(filename))
            continue


        if len(sim_ds.index.index) > 1:
            run_index = -1
        else:
            run_index = None

        r_k, C, _, _, _ = tools.getAvgCorrFunction(sim_ds, corrConfig, run_index=run_index)

        data = np.vstack((r_k, C)).T

        np.savetxt(os.path.join(output_directory, filename), data)

        print("Saved to file {}".format(filename))



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":

    simulation_paths = []
    directory = 'data_for_thesis/increasing_disorder_500K/'

    for entry in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, entry)):
            simulation_paths.append(os.path.join(directory, entry))

    generateCorrFunctions(simulation_paths, directory)
