import analysisHelpers as tools
import numpy as np
import matplotlib.pyplot as plt
import flatspin as fs
import os

corrConfig = {
    'dr':             0.5,     # units of lattice spacing
    'dtheta':         90/70,   # degrees
    'N_points_avg':   1,       # number of timeframes used to make thermal avg
    'neighbor_dist':  10,  # units of lattice spacing. Distance within correlation should be checked
}

def plotCorrSumRandomSpin(fs_path):
    if fs_path.endswith(".csv"):
        print("Implement load csv")
        data = np.loadtxt(fs_path)
        print("data.shape")
        print(data.shape)
        temp        = data[:,0]
        corrSums    = data[:,1]

    else:
        print("Loading {}".format(fs_path))
        ds = fs.data.Dataset.read(fs_path)
        print(ds.index)

        corrSums = np.zeros(len(ds.index.index))

        for i in range(len(ds.index.index)):
            r_k, C, corrSums[i], _, spinConfiguration = tools.getAvgCorrFunction(ds, i, corrConfig)
            print("CorrSum {}:\t{}\n".format(i, corrSums[i]))

        if 'T_end' in ds.index.columns:
            temp = ds.index['T_end'].to_numpy()
        else:
            temp = ds.index['temp'].to_numpy()
        fname = os.path.join("wip", "corrSum_" + os.path.basename(fs_path) + ".csv")
        np.savetxt(fname, np.vstack((temp, corrSums)).T)
        print("Stored data in file {}".format(fname))

    print("temp")
    print(temp)
    print()
    print("corrSums")
    print(corrSums)
    print()
    print("**************")
    print("Average corrSum = {}".format(np.mean(corrSums)))
    print("**************")
    print()

    plt.scatter(temp, corrSums, alpha=0.5)
    # plt.axhline(y=np.mean(corrSums), linestyle='--', color='black')
    plt.grid()
    plt.xlabel('temp')
    plt.ylabel('corrSum')
    plt.show()








# plotCorrSumRandomSpin("flatspin_temp_sweeps/randomInit")
# plotCorrSumRandomSpin("flatspin_temp_sweeps/I40_alpha_0.0187_temps_1-2000_runs_16")
# plotCorrSumRandomSpin("wip/corrSum_I40_alpha_0.0187_temps_1-2000_runs_16.csv")
plotCorrSumRandomSpin("flatspin_temp_sweeps/I40_alpha_0.0187_temps_1-3000_runs_4x2/")
