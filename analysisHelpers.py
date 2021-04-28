
import os
import csv
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
colors = [v for k, v in mcolors.TABLEAU_COLORS.items()]



""""""""" CONSTANTS """""""""
m       = 860e3 * 3*80*220e-27  # magnetic moment, one nanomagnet
k_B     = 1.38064852e-23        # m2 kg s-2 K-1, Boltzmann
""""""""""""""""""""""""""""""

def flucDissSusceptibility(temp, C_sum):
    return m**2 / (k_B * temp) * C_sum


def expfunc(r, r0):
    return np.exp(-r / r0)


def getDistanceMatrix(pos):
    """ Returns a numpy array with dimensions [N, N, 2] where N is the
        total number of magnets and the third dimension correspond to
        row, column distance. distances(i,j) gives the vector going from
        magnet i to magnet j. """

    # coords, angles = system._init_geometry()
    n_magnets = len(pos) # system.N
    distances = np.zeros((n_magnets, n_magnets, 2))

    for i in range(n_magnets):
        for j in range(n_magnets):
            distances[i,j,0] = pos[j][0] - pos[i][0]
            distances[i,j,1] = pos[j][1] - pos[i][1]
    return distances


def isFloat(val):
    try:
        float(val)
        return True
    except:
        return False


def readParams(fs_data):
    params = {}
    for k,v in fs_data['params']:
        if isFloat(v):
            params[k] = float(v)
        else:
            params[k] = v
    return params


def getNeighborList(pos, params, neighborDistance = None):
    # Calculate neighborhood matrix
    neighbors = []
    num_neighbors = 0

    # Construct KDTree for every position
    tree = cKDTree(pos)

    if neighborDistance != None:
        params['neighbor_distance'] = neighborDistance

    nd = params['lattice_spacing'] * params['neighbor_distance']
    nd += 1e-5 # pad to avoid rounding errors

    for i in range(len(pos)):
        p = pos[i]
        n = tree.query_ball_point([p], nd)[0]
        n.remove(i)
        neighbors.append(n)
        num_neighbors = max(num_neighbors, len(n))

    # Neighborhood list, -1 marks end of each list
    neighbor_list = np.full((len(pos), num_neighbors), -1, dtype=np.int32)
    for i, neighs in enumerate(neighbors):
        neighbor_list[i,:len(neighs)] = neighs

    return neighbor_list

def getCorrelationValue(spinConfiguration, i, j, r_ij, angle_i, angle_j):
    """ Compute S_i * S_j. +1 for lowest energy, -1 for highest energy. """
    if i==j:
        """ Correlation with itself is always +1 """
        return 1

    m_i = spinConfiguration[i]*np.array([np.cos(angle_i), np.sin(angle_i)])
    m_j = spinConfiguration[j]*np.array([np.cos(angle_j), np.sin(angle_j)])

    # Eq. 3 in flatspin paper
    r_ij_len = np.linalg.norm(r_ij)
    h_dipole = 3 * r_ij * ( m_j @ r_ij ) / r_ij_len**5 - m_j / r_ij_len**3

    # Dipole field parallel to m_i
    h_dipole_parallel = h_dipole@m_i

    if abs(h_dipole_parallel) < 1e-20:
        """ If degenerate states (i.e. 45 deg pin-wheel). """
        """ Rationale: just need to be consequent for correlation function calcualtions """
        return -1 + 2*((spinConfiguration[j]*angle_j - spinConfiguration[i]*angle_i)*180/np.pi == 90)

    else:
        return -1 + 2*(h_dipole_parallel > 0)


def getTheta(coordsA, coordsB, originRot, verbose=False):
    deltaX = abs(coordsB[0] - coordsA[0])
    deltaY = abs(coordsB[1] - coordsA[1])

    originRot %= np.pi
    if (deltaX == 0):
        if (deltaY == 0):
            return np.nan
        else:
            phi = np.pi / 2
    else:
        phi = np.arctan(deltaY / deltaX)
    theta = abs(originRot - phi)

    if theta > np.pi / 2:
        theta %= np.pi / 2
    return theta


def getGeometry(geometry):
    pos, angle = [], []
    for x,y,alpha in geometry:
        pos.append((x,y))
        angle.append(alpha)
    pos = np.array(pos).astype(float)
    angle = np.array(angle).astype(float)
    return pos, angle


def getAvgCorrFunction(fs_data, corrConfig):
    dr            = corrConfig['dr']
    dtheta        = corrConfig['dtheta']
    N_points_avg  = corrConfig['N_points_avg']
    neighbor_dist = corrConfig['neighbor_dist']

    """ Called from tempsweep. Returns 1d avg correlation function. """

    if N_points_avg > 1:
        raise NotImplementedError("N_points_avg more than 1 has not been implemented yet")

    pos, angle = getGeometry(fs_data['geometry'])

    # Compute distance matrix
    distances       = getDistanceMatrix(pos)
    absDistances    = abs(distances)

    # get dict of params
    runParams = readParams(fs_data)

    # get list of neighbors for each magnet
    neighborList = getNeighborList(pos, runParams, neighborDistance=neighbor_dist)

    # Prepare array to store correlation values
    C = np.zeros((N_points_avg,
                  round(runParams['lattice_spacing'] * (runParams['neighbor_distance'] + 4) / dr) ,  # number of radial bins
                  round(90/dtheta)                                                                   # number of angular bins
                  ))

    timeframes = [len(fs_data['spin'])-1]

    for ti, t in enumerate(timeframes):
        spinConfiguration = np.array(list(fs_data['spin'][t]))[1:] # [1:] to discard timestep in first position
        counter           = np.zeros(C[ti,:,:].shape)

        for i in tqdm(range(len(pos))):
            # Correlation with self is always +1
            C[ti, 0, 0]   += 1
            counter[0, 0] += 1

            for j in neighborList[i][neighborList[i]>=0]:
                dist = absDistances[i,j]
                spinCorrelation = getCorrelationValue(spinConfiguration, i, j, distances[i,j], angle[i], angle[j])
                theta = getTheta(pos[i], pos[j], angle[i])
                r = np.sqrt(dist[0]**2 + dist[1]**2)

                nBinsTheta = 90 / dtheta


                index = np.array([  round( r / dr ),
                                    min( C.shape[2]-1, int(theta / (np.pi/2) * nBinsTheta))
                                    ])
                #print("r", r)
                #print("index", index)
                C[ti, index[0], index[1]] += spinCorrelation
                counter[index[0], index[1]] += 1

        C[ti, (counter == 0)] = np.nan
        C[ti, :, :] /= counter
        C[ti, :, :] = abs(C[ti, :, :])

    counter[counter == 0] = np.nan
    avgPairsInBin = np.nanmean(counter)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        C_sum = np.nansum(np.nanmean(C, axis=0))
        # collapse polar dimension, only keeoping r
        C = np.nanmean(C, axis=2)
        # average over random timeframes
        C = np.nanmean(C, axis=0)

    r_k = np.arange(0, C.shape[0]*dr, dr)#[:C.shape[0]]
    nan_index = np.argwhere(np.isnan(C))
    r_k = np.delete(r_k, nan_index)
    C = np.delete(C, nan_index)

    return r_k, C, C_sum, avgPairsInBin, spinConfiguration



def readIndex(input_path):
    # Read index.csv into a list of dicts
    all_runs = []
    with open(input_path + '/index.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        columns = []
        for i, line in enumerate(csv_reader):
            if i == 0:
                columns = line[1:]
            else:
                entry = {}
                for i, col in enumerate(columns):
                    entry[col] = line[i]
                all_runs.append(entry)
    return all_runs


def getEndTemp(temp_string):
    temp = temp_string[1:-1]
    temp = float(temp.split(', ')[1])
    return temp


def getCriticalTemp(temps, susceptibilities):
    linearFit = np.polyfit(temps, 1/susceptibilities, 1)
    C_curie   = 1 / linearFit[0]
    T_c       = - linearFit[1] / linearFit[0]
    return T_c, C_curie


def curieWeissSusceptibility(temp, C, T_C):
    return C / (temp - T_C)


def corrLengthPowerLaw(X, A, nu):
    # https://en.wikipedia.org/wiki/Critical_exponent
    temp, T_c = X
    T_c = T_c[0]

    zeta = np.zeros(temp.shape)

    for i, T in enumerate(temp):
        tau = (T-T_c) / T_c

        if tau > 0:
            # disordered phase
            zeta[i] = A * ( tau )**(-nu)
        elif tau < 0:
            # ordered phase
            zeta[i] = A * ( -tau )**(-nu)
        else:
            print("Corr length diverges at T=T_c")
            zeta[i] = np.inf
    return zeta


def getCriticalExponent(temps, corrLengths, T_c):
    try:
        popt, pcov = curve_fit(corrLengthPowerLaw, (temps, T_c*np.ones(temps.shape)), corrLengths)
        A, nu = popt
    except RuntimeError:
        print("RuntimeError: corrLengthPowerLaw could not be estimated")
        A, nu = (np.nan, np.nan)
    return A, nu


def plotSpinSystem(spinConfiguration, pos, angle, title="", labelIndex=False, magnet_width_lattice=80/320, magnet_length_lattice=220/320, colorCorrelation=None, colorSpin=True, axObject=None, removeFrame=False, customColors=None):

    if axObject==None:
        fig, axObject = plt.subplots(1, 1)

    dist = getDistanceMatrix(pos)

    for i in range(pos.shape[0]):
        elementRot = angle[i]
        if spinConfiguration[i] == -1:
            elementRot += np.pi

        dx = magnet_length_lattice*np.cos(elementRot)
        dy = magnet_length_lattice*np.sin(elementRot)

        x = pos[i,0] - dx/2
        y = pos[i,1] - dy/2

        if labelIndex:
            axObject.text(pos[i,0], pos[i,1], i)

        if colorCorrelation != None:
            if i == colorCorrelation:
                color = 'blue'
            else:
                color = 'green' if getCorrelationValue(spinState, colorCorrelation, i, dist[colorCorrelation, i], angle[colorCorrelation], angle[i])==1 else 'red'
        elif colorSpin:
            elementRot = elementRot % (2*np.pi)
            if elementRot < 0:
                elementRot += (2*np.pi)
            color = colorsys.hsv_to_rgb(elementRot/(2*np.pi) ,1,1)
        elif customColors is not None:
            if i in customColors.keys():
                color = customColors[i]
            else:
                color = 'gray'
        else:
            color = 'gray'

        axObject.arrow(x ,y, dx, dy, length_includes_head = True,
                width=0.12*magnet_length_lattice, fc=color, ec=color)

    axObject.axis('equal')
    axObject.set_title(title)

    if removeFrame:
        axObject.axis('off')


def getRowsCols(totalNum):
    if totalNum == 1:
        rows, cols = (1,1)
    elif totalNum == 2:
        rows, cols = (1,2)
    elif totalNum <= 4:
        rows, cols = (2,2)
    elif totalNum <= 4:
        rows, cols = (2,2)
    elif totalNum <= 6:
        rows, cols = (3,2)
    elif totalNum <= 9:
        rows, cols = (3,3)
    elif totalNum <= 16:
        rows, cols = (4,4)
    elif totalNum <= 25:
        rows, cols = (5,5)
    return rows, cols


def plotASEs(input_path, all_runs, filenameBase, spinConfigs, temps=None, saveFile=False, directory='./'):
    fs_data = np.load(input_path + '/' + all_runs[-1]['outdir'], allow_pickle=True)
    pos, angle = getGeometry(fs_data['geometry'])

    rows, cols = getRowsCols(len(all_runs))

    # plot position index
    plotPosIndex = range(1,len(all_runs) + 1)

    fig = plt.figure(1, figsize=(30,30))
    for i in range(len(all_runs)):
        ax = fig.add_subplot(rows,cols,plotPosIndex[i])
        plotSpinSystem(spinConfigs[i], pos, angle, title=all_runs[i]['temp'], axObject=ax, labelIndex=False, colorCorrelation=None, colorSpin=True, removeFrame=True)

    if saveFile:
        filename = filenameBase + "-plots-arrows" + ".pdf"
        fig.savefig(filename, format = 'pdf', dpi=300, transparent=False)
        plt.clf()
        print("Stored file", filename)
    else:
        plt.show()


def plotAnalysis(input_path, all_runs, filenameBase, temps, r, corrFunctions, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, E_dips, timesteps, saveFile=False, directory='./'):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    for i in range(len(all_runs)):
        # Plot C vs r
        ax1.plot(r, corrFunctions[i], 'o', label=r'T={}, $\zeta={}$'.format(round(temps[i],2), round(corrLengths[i],2)), color=colors[i%len(colors)])
        ax1.plot(np.linspace(0,r[-1],100), np.exp(-np.linspace(0,r[-1],100) / corrLengths[i] ), '--', color=colors[i%len(colors)])

        # Plot E_dip
        ax4.plot(timesteps, E_dips[i], label=r'T={}'.format(round(temps[i],2)))

    ax1.legend()
    ax1.set_xlabel("r [a]")
    ax1.set_ylabel("C")
    ax1.set_ylim(0,1)
    ax1.axhline(y=1/np.exp(1), linestyle='--', color="black")

    # Plot corr lengths
    ax2.plot(temps, corrLengths, 'o', label="from exp curve fit")
    ax2.plot(np.linspace(0, 1.1*temps[-1], 100), corrLengthPowerLaw((np.linspace(0, 1.1*temps[-1], 100), T_c*np.ones(100)), A, nu), '-', label=r"power law ($\nu={}$, $A={}$)".format(round(nu,2), round(A,2)))
    ax2.set_xlabel("Temp")
    ax2.set_ylabel(r"$\zeta$")
    ax2.set_ylim(-0.1, 1.5*np.amax(corrLengths))
    ax2.legend()

    # Plot susceptibilities
    ax3.plot(np.linspace(temps[0], 1.1*temps[-1], 100), 1/curieWeissSusceptibility(np.linspace(temps[0], 1.1*temps[-1], 100), C_curie, T_c), '-', label=r"Curie-Weiss $T_C={}$K".format(round(T_c, 2)))
    ax3.plot(temps, 1/susceptibilities, 'o', label="Flux-Dissip theorem")
    ax3.set_xlabel("T [K]")
    ax3.set_ylabel(r"$\chi^{-1}$")
    ax3.set_ylim(min(1.1*min(1/susceptibilities), 0.9*min(1/susceptibilities)), 1.1*max(1/susceptibilities))
    ax3.set_xlim(0.9*min(temps), 1.1*max(temps))
    ax3.legend()

    ax4.legend()
    ax4.set_xlabel("time")
    ax4.set_ylabel("E_dip")

    if saveFile:
        filename = filenameBase + "-plots-analysis" + ".png"
        fig.savefig(filename, format = 'png', dpi=300, transparent=False)
        print("Stored file", filename)
        plt.clf()
    else:
        plt.show()



def processResults(corrConfig, temps, corrLengths, corrLengthsVar, corrSums, susceptibilities, T_c, C_curie, A, nu, writeToFile=False, filenameBase=None, printResults=True, input_path=None):
    tempSweepResults = pd.DataFrame(
        data={'temps': temps,
              'corrLengths': corrLengths,
              'corrLengthsVar': corrLengthsVar,
              'corrSums': corrSums,
              'susceptibilities': susceptibilities,
             })
    parameterResults = {"T_c": T_c, "C_curie": C_curie, "A": A, "nu": nu}

    if printResults:
        print(tempSweepResults, '\n')
        for k,v in parameterResults.items():
            print("{:<10} {:>8}".format(k, v))
        print()

    if writeToFile:
        filename = filenameBase + "-analysis-report" + ".csv"
        with open(filename, "w") as f:
            writer = csv.writer(f)

            writer.writerow(['Corr config'])
            for k,v in corrConfig.items():
                writer.writerow([k, v])
            writer.writerow([''])
            writer.writerow(['General info'])
            writer.writerow(["input_path", input_path])
            writer.writerow(["report_created", datetime.now()])
            writer.writerow([''])
            writer.writerow(['Parameter results'])
            for k,v in parameterResults.items():
                writer.writerow([k, v])
            writer.writerow([''])
            writer.writerow(['Temp sweep data'])
            writer.writerow(["temps", "corrLengths", "corrLengthsVar", "corrSums", "susceptibilities"])
            for i in range(len(temps)):
                writer.writerow([temps[i], corrLengths[i], corrLengthsVar[i], corrSums[i], susceptibilities[i]])
            writer.writerow([])
        print("Wrote report to file", filename)
    return tempSweepResults, parameterResults


def getAnalysisId(out_directory):
    maxID = 0
    for fname in os.listdir(out_directory):
        if fname == "":
            continue
        thisID = fname.split('_')[0]
        if isFloat(thisID):
            maxID = max(maxID, int(thisID))
    return maxID + 1


def getRunName(input_path, temps):
    elements = input_path.split('/')[-1].split('_')
    runName = ""
    for i in range(len(elements)):
        if i >= 1:
            if elements[i-1] == 'temps':
                elements[i] = str(round(temps[0])) + "-" + str(round(temps[-1]))
            elif elements[i-1] == 'runs':
                elements[i] = str(len(temps))
        runName += elements[i] + "_"
    return runName[:-1]


def printConfig(corrConfig):
    print("Configuration:")
    for k,v in corrConfig.items():
        print("\t{:<20} {:>6}".format(k, round(v,4)))
