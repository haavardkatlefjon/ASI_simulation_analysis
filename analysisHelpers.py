
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
import flatspin.data as fsd
import colorsys

colors = [v for k, v in mcolors.TABLEAU_COLORS.items()]



""""""""" CONSTANTS """""""""
m       = 860e3 * 3*80*220e-27  # magnetic moment, one nanomagnet
k_B     = 1.38064852e-23        # m2 kg s-2 K-1, Boltzmann
""""""""""""""""""""""""""""""

def flucDissSusceptibility(temp, C_sum):
    return m**2 / (k_B * temp) * C_sum

def invSusceptibilityStd(temp, corrSums):
    chi = flucDissSusceptibility(temp, corrSums[:,0])
    # Error propagation formula
    std = (1/chi) * (1/corrSums[:,0]) * corrSums[:,1]
    return std

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

def getNeighborList(pos, params, neighborDistance = None):
    # Calculate neighborhood matrix
    neighbors = []
    num_neighbors = 0

    # Construct KDTree for every position
    tree = cKDTree(pos)

    if neighborDistance == None:
        neighborDistance = params['neighbor_distance']

    nd = params['lattice_spacing'] * neighborDistance
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
        return -1 + (2*round((spinConfiguration[j]*angle_j - spinConfiguration[i]*angle_i)*180/np.pi) == 90)

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


def getAvgCorrFunction(sweep_ds, corrConfig, run_index=None, manual_spin_config=None):
    """ Called from tempsweep. Returns 1d avg correlation function. """

    # polar correlation function config
    dr            = corrConfig['dr']
    dtheta        = corrConfig['dtheta']
    N_points_avg  = corrConfig['N_points_avg']
    neighbor_dist = corrConfig['neighbor_dist']

    if N_points_avg > 1:
        raise NotImplementedError("N_points_avg more than 1 has not been implemented yet")

    # read ASE geometry
    if run_index == None:
        pos, angle = fsd.read_geometry(sweep_ds.tablefile('geometry'))
    else:
        print(run_index)
        print("run_index")
        pos, angle = fsd.read_geometry(sweep_ds.tablefile('geometry')[run_index])

    # convert coordinated to units of lattice spacing
    pos /= sweep_ds.params['lattice_spacing']
    sweep_ds.params['lattice_spacing'] = 1

    # Compute distance matrix
    distances       = getDistanceMatrix(pos)
    absDistances    = abs(distances)

    absEuclidianDist = np.sqrt(absDistances[:,:,0]**2 + absDistances[:,:,1]**2)
    absEuclidianDist = absEuclidianDist[absEuclidianDist>0]

    r_max = np.amax(absEuclidianDist)
    print("min euclid dist", np.amin(absEuclidianDist))
    print("max euclid dist", r_max)

    if True:
        dr = dr * np.amin(absEuclidianDist)
        neighbor_dist = 0.25 * r_max # 25 * np.amin(absEuclidianDist)

    if False:
        n_macrospins = len(pos)
        size_x = np.amax(pos[:,0]) - np.amin(pos[:,0])
        size_y = np.amax(pos[:,1]) - np.amin(pos[:,1])
        system_area = size_x*size_y
        macrospin_density = n_macrospins / system_area

        print("Number of magnets ={}".format(n_macrospins))
        print("System size = {}x{}".format(size_x, size_y))
        print("Density = {}".format(macrospin_density))
        print("dr = 0.3 / sqrt(density) = {}".format(dr / np.sqrt(macrospin_density)))
        print("neighbor_dist = 10 / sqrt(density) = {}".format(10 / np.sqrt(macrospin_density)))
        # dr = dr * np.amin(absEuclidianDist)
        dr = dr / np.sqrt(macrospin_density)
        neighbor_dist = 10 / np.sqrt(macrospin_density)

    print("dr = {}".format(dr))
    print("neighbor_dist = {}".format(neighbor_dist))

    if neighbor_dist == np.inf:
        # using maximum separation as proxy for np.inf neighbor distance
        r_max = np.amax(absEuclidianDist)
        neighbor_dist = 0.25 * r_max

    # get list of neighbors for each magnet
    neighborList = getNeighborList(pos, sweep_ds.params, neighborDistance=neighbor_dist)

    # Prepare array to store correlation values
    C = np.zeros((N_points_avg,
                  round(sweep_ds.params['lattice_spacing'] * 1.1 * neighbor_dist / dr)+2 ,   # number of radial bins, 10% extra
                  round(90/dtheta)+2                                                         # number of angular bins
                  ))

    C_sum = np.zeros(N_points_avg)

    if run_index == None:
        allSpinConfiguration = fsd.read_table(sweep_ds.tablefile("spin"))
    else:
        allSpinConfiguration = fsd.read_table(sweep_ds.tablefile("spin")[run_index])

    timeframes = [len(allSpinConfiguration.index)-1]

    for ti, t in enumerate(timeframes):
        try:
            manual_spin_config[0]
            spinConfiguration = manual_spin_config
        except:
            spinConfiguration = allSpinConfiguration.iloc[t].to_numpy()[1:]

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
                                    ], dtype=np.int32)

                C[ti, index[0], index[1]]   += spinCorrelation
                counter[index[0], index[1]] += 1

        C[ti, (counter == 0)] = np.nan
        C[ti, :, :] /= counter

        C_sum[ti] = np.nansum(C[ti, :, :]) # sum over the full array (within neighborhood). sum -> 0 as r -> inf.
        # See Eq. 9.4.5 in https://phys.libretexts.org/Bookshelves/Thermodynamics_and_Statistical_Mechanics/Book%3A_Statistical_Mechanics_(Styer)/09%3A_Strongly_Interacting_Systems_and_Phase_Transitions/9.04%3A_Correlation_Functions_in_the_Ising_Model

        C[ti, :, :] = abs(C[ti, :, :])

    nonempty_bins_count = np.sum(counter > 0)
    counter[counter == 0] = np.nan
    avgPairsInBin = np.nanmean(counter)
    print("avgPairsInBin", avgPairsInBin)
    print("nonempty_bins_count {} ({}% of available bins are not empty)".format(nonempty_bins_count, round(nonempty_bins_count*100/(C.shape[1]*C.shape[2]),2)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # C_sum = np.nansum(np.nanmean(C, axis=0))
        # collapse polar dimension, only keeping r
        C = np.nanmean(C, axis=2)
        # average over random timeframes
        C = np.nanmean(C, axis=0)

    r_k = np.linspace(0, C.shape[0]*dr, C.shape[0])
    nan_index = np.argwhere(np.isnan(C))
    r_k = np.delete(r_k, nan_index)
    C = np.delete(C, nan_index)
    C_sum = np.mean(C_sum)
    print("C_sum before normalization {}".format(C_sum))
    # C_sum /= nonempty_bins_count
    #print("C_sum after normalization {}".format(C_sum))

    return r_k, C, C_sum, avgPairsInBin, spinConfiguration


def getLatticeCorrelationFunction(sim_ds):

    if len(sim_ds.index.index) > 1:
        run_index = -1
    else:
        run_index = None
    # run_index = 20

    resolution = 0.5

    # read ASI geometry
    if run_index == None:
        pos, angle = fsd.read_geometry(sim_ds.tablefile('geometry'))
    else:
        pos, angle = fsd.read_geometry(sim_ds.tablefile('geometry')[run_index])

    # Compute distance matrix
    distances       = getDistanceMatrix(pos)
    absDistances    = abs(distances)

    absEuclidianDist = np.sqrt(absDistances[:,:,0]**2 + absDistances[:,:,1]**2)
    absEuclidianDist = absEuclidianDist[absEuclidianDist>0]

    r_max = np.amax(absEuclidianDist)
    neighbor_dist = 0.25 * r_max
    neighbor_dist = 10

    # get list of neighbors for each magnet
    neighborList = getNeighborList(pos, sim_ds.params, neighborDistance=neighbor_dist)

    # Allocate array for C
    C = np.zeros((int(round(sim_ds.params['lattice_spacing'] * neighbor_dist / resolution)) + 2,
                  int(round(sim_ds.params['lattice_spacing'] * neighbor_dist / resolution)) + 2 ))

    counter = np.zeros(C.shape)

    if run_index == None:
        spinConfig = fsd.read_table(sim_ds.tablefile("spin"))
    else:
        spinConfig = fsd.read_table(sim_ds.tablefile("spin")[run_index])

    spinConfig = spinConfig.iloc[-1].to_numpy()[1:]

    for i in tqdm(range(len(pos))):
        #print()
        #print("Magnet", i)
        # Correlation with self is always +1
        C[0, 0]   += 1
        counter[0, 0] += 1


        #print("Neighbors")
        #print(neighborList[i][neighborList[i]>=0])

        for j in neighborList[i][neighborList[i]>=0]:
            #print("Neighbor", j)
            dist = absDistances[i,j]
            spinCorrelation = getCorrelationValue(spinConfig, i, j, distances[i,j], angle[i], angle[j])

            if abs(angle[i] - np.pi/2) < 1e-3:
                dist_tmp_x = dist[1]
                dist_tmp_y = dist[0]
                dist = np.array([dist_tmp_x, dist_tmp_y])
                #print("Switching, dist =", dist)


            index = np.array([  int(round( dist[0] / resolution )),
                                int(round( dist[1] / resolution ))], dtype=np.int32)

            C[index[0], index[1]]   += spinCorrelation
            counter[index[0], index[1]] += 1

    C[counter == 0] = np.nan
    C /= counter
    C = abs(C)

    # 2D corr function finished. Now, compact into 1D
    delta = 0.1
    r = np.arange(0, neighbor_dist, delta)
    C_avg = np.zeros(r.shape)
    N_pair = np.zeros(r.shape)

    for ri, r_k in enumerate(r):
        # print("Checking entries for r={}".format(r_k))
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if np.isnan(C[i,j]):
                    continue
                rij = np.sqrt((i*resolution)**2 + (j*resolution)**2)
                if rij > r_k - delta/2 and rij < r_k + delta/2:
                    C_avg[ri]  += C[i,j]
                    N_pair[ri] += 1

    N_pair[N_pair==0] = np.nan
    C_avg /= N_pair

    # Delete nan entries to avoid fuckups when doing curve fit
    nan_index = np.argwhere(np.isnan(C_avg))
    r = np.delete(r, nan_index)
    C_avg = np.delete(C_avg, nan_index)

    return r, C_avg

def getEndTemp(temp_string):
    if len(temp_string) > 1000:
        temp_string = temp_string[-100:]
        temp_string = temp_string.split(',')[-1]
        temp_string = temp_string.strip()
        temp_string = temp_string.replace(')','')
        temp_string = temp_string.replace(']','')
        temp = float(temp_string)
    else:
        temp = temp_string[1:-1]
        temp = float(temp.split(', ')[1])
    return temp


def getTemps(sweep_ds):
    if 'T_end' in sweep_ds.index.columns:
        temps = sweep_ds.index['T_end'].to_numpy()
        temps = np.unique(temps) # unique temps only
    else:
        temps = np.array([tools.getEndTemp(sweep_ds.index.iloc[i]['temp']) for i in range(len(sweep_ds.index.index))])
    return temps


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
        popt, pcov = curve_fit(corrLengthPowerLaw, (temps, T_c*np.ones(temps.shape)), corrLengths, p0=[5.0, 0.5], bounds=([0.0, 0.0], [np.inf, 10.0]))
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

def getSubplotTitle(tempString):
    if isFloat(tempString):
        return tempString

    if len(tempString) > 1000:
        title = 'T={}'.format(round(getEndTemp(tempString),1))
    else:
        start = tempString.index('[')+1
        end = tempString.index(']')
        title = tempString[start:end]
        title = str([round(float(T),1) for T in title.split(',')])
    return title

def plotASEs(sweep_ds, filenameBase, spinConfigs, temps=None, saveFile=False, directory=''):
    pos, angle = fsd.read_geometry(sweep_ds.tablefile('geometry')[0])
    rows, cols = getRowsCols(len(temps))

    # plot position index
    plotPosIndex = range(1,len(temps) + 1)
    stepsize = 1

    if 'group_id' in sweep_ds.index.columns:
        stepsize = len(np.unique(sweep_ds.index['group_id'].to_numpy()))

    fig = plt.figure(1, figsize=(30,30))
    for i in range(len(temps)):
        ax = fig.add_subplot(rows,cols,plotPosIndex[i])
        title = getSubplotTitle(temps[i])
        plotSpinSystem(spinConfigs[i], pos, angle, title=title, axObject=ax, labelIndex=False, colorCorrelation=None, colorSpin=True, removeFrame=True)

    if saveFile:
        filename = filenameBase + "-plots-arrows" + ".pdf"
        fig.savefig(filename, format = 'pdf', dpi=300, transparent=False)
        plt.clf()
        print("Stored file", filename)
    else:
        plt.show()


def plotAnalysis(sweep_ds, filenameBase, temps, r, corrFunctions, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, invSusceptibilitiesStd=[None], saveFile=False, directory=''):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    for i in range(len(temps)):
        # Plot C vs r
        ax1.plot(r, corrFunctions[i], 'o', label=r'T={}, $\zeta={}$'.format(round(temps[i],2), round(corrLengths[i],2)), color=colors[i%len(colors)])
        ax1.plot(np.linspace(0,r[-1],100), np.exp(-np.linspace(0,r[-1],100) / corrLengths[i] ), '--', color=colors[i%len(colors)])

        # Plot E_dip
        energy = fsd.read_table(sweep_ds.tablefile("energy")[i])
        ax4.plot(energy['t'], energy['E_dip'], label=r'T={}'.format(round(temps[i],2)))

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

    if len(invSusceptibilitiesStd) > 1:
        ax3.errorbar(temps, 1/susceptibilities, yerr=invSusceptibilitiesStd, label="Flux-Dissip theorem", fmt='o', capsize=5)
        # errorbar(x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, *, data=None, **kwargs)[source]¶
    else:
        ax3.plot(temps, 1/susceptibilities, 'o', label="Flux-Dissip theorem")

    ax3.set_xlabel("T [K]")
    ax3.set_ylabel(r"$\chi^{-1}$")
    ax3.set_ylim(min(1.1*min(1/susceptibilities), 0.9*min(1/susceptibilities)), 1.1*max(1/susceptibilities))
    ax3.set_xlim(0.9*min(temps), 1.1*max(temps))
    ax3.legend()

    ax4.legend()
    ax4.set_xlabel("time")
    ax4.set_ylabel("E_dip")

    fig.tight_layout()

    if saveFile:
        filename = filenameBase + "-plots-analysis" + ".png"
        fig.savefig(filename, format = 'png', dpi=300, transparent=False)
        print("Stored file", filename)
        plt.clf()
    else:
        plt.show()


def plotAnalysisSimplified(filenameBase, temps, corrLengths, corrLengthsVar, susceptibilities, T_c, C_curie, A, nu, invSusceptibilitiesStd, saveFile=False, directory=''):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))


    corrLengths = np.log(corrLengths)
    ylabel = r"log($\zeta$)"
    tempArray   = np.linspace(0, 1.1*temps[-1], 100)
    powerLawFit = np.log(corrLengthPowerLaw((tempArray, T_c*np.ones(len(tempArray))), A, nu))

    # Plot corr lengths
    ax1.plot(temps, corrLengths, 'o', label="from exp curve fit")
    ax1.plot(tempArray, powerLawFit, '-', label=r"power law ($\nu={}$, $A={}$)".format(round(nu,2), round(A,2)))
    ax1.set_xlabel("Temp")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(0.9*min(temps), 1.1*max(temps))
    ax1.set_ylim(-0.1, 1.5*np.amax(corrLengths))
    ax1.legend()

    # Plot susceptibilities
    ax2.plot(np.linspace(temps[0], 1.1*temps[-1], 100), 1/curieWeissSusceptibility(np.linspace(temps[0], 1.1*temps[-1], 100), C_curie, T_c), '-', label=r"Curie-Weiss $T_C={}$K".format(round(T_c, 2)))

    if len(invSusceptibilitiesStd) > 1:
        ax2.errorbar(temps, 1/susceptibilities, yerr=invSusceptibilitiesStd, label="Flux-Dissip theorem", fmt='o', capsize=5)
        # errorbar(x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, *, data=None, **kwargs)[source]¶
    else:
        ax2.plot(temps, 1/susceptibilities, 'o', label="Flux-Dissip theorem")

    ax2.set_xlabel("T [K]")
    ax2.set_ylabel(r"$\chi^{-1}$")
    ax2.set_ylim(min(1.1*min(1/susceptibilities), 0.9*min(1/susceptibilities)), 1.1*max(1/susceptibilities))
    ax2.set_xlim(0.9*min(temps), 1.1*max(temps))
    ax2.axhline(y=0.0, linestyle='--', color='gray')
    ax2.legend()

    ax1.grid()
    ax2.grid()

    fig.tight_layout()

    if saveFile:
        filename = filenameBase + "-plots-analysis" + ".png"
        fig.savefig(filename, format = 'png', dpi=300, transparent=False)
        print("Stored file", filename)
        plt.clf()
    else:
        plt.show()




def processResults(corrConfig, temps, corrFunctions, corrLengths, corrLengthsVar, corrSums, susceptibilities, T_c, C_curie, A, nu, writeToFile=False, filenameBase=None, printResults=True, input_path=None):
    tempSweepResults = pd.DataFrame(
        data={'temps': temps,
              'corrLengths': corrLengths,
              'corrLengthsVar': corrLengthsVar,
              'corrSums': corrSums[:,0],
              'corrSumsStd': corrSums[:,1],
              'susceptibilities': susceptibilities,
              'corrFunctions': str([str(C) for C in corrFunctions]),
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
            writer.writerow(["temps", "corrLengths", "corrLengthsVar", "corrSums", "corrSumsStd", "susceptibilities", "corrFunctions"])
            for i in range(len(temps)):
                writer.writerow([temps[i], corrLengths[i], corrLengthsVar[i], corrSums[i,0], corrSums[i,1], susceptibilities[i], str(corrFunctions[i])])
            writer.writerow([])
        print("Wrote report to file", filename)

        tempSweepResults = tempSweepResults.drop(columns=['corrFunctions'])
        tempSweepResults.to_csv(filenameBase + '-data.csv', index=False)

    return tempSweepResults, parameterResults


def getAnalysisId(out_directory):
    if out_directory == '':
        return None
    maxID = 0
    for fname in os.listdir(out_directory):
        if fname == "":
            continue
        thisID = fname.split('_')[0]
        if isFloat(thisID):
            maxID = max(maxID, int(thisID))
    return maxID + 1


#def existingAnalysis(id, directory="analysis_output"):
def existingAnalysis(reportName, directory="analysis_output"):
    for fname in os.listdir(directory):
        test_id = fname.split("_")[0]
        if test_id == str(id) and fname[-3:] == "csv":
            return os.path.join(directory, fname)
    return

def readData(path, args):
    data = {}
    with open(path) as csv_file:
        readingData = False
        foundStart = False
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row == [] or row == [''] or row == None or row == '':
                continue
            elif row[0] == 'Temp sweep data':
                foundStart = True
                continue
            elif foundStart:
                dataKeys = row
                for key in dataKeys:
                    data[key] = []
                foundStart = False
                readingData = True
                continue
            elif readingData:
                for i, c in enumerate(row):
                    if isFloat(c):
                        data[dataKeys[i]].append(float(c))
                    else:
                        data[dataKeys[i]].append(c)
    data['temps'] = np.array(data['temps'])
    data['corrLengths'] = np.array(data['corrLengths'])
    data['corrLengthsVar'] = np.array(data['corrLengthsVar'])

    corrSumMean = []
    corrSumStd = []
    for v in data['corrSums']:
        v = v[1:-1].split()
        corrSumMean.append(float(v[0]))
        corrSumStd.append(float(v[1]))
    data['corrSumMean'] = np.array(corrSumMean)
    data['corrSumStd'] = np.array(corrSumStd)

    if args.temp != None:
        try:
            temp = args.temp.split(':')
            mask = np.array([True for i in range(len(data['temps']))])
            if temp[0] != '':
                mask *= data['temps'] > float(temp[0])
            if temp[1] != '':
                mask *= data['temps'] < float(temp[1])

            sliceStart = np.where(mask)[0][0]
            sliceEnd = np.where(mask)[0][-1]+1

            for key in data.keys():
                data[key] = data[key][sliceStart:sliceEnd]

        except Exception as e:
            print("Invalid index. Should be Python list slicing format start:end")
            sys.exit(1)

    return data

def getRunName(input_path, temps):
    elements = os.path.basename(input_path).split('_')

    runName = ""
    for i in range(len(elements)):
        if i >= 1:
            if elements[i-1] == 'temp':
                elements[i] = str(round(temps[0])) + "-" + str(round(temps[-1]))
            elif elements[i-1] == 'runs':
                elements[i] = str(len(temps))
        runName += elements[i] + "_"
    return runName[:-1]


def printConfig(corrConfig):
    print("Configuration:")
    for k,v in corrConfig.items():
        print("\t{:<20} {:>6}".format(k, round(v,4)))


def getValidIndices(corrLengths):
    indices = np.array([False for i in range(len(corrLengths))])

    indices[-1] = True
    for i in reversed(range(len(corrLengths)-1)):
        if corrLengths[i] > corrLengths[i+1] * 0.8:
            indices[i] = True
            print("corr[{}] > corr[{}]*0.8".format(i, i+1))
        else:
            break
    return indices
