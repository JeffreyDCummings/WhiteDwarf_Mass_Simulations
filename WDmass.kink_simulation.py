""" This program performs monte-carlo simulations for a sample of stars of size STARNUM and
 evolves them with appropriate Fe/H based on galactic relations, assumes uniform star formation
 of 10 Gyr, and then produces the WD mass distribution with a cutoff of Mv=12 for the 3
 different types of IFMRs.  Synthetic errors are then added, and these are then compared to
 the observed SDSS field WD-mass distribution. """
import math
import numpy as np
import matplotlib.pyplot as plt

IMF_CO = -1.55
STARNUM = 30000
OBS_SIZE = 4550
MAX_AGE = 10000
ERRSCALE = 0.02

np.random.seed()
DATAOUT1 = open("WDsynthesiskink.txt", "w")
DATAOUT2 = open("WDsynthesisplateau.txt", "w")
DATAOUT4 = open("WDsynthesislinear.txt", "w")

def faint(mass):
    """ Returns the cooling age at which the WD reaches Mv = 12 for cutting off
     everything fainter."""
    return 5.196285E2 - mass * 3.920262E3 + mass**2 * 4.422427E3 - mass**3 * 1.383987E3

def age_met():
    """ Returns the total age of the star, plus based on the metallicity vs age relation
     based on Noguchi (2018) Figure 1 provides the initial Fe/H. """
    age = np.random.random()*MAX_AGE
    if age < 5300:
        feh = 0.05-age*3.77358E-05
    else:
        feh = 1.631206E-1+age*1.382979E-4-age**2*3.546099E-8
    z_lin = 10**feh*0.0152
    mass0 = np.random.random()
    m_i = ((6.5**IMF_CO - 0.83**IMF_CO) * mass0 + 0.83**IMF_CO)**(1/IMF_CO)
    return age, m_i, feh, z_lin

def z_0001(m_i):
    """ Returns the evolutionary timescale for a star of mass m_i at metallicity Z = 0.0001."""
    return 10**(1.295287E1 - 6.213973E0 * m_i + 4.679919E0 * m_i**2 - 2.183321E0 * m_i**3 +\
     6.126135E-1 * m_i**4 - 1.004211E-1 * m_i**5 + 8.837004E-3 * m_i**6 - 3.219222E-4 * m_i**7)\
     / 1E6

def z_001(m_i):
    """ Returns the evolutionary timescale for a star of mass m_i at metallicity Z = 0.001."""
    if m_i >= 1.65:
        return 10**(1.105521E1 - m_i * 1.671426E0 + m_i**2 * 3.973562E-1 - m_i**3 * 4.908292E-2\
         + m_i**4 * 2.384426E-3) / 1E6
    return 10**(-8.786352E-1+m_i*5.496717E1-m_i**2*1.014672E2+m_i**3*8.844773E1-\
     m_i**4*3.753581E1+6.255108E0*m_i**5) / 1E6

def z_004(m_i):
    """ Returns the evolutionary timescale for a star of mass m_i at metallicity Z = 0.004."""
    if m_i >= 1.75:
        return 10**(1.097602E1 - m_i * 1.440251E0 + m_i**2 * 2.920685E-1 - m_i**3 * 3.110539E-2\
         + m_i**4 * 1.320956E-3) / 1E6
    return 10**(-1.542074E0 + m_i * 5.502736E1 - m_i**2 * 9.596637E1 + m_i**3 * 7.912474E1\
     - m_i**4 * 3.177365E1 + 5.012728E0 * m_i**5) / 1E6

def z_010(m_i):
    """ Returns the evolutionary timescale for a star of mass m_i at metallicity Z = 0.010."""
    if m_i >= 1.85:
        return 10**(1.079280E1 - 1.101505E0 * m_i + 1.507980E-1 * m_i**2 - 8.075335E-3 * m_i**3)\
         / 1E6
    return 10**(1.238925E1 - 1.483789E1 * m_i + 4.622215E1 * m_i**2 - 6.997709E1 * m_i**3 +\
     5.317299E1 * m_i**4 - 1.990292E1 * m_i**5 + 2.933540E0 * m_i**6)/ 1E6

def z_014(m_i):
    """ Returns the evolutionary timescale for a star of mass m_i at metallicity Z = 0.014."""
    if m_i >= 1.9:
        return 10**(1.083717E1 - 1.099328E0 * m_i + 1.482228E-1 * m_i**2 - 7.872740E-3 * m_i**3)\
         / 1E6
    return 10**(-3.543566E0 + 5.846804E1 * m_i - 9.308277E1 * m_i**2 + 7.044336E1 * m_i**3 -\
     2.601079E1 * m_i**4 + 3.776767E0 * m_i**5) / 1E6

def evolutionary_timescale(z_lin, m_i, age):
    """ Based on the timescale grid generated in the previous functions, this interpolates
     between them to output the evolutionary timescale for a star of mass m_i and
     metallicity feh. """
    if z_lin < 0.001:
        lifem3lin = z_001(m_i)
        lifem4lin = z_0001(m_i)
        lifefeh = lifem4lin + (lifem3lin - lifem4lin) * (z_lin - 0.0001) / 0.0009
    elif z_lin < 0.004:
        lifem2lin = z_004(m_i)
        lifem3lin = z_001(m_i)
        lifefeh = lifem3lin + (lifem2lin - lifem3lin) * (z_lin - 0.001) / 0.003
    elif z_lin < 0.01:
        lifem1lin = z_010(m_i)
        lifem2lin = z_004(m_i)
        lifefeh = lifem2lin + (lifem1lin - lifem2lin) * (z_lin - 0.004) / 0.006
    else:
        lifem1lin = z_010(m_i)
        life0lin = z_014(m_i)
        lifefeh = lifem1lin + (life0lin - lifem1lin) * (z_lin - 0.01) / 0.04
    cooling_age = lifefeh - age
    return cooling_age, lifefeh

def linear_ifmr(m_i):
    """ The IFMR equations with a simple linear fit to the low-mass region."""
    if m_i < 3.125:
        return 4.882269E-1 + m_i * 9.200655E-2
    if 3.125 <= m_i < 3.54:
        return 0.210 + m_i * 0.181
    return 4.709281E-1 + m_i * 1.070174E-1

def kink_ifmr(m_i):
    """ The IFMR equations with our final kink at the low-mass region."""
    if m_i < 1.51:
        return 4.421571E-1 + m_i * 1.069327E-1
    if 1.51 <= m_i <= 1.845:
        return 1.279400E-3 + m_i * 3.986222E-1
    if 1.845 < m_i <= 2.21:
        return -3.416459E-1 * m_i + 1.366503E0
    if 2.21 < m_i <= 3.54:
        return 0.210 + m_i * 0.181
    return 4.709281E-1+m_i*1.070174E-1

def plateau_ifmr(m_i):
    """ The IFMR equations with a plateau region between m_i of 1.845 and 2.91. """
    if m_i < 1.51:
        return 4.421571E-1 + m_i * 1.069327E-1
    if 1.51 <= m_i <= 1.845:
        return 1.279400E-3 + m_i * 3.986222E-1
    if 1.845 < m_i <= 2.91:
        return 0.736737359
    if 2.91 < m_i <= 3.54:
        return 0.210 + m_i * 0.181
    return 4.709281E-1+m_i*1.070174E-1

def star_generator():
    """ Generate the synthetic stars and evolve them through the multiple relations and write them
     to file. """
    star_count = 0
    mi_arr = np.array([])
    mf_lin = np.array([])
    mf_kink = np.array([])
    mf_plat = np.array([])
    while star_count <= STARNUM:
        age, m_i, feh, z_lin = age_met()
        # Mass vs Lifetime relations based on PARSEC isochrones.
        cooling_age, lifefeh = evolutionary_timescale(z_lin, m_i, age)
        if cooling_age < 0:
            mf_kink_in = kink_ifmr(m_i)
            faint_cooling_limit = faint(mf_kink_in)
            if cooling_age > faint_cooling_limit:
                masserr = np.random.normal(0, ERRSCALE * (1 + 1.5 *\
                 cooling_age / faint_cooling_limit))
                mf_kink = np.append(mf_kink, mf_kink_in + masserr)
                mf_lin = np.append(mf_lin, linear_ifmr(m_i) + masserr)
                mf_plat = np.append(mf_plat, plateau_ifmr(m_i) + masserr)
                mi_arr = np.append(mi_arr, m_i)
                print(mf_kink[star_count], m_i, lifefeh, age, masserr, feh, file=DATAOUT1)
                print(mf_plat[star_count], m_i, lifefeh, age, masserr, feh, file=DATAOUT2)
                print(mf_lin[star_count], m_i, lifefeh, age, masserr, feh, file=DATAOUT4)
                star_count += 1
    DATAOUT1.close()
    DATAOUT2.close()
    DATAOUT4.close()
    return mf_lin, mf_plat, mf_kink, mi_arr

def plot_ifmr(mf_lin, mf_plat, mf_kink, mi_arr):
    """ Plot the IFMR with flipped axes and added synthetic WD-mass errors. """
    plt.figure(0, figsize=(12, 10))
    plt.subplot(211)
    plt.ylabel("M$_{initial}$", fontsize=14)
    plt.xlim(0.44, 1.25)
    plt.scatter(mf_lin, mi_arr, s=3, alpha=0.09, color='purple')
    plt.scatter(mf_plat, mi_arr, s=3, alpha=0.09, color='red')
    plt.scatter(mf_kink, mi_arr, s=3, alpha=0.09, color='black')

def plot_dist(mf_lin, mf_plat, mf_kink):
    """ Plot the WD mass distribution and output the synthetic and observed bin counts. """
    plt.subplot(212)
    plt.xlabel("M$_{final}$", fontsize=14)
    plt.ylabel("Number", fontsize=14)
    plt.xlim(0.44, 1.25)
    bins = np.linspace(0.30, 1.25, 20)
    bins_cut = np.linspace(0.55, 1.25, 15)
    sdss_dist = np.loadtxt("SDSS.finalmass.txt", usecols=0)
    count_kink, _, _ = plt.hist(mf_kink, bins_cut, alpha=0.3, density=True, color='black')
    count_plat, _, _ = plt.hist(mf_plat, bins_cut, alpha=1, density=True, color='red',\
     histtype='step', linewidth=2)
    count_lin, _, _ = plt.hist(mf_lin, bins_cut, alpha=1, density=True, color='purple',\
     histtype='step', linewidth=2)
    count_sdss_cut, _, _ = plt.hist(sdss_dist, bins_cut, alpha=1, density=True, color='green',\
     histtype='step', linewidth=2)
    count_sdss, bin_edges = np.histogram(sdss_dist, bins, density=True)
    total_count = np.sum(count_sdss)
    return count_kink, count_plat, count_lin, count_sdss_cut, count_sdss, total_count, bins,\
     bins_cut, bin_edges

def stats_plot(kink_dist, plateau_dist, linear_dist):
    """ Plot on the figures the averaged differential statistics for the syntheses relative
     to the observations. """
    plt.text(0.87, 5, "Kink STDEV = "+str(round(np.average(kink_dist), 2))+" +/- "\
     +str(round(np.std(kink_dist), 2)), fontsize=14, color='black')
    plt.text(0.87, 4, "Plateau STDEV = "+str(round(np.average(plateau_dist), 2))+" +/- "\
     +str(round(np.std(plateau_dist), 2)), fontsize=14, color='red')
    plt.text(0.87, 3, "Linear STDEV = "+str(round(np.average(linear_dist), 2))+" +/- "\
     +str(round(np.std(linear_dist), 2)), fontsize=14, color='purple')
    plt.text(0.87, 2.15, r"Relative to SDSS DR10 M$_i$ from 0.55 to 1.2 M$_\odot$",\
     fontsize=12, color='green')
    plt.text(0.87, 1.8, "using same numbers as observed.", fontsize=12, color='green')

def statistics(count_kink, count_plat, count_lin, count_sdss, count_sdss_cut, total_count,\
 bins, bins_cut, bin_edges, mf_kink, mf_plat, mf_lin):
    """ Calculate the relative scales of the classes, and calculate the statistics for each
     set of syntheses as large as the observed set, and output the sets of these stats. """
    for step, bin_step in enumerate(bins):
        if round(bin_step, 2) == 0.55:
            high_count = np.sum(count_sdss[step:])
            break
    res1, res2, res4 = 0, 0, 0
    for step_cut in range(len(bins_cut)-1):
        res1 += (count_kink[step_cut] - count_sdss[step_cut] / (total_count / high_count))
        res2 += (count_plat[step_cut] - count_sdss[step_cut] / (total_count / high_count))
        res4 += (count_lin[step_cut] - count_sdss[step_cut] / (total_count / high_count))
    plt.step(bin_edges[1:6], count_sdss[0:5] * (total_count / high_count), color='green',\
     linewidth=2)
    kink_dist, plateau_dist, linear_dist = [], [], []
    for synth_group in range(int(STARNUM/OBS_SIZE)):
        count1, _ = np.histogram(mf_kink[synth_group * OBS_SIZE:(OBS_SIZE * (synth_group + 1)\
         - 1)], bins_cut, density=True)
        count2, _ = np.histogram(mf_plat[synth_group * OBS_SIZE:(OBS_SIZE * (synth_group + 1)\
         - 1)], bins_cut, density=True)
        count4, _ = np.histogram(mf_lin[synth_group * OBS_SIZE:(OBS_SIZE * (synth_group + 1)\
         - 1)], bins_cut, density=True)
        res1, res2, res4 = 0, 0, 0
        for step_cut in range(len(bins_cut)-1):
            res1 += (count1[step_cut] - count_sdss_cut[step_cut])**2
            res2 += (count2[step_cut] - count_sdss_cut[step_cut])**2
            res4 += (count4[step_cut] - count_sdss_cut[step_cut])**2
        kink_dist.append(math.sqrt(res1/len(bins_cut)))
        plateau_dist.append(math.sqrt(res2/len(bins_cut)))
        linear_dist.append(math.sqrt(res4/len(bins_cut)))
    stats_plot(kink_dist, plateau_dist, linear_dist)

def main():
    """ The main set of functions to produce the monte-carlo simulations and statistics. """
    mf_lin, mf_plat, mf_kink, mi_arr = star_generator()
    plot_ifmr(mf_lin, mf_plat, mf_kink, mi_arr)
    count_kink, count_plat, count_lin, count_sdss_cut, count_sdss, total_count, bins, bins_cut,\
     bin_edges = plot_dist(mf_lin, mf_plat, mf_kink)
    statistics(count_kink, count_plat, count_lin, count_sdss, count_sdss_cut, total_count, bins,\
     bins_cut, bin_edges, mf_kink, mf_plat, mf_lin)
    plt.tight_layout()
    plt.show()

main()
