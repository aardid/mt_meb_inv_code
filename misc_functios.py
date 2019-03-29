"""
- Module Extras: Miscelanous functions to be used un main.py
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
"""
# ==============================================================================
#  import
# ==============================================================================
import numpy as np
import os

# ==============================================================================
#  load files
# ==============================================================================

def load_sta_est_par(station_objects):
    # import file output from mcmc of estimated parameters (est_par.dat) for
    # for each station, and assign those values to station object attributes.
    for sta_obj in station_objects: 
        # load pars
        est_par = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'est_par.dat')
        est_par = est_par[:,1:]
        idx_first_per = 3
        len_line = len(est_par[0][:])  
        sta_obj.z1_pars = [float(est_par[0][0]), float(est_par[0][1]),float(est_par[0][2]),\
                            [float(est_par[0][i]) for i in np.arange(idx_first_per,len_line)]]
        sta_obj.z2_pars = [float(est_par[1][0]), float(est_par[1][1]),float(est_par[1][2]),\
                            [float(est_par[1][i]) for i in np.arange(idx_first_per,len_line)]]
        sta_obj.r1_pars = [float(est_par[2][0]), float(est_par[2][1]),float(est_par[2][2]),\
                            [float(est_par[2][i]) for i in np.arange(idx_first_per,len_line)]]
        sta_obj.r2_pars = [float(est_par[3][0]), float(est_par[3][1]),float(est_par[3][2]),\
                            [float(est_par[3][i]) for i in np.arange(idx_first_per,len_line)]]
        sta_obj.r3_pars = [float(est_par[4][0]), float(est_par[4][1]),float(est_par[4][2]),\
                            [float(est_par[4][i]) for i in np.arange(idx_first_per,len_line)]]