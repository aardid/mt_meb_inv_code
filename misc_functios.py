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

# ==============================================================================
# PIECEWISE LINEAR INTERPOLATION
# ==============================================================================
# perform piecewise linear interpolation
def piecewise_interpolation(xi, yi, xj):
        """Fit straight line segments between neighbouring data pairs.
        input:
        xi: data x axis
        yi: data y axis
        xj: specify the interpolation locations
        output:
        yj: interpolated data at xj positions
        """
        # for each subinterval
        yj = []
        for xi1, yi1, xi2, yi2 in zip(xi[:-1], yi[:-1], xi[1:], yi[1:]):
                # compute gradient and intercept
                mi,ci = mx_c(xi1,yi1,xi2,yi2)
                # find interpolating points in subinterval
                inds = np.where((xj>=xi1)&(xj<xi2))
                # evaluate piecewise interpolating function at points
                yj += list(mi*xj[inds] + ci)		

        return yj
# linear interpolation between points 
def mx_c(x1,y1,x2,y2):
	"""Returns gradient and y-intercept for straight line segment between the points (X1,Y1) and (X2,Y2)
	"""
	# gradient
	m = (y2-y1)/(x2-x1)
	# y-intercept
	c = y1-m*x1
	return m,c