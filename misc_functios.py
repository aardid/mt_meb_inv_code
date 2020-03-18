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

def load_sta_est_par(station_objects, autocor_accpfrac = None):
        '''
        import file output from mcmc of estimated parameters (est_par.dat) for
        for each station, and assign those values to station object attributes.
        autocor_accpfrac: is True, assign autocor_accpfrac values
        '''
        for sta_obj in station_objects: 
                # load pars
                # if station comes from edi file (.edi)
                if sta_obj.name[-4:] == '.edi': 
                        est_par = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'est_par.dat')
                else: 
                        est_par = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name+os.sep+'est_par.dat')
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

                if autocor_accpfrac: 
                        if sta_obj.name[-4:] == '.edi': 
                                act_af = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'autocor_accpfrac.txt')
                        else: 
                                act_af = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name+os.sep+'autocor_accpfrac.txt')
                        sta_obj.af_mcmcinv = act_af[0,:]
                        sta_obj.act_mcmcinv = act_af[1,:]

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

def find_nearest(array, value):
    """
    Find nearest to value in an array
    
    Input:
    - array: numpy array to look in
    - value: value to search
    
    Output:
    - array[idx]: closest element in array to value
    - idx: index of array[idx] in array

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    try:    
        return array[idx], idx
    except:
        return array[0][idx], idx            