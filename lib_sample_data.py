"""
- Module sample data: Library to manipulate sample data
    - Spline cubic interpolation 

# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
.. conventions::
	: 
"""

import numpy as np
import sys
from scipy.linalg import inv
from numpy.linalg import norm, solve

# ==============================================================================
#  Cubic spline interpolation functions
# ==============================================================================

def cubic_spline_interpolation(xi,yi,xj, rev = None): 
    ''' Cubic spline interpolation. 
        
        Parameters
        ----------
        xi : np.array
            Locations of data points in x axis
        yi : np.array
            Locations of data points in y axis
        xj : np.array
            Locations of interpolate points in x axis
        rev: bool
            Reversed: if xi and xj goes from mayor to minor values
        
        Returns
        -------
        yj : np.array
            Locations of interpolate points in y axis
        
        Notes
        -------
        xi and xj must go in the same order (from minor to mayor or opposite).
        Return array is consistent with the order.  

    '''
    # sort xi,yi,xj if rev is True
    if rev:  
        xi_aux = np.zeros(len(xi))
        yi_aux = np.zeros(len(yi))
        count = 0
        for i,j in zip(reversed(xi),reversed(yi)):
            xi_aux[count] = i
            yi_aux[count] = j
            count+=1
        xj_aux = np.zeros(len(xj))
        count = 0
        for k in reversed(xj):
            xj_aux[count] = k
            count+=1
        xi = xi_aux
        yi = yi_aux
        xj = xj_aux

    # i. assemble the coefficient matrix
    A = spline_coefficient_matrix(xi)
    # ii. assemble the RHS vector
    b = spline_rhs(xi,yi)
    # iii. DISPLAY A and b 
    # display_matrix_equation(A,b)
    # iv. solve matrix equations to obtain spline polynomial coefficients
    ak = solve(A,b)	
    # v. interpolate points
    yj = spline_interpolate(xj, xi, ak)

    if rev:
        yj_aux = np.zeros(len(yj))
        count = 0
        for k in reversed(yj):
            yj_aux[count] = k
            count+=1
        yj = yj_aux
        return yj

    return yj

#  assemble the coefficient matrix					 ----------
def spline_coefficient_matrix(xi):	
	''' Assembles coefficients for spline polynomial problem.
		
		Parameters
		----------
		xi : np.array
			Locations of data points.
		
		Returns
		-------
		A : np.array
			Matrix of spline coefficients.
	'''
	# create zero array with correct dimensions
	N = len(xi)
	A = np.zeros((4*(N-1),4*(N-1)))
	j = 0 	# row (equation) counter for matrix
	
	# for each subinterval, add two equations
	for i in range(N-1):
		# compute interval width term
		ci = 1./(xi[i+1]-xi[i])
		# add first equation (for LHS boundary of subint)
		A[j,4*i:4*(i+1)] = [1, 0, 0, 0]
		j = j+1
		# add second equation (for RHS boundary of subint)
		A[j,4*i:4*(i+1)] = [1, 1./ci, 1./ci**2, 1./ci**3]
		j += 1
	
	# for each subinterval boundary, add two equations
	for i in range(N-2):
		# compute interval width term
		ci = 1./(xi[i+1]-xi[i])
		# add first equation (for 1st derivative at boundary)
		A[j,4*i:4*(i+2)] = [0, 1, 2./ci, 3./ci**2, 0, -1, 0, 0]
		j += 1
		# add first equation (for 2st derivative at boundary)
		A[j,4*i:4*(i+2)] = [0, 0, 2., 6./ci, 0, 0, -2, 0]
		j += 1
	
	# for beginning and end nodes, natural spline condition
	A[j,:4] = [0,0,2,0]
	j += 1
	ci = 1./(xi[-1]-xi[-2])
	A[j,4*(N-2):4*(N-1)] = [0,0,2,6./ci]
	j += 1
			
	return A

# assemble the RHS vector					 ----------
def spline_rhs(xi, yi):
	''' Assembles righthandside values for spline problem.
		
		Parameters
		----------
		xi : np.array
			Locations of data points.
		yi : np.array
			Values of data points.
		
		Returns
		-------
		b : np.array
			RHS vector.
	'''
	# create zero array with correct dimensions
	N = len(xi)
	b = np.zeros(4*(N-1))
	j = 0 	# row (equation) counter for matrix
	
	# for each subinterval, add two equations
	for i in range(N-1):
		# add first equation (for LHS boundary of subint)
		b[j] = yi[i]
		j += 1
		# add second equation (for RHS boundary of subint)
		b[j] = yi[i+1]
		j += 1
	
	# for each subinterval boundary, add two equations
	for i in range(0,N-2):
		# add first equation (for 1st derivative at boundary)
		b[j] = 0.
		j += 1
		# add first equation (for 2st derivative at boundary)
		b[j] = 0.
		j += 1
	
	# for beginning and end nodes, natural spline condition
	b[j] = 0.
	j += 1
	b[j] = 0.
	j += 1
	
	return b
	
#					 ----------
def spline_interpolate(xj, xi, ak):
	''' Performs interpolation.
		
		Parameters
		----------
		xj : np.array
			Locations of interpolation points.
		xi : np.array
			Locations of data points.
		ak : np.array
			Spline coefficients.
		
		Returns
		-------
		yj : np.array
			Interpolation values.
			
		Notes
		-----
		You may assume that the interpolation points XJ are in ascending order.
	'''
	
	# initialise with first polynomial
	i = 1
	aki = ak[4*(i-1):4*i]
	
	# loop through interpolation points
	yj = []
	for xji in xj:	
		# check if need to move onto next polynomial
		while xji > xi[i]:
			i += 1
			aki = ak[4*(i-1):4*i]
		
		# evaluate polynomial
		yj.append(polyval(aki,xji-xi[i-1]))
		
	return np.array(yj)	
	
# DISPLAY A and b 
def display_matrix_equation(A,b):
	''' Prints the matrix equation Ax=b to the screen.
	
		Parameters
		----------
		A : np.array
			Matrix.
		b : np.array
			RHS vector.
			
		Notes
		-----
		This will look horrendous for anything more than two subintervals.	
	'''
	
	# problem dimension
	n = A.shape[0]
	
	# warning
	if n > 8:
		print('this will not format well...')
		
	print(' _'+' '*(9*n-1) +'_  _       _   _        _')
	gap = ' '
	for i in range(n):
		if i == n - 1:
			gap = '_'
		str = '|{}'.format(gap)
		str += ('{:+2.1e} '*n)[:-1].format(*A[i,:])
		str += '{}||{}a_{:d}^({:d})'.format(gap,gap,i%4,i//4+1)+'{}|'.format(gap)
		if i == n//2 and i%2 == 0:
			str += '='
		else:
			str += ' '
		str += '|{}{:+2.1e}{}|'.format(gap,b[i],gap)
		print(str)
	
# 
def get_data():
	# returns a data vector used during this lab
	xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
	yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
	return xi,yi
		
# 
def ak_check():
	# returns a vector of predetermined values
	out = np.array([2.47e+01, -4.075886048665986e+00,0.,8.758860486659859e-01,2.15e+01,
		-1.448227902668027e+00,2.627658145997958e+00,-1.079430243329928e+00,2.16e+01,
		5.687976593381042e-01,-6.106325839918264e-01,5.358287012458253e-01,2.22e+01,
		1.170464160078432e+00,1.157602130119396e+00,-2.936967278262911e-01,2.82e+01,
		1.862652894849505e-01,-1.485668420317224e+00,1.677900564431842e-01,2.63e+01,
		-2.825777017172887e+00,-8.312872001888050e-01,1.079137281294699e+00,4.17e+01,
		2.313177016138269e+01,9.204689515851896e+00,-6.136459677234598e+00])
	return out
	
# 
def polyval(a,xi):
	''' Evaluates polynomial.
		
		Parameters
		----------
		A : np.array
			Vector of polynomial coefficients, increasing order.
		xi : np.array
			Points at which to evaluate polynomial.
		
		Returns
		-------
		yi : np.array
			Evaluated polynomial.
		
	'''
	# initialise output at correct length
	yi = 0.*xi
	
	# loop over polynomial coefficients
	for i,ai in enumerate(a):
		yi = yi + ai*xi**i
		
	return yi

# solve matrix equations to obtain spline polynomial coefficients
def solve_A_b(A,b): 
    '''
    '''
    ak = solve(A,b)
    return ak

# ==============================================================================
#  Polynomial fitting
# ==============================================================================

# ==============================================================================
#  Piecewise linear interpolation
# ==============================================================================

