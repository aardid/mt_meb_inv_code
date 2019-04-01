"""
- Module MCMC meb: Library for model MeB profiles in wells
    - fit meb curves a funcion (square, normal)
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
.. conventions::
	: z1 and z2 are the distances to zero
    : cc clay cap
    : distances in meters
    : MeB methylene blue
"""
import numpy as np
#from math import sqrt, pi
from cmath import exp, sqrt 
import os, shutil, time, math, cmath
import corner, emcee
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import multiprocessing 
import csv
from scipy import signal

textsize = 15.
min_val = 1.e-7

# ==============================================================================
#  MCMC meb class
# ==============================================================================

class mcmc_meb(object):
    """
    This class is for model MeB profiles 
	====================    ========================================== =====================
    Attributes              Description                                default
    =====================   ========================================== =====================
	obj                     well object containing meb data 
    name                    name of the well to model
	meb_prof				methylene-blue (MeB) prof. % (array of %)
	meb_depth				methylene-blue (MeB) prof. depths 
    work_dir                directory to save the inversion outputs     '.'
    type_func               type of function to model meb data          'square'
    num_cc                  number of clay caps to model                1
    norm                    norm to measure the fit in the inversion    2.
	time				    time consumed in inversion
    max_depth               maximum depth for profile model             2000.
    delta_rs_depth          sample depth to resample the meb prof.      50.  
    meb_prof_rs             resample methylene-blue (MeB) prof. %
    meb_depth_rs            resample methylene-blue (MeB) prof. depths
    walk_jump               burn-in jumps for walkers                   2000
    ini_mod                 inicial model [z1,z2,%]                     [200,200,20]
    z1_pars                 distribution parameters z1 in square func. 
                            (representive of top boundary of cc)
                            from mcmc chain results: [a,b,c,d]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)
    z2_pars                 distribution parameters z2 in square func. 
                            (representive of bottom boundary of cc)
                            from mcmc chain results: [a,b,c,d]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)
    pc_pars                 distribution parameters % of clay in square func.  
                            (representive of bottom boundary of cc)
                            from mcmc chain results: [a,b,c,d]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)

    =====================   =================================================================
    Methods                 Description
    =====================   =================================================================
    rs_prof                 resample the MeB profile
    run_mcmc
    ln_prob
    prob_likelihood
    square_fn
    plot_results
    sample_post
    """
    def __init__(self, wl_obj, name= None, work_dir = None, type_func = None , norm = None, \
        num_cc = None, max_depth = None, delta_rs_depth = None, walk_jump = None, ini_mod = None):
	# ==================== 
    # Attributes            
    # ===================== 
        self.name = wl_obj.name
        self.meb_prof =  wl_obj.meb_prof
        self.meb_depth =  wl_obj.meb_depth
        if work_dir is None: 
            self.work_dir = '.'
        if type_func is None: 
            self.type_func = 'square'
        if num_cc is None: 
            self.num_cc = 1
        if norm is None: 
            self.norm = 2.     
        if walk_jump is None: 
            self.walk_jump = 5000      
        if ini_mod is None: 
            if self.num_cc == 1:
                self.ini_mod = [400,600,20]
        if max_depth is None: 
            self.max_depth = 2000.
        if delta_rs_depth is None: 
            self.delta_rs_depth = 20.
        self.meb_prof_rs = None
        self.meb_depth_rs = None   
        self.time = None  
        self.path_results = None
        self.z1_pars = None
        self.z2_pars = None
        self.pc_pars = None

    # ===================== 
    # Methods   
    # =====================            
    
    def resample_meb_prof(self):
        # default values
        ini_depth = 0.
        def_per_c = 2.
        # create new z depths axis (resample)
        z_rs = np.arange(ini_depth, self.max_depth + self.delta_rs_depth, self.delta_rs_depth) # new z axis
        per_c_rs = np.zeros(len(z_rs))
        # Resample profile 
        i = 0
        for z in z_rs:
            idx = np.argmin(abs(self.meb_depth-z))
            if abs(z-self.meb_depth[idx])<20.:
                per_c_rs[i] = self.meb_prof[idx] 
            else: 
                per_c_rs[i] = def_per_c
            i+=1
        # add new profile to object attributes
        self.meb_depth_rs = z_rs
        self.meb_prof_rs = per_c_rs

    def square_fn(self, pars, x):
        # fill this        

    def lnprob(self, pars, obs):
		## Parameter constrain
        if (any(x<0 for x in pars)):
            return -np.Inf
        # estimate parameters by forward function 
        per_c_est =  self.square_fn(*pars,self.meb_depth_rs)
        # calculate prob without priors
        
        ---------------------------------------------------------------------
        # calculate prob without priors
        prob = prob_likelihood(Z_est, rho_ap_est, phi_est)
        if prob!=prob: # assign non values to -inf
            return -np.Inf
        return prob
    
    def run_mcmc(self):
        nwalkers= 20               # number of walkers
        if self.num_cc == 1:
            ndim = 3               # parameter space dimensionality
	 	## Timing inversion
        start_time = time.time()
        ## Resample the MeB profile
        self.resample_meb_prof()
        # create the emcee object (set threads>1 for multiprocessing)
        data = np.array([self.meb_depth_rs, self.meb_prof_rs]).T
        cores = multiprocessing.cpu_count()
        # Create sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=cores-1, args=[data,])




    #     -----------------------------------------------------------------------



    #     cores = multiprocessing.cpu_count()
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=cores-1, args=[data,])
	# 	# set the initial location of the walkers
    #     pars = self.ini_mod  # initial guess
    #     p0 = np.array([pars + 0.5e2*np.random.randn(ndim) for i in range(nwalkers)])  # add some noise
    #     p0 = np.abs(p0)
    #     #p0 = emcee.utils.sample_ball(p0, [20., 20.,], size=nwalkers)

    #     Z_est, rho_ap_est, phi_est = self.MT1D_fwd_3layers(*pars,self.T_obs)

    #     # p0_log = np.log(np.abs(p0))
	# 	# set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
    #     pos,prob,state=sampler.run_mcmc(np.abs(p0),self.walk_jump)
	# 	# sampler.reset()

    #     f = open("chain.dat", "w")
    #     nk,nit,ndim=sampler.chain.shape
    #     for k in range(nk):
    #     	for i in range(nit):
    #     		f.write("{:d} {:d} ".format(k, i))
    #     		for j in range(ndim):
    #     			f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
    #     		f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
    #     f.close()
    #     # assign results to station object attributes 
    #     self.time = time.time() - start_time # enlapsed time
    #     ## move chain.dat to file directory
    #     # shutil.rmtree('.'+os.sep+str('mcmc_inversions'))
    #     if not os.path.exists('.'+os.sep+str('mcmc_inversions')):
    #         os.mkdir('.'+os.sep+str('mcmc_inversions'))
    #         os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion'))
    #     elif not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+self.name):
    #         os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+self.name)
    #     if not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion')):
    #         os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion'))
            
    #     self.path_results = '.'+os.sep+str('mcmc_inversions')+os.sep+self.name
    #     shutil.move('chain.dat', self.path_results+os.sep+'chain.dat')

    #     # # save text file with inversion parameters
    #     a = ["Station name","Number of layers","Inverted data","Norm","Priors","Time(s)"] 
    #     b = [self.name,self.num_lay,self.inv_dat,self.norm,self.prior,int(self.time)]
    #     with open('inv_par.txt', 'w') as f:
    #         writer = csv.writer(f, delimiter='\t')
    #         writer.writerows(zip(a,b))
    #     f.close()
    #     shutil.move('inv_par.txt',self.path_results+os.sep+'inv_par.txt') 