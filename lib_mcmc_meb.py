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
#from scipy.interpolate import interp1d
import scipy.stats as stats
import multiprocessing 
import csv
from scipy import signal
# import pandas as pd
from misc_functios import *

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
    norm                    norm to measure the fit in the inversion    1.
	time				    time consumed in inversion
    max_depth               maximum depth for profile model             2000.
    delta_rs_depth          sample depth to resample the meb prof.      50.  
    meb_prof_rs             resample methylene-blue (MeB) prof. %
    meb_depth_rs            resample methylene-blue (MeB) prof. depths
    walk_jump               burn-in jumps for walkers                   2000
    ini_mod                 inicial model [z1,z2,%]                     [200,400,20]
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
            self.norm = 1.     
        if walk_jump is None: 
            self.walk_jump = 3000
        else: 
            self.walk_jump = walk_jump   
        if ini_mod is None: 
            if self.num_cc == 1:
                self.ini_mod = [200,150,10]
        if max_depth is None: 
            self.max_depth = 3000.
        if delta_rs_depth is None: 
            self.delta_rs_depth = 1.
        self.meb_prof_rs = None
        self.meb_depth_rs = None   
        self.time = None  
        self.z1_pars = None
        self.z2_pars = None
        self.pc_pars = None
        self.path_results = None

    # ===================== 
    # Methods   
    # =====================            

    def resample_meb_prof_3(self):
        # default values
        ini_depth = 0.
        def_per_c = 2.
        ## create vectors to fill
        depths_aux = self.meb_depth
        meb_aux = self.meb_prof
        ## check if first observation is bigger than 2%. if is it, instert a previus 2% point,
        # separate by the typical sample rate of 50m
        if meb_aux[0] > 2.:
            if depths_aux[0] < 50.:
                depths_aux.insert(0,depths_aux[0]-50.)
                meb_aux.insert(0,def_per_c)
            else:
                depths_aux.insert(0,depths_aux[0]-25.)
                meb_aux.insert(0,def_per_c)
        ## check if last observation is bigger than 2%. if is it, instert a last 2% point 
        if meb_aux[len(meb_aux)-1] > 2.:
            depths_aux.insert(len(depths_aux),depths_aux[-1]+50.)
            meb_aux.insert(len(meb_aux),def_per_c)

        ## insert initial values (depth 0)
        depths_aux.insert(0,ini_depth)
        meb_aux.insert(0,def_per_c)
        ## insert final value (depth self.max_depth)
        depths_aux.insert(len(depths_aux),self.max_depth)
        meb_aux.insert(len(meb_aux),def_per_c)

        # create new z depths axis (resample)
        z_rs = np.arange(ini_depth, self.max_depth + self.delta_rs_depth, self.delta_rs_depth) # new z axis
        per_c_rs = np.zeros(len(z_rs))
        # Resample profile 
        per_c_rs = piecewise_interpolation(depths_aux, meb_aux, z_rs)
        per_c_rs.insert(-1,def_per_c)
        self.meb_depth_rs = z_rs
        self.meb_prof_rs = np.asarray(per_c_rs)

    def square_fn(self, pars, x_axis = None, y_base = None):
        # set base value 
        if y_base == None:
            y_base = 0.
        # vector to fill
        y_axis = np.zeros(len(x_axis))
        # pars = [x1,x2,y1]
        # find indexs in x axis
        idx1 = np.argmin(abs(x_axis-pars[0]))
        idx2 = np.argmin(abs(x_axis-pars[1]))
        # fill y axis and return 
        y_axis[0:idx1] = y_base
        y_axis[idx1:idx2+1] = pars[2]
        y_axis[idx2+1:] = y_base
        return y_axis

    def prob_likelihood(self, est, obs):
        # log likelihood for the model, given the data
        v = 0.15
        # fitting estimates (square function) with observation (meb profile)
        #fit = abs(est - obs)
        #prob = (-np.sum(fit)**self.norm)/v 
        log_fit = abs(np.log10(est) - np.log10(obs))
        prob = (-np.sum(log_fit)**self.norm)/v
        return prob

    def lnprob(self, pars, obs):
		## Parameter constraints
        if (any(x<0 for x in pars)): # positive parameters
            return -np.Inf
        if (pars[0] >= self.meb_depth_rs[self.inds[-2]]): # z2 smaller than maximum depth of meb prof
            return -np.Inf
        if (pars[1] >= self.meb_depth_rs[self.inds[-2]]): # z2 smaller than maximum depth of meb prof
            return -np.Inf
        if pars[0] >= pars[1]: # z1 smaller than z1 
            return -np.Inf
        if pars[2] >= 35.:# max(self.meb_prof)+5.: # percentage range 
            return -np.Inf
        ## estimate square function of clay content given pars [z1,z2,%] 
        sq_prof_est =  self.square_fn(pars, x_axis=self.meb_depth_rs[self.inds], y_base = 2.)
        ## calculate prob and return  
        prob = self.prob_likelihood(sq_prof_est,self.meb_prof_rs[self.inds])
        ## check if prob is nan
        if prob!=prob: # assign non values to -inf
            return -np.Inf
        return prob
    
    def run_mcmc(self):

        nwalkers= 24               # number of walkers
        if self.num_cc == 1:
            ndim = 3               # parameter space dimensionality
	 	## Timing inversion
        start_time = time.time()
        ## Set data for inversion 
        for i in range(len(self.meb_prof)):
            if self.meb_prof[i] < 2.:
                self.meb_prof[i] = 2.

        ##Resample the MeB profile
        self.resample_meb_prof_3()
        ##extract index of observes in the resample vector 
        self.inds = []
        for i in range(len(self.meb_depth_rs)):
            for j in range(len(self.meb_depth)):
                 if self.meb_depth_rs[i] == np.floor(self.meb_depth[j]):
                    if i not in self.inds:
                        self.inds.append(i)
        # create the emcee object (set threads>1 for multiprocessing)
        data = self.meb_prof_rs.T
        cores = multiprocessing.cpu_count()
        # Create sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=cores-1, args=[data])
        # set the initial location of the walkers
        pars = self.ini_mod  # initial guess
        p0 = np.array([pars + 10.*np.random.randn(ndim) for i in range(nwalkers)])  # add some noise
        p0 = np.abs(p0)
	 	# set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
        sq_prof_est =  self.square_fn(pars, x_axis=self.meb_depth_rs, y_base = 2.)
        pos,prob,state=sampler.run_mcmc(np.abs(p0),self.walk_jump)

        f = open("chain.dat", "w")
        nk,nit,ndim=sampler.chain.shape
        for k in range(nk):
        	for i in range(nit):
        		f.write("{:d} {:d} ".format(k, i))
        		for j in range(ndim):
        			f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
        		f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
        f.close()
        # assign results to station object attributes 
        self.time = time.time() - start_time # enlapsed time

        ## move chain.dat to file directory
        # shutil.rmtree('.'+os.sep+str('mcmc_meb'))
        if not os.path.exists('.'+os.sep+str('mcmc_meb')):
            os.mkdir('.'+os.sep+str('mcmc_meb'))
            os.mkdir('.'+os.sep+str('mcmc_meb')+os.sep+str('00_global_inversion'))
        if not os.path.exists('.'+os.sep+str('mcmc_meb')+os.sep+self.name):
            os.mkdir('.'+os.sep+str('mcmc_meb')+os.sep+self.name)
        if not os.path.exists('.'+os.sep+str('mcmc_meb')+os.sep+str('00_global_inversion')):
            os.mkdir('.'+os.sep+str('mcmc_meb')+os.sep+str('00_global_inversion'))
            
        self.path_results = '.'+os.sep+str('mcmc_meb')+os.sep+self.name
        shutil.move('chain.dat', self.path_results+os.sep+'chain.dat')

        # # save text file with inversion parameters
        #a = ["Station name","Number of layers","Inverted data","Norm","Priors","Time(s)"] 
        #b = [self.name,self.num_lay,self.inv_dat,self.norm,self.prior,int(self.time)]
        #with open('inv_par.txt', 'w') as f:
        #    writer = csv.writer(f, delimiter='\t')
        #    writer.writerows(zip(a,b))
        #f.close()
        #shutil.move('inv_par.txt',self.path_results+os.sep+'inv_par.txt') 

    def plot_results_mcmc(self, corner_plt = False, walker_plt = True): 
        chain = np.genfromtxt(self.path_results+os.sep+'chain.dat')
        if corner_plt: 
        # show corner plot
            weights = chain[:,-1]
            weights -= np.max(weights)
            weights = np.exp(weights)
            labels = ['z1','z2','% clay']
            fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30)
            plt.savefig(self.path_results+os.sep+'corner_plot.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            #plt.close(fig)
        if walker_plt:
            labels = ['z1','z2','% clay']
            npar = int(chain.shape[1] - 3)
            f,axs = plt.subplots(npar,1)
            f.set_size_inches([8,8])
            for i,ax,label in zip(range(npar),axs,labels):
            	for j in np.unique(chain[:,0]):
            		ind = np.where(chain[:,0] == j)
            		it = chain[ind,1]
            		par = chain[ind,2+i]
            		ax.plot(it[0],par[0],'k-')
            	ax.set_ylabel(label)
            plt.savefig(self.path_results+os.sep+'walkers.png', dpi=300)
        chain = None
        plt.close('all')

    def sample_post(self, plot_fit = True, exp_fig = None): 
		######################################################################
		# reproducability
        np.random.seed(1)
		# load in the posterior
        chain = np.genfromtxt(self.path_results+os.sep+'chain.dat')
        walk_jump = chain[:,0:2]
        params = chain[:,2:-1]

		# define parameter sets for forward runs
        Nruns = 500
        pars = []
        pars_order = []
        # generate Nruns random integers in parameter set range (as a way of sampling this dist)
        Nsamples = 0
        
        while Nsamples != Nruns:
            id = np.random.randint(0,params.shape[0]-1)
            # condition for sample: prob dif than -inf and jump after 2/3 of total (~converged ones)
            if (chain[id,-1] != -np.inf and chain[id,1] > int(self.walk_jump*2/3)): 
                #par_new = [params[id,0], params[id,1], params[id,2], params[id,3], params[id,4]]
                pars.append([Nsamples, params[id,0], params[id,1], params[id,2]])
                pars_order.append([chain[id,0], chain[id,1],chain[id,2], chain[id,3],chain[id,4],\
                    chain[id,5]])
                Nsamples += 1

		# Write in .dat paramateres sampled in order of fit (best fit a the top)
        pars_order= np.asarray(pars_order)
        pars_order = pars_order[pars_order[:,-1].argsort()[::]]        		
        f = open(self.path_results+os.sep+"chain_sample_order.dat", "w")
        for l in reversed(range(len(pars_order[:,1]))):
            j= l - len(pars_order[:,1])
            f.write("{:4.0f}\t {:4.0f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\n"\
                .format(pars_order[j,0], pars_order[j,1], \
                pars_order[j,2], pars_order[j,3], pars_order[j,4],\
                pars_order[j,5]))
        f.close()

        if plot_fit: 
            f,(ax1) = plt.subplots(1,1)
            f.set_size_inches(6,8)
            f.suptitle(self.name, fontsize=20)
            ax1.set_xscale("linear")
            ax1.set_yscale("linear") 
            ax1.set_xlim([0, 20])
            #ax1.set_ylim([250,270])
            #ax1.set_ylim([self.meb_depth[1],self.meb_depth[-2]])
            ax1.set_ylim([0,self.meb_depth[-2]+200.])
            ax1.set_xlabel('MeB [%]', fontsize=18)
            ax1.set_ylabel('Depth [m]', fontsize=18)
            ax1.grid(True, which='both', linewidth=0.4)
            ax1.invert_yaxis()

            for par in pars:
                if all(x > 0. for x in par):
                    sq_prof_est =  self.square_fn(par[1:], x_axis=self.meb_depth_rs, y_base = 2.)
                    ax1.plot(sq_prof_est, self.meb_depth_rs,'g-', lw = 1.0, alpha=0.5, zorder=0)
            ax1.plot(sq_prof_est, self.meb_depth_rs,'g-', lw = 1.0, alpha=0.5, zorder=0, label = 'samples')
            #plot observed
            ax1.plot(self.meb_prof, self.meb_depth,'b*', lw = 2.5, alpha=1.0, zorder=0, label = 'observed')
            ax1.plot(self.meb_prof_rs, self.meb_depth_rs,'--', lw = 1.5, alpha=0.7, zorder=0, label = 'obs. r-sample')
            ax1.legend(loc='lower right', shadow=False, fontsize='small')
            plt.savefig(self.path_results+os.sep+'fit.png', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        if exp_fig == None:
            plt.close(f)
            plt.clf()
        if exp_fig:  # True: return figure
           return f

    def model_pars_est(self, path = None, plot_dist = False, plot_hist = True):

        if path is None: 
            path =  self.path_results
        # import chain and estimated model parameters
        meb_mod_samples = np.genfromtxt(path+os.sep+"chain_sample_order.dat")
        num_samples = len(meb_mod_samples[0:,1])
        ## CC boundary parameters
        z1_s = meb_mod_samples[0:num_samples,2]
        z2_s = meb_mod_samples[0:num_samples,3]
        p1_s = meb_mod_samples[0:num_samples,4]

        # calculate distribution parameters (for model parameters)
        percentil_range = np.arange(5,100,5) # 5%...95% (19 elements)
        self.z1_pars = [np.mean(z1_s), np.std(z1_s), np.median(z1_s), \
            [np.percentile(z1_s,i) for i in percentil_range]]
        self.z2_pars = [np.mean(z2_s), np.std(z2_s), np.median(z2_s), \
            [np.percentile(z2_s,i) for i in percentil_range]]
        self.p1_pars = [np.mean(p1_s), np.std(p1_s), np.median(p1_s), \
            [np.percentile(p1_s,i) for i in percentil_range]]

        # print a .dat with parameters estimated
        f = open("est_par.dat", "w")
        f.write("# Par\tmean\tstd\tmedian\tperc: 5%, 10%, ..., 95%\n")
        # depth of first boundary: z1 
        f.write("z1:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.z1_pars[0],self.z1_pars[1],self.z1_pars[2]))
        for per in self.z1_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")
        # depth of second boundary: z2  
        f.write("z2:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.z2_pars[0],self.z2_pars[1],self.z2_pars[2]))
        for per in self.z2_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")
        # percetage of clay (second layer) 
        f.write("pC:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.p1_pars[0],self.p1_pars[1],self.p1_pars[2]))
        for per in self.p1_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")

        f.close()
        shutil.move('est_par.dat',path+os.sep+"est_par.dat")

        if plot_dist: 
            f,(ax1,ax2,ax3) = plt.subplots(1,3)
            f.set_size_inches(12,4)
            f.suptitle(self.name, fontsize=12, y=.995)
            f.tight_layout()

            #hist = np.histogram(z1_s)
            #ax1.plot(self.z1_pars[3][:],,'b-', lw = 2.5, alpha=1.0, zorder=0, label = '')
            
            y_axis = np.arange(5,55,5) # 5%...50% (9 elements)
            y_axis = np.append(y_axis,np.arange(45,0,-5)) # 50%...95% (10 elements)
            y_axis = 100/np.sum(y_axis) * y_axis 
            
            #y_axis_r1 = y_axis/(self.z1_pars[3][:] - np.min(self.z1_pars[3][:]))  
            #y_axis_r2 = y_axis/(self.z2_pars[3][:] - np.min(self.z2_pars[3][:])) 
            #y_axis_p1 = y_axis/(self.p1_pars[3][:] - np.min(self.p1_pars[3][:]))  
  
            #y_axis_r1 = 100/(y_axis*y_axis_r1)

            ax1.plot(self.z1_pars[3][:], y_axis,'b*', lw = 1.5, alpha=1.0, zorder=0, label = '')
            ax1.plot(self.z1_pars[3][:], y_axis,'k--', lw = 0.5, alpha=1.0, zorder=0, label = '')
            ax1.set_xlabel('z1 [m]', fontsize=10)
            ax1.set_ylabel('~ Prob', fontsize=10)
            ax1.grid(True, which='both', linewidth=0.1)

            ax2.plot(self.z2_pars[3][:], y_axis,'b*', lw = 1.5, alpha=1.0, zorder=0, label = '')
            ax2.plot(self.z2_pars[3][:], y_axis,'k--', lw = 0.5, alpha=1.0, zorder=0, label = '')
            ax2.set_xlabel('z2 [m]', fontsize=10)
            ax2.set_ylabel('~ Prob', fontsize=10)
            ax2.grid(True, which='both', linewidth=0.1)

            ax3.plot(self.p1_pars[3][:], y_axis,'b*', lw = 1.5, alpha=1.0, zorder=0, label = '')
            ax3.plot(self.p1_pars[3][:], y_axis,'k--', lw = 0.5, alpha=1.0, zorder=0, label = '')
            ax3.set_xlabel('clay content [%]', fontsize=10)
            ax3.set_ylabel('~ Prob', fontsize=10)
            ax3.grid(True, which='both', linewidth=0.1)
            plt.tight_layout()
            plt.savefig(self.path_results+os.sep+'par_dist.png', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            plt.close(f)
            plt.clf()
        
        if plot_hist:

            f,(ax1,ax2,ax3) = plt.subplots(1,3)
            f.set_size_inches(12,4)
            f.suptitle(self.name, fontsize=12, y=.995)
            f.tight_layout() 

            n,id,z1,z2,mb,llk = np.genfromtxt(path+os.sep+'chain_sample_order.dat').T
            # z1
            bins = np.linspace(np.min(z1), np.max(z1), int(np.sqrt(len(n))))
            h,e = np.histogram(z1, bins)
            m = 0.5*(e[:-1]+e[1:])
            ax1.bar(e[:-1], h, e[1]-e[0])
            ax1.set_xlabel('z1 [m]', fontsize=10)
            ax1.set_ylabel('freq.', fontsize=10)
            ax1.grid(True, which='both', linewidth=0.1)
            # z2
            bins = np.linspace(np.min(z2), np.max(z2), int(np.sqrt(len(n))))    
            h,e = np.histogram(z2, bins) 
            m = 0.5*(e[:-1]+e[1:])
            ax2.bar(e[:-1], h, e[1]-e[0])
            ax2.set_xlabel('z2 [m]', fontsize=10)
            ax2.set_ylabel('freq.', fontsize=10)
            ax2.grid(True, which='both', linewidth=0.1)
            # mb
            bins = np.linspace(np.min(mb), np.max(mb), int(np.sqrt(len(n))))
            h,e = np.histogram(mb, bins)
            m = 0.5*(e[:-1]+e[1:])
            ax3.bar(e[:-1], h, e[1]-e[0])
            ax3.set_xlabel('mb [%]', fontsize=10)
            ax3.set_ylabel('freq.', fontsize=10)
            ax3.grid(True, which='both', linewidth=0.1)

            plt.tight_layout()
            plt.savefig(self.path_results+os.sep+'par_hist.png', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            plt.close(f)
            plt.clf()