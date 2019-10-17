"""
- Module MMCMC inversion: Library for invert MT data. 
    - 1D contraint MCMC inverion
    - Constraint by information in wells 
    - Consraint built by priors
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
.. conventions::
	: number of layer 3 (2 layers + half-space)
"""
import numpy as np
#from math import sqrt, pi
from cmath import exp, sqrt 
import os, shutil, time, math, cmath
import corner, emcee
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from Maping_functions import *
from scipy.stats import norm
import multiprocessing 
from misc_functios import find_nearest
import csv
import matplotlib.mlab as mlab

textsize = 15.
min_val = 1.e-7


# ==============================================================================
#  MCMC class
# ==============================================================================

class mcmc_inv(object):
    """
    This class is for invert MT stations. 
	====================    ========================================== =====================
    Attributes              Description                                default
    =====================   ========================================== =====================
	obj                     MT stations to inverted. It's an objec of
    T_obs                   periods recorded
    rho_app_obs             apparent resistivity observed (4 comp.)
    rho_app_obs_er          error of apparent resistivity observed (4 comp.)
    phase_obs               phase (degree) observed (4 comp.)
    phase_obs_er            error of phase (degree) observed (4 comp.)
    work_dir                directory to save the inversion outputs     '.'
                            Station class (see lib_MT_station.py).
    num_lay                 number of layers in the inversion           3
    inv_dat		 			weighted data to invert [a,b,c,d,e,f,g]     [1,0,1,0,0,0,0]
                            a: app. res. TE mode (Zxy)
                            b: phase of TE mode (Zxy) 
                            c: app. res. TE mode (Zyx)
                            d: phase of TE mode (Zyx)
                            e: maximum value in Z
                            f: determinat of Z
                            g: sum of squares elements of Z
    norm                    norm to measure the fit in the inversion    2.
    range_p                 filter range of periods to invert           [0,inf]
	time				    time consumed in inversion 
    data_error              cosider data error for inversion 
                            (std. dev. of app. res and phase)
    prior                   consider priors (boolean). For uniform      False
                            priors to set a range on parameters use 
                            'uniform'. For normal priors built from 
                            clay content data use 'normal'. 
    prior_type              set up based if 'prior' is True, filled
                            with the type of prior inputed in 'prior' 
    prior_input             Values for priors. If priors are 'Uniform'
                            inout should be an array with the limits 
                            for each parameter. 
                            numpy array(5) of arrays(2): [a,b,c,d,e]
                            a: [z1_min, z1_max] range for thickness 
                                of layer 1
                            b: [z2_min, z2_max] range for thickness 
                                of layer 2  
                            c: [r1_min, r1_max] range for resistivity 
                                of layer 1  
                            d: [r2_min, r2_max] range for resistivity 
                                of layer 2  
                            e: [r3_min, r3_max] range for resistivity 
                                of layer 3
    nwalkers                number of walkers                           40                       
    walk_jump               burn-in jumps for walkers                   4000
    ini_mod                 inicial model                               [200,100,150,50,500]
    z1_pars                 distribution parameters for layer 1 
                            thickness (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)
    z2_pars                 distribution parameters for layer 2 
                            thickness (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    r1_pars                 distribution parameters for layer 1 
                            resistivity (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    r2_pars                 distribution parameters for layer 2 
                            resistivity (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    r3_pars                 distribution parameters for layer 3 
                            resistivity (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description
    autocor_tim             autocorrelation time (steps) in mcmc inv. 
                            [par1,...,par5]
    aceprat                 acceptance ratio in mcmc inv.
                            [par1,...,par5]
    prior_meb               consider MeB priors (boolean), for z1 and       False
                            z2 pars
    prior_meb_weigth        weigth of meb prior in posterio                 0.1
    prior_meb_wl_names      wells considered for MeB prior            
    prior_meb_pars          mean and std of normal dist. priors for 
                            pars z1 and z2 based on MeB data mcmc inv.
                            [[z1_mean,z1_std],[z2_mean,z2_std]]
    prior_meb_wl_dist       distant to nearest wells consider for 
                            prior_meb_pars
                            [float1, ..., float2]
    autocor_accpfrac        calc. and plot autocorrelation and aceptance fraction 
    =====================   =================================================================
    Methods                 Description
    =====================   =================================================================

    """
    def __init__(self, sta_obj, dim_inv = None, name= None, work_dir = None, num_lay = None , norm = None, \
        prior = None, prior_input = None, prior_meb = None, prior_meb_weigth = None, nwalkers = None, \
            walk_jump = None, inv_dat = None, ini_mod = None, range_p = None, autocor_accpfrac = None, data_error = None):
	# ==================== 
    # Attributes            
    # ===================== 
        if dim_inv is None:
            self.dim_inv = 3
        else: 
            self.dim_inv = dim_inv 
        # if station comes from edi file (.edi)
        if sta_obj.name[-4:] == '.edi': 
            self.name = sta_obj.name[:-4]
        else: 
            self.name = sta_obj.name
        self.T_obs = sta_obj.T
        self.rho_app_obs = sta_obj.rho_app
        self.rho_app_obs_er = sta_obj.rho_app_er
        self.phase_obs = sta_obj.phase_deg
        self.phase_obs_er = sta_obj.phase_deg_er
        self.max_Z_obs = sta_obj.max_Z
        self.det_Z_obs = sta_obj.det_Z
        self.ssq_Z_obs = sta_obj.ssq_Z
        self.norm = norm
        if range_p is None:
            self.range_p = [np.min(self.T_obs),np.max(self.T_obs)]
        else:
            self.range_p = range_p
        if work_dir is None: 
            self.work_dir = '.'
        if num_lay is None: 
            self.num_lay = 3
        if inv_dat is None: 
            self.inv_dat = [1,0,1,0,0,0,0]
        else:
            self.inv_dat = inv_dat
        if norm is None: 
            self.norm = 2.     
        if prior is None: 
            self.prior = False
        else: 
            if (prior is not 'uniform' and prior is not 'meb'): 
                raise 'invalid prior type' 
            if prior == 'uniform': 
                self.prior = True
                self.prior_type = prior
                self.prior_input = prior_input
                if prior_input is None: 
                    raise 'no input for priors'
                if len(prior_input) != 5:
                    raise 'incorrect input format: 5 ranges'
                for i in range(len(prior_input)): 
                    if len(prior_input[i]) != 2: 
                        raise 'incorrect input format:  = [[z1_mean,z1_std],[z2_mean,z2_std]]min and max for each range'
        if data_error: 
            self.data_error = data_error
        else: 
            self.data_error = None
        if prior_meb is None:
            self.prior_meb = False
        else:  
            self.prior_meb = True
            self.prior_meb_wl_names = sta_obj.prior_meb_wl_names
            self.prior_meb_pars = sta_obj.prior_meb  #
            self.prior_meb_wl_dist = sta_obj.prior_meb_wl_dist
            
            if prior_meb_weigth:
                self.prior_meb_weigth = prior_meb_weigth
            else: 
                self.prior_meb_weigth = 0.1

        if nwalkers is None: # number of walkers 
            self.nwalkers = 100
        else:
            self.nwalkers = nwalkers

        if walk_jump is None: 
            self.walk_jump = 3000
        else: 
            self.walk_jump = walk_jump        
        
        if ini_mod is None: 
            if self.num_lay == 3:
                self.ini_mod = [300,200,500,2.5,200]
            if self.num_lay == 4:
                self.ini_mod = [200,100,200,150,50,500,1000]  
        if ini_mod: 
            self.ini_mod = ini_mod
        self.time = None  
        self.path_results = None
        self.z1_pars = None
        self.z2_pars = None
        self.r1_pars = None
        self.r2_pars = None
        self.r3_pars = None

        self.autocor_accpfrac = autocor_accpfrac
    # ===================== 
    # Methods               
    # =====================
    def inv(self):
        # files structure
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')):
            os.mkdir('.'+os.sep+str('mcmc_inversions'))
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion')):
            os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion'))
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+self.name):
            os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+self.name)
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion')):
            os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion'))
        self.path_results = '.'+os.sep+str('mcmc_inversions')+os.sep+self.name

        # Create chain.dat
        if self.num_lay == 3:
            ndim = 5               # parameter space dimensionality
        if self.num_lay == 4:
            ndim = 7               # parameter space dimensionality
		## Timing inversion
        start_time = time.time()
        # create the emcee object (set threads>1 for multiprocessing)
        data = np.array([self.T_obs,\
                    self.rho_app_obs[1],self.phase_obs[1],\
                    self.rho_app_obs[2],self.phase_obs[2],\
                    self.max_Z_obs,self.det_Z_obs,self.ssq_Z_obs]).T
        cores = multiprocessing.cpu_count()
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob, threads=cores-1, args=[data,])
		# set the initial location of the walkers
        pars = self.ini_mod  # initial guess
        p0 = np.array([pars + 0.5e0*np.random.randn(ndim) for i in range(self.nwalkers)])  # add some noise
        p0 = np.abs(p0)
        #p0 = emcee.utils.sample_ball(p0, [20., 20.,], size=nwalkers)

        #Z_est, rho_ap_est, phi_est = self.MT1D_fwd_3layers(*pars,self.T_obs)

        # p0_log = np.log(np.abs(p0))
		# set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
        pos,prob,state=sampler.run_mcmc(np.abs(p0),self.walk_jump)
		# sampler.reset()

        # check the mean acceptance fraction of the ensemble 

        if self.autocor_accpfrac: 
            # check integrated autocorrelation tim
            f = open("autocor_accpfrac.txt", "w")
            f.write("# acceptance fraction: mean std thick1 thick2 res1 res2 res3 \n# autocorrelation time: mean std thick1 thick2 res1 res2 res3\n")
            acpfrac = sampler.acceptance_fraction
            self.aceprat = acpfrac
            f.write("{:2.2f}\t{:2.2f}\t".format(np.mean(acpfrac),np.std(acpfrac)))
            for j in range(ndim):
                f.write("{:2.2f}\t".format(acpfrac[j]))
            f.write("\n")
            autocor = sampler.get_autocorr_time(low=10, high=None, step=1, c=1, fast=False) # tau
            self.autocor_tim = autocor
            f.write("{:2.2f}\t{:2.2f}\t".format(np.mean(autocor),np.std(autocor)))
            for j in range(ndim):
                f.write("{:2.2f}\t".format(autocor[j]))
            f.close()
            #print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            #print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(low=10, high=None, step=1, c=1, fast=False))))
            shutil.move('autocor_accpfrac.txt', self.path_results+os.sep+'autocor_accpfrac.txt')

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
        shutil.move('chain.dat', self.path_results+os.sep+'chain.dat')

        # # save text file with inversion parameters
        if self.prior_meb:
            a = ["Station name","Number of layers","Inverted data","Norm","Priors","Time(s)","MeB wells for prior", "Dist. (km) to MeB wells"] 
            b = [self.name,self.num_lay,self.inv_dat,self.norm,self.prior,int(self.time),self.prior_meb_wl_names, self.prior_meb_wl_dist] 
        else:
            a = ["Station name","Number of layers","Inverted data","Norm","Priors","Time(s)"] 
            b = [self.name,self.num_lay,self.inv_dat,self.norm,self.prior,int(self.time)]
        
        with open('inv_par.txt', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(a,b))
        f.close()
        shutil.move('inv_par.txt',self.path_results+os.sep+'inv_par.txt')

    def lnprob(self, pars, obs):
        ## function to calculate likelihood probability
        def prob_likelihood(Z_est, rho_ap_est, phi_est):
		    # log likelihood for the model, given the data
            v_vec = np.ones(len(self.T_obs))
            variance = self.data_error
            # set weigths for app rest. and phase in the obj. fn. 
            if variance: 
                w_app_res = 1. 
                if self.inv_dat[2] == 0: # no TM
                    w_phase = .01
                if self.inv_dat[2] == 1: # no TM
                    w_phase = .0005

             #v_vec[21:] = np.inf 
            # filter range of periods to work with
            if self.range_p: 
                ## range_p = [0.01, 1.0] s
                ## to do: replace in v_vec with '1' for positions of periods to invert and 'inf' otherwise.  
                # find values of range_p in vector of periods 
                short_p, short_p_idx = find_nearest(self.T_obs, self.range_p[0])
                long_p, long_p_idx = find_nearest(self.T_obs, self.range_p[1])
                # fill v_vec 
                v_vec[:short_p_idx] = np.inf
                v_vec[long_p_idx:] = np.inf
            ############################################################
            ### TE(xy): fitting sounding curves 
            v = .1
            TE_apres = self.inv_dat[0]*-np.sum(((np.log10(obs[:,1]) \
                        -np.log10(rho_ap_est))/v_vec)**self.norm) /v
            if variance:
                #v = np.median(2*np.log10(self.rho_app_obs_er[1])**2) 
                #TE_apres = self.inv_dat[0]*-np.sum(((np.log10(obs[:,1]) \
                #            -np.log10(rho_ap_est))/v_vec)**self.norm) /v
                TE_apres = self.inv_dat[0]*-np.sum(((np.log10(obs[:,1]) \
                        -np.log10(rho_ap_est))/v_vec)**self.norm / (2*np.log10(self.rho_app_obs_er[1])**2)) #/v
                # apply weigth 
                TE_apres = TE_apres*w_app_res
       
            v = 100          
            TE_phase = self.inv_dat[1]*-np.sum(((obs[:,2] \
                        -phi_est)/v_vec)**self.norm )/v 
            if variance:
                v = np.median(2*self.phase_obs_er[1]**2)          
                TE_phase = self.inv_dat[1]*-np.sum(((obs[:,2] \
                            -phi_est)/v_vec)**self.norm )/v 
                #TE_phase = self.inv_dat[1]*-np.sum(((obs[:,2] \
                #        -phi_est)/v_vec)**self.norm / (2*self.phase_obs_er[1]**2)) #/v 
                # apply weigth 
                TE_phase = TE_phase*w_phase
            
            ### TM(yx): fitting sounding curves 
            #v = self.rho_app_obs_er[2]**2
            v = .1
            TM_apres = self.inv_dat[2]*-np.sum(((np.log10(obs[:,3]) \
                        -np.log10(rho_ap_est))/v_vec)**self.norm )/v
            if variance:

                TM_apres = self.inv_dat[2]*-np.sum(((np.log10(obs[:,3]) \
                        -np.log10(rho_ap_est))/v_vec)**self.norm / (2*np.log10(self.rho_app_obs_er[2])**2)) #/v
                # apply weigth 
                TM_apres = TM_apres*w_app_res

            v = 100
            TM_phase = self.inv_dat[3]*-np.sum(((obs[:,4] \
                        -phi_est)/v_vec)**self.norm )/v 
            if variance:
                v = np.median(2*self.phase_obs_er[2]**2)          
                TE_phase = self.inv_dat[3]*-np.sum(((obs[:,4] \
                            -phi_est)/v_vec)**self.norm )/v 
                #TE_phase = self.inv_dat[1]*-np.sum(((obs[:,2] \
                #        -phi_est)/v_vec)**self.norm / (2*self.phase_obs_er[1]**2)) #/v 
                # apply weigth 
                TE_phase = TE_phase*w_phase
            
            ############################################################
            # fitting maximum value of Z
            if self.dim_inv == 2:
                max_Z = 0.
            else:
                max_Z = self.inv_dat[4]*-np.sum(((np.log10(obs[:,5]) \
                            -np.log10(np.absolute(Z_est)))/v_vec)**self.norm)/v
                
            # fitting determinant of Z
            if self.dim_inv == 2:
                det_Z = 0.
            else:
                # divide det() in magnitud and phase 
                det_Z_amp = self.inv_dat[5]*-np.sum(((np.log10(np.absolute(obs[:,6])) \
                            -np.log10(np.absolute(Z_est)))/v_vec)**self.norm)/v

                det_Z_pha = self.inv_dat[5]*-np.sum(((np.angle(obs[:,6],deg=True) \
                            -np.angle(Z_est, deg=True))/v_vec)**self.norm)/v
    
                det_Z = det_Z_amp

            # fitting ssq of Z
            # fitting determinant of Z
            if self.dim_inv == 2:
                ssq_Z = 0.
            else:
                ssq_Z = self.inv_dat[6]*-np.sum(((np.log10(obs[:,7]) \
                            -np.log10(np.absolute(Z_est)))/v_vec)**self.norm)/v

            return TE_apres + TE_phase +  TM_apres + TM_phase + max_Z + ssq_Z + det_Z 
        
        # pars = np.exp(pars)  # pars was define in log space, here we take it back to linear space
		## Parameter constrain
        if (any(x<0 for x in pars)):
            return -np.Inf
        else:
            if self.prior: # with priors
                if self.prior_type == 'uniform': 
                    for i in range(len(pars)): # Check if parameters are inside the range
                        if (pars[i] < self.prior_input[i][0] or pars[i] > self.prior_input[i][1]): 
                            return -np.Inf  # if pars are outside range, return -inf
                    #if (pars[1]*pars[3]<500. or pars[1]*pars[3]>2000.): # constrain correlation between rest and thick (alpha) 
                    #    return -np.Inf  # if product between rest and thick of second layer is outside range
                    Z_est, rho_ap_est, phi_est = self.MT1D_fwd_3layers(*pars,self.T_obs)
                    # calculate prob without priors
                    prob = prob_likelihood(Z_est, rho_ap_est, phi_est)

                    if self.prior_meb: 
                        #prob = prob
                        #v = 0.15
                        #dist = np.min(self.prior_meb_wl_dist) # distant to nearest well [km]
                        #if dist > 1.:
                        #    dist = np.inf
                        #weight = np.exp(-dist+2.) # km
                        weight_z1 = self.prior_meb_weigth
                        weight_z2 = weight_z1
                        # prior over z1 (thickness layer 1)
                        prob += weight_z1**1. *-((self.prior_meb_pars[0][0]) - pars[0])**self.norm \
                            /self.prior_meb_pars[0][1]**2 
                        # prior over z2 (thickness layer 2)
                        #prob += weight**0. *-((self.prior_meb_pars[1][0] - self.prior_meb_pars[0][0]) - pars[1])**self.norm \
                        prob += weight_z2**1. *-((self.prior_meb_pars[1][0]) - pars[1])**self.norm \
                            /self.prior_meb_pars[1][1]**2

            else: # without priors
                # estimate parameters
                Z_est, rho_ap_est, phi_est = self.MT1D_fwd_3layers(*pars,self.T_obs)
                # calculate prob without priors
                prob = prob_likelihood(Z_est, rho_ap_est, phi_est)
            if prob!=prob: # assign non values to -inf
            	return -np.Inf
            return prob

    def MT1D_fwd_3layers(self,h_1,h_2,rho_1,rho_2,rho_hs,T):  # 2 layers and half-space
        # Base parameters:
        mu=4*math.pi*(10**-7)             # Electrical permitivity [Vs/Am]
        # Vectors to be filled:
        rho_ap = T.copy()
        phi = T.copy()
        Z =  []
        h = np.array([self.num_lay-2,1])
        rho = np.array([self.num_lay-1,1])
        # Variables:
        n_layers= self.num_lay-1 # number of layers
        h = [h_1, h_2]
        rho = [rho_1, rho_2, rho_hs]
        # Recursion
        for k in range(0,len(T)):
            pe=T[k]
            omega=(2*math.pi)/pe
            # Half space parameters
            gamM = cmath.sqrt(1j*omega*mu*(1/rho[2]))
            C = 1/gamM
            # Interaction: inferior layer -> superior layer
            for l in range(0,n_layers):
                gam = cmath.sqrt(1j*omega*mu*(1/rho[(n_layers-1)-l]))
                r = (1-(gam*C))/(1+(gam*C))
                C=(1-r*cmath.exp(-2*gam*h[n_layers-(l+1)]))/(gam*(1+r*cmath.exp(-2*gam*h[n_layers-(l+1)])))
            Z.append(1j*omega*C)                                                 # Impedance
            phi[k]= (math.atan(Z[k].imag/Z[k].real))*360/(2*math.pi)        # Phase in degrees
            rho_ap[k]=(mu/omega)*(abs(Z[k])**2)                             # Apparent resistivity 
        Z = np.asarray(Z)
        return Z, rho_ap, phi

    def lnprior_station(self,*pars):
		# pars = [thickness_layer1[0],thickness_layer2[1],rho_layer1[2],rho_layer2[3],rho_hs[4]]

		# # 1th: high resistivity, around 600 ohm m 
        # lp_thick_layer1 = -(pars[0] - abs(obj.l1_thick_prior))**2/ (2*obj.sigma_thick_l1**2)
        # lp_rho_layer1 = -(pars[2] - obj.resistivity_prior_l1_l3[0])**2/ (2*500**2)
					
		# # 2th layer: product between thickness and resistivity
		# #lp_alpha_l2 = -(pars[1]*pars[3] - well_alpha_2)**2/ (2*1000**2)
        # lp_rho_layer2 = -(pars[3] - obj.clay_cap_resistivity_prior)**2/ (2*50**2)
        # lp_thick_layer2 = -(pars[1] - abs(obj.clay_cap_thick_prior))**2/ (2*obj.sigma_thick_l2**2)
					
	    # # 3th: high resistivity, around 600 ohm m 	
        # lp_rho_hs = -(pars[4] - obj.resistivity_prior_l1_l3[1])**2/ (2*500**2)
					
        return  # lp_thick_layer1 + lp_rho_layer1 + lp_rho_layer2 + lp_thick_layer2 + lp_rho_hs

    def plot_results_mcmc(self, chain_file = None, corner_plt = False, walker_plt = True): 
        chain = np.genfromtxt(self.path_results+os.sep+chain_file)
        if corner_plt: 
        # show corner plot
            weights = chain[:,-1]
            weights -= np.max(weights)
            weights = np.exp(weights)
            labels = ['$z_1$','$z_2$',r'$\rho_1$',r'$\rho_2$',r'$\rho_3$']
            fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30, label_kwargs=dict(fontsize= textsize))
            #ax.tick_params(axis='both', labelsize=textsize)
            plt.savefig(self.path_results+os.sep+'corner_plot.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            #plt.close(fig)
        if walker_plt:
            labels = ['$z_1$','$z_2$',r'$\rho_1$',r'$\rho_2$',r'$\rho_3$']
            npar = int(chain.shape[1] - 3)
            f,axs = plt.subplots(npar,1)
            f.set_size_inches([8,8])
            for i,ax,label in zip(range(npar),axs,labels):
                for j in np.unique(chain[:,0]):
                    ind = np.where(chain[:,0] == j)
                    it = chain[ind,1]
                    par = chain[ind,2+i]
                    ax.plot(it[0],par[0],'k-')
                ax.set_ylabel(label, size = textsize)
                ax.tick_params(axis='both', labelsize=textsize)
                ax.set_xticklabels([])
            ax.set_xticklabels(np.arange(-250,2250,250))
            plt.tight_layout()
            plt.savefig(self.path_results+os.sep+'walkers.png', dpi=300)
        chain = None
        plt.close('all')

    def sample_post(self, idt_sam = None, plot_fit = True, exp_fig = None, plot_model = None): 
        """
        idt_sam: sample just independet samples by autocorrelation in time criteria 
        """
        if plot_fit is None:
            plot_fit = ['appres', 'phase']
		######################################################################
		# reproducability
        np.random.seed(1)
		# load in the posterior
        chain = np.genfromtxt(self.path_results+os.sep+'chain.dat')
        #walk_jump = chain[:,0:2]
        

		# define parameter sets for forward runs
        Nruns = 700
        pars = []
        pars_order = []
        Nsamples = 0
        Nsamples_vec = []

        Nsamples_tot = self.nwalkers * self.walk_jump # total number of samples
        Nburnin = int(self.walk_jump*.5) # Burnin section (% of the walker jumps)

        if idt_sam: 
            # sample only independet samples (by autocorrelation in time estimation)     
            # explain: 
            act_mean = int(np.mean(self.autocor_tim[:-1]))
            mult_act = [i for i in np.arange(Nburnin,self.walk_jump,act_mean)] # multiples of act (mean)from burnin position 
            #print(act_mean)
            #print(mult_act)
            # loop over the chain an filter only independet samples 
            row_count = 0
            for w in range(self.nwalkers):
                for i in range(self.walk_jump):
                    if i in mult_act:
                        pars.append([Nsamples, chain[row_count,2],chain[row_count,3], chain[row_count,4], chain[row_count,5],\
                            chain[row_count,6]])
                        pars_order.append([chain[row_count,0], chain[row_count,1],chain[row_count,2], chain[row_count,3],chain[row_count,4],\
                            chain[row_count,5],chain[row_count,6],chain[row_count,7]])
                        Nsamples += 1
                    row_count+= 1
            #print('Number of independet samples: {:} from {:}'.format(Nsamples, Nsamples_tot))
            Nsamples_vec.append(Nsamples)

            f = open(self.path_results+os.sep+"samples_info.txt", "w")
            f.write('# INDEPENDENT SAMPLES INFORMATION\n# Number of independet samples:\n# Total number of samples:\n# Number of walkers \n# Number of walkers jumps\n')
            f.write('{:}\n'.format(Nsamples))
            f.write('{:}\n'.format(Nsamples_tot))
            f.write('{:}\n'.format(self.nwalkers))
            f.write('{:}\n'.format(self.walk_jump))
            f.close()

        else: 
            # generate Nruns random integers in parameter set range (as a way of sampling this dist)
            # sample considering just burnin
            while Nsamples != Nruns:
                params = chain[:,2:-1]
                id = np.random.randint(0,params.shape[0]-1)
                # condition for sample: prob dif than -inf and jump after 2/3 of total (~converged ones)
                if (chain[id,7] != -np.inf and chain[id,1] > int(self.walk_jump*2/3)) : 
                    #par_new = [params[id,0], params[id,1], params[id,2], params[id,3], params[id,4]]
                    pars.append([Nsamples, params[id,0], params[id,1], params[id,2], params[id,3], \
                        params[id,4]])
                    pars_order.append([chain[id,0], chain[id,1],chain[id,2], chain[id,3],chain[id,4],\
                        chain[id,5],chain[id,6],chain[id,7]])
                    Nsamples += 1

		# Write in .dat paramateres sampled in order of fit (best fit a the top)
        pars_order= np.asarray(pars_order)
        pars_order = pars_order[pars_order[:,7].argsort()[::]]        		
        f = open(self.path_results+os.sep+"chain_sample_order.dat", "w")
        for l in reversed(range(len(pars_order[:,1]))):
            j= l - len(pars_order[:,1])
            f.write("{:4.0f}\t {:4.0f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\n"\
                .format(pars_order[j,0], pars_order[j,1], \
                pars_order[j,2], pars_order[j,3], pars_order[j,4],\
                pars_order[j,5], pars_order[j,6], pars_order[j,7]))
        f.close()

        if plot_model:
            def square_fn(pars, x_axis):
                """
                Calcule y axis for square function over x_axis
                with corners given by pars, starting at y_base. 

                pars = [x1,x2,y0,y1,y2] 
                y_base = 0 (default)
                """
                # vector to fill
                y_axis = np.zeros(len(x_axis))
                # pars = [x1,x2,y1]
                # find indexs in x axis
                idx1 = np.argmin(abs(x_axis-pars[0]))
                idx2 = np.argmin(abs(x_axis-pars[1]))
                # fill y axis and return 
                y_axis[0:idx1] = pars[2]
                y_axis[idx1:idx2+1] = pars[3]
                y_axis[idx2+1:] = pars[4]
                return y_axis
            # depths to plot
            z_model = np.arange(0.,1500.,2.)
            #f,(ax) = plt.subplots(1,1)
            #f.set_size_inches(8,3) 
            f = plt.figure(figsize=(10, 11))
            #f.suptitle('Model: '+self.name, size = textsize)
            gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[2, 1, 1])
            # model 
            ax = f.add_subplot(gs[0, :])
            for par in pars_order:
                #if all(x > 0. for x in par):
                sq_prof_est = square_fn([par[2],par[2]+par[3],par[4],par[5],par[6]], x_axis=z_model)
                ax.semilogy(z_model,sq_prof_est,'c-', lw = 0.7, alpha=0.2, zorder=1)
            ax.plot(z_model,sq_prof_est,'c-', lw = 1.5, alpha=0.5, zorder=0, label = 'model samples')

            # labels
            #ax.set_xlim([np.min(z_model), np.max(z_model)])
            #ax.set_xlim([0,np.mean(pars_order[2]) + np.mean(pars_order[3]) + 100.])
            ax.set_xlim([0, 1.e3])
            ax.set_ylim([1E-1,1e3])
            #ax.set_ylim([1E-1,1.5e3])
            ax.set_xlabel('depth [m]', size = textsize)
            ax.set_ylabel(r'$\rho$ [$\Omega$ m]', size = textsize)
            #ax.legend()
            ### layout figure
            ax.grid(True, which='both', linewidth=0.1)
            #plt.tight_layout()
            #plt.savefig(self.path_results+os.sep+'model_samples.png', dpi=300, facecolor='w', edgecolor='w',
			#		orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            #plt.clf()

            if True:
                # plot histograms for pars
                ax1 = f.add_subplot(gs[1, 0])
                ax2 = f.add_subplot(gs[1, 1])
                ax3 = f.add_subplot(gs[2, 0])
                ax4 = f.add_subplot(gs[2, 1])
                ax5 = f.add_subplot(gs[2, 2])
                #f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
                #f.set_size_inches(12,2)
                #f.set_size_inches(10,2)  
                texts = textsize
                #f.suptitle(self.name, size = textsize)
                # z1
                z1 = pars_order[:,2]
                bins = np.linspace(np.min(z1), np.max(z1), int(np.sqrt(len(z1))))
                h,e = np.histogram(z1, bins, density = True)
                m = 0.5*(e[:-1]+e[1:])
                ax1.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
                ax1.set_xlabel('$z_1$ [m]', size = texts)
                ax1.set_ylabel('freq.', size = texts)
                #ax1.grid(True, which='both', linewidth=0.1)
                # plot normal fit 
                (mu, sigma) = norm.fit(z1)
                y = mlab.normpdf(bins, mu, sigma)
                ax1.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
                ax1.plot(bins, y, 'r-', linewidth=1, label = 'normal fit') #$\mu$:{:3.1f},$\sigma$:{:2.1f}'.format(mu,sigma))
                # plot lines for mean of z1 and z2 (top plot)
                ax.plot([mu,mu],[1e-1,1e3],'r--', linewidth=1.5, alpha = .5, zorder = 0) #, label = r'$\mu$ of  $\rho_3$'
                ax1.plot([mu,mu],[0,max(y)],'r--', label = '$\mu$ of  $z_1$', linewidth=1.0)

                # z2
                z2 = pars_order[:,3]
                bins = np.linspace(np.min(z2), np.max(z2), int(np.sqrt(len(z2))))
                h,e = np.histogram(z2, bins, density = True)
                m = 0.5*(e[:-1]+e[1:])
                ax2.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
                ax2.set_xlabel('$z_2$ [m]', size = texts)
                ax2.set_ylabel('freq.', size = texts)
                #ax1.grid(True, which='both', linewidth=0.1)
                # plot normal fit 
                (mu2, sigma) = norm.fit(z2)
                y = mlab.normpdf(bins, mu2, sigma)
                ax2.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu2,sigma), fontsize = textsize, color='gray')#, y=0.8)
                ax2.plot(bins, y, 'r-', linewidth=1, label = 'normal fit')
                # plot lines for mean of z1 and z2 (top plot)
                ax.plot([mu+mu2,mu+mu2],[1e-1,1e3],'b--', linewidth=1.5, alpha = .5, zorder = 0) #, label = r'$\mu$ of  $\rho_3$'
                ax2.plot([mu2,mu2],[0,max(y)],'b--', label = '$\mu$ of  $z_2$', linewidth=1.0)
                # axis for main plot 
                ax.set_xlim([0, mu+mu2+200])

                # r1
                r1 = pars_order[:,4]
                bins = np.linspace(np.min(r1), np.max(r1), int(np.sqrt(len(r1))))
                h,e = np.histogram(r1, bins, density = True)
                m = 0.5*(e[:-1]+e[1:])
                ax3.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
                ax3.set_xlabel(r'$\rho_1$ [$\Omega$ m]', size = texts)
                ax3.set_ylabel('freq.', size = texts)
                #ax1.grid(True, which='both', linewidth=0.1)
                # plot normal fit 
                (mu, sigma) = norm.fit(r1)
                y = mlab.normpdf(bins, mu, sigma)
                ax3.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
                ax3.plot(bins, y, 'r-', linewidth=1)
                # plot lines for mean 
                ax.plot([0,1e3],[mu,mu],'g--', linewidth=1.5, alpha = .5, zorder = 0) #, label = r'$\mu$ of  $\rho_3$'
                ax3.plot([mu,mu],[0,max(y)],'g--', label = r'$\mu$ of  $\rho_1$', linewidth=1.0)

                # r2
                r2 = pars_order[:,5]
                bins = np.linspace(np.min(r2), np.max(r2), int(np.sqrt(len(r2))))
                h,e = np.histogram(r2, bins, density = True)
                m = 0.5*(e[:-1]+e[1:])
                ax4.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
                ax4.set_xlabel(r'$\rho_2$ [$\Omega$ m]', size = texts)
                ax4.set_ylabel('freq.', size = texts)
                #ax1.grid(True, which='both', linewidth=0.1)
                # plot normal fit 
                (mu, sigma) = norm.fit(r2)
                y = mlab.normpdf(bins, mu, sigma)
                ax4.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
                ax4.plot(bins, y, 'r-', linewidth=1)
                # plot lines for mean 
                ax.plot([0,1e3],[mu,mu],'m--', linewidth=1.5, alpha = .5, zorder = 0) #, label = r'$\mu$ of  $\rho_3$'
                ax4.plot([mu,mu],[0,max(y)],'m--', label = r'$\mu$ of  $\rho_2$', linewidth=1.0)

                # r3
                r3 = pars_order[:,6]
                bins = np.linspace(np.min(r3), np.max(r3), int(np.sqrt(len(r3))))
                h,e = np.histogram(r3, bins, density = True)
                m = 0.5*(e[:-1]+e[1:])
                ax5.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
                ax5.set_xlabel(r'$\rho_3$ [$\Omega$ m]', size = texts)
                ax5.set_ylabel('freq.', size = texts)
                #ax1.grid(True, which='both', linewidth=0.1)
                # plot normal fit 
                (mu, sigma) = norm.fit(r3)
                y = mlab.normpdf(bins, mu, sigma)
                ax5.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
                ax5.plot(bins, y, 'r-', linewidth=1)
                # plot lines for mean 
                ax.plot([0,1e3],[mu,mu],'y--', linewidth=1.5,alpha = .5, zorder = 0) #, label = r'$\mu$ of  $\rho_3$'
                ax5.plot([mu,mu],[0,max(y)],'y--', label = r'$\mu$ of  $\rho_3$', linewidth=1.5)

                ### layout figure
                ax.tick_params(labelsize=textsize)
                ax1.tick_params(labelsize=textsize)
                ax2.tick_params(labelsize=textsize)
                ax3.tick_params(labelsize=textsize)
                ax4.tick_params(labelsize=textsize)
                ax5.tick_params(labelsize=textsize)

                ax.legend(loc = 'upper right', fontsize=textsize, fancybox=True, framealpha=0.5)
                #ax1.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
                #ax2.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
                #ax3.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
                #ax4.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
                #ax5.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
                plt.tight_layout()

                # legend subplot 
                ax6 = f.add_subplot(gs[1, 2])
                #ax6.plot([],[],'c-', label = 'model samples')
                ax6.plot([],[],'r--', label = '$\mu$ of  $z_1$')
                ax6.plot([],[],'b--', label =  '$\mu$ of  $z_2$')
                ax6.plot([],[],'g--', label =  r'$\mu$ of  $\rho_1$')
                ax6.plot([],[],'m--', label =  r'$\mu$ of  $\rho_2$')
                ax6.plot([],[],'y--', label =  r'$\mu$ of  $\rho_3$')
                ax6.plot([],[],'r-', label = 'normal fit    ')
                ax6.axis('off')
                ax6.legend(loc = 'upper right', fontsize=textsize, fancybox=True, framealpha=1.)

                #plt.show()
                plt.savefig(self.path_results+os.sep+'model_samples.png', dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
                #plt.clf()

        if plot_fit: 
            g,(ax, ax1) = plt.subplots(2,1)
            g.set_size_inches(8,8) 
            g.suptitle(self.name, size = textsize)#, y=1.08)

            ### ax: apparent resistivity
            ax.set_xlim([np.min(self.T_obs), np.max(self.T_obs)])
            ax.set_xlim([1E-3,1e3])
            ax.set_ylim([1e0,1e3])
            #ax.set_xlabel('period [s]', size = textsize)
            ax.set_ylabel(r'$\rho_{app}$ [$\Omega$ m]', size = textsize)
            #ax.set_title('Apparent Resistivity (TM and TE)', size = textsize)
            # plot samples
            for par in pars:
                if all(x > 0. for x in par):
                    Z_vec_aux,app_res_vec_aux, phase_vec_aux = \
                        self.MT1D_fwd_3layers(*par[1:6],self.T_obs)
                    ax.loglog(self.T_obs, app_res_vec_aux,'b-', lw = 0.1, alpha=0.2, zorder=0)
            ax.loglog(self.T_obs, app_res_vec_aux,'b-', lw = 0.5, alpha=0.8, zorder=0, label = 'sample')
            #plot observed
            ax.loglog(self.T_obs, self.rho_app_obs[1],'r*', lw = 1.5, alpha=0.7, zorder=0, label = 'observed $Z_{xy}$')
            ax.errorbar(self.T_obs,self.rho_app_obs[1],self.rho_app_obs_er[1], fmt='r*')
            ax.loglog(self.T_obs, self.rho_app_obs[2],'g*', lw = 1.5, alpha=0.7, zorder=0, label = 'observed $Z_{yx}$')
            ax.errorbar(self.T_obs,self.rho_app_obs[2],self.rho_app_obs_er[2], fmt='g*')
            ax.legend(fontsize=textsize, loc = 1, fancybox=True, framealpha=0.8)
            ### ax: phase
            ax1.set_xlim([np.min(self.T_obs), np.max(self.T_obs)])
            ax1.set_xlim([1E-3,1e3])
            ax1.set_ylim([0.e0,1.1e2])
            ax1.set_xlabel('period [s]', size = textsize)
            ax1.set_ylabel('$\phi$ [°]', size = textsize)
            #ax1.set_title('Phase (TM and TE)', size = textsize)
            # plot samples
            for par in pars:
                if all(x > 0. for x in par):
                    Z_vec_aux,app_res_vec_aux, phase_vec_aux = \
                        self.MT1D_fwd_3layers(*par[1:6],self.T_obs)
                    ax1.plot(self.T_obs, phase_vec_aux,'b-', lw = 0.1, alpha=0.2, zorder=0)
            ax1.plot(self.T_obs, phase_vec_aux,'b-', lw = 0.5, alpha=0.8, zorder=0, label = 'sample')
            #plot observed
            ax1.plot(self.T_obs, self.phase_obs[1],'r*', lw = 1.5, alpha=0.7, zorder=0, label = 'observed $Z_{xy}$')
            ax1.errorbar(self.T_obs,self.phase_obs[1],self.phase_obs_er[1], fmt='r*')
            ax1.plot(self.T_obs, self.phase_obs[2],'g*', lw = 1.5, alpha=0.7, zorder=0, label = 'observed $Z_{yx}$')
            ax1.errorbar(self.T_obs,self.phase_obs[2],self.phase_obs_er[2], fmt='g*')
            ax1.legend(fontsize=textsize, loc = 1, fancybox=True, framealpha=0.8)
            ax1.set_xscale('log')
            # plot reference for periods consider in inversion (range_p)
            ax.plot([self.range_p[0],self.range_p[0]],[1.e0,1.e3],'y--',linewidth=0.5, alpha = .5)
            ax.plot([self.range_p[1],self.range_p[1]],[1.e0,1.e3],'y--',linewidth=0.5, alpha = .5)
            ax1.plot([self.range_p[0],self.range_p[0]],[1.e0,1.e3],'y--',linewidth=0.5, alpha = .5)
            ax1.plot([self.range_p[1],self.range_p[1]],[1.e0,1.e3],'y--',linewidth=0.5, alpha = .5)
            ### layout figure
            #
            ax.grid()
            ax1.grid()
            ax.tick_params(labelsize=textsize)
            ax1.tick_params(labelsize=textsize)
            #plt.tight_layout()

            plt.savefig(self.path_results+os.sep+'app_res_fit.png', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            

        if exp_fig == None:
            plt.close('all')
            #plt.clf()
        if exp_fig:  # True: return figure
            return f, g

    def model_pars_est(self, path = None):

        if path is None: 
            path =  self.path_results
        # import chain and estimated model parameters
        rest_mod_samples = np.genfromtxt(path+os.sep+"chain_sample_order.dat")
        num_samples = len(rest_mod_samples[0:,1])
        ## CC boundary parameters
        z1_s = rest_mod_samples[0:num_samples,2]
        z2_s = rest_mod_samples[0:num_samples,3]
        r1_s = rest_mod_samples[0:num_samples,4]
        r2_s = rest_mod_samples[0:num_samples,5]
        r3_s = rest_mod_samples[0:num_samples,6]

        # calculate distribution parameters (for model parameters)
        percentil_range = np.arange(5,100,5) # 5%...95% (19 elements)
        self.z1_pars = [np.mean(z1_s), np.std(z1_s), np.median(z1_s), \
            [np.percentile(z1_s,i) for i in percentil_range]]
        self.z2_pars = [np.mean(z2_s), np.std(z2_s), np.median(z2_s), \
            [np.percentile(z2_s,i) for i in percentil_range]]
        self.r1_pars = [np.mean(r1_s), np.std(r1_s), np.median(r1_s), \
            [np.percentile(r1_s,i) for i in percentil_range]]
        self.r2_pars = [np.mean(r2_s), np.std(r2_s), np.median(r2_s), \
            [np.percentile(r2_s,i) for i in percentil_range]]
        self.r3_pars = [np.mean(r3_s), np.std(r3_s), np.median(r3_s), \
            [np.percentile(r3_s,i) for i in percentil_range]]

        # print a .dat with parameters estimated
        f = open("est_par.dat", "w")
        f.write("# Par\tmean\tstd\tmedian\tperc: 5%, 10%, ..., 95%\n")
        # layer 1 thickness 
        f.write("z1:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.z1_pars[0],self.z1_pars[1],self.z1_pars[2]))
        for per in self.z1_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")
        # layer 2 thickness 
        f.write("z2:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.z2_pars[0],self.z2_pars[1],self.z2_pars[2]))
        for per in self.z2_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")
        # layer 1 resistivity 
        f.write("r1:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.r1_pars[0],self.r1_pars[1],self.r1_pars[2]))
        for per in self.r1_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")
        # layer 2 resistivity 
        f.write("r2:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.r2_pars[0],self.r2_pars[1],self.r2_pars[2]))
        for per in self.r2_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")
        # layer 1 resistivity 
        f.write("r3:\t{:5.2f}\t{:5.2f}\t{:5.2f}\t"\
            .format(self.r3_pars[0],self.r3_pars[1],self.r3_pars[2]))
        for per in self.r3_pars[3]:
            f.write("{:5.2f}\t".format(per))
        f.write("\n")

        f.close()
        shutil.move('est_par.dat',path+os.sep+"est_par.dat")

        if self.prior_meb: 
           ## plot posterior and prior distribution side-by-side
            f = plt.figure(figsize=[7.5,5.5])
            ax=plt.subplot(2, 1, 1)
            ax1=plt.subplot(2, 1, 2)
            
            # z1 
            min_x = self.z1_pars[0] - 3*self.z1_pars[1] 
            min_aux = self.prior_meb_pars[0][0] - 3*self.prior_meb_pars[0][1]
            if min_x > min_aux: 
                min_x = min_aux
            max_x = self.z1_pars[0] + 3*self.z1_pars[1] 
            max_aux = self.prior_meb_pars[0][0] + 3*self.prior_meb_pars[0][1]
            if max_x < max_aux: 
                max_x = max_aux
            x_axis = np.arange(min_x, max_x, 1.)
            ax.plot(x_axis, norm.pdf(x_axis,self.z1_pars[0],self.z1_pars[1] ), label = '~ Posterior')
            ax.plot(x_axis, norm.pdf(x_axis,self.prior_meb_pars[0][0],self.prior_meb_pars[0][1] ), label = '~ Prior')

            ax.set_xlabel('$z_1$ [m]', size = textsize)
            ax.set_ylabel('pdf', size = textsize)
            ax.set_title('Posterior vs Prior: '+ self.name, size = textsize)
            ax.legend(loc=1, prop={'size': 10})	

            # z2
            min_x = self.z2_pars[0] - 3*self.z2_pars[1] 
            min_aux = self.prior_meb_pars[1][0] - 3*self.prior_meb_pars[1][1]
            if min_x > min_aux: 
                min_x = min_aux
            max_x = self.z2_pars[0] + 3*self.z2_pars[1] 
            max_aux = self.prior_meb_pars[1][0] + 3*self.prior_meb_pars[1][1]
            if max_x < max_aux: 
                max_x = max_aux
            x_axis = np.arange(min_x, max_x, 1.)
            ax1.plot(x_axis, norm.pdf(x_axis,self.z2_pars[0],self.z2_pars[1] ), label = '~ Posterior')
            ax1.plot(x_axis, norm.pdf(x_axis,self.prior_meb_pars[1][0],self.prior_meb_pars[1][1] ), label = '~ Prior')

            #ax1.set_title('Posterior vs Prior: z2 bottom .bound'+ self.name, size = textsize)            
            ax1.set_xlabel('$z_2$ [m]', size = textsize)
            ax1.set_ylabel('pdf', size = textsize)
            #ax.set_title('Posterior vs Prior: '+ self.name, size = textsize)
            ax1.legend(loc=1, prop={'size': 10})
            plt.tight_layout()
            plt.savefig(self.path_results+os.sep+'post_vs_prior.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            plt.close()

            KL_div = True
            if KL_div:
                # create file
                f = open("KL_value.txt", "w")
                f.write("# Kullback-Leibler divergence value in station position (for z1 and z1 parameters)\n")
                # calc. value for station 
                # z1
                P = norm.pdf(x_axis,self.z1_pars[0],self.z1_pars[1])
                Q = norm.pdf(x_axis,self.prior_meb_pars[0][0],self.prior_meb_pars[0][1])
                KL_z1 = np.sum([P[i]*np.log(P[i]/Q[i]) for i in range(len(P)) if Q[i] > 0])
                # z2
                P = norm.pdf(x_axis,self.z2_pars[0],self.z2_pars[1])
                Q = norm.pdf(x_axis,self.prior_meb_pars[1][0],self.prior_meb_pars[1][1])
                KL_z2 = np.sum([P[i]*np.log(P[i]/Q[i]) for i in range(len(P)) if Q[i] > 0])
                # save value in txt 
                f.write("{:2.2f}\n".format(KL_z1))
                f.write("{:2.2f}\n".format(KL_z2))
                # close KL_div file       
                f.close()
                # move file 
                shutil.move('KL_value.txt', self.path_results+os.sep+'KL_value.txt')

    # ===================== 
    # Functions               
    # =====================

# ==============================================================================
#  Functions
# ==============================================================================

def calc_prior_meb_quadrant(station_objects, wells_objects, slp = None): 
    """
    Function that calculate MeB prior for each MT station based on MeB mcmc results on wells. 
    First, for each quadrant around the station, the nearest well with MeB data is found. 
    Second, using the MeB mcmcm results, the prior is calculated as a weigthed average of the nearest wells. 
    Third, the results are assigned as attributes to the MT objects. 
    KL_div: calculate Kullback-Leibler divergence. Save in txt. 
    slp: slope of std increase as fn. of distance from well (default .25)
    Attributes generated:
    sta_obj.prior_meb_wl_names      : list of names of nearest wells with MeB 
                                    ['well 1',... , 'ẃell 4']
    sta_obj.prior_meb               : values of normal prior for boundaries of clay cap (top and boundarie)
                                    [[z1_mean,z1_std],[z2_mean,z2_std]]
    .. conventions::
	: z1 and z2 in MT object refer to thickness of two first layers
    : z1 and z2 in results of MeB mcmc inversion refer to depth of the top and bottom boundaries of CC (second layer)
    : cc clay cap
    : distances in meters
    : MeB methylene blue
    """
    if slp is None:
        slp = .25

    for sta_obj in station_objects:
        dist_pre_q1 = []
        dist_pre_q2 = []
        dist_pre_q3 = []
        dist_pre_q4 = []
        #
        name_aux_q1 = [] 
        name_aux_q2 = []
        name_aux_q3 = []
        name_aux_q4 = []
        wl_q1 = []
        wl_q2 = []
        wl_q3 = []
        wl_q4 = []
        for wl in wells_objects:
            if wl.meb:
                # search for nearest well to MT station in quadrant 1 (Q1)
                if (wl.lat_dec > sta_obj.lat_dec and wl.lon_dec > sta_obj.lon_dec): 
                    # distance between station and well
                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                    if not dist_pre_q1:
                        dist_pre_q1 = dist
                    # check if distance is longer than the previous wel 
                    if dist <= dist_pre_q1: 
                        name_aux_q1 = wl.name
                        wl_q1 = wl
                        dist_pre_q1 = dist
                # search for nearest well to MT station in quadrant 2 (Q2)
                if (wl.lat_dec < sta_obj.lat_dec and wl.lon_dec > sta_obj.lon_dec): 
                    # distance between station and well
                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                    if not dist_pre_q2:
                        dist_pre_q2 = dist
                    # check if distance is longer than the previous wel 
                    if dist <= dist_pre_q2: 
                        name_aux_q2 = wl.name
                        wl_q2 = wl
                        dist_pre_q2 = dist
                # search for nearest well to MT station in quadrant 3 (Q3)
                if (wl.lat_dec < sta_obj.lat_dec and wl.lon_dec < sta_obj.lon_dec): 
                    # distance between station and well
                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                    if not dist_pre_q3:
                        dist_pre_q3 = dist
                    # check if distance is longer than the previous wel 
                    if dist <= dist_pre_q3: 
                        name_aux_q3 = wl.name
                        wl_q3 = wl
                        dist_pre_q3 = dist
                # search for nearest well to MT station in quadrant 4 (Q4)
                if (wl.lat_dec > sta_obj.lat_dec and wl.lon_dec < sta_obj.lon_dec): 
                    # distance between station and well
                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                    if not dist_pre_q4:
                        dist_pre_q4 = dist
                    # check if distance is longer than the previous wel 
                    if dist <= dist_pre_q4: 
                        name_aux_q4 = wl.name
                        wl_q4 = wl
                        dist_pre_q4 = dist

        # save names of nearest wells to be used for prior
        sta_obj.prior_meb_wl_names = [name_aux_q1, name_aux_q2, name_aux_q3, name_aux_q4]
        sta_obj.prior_meb_wl_names = list(filter(None, sta_obj.prior_meb_wl_names))
        near_wls = [wl_q1,wl_q2,wl_q3,wl_q4] #list of objects (wells)
        near_wls = list(filter(None, near_wls))
        dist_wels = [dist_pre_q1,dist_pre_q2,dist_pre_q3,dist_pre_q4]
        dist_wels = list(filter(None, dist_wels))
        sta_obj.prior_meb_wl_dist = dist_wels

        # Calculate prior values for boundaries of the cc in station
        # prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
        # z1
        z1_mean_prior = np.zeros(len(near_wls))
        z1_std_prior = np.zeros(len(near_wls))
        z2_mean_prior = np.zeros(len(near_wls))
        z2_std_prior = np.zeros(len(near_wls))
        #
        z1_std_prior_incre = np.zeros(len(near_wls))
        z2_std_prior_incre = np.zeros(len(near_wls))
        count = 0
        # extract meb mcmc results from nearest wells 
        for wl in near_wls:
            # extract meb mcmc results from file 
            meb_mcmc_results = np.genfromtxt(wl.path_mcmc_meb+os.sep+"est_par.dat")
            # values for mean a std for normal distribution representing the prior
            z1_mean_prior[count] = meb_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
            z1_std_prior[count] =  meb_mcmc_results[0,2] # std z1
            z2_mean_prior[count] = meb_mcmc_results[1,1] # mean [1] z2 # median [3] z1
            z2_std_prior[count] =  meb_mcmc_results[1,2] # std z2
            # calc. increment in std. in the position of the station
            # std. dev. increases as get farder from the well. It double its values per 2 km.
            z1_std_prior_incre[count] = z1_std_prior[count] * (sta_obj.prior_meb_wl_dist[count]*slp  + 1.)
            z2_std_prior_incre[count] = z2_std_prior[count] * (sta_obj.prior_meb_wl_dist[count]*slp  + 1.)
            # load pars in well 
            count+=1

        # calculete z1 normal prior parameters
        dist_weigth = [1./d for d in sta_obj.prior_meb_wl_dist]
        z1_mean = np.dot(z1_mean_prior,dist_weigth)/np.sum(dist_weigth)
        # std. dev. increases as get farder from the well. It double its values per km.  
        z1_std = np.dot(z1_std_prior_incre,dist_weigth)/np.sum(dist_weigth)
        # calculete z2 normal prior parameters
        # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
        #z2_mean_prior = z2_mean_prior - z1_mean_prior
        #print(z2_mean_prior)
        z2_mean = np.dot(z2_mean_prior,dist_weigth)/np.sum(dist_weigth)
        #z2_mean = z2_mean 
        if z2_mean < 0.:
            raise ValueError
        z2_std = np.dot(z2_std_prior_incre,dist_weigth)/np.sum(dist_weigth)
        # assign result to attribute
        sta_obj.prior_meb = [[z1_mean,z1_std],[z2_mean - z1_mean,z2_std]]

        # create plot of meb prior and save in station folder. 

        f = plt.figure(figsize=[7.5,5.5])
        ax=plt.subplot(2, 1, 1)
        ax1=plt.subplot(2, 1, 2)
        # plot for z1
        min_x = 0#min([z1_mean - 3*z1_std, min(z1_mean_prior) - 3*max(z1_std_prior)])
        max_x = max([z1_mean - 3*z1_std, max(z1_mean_prior) + 3*max(z1_std_prior)])
        x_axis = np.arange(min_x, max_x, 1.)
        count = 0
        for wl in near_wls:
            # values for mean a std for normal distribution representing the prior
            #z1_mean_prior[count] = meb_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
            #z1_std_prior[count] =  meb_mcmc_results[0,2] # std z1
            ax.plot(x_axis, norm.pdf(x_axis,z1_mean_prior[count],z1_std_prior[count]), label = '$z_1$ well: '+wl.name)
            count+=1
        ax.plot(x_axis, norm.pdf(x_axis,z1_mean,z1_std),label = 'prior $z_1$')
        #plt.plot(x_axis, norm.pdf(x_axis,sta_obj.prior_meb[1][0],sta_obj.prior_meb[1][1]), label = 'prior z2')
        ax.set_xlabel('$z_1$ [m]', size = textsize)
        ax.set_ylabel('pdf', size = textsize)
        ax.set_title('MeB prior for MT station '+ sta_obj.name[:-4], size = textsize)
        ax.legend(loc=1, prop={'size': textsize})	
        
        # plot for z2
        min_x = min([z2_mean - 3*z2_std, min(z2_mean_prior) - 3*max(z2_std_prior)])
        max_x = max([z2_mean - 3*z2_std, max(z2_mean_prior) + 3*max(z2_std_prior)])
        x_axis = np.arange(min_x, max_x, 1.)
        count = 0
        for wl in near_wls:
            # values for mean a std for normal distribution representing the prior
            #z1_mean_prior[count] = meb_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
            #z1_std_prior[count] =  meb_mcmc_results[0,2] # std z1
            ax1.plot(x_axis, norm.pdf(x_axis,z2_mean_prior[count],z2_std_prior[count]), label = '$z_2$ well: '+wl.name)
            count+=1
        ax1.plot(x_axis, norm.pdf(x_axis,z2_mean,z2_std),label = 'prior $z_2$')
        #plt.plot(x_axis, norm.pdf(x_axis,sta_obj.prior_meb[1][0],sta_obj.prior_meb[1][1]), label = 'prior z2')
        ax1.set_xlabel('$z_2$ [m]', size = textsize)
        ax1.set_ylabel('pdf', size = textsize)
        #ax1.set_title('MeB prior for station: '+ sta_obj.name, size = textsize)
        ax1.legend(loc=1, prop={'size': textsize})

        ax.tick_params(axis="x", labelsize=textsize)
        ax.tick_params(axis="y", labelsize=textsize)

        ax1.tick_params(axis="x", labelsize=textsize)
        ax1.tick_params(axis="y", labelsize=textsize)
        
        plt.tight_layout()
        sta_obj.path_results = '.'+os.sep+str('mcmc_inversions')+os.sep+sta_obj.name[:-4]
        plt.savefig(sta_obj.path_results+os.sep+'meb_prior.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        plt.close()

def plot_bound_uncert(station_objects, file_name = None): 
    '''Plot uncertainty in boundary depths
        Average of standard deviations 
    '''
    # calc mean of stds for pars z1 and z2
    # import std
    stds_z1 = [sta.z1_pars[1] for sta in station_objects]
    stds_z2 = [sta.z2_pars[1] for sta in station_objects]
    mean_std_z1 = np.mean(stds_z1)
    mean_std_z2 = np.mean(stds_z2)
    # create figure 
    #f = plt.figure(figsize=[9.0,2.5])
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=[7.5,2.5])
    #ax=plt.subplot(1, 2, 1, constrained_layout=True)
    #ax1=plt.subplot(1, 2, 2, constrained_layout=True)
    # plot normal distributions
    # z1
    bins = np.linspace(-mean_std_z1*4, mean_std_z1*4, 100)
    mu, sigma = 0, mean_std_z1
    y = mlab.normpdf(bins, mu, sigma)
    axs[0].plot(bins, y, 'r-', linewidth=2)
    # plot refence lines
    xticks = [0]
    for i in range(3):
        axs[0].plot([mean_std_z1*(i+1), mean_std_z1*(i+1)],[0,max(y)],'--',linewidth=1, label = r'{}$\sigma$'.format(i+1))
        xticks.append(int(np.floor(mean_std_z1*(i+1))))
    axs[0].legend()
    axs[0].set_xlabel('z1 [m]', size = textsize)
    axs[0].set_ylabel('pdf', size = textsize)
    axs[0].set_xticks(xticks)
    axs[0].grid(linewidth=.5)
    # z2
    bins = np.linspace(-mean_std_z2*4, mean_std_z2*4, 100)
    mu, sigma = 0, mean_std_z2
    y = mlab.normpdf(bins, mu, sigma)
    axs[1].plot(bins, y, 'b-', linewidth=2)
    # plot refence lines
    xticks = [0]
    for i in range(3):
        axs[1].plot([mean_std_z2*(i+1), mean_std_z2*(i+1)],[0,max(y)],'--',linewidth=1, label = r'{}$\sigma$'.format(i+1))
        xticks.append(int(np.floor(mean_std_z2*(i+1))))
    axs[1].legend()
    axs[1].set_xlabel('z2 [m]', size = textsize)
    axs[1].set_ylabel('pdf', size = textsize)
    axs[1].set_xticks(xticks)
    axs[1].grid(linewidth=.5)
    #plt.tight_layout()

    fig.suptitle('Uncertainty in boundary depths', fontsize=16)
    # save figure
    #plt.show()
    plt.savefig('.'+os.sep+file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
    plt.close()
    





            


            