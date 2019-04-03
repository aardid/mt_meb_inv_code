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
import multiprocessing 
import csv

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
    phase_obs               phase (degree) observed (4 comp.)
    work_dir                directory to save the inversion outputs     '.'
                            Station class (see lib_MT_station.py).
    num_lay                 number of layers in the inversion           3
    inv_dat		 			weighted data to invert [a,b,c,d,e]         [1,1,0,0,0]
                            a: app. res. and phase TE mode 
                            b: app. res. and phase TM mode
                            c: maximum value in Z
                            d: determinat of Z
                            e: sum of squares elements of Z
    norm                    norm to measure the fit in the inversion    2.
	time				    time consumed in inversion 
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

    =====================   =================================================================
    Methods                 Description
    =====================   =================================================================

    """
    def __init__(self, sta_obj, name= None, work_dir = None, num_lay = None , norm = None, \
        inv_dat = None, prior = None, prior_input = None, walk_jump = None, ini_mod = None):
	# ==================== 
    # Attributes            
    # ===================== 
        self.name = sta_obj.name[:-4]
        self.T_obs = sta_obj.T
        self.rho_app_obs = sta_obj.rho_app
        self.phase_obs = sta_obj.phase_deg
        self.max_Z_obs = sta_obj.max_Z
        self.det_Z_obs = sta_obj.det_Z
        self.ssq_Z_obs = sta_obj.ssq_Z
        self.norm = norm
        if work_dir is None: 
            self.work_dir = '.'
        if num_lay is None: 
            self.num_lay = 3
        if inv_dat is None: 
            self.inv_dat = [1,1,0,0,0] 
        if norm is None: 
            self.norm = 2.     
        if prior is None: 
            self.prior = False
        else: 
            if (prior is not 'uniform' and prior is not 'normal'): 
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
                        raise 'incorrect input format: min and max for each range'
            if prior == 'normal': 
                self.prior = True
                self.prior_type = prior
                self.prior_input = prior_input
        if walk_jump is None: 
            self.walk_jump = 5000      
        if ini_mod is None: 
            if self.num_lay == 3:
                self.ini_mod = [500,200,500,5,200]
            if self.num_lay == 4:
                self.ini_mod = [200,100,200,150,50,500,1000]    
        self.time = None  
        self.path_results = None
        self.z1_pars = None
        self.z2_pars = None
        self.r1_pars = None
        self.r2_pars = None
        self.r3_pars = None
    # ===================== 
    # Methods               
    # =====================
    def inv(self):
        nwalkers= 30               # number of walkers
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=cores-1, args=[data,])
		# set the initial location of the walkers
        pars = self.ini_mod  # initial guess
        p0 = np.array([pars + 0.5e2*np.random.randn(ndim) for i in range(nwalkers)])  # add some noise
        p0 = np.abs(p0)
        #p0 = emcee.utils.sample_ball(p0, [20., 20.,], size=nwalkers)

        #Z_est, rho_ap_est, phi_est = self.MT1D_fwd_3layers(*pars,self.T_obs)

        # p0_log = np.log(np.abs(p0))
		# set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
        pos,prob,state=sampler.run_mcmc(np.abs(p0),self.walk_jump)
		# sampler.reset()

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
        # shutil.rmtree('.'+os.sep+str('mcmc_inversions'))
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')):
            os.mkdir('.'+os.sep+str('mcmc_inversions'))
            os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion'))
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+self.name):
            os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+self.name)
        if not os.path.exists('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion')):
            os.mkdir('.'+os.sep+str('mcmc_inversions')+os.sep+str('00_global_inversion'))
            
        self.path_results = '.'+os.sep+str('mcmc_inversions')+os.sep+self.name
        shutil.move('chain.dat', self.path_results+os.sep+'chain.dat')

        # # save text file with inversion parameters
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
            v = 0.15
            v_vec = np.ones(len(self.T_obs))
            #if (self.name == 'WT505a' or self.name == 'WT117b' or self.name == 'WT222a' or self.name == 'WT048a'):
            #v_vec[21:] = np.inf 
            # fitting sounding curves for TE(xy)
            TE_sc = self.inv_dat[0]*-np.sum(((np.log10(obs[:,1]) \
                        -np.log10(rho_ap_est))/v_vec)**self.norm)/v \
                    #+self.inv_dat[0]*-np.sum(((np.log10(obs[:,2]) \
                    #    -np.log10(phi_est))/v_vec)**self.norm)/v 
            # problem with np.log10(phi_est), (not resolved)

            # fitting sounding curves for TM(yx)
            TM_sc = self.inv_dat[1]*-np.sum(((np.log10(obs[:,3]) \
                        -np.log10(rho_ap_est))/v_vec)**self.norm)/v \
                    #+self.inv_dat[1]*-np.sum(((np.log10(obs[:,4]) \
                    #    -np.log10(phi_est))/v_vec)**self.norm)/v 
            # fitting maximum value of Z
            max_Z = self.inv_dat[2]*-np.sum(((np.log10(obs[:,5]) \
                        -np.log10(abs(Z_est)))/v_vec)**self.norm)/v
            # fitting determinant of Z
            #det_Z = self.inv_dat[3]*-np.sum(((-1*np.log10(obs[:,4]) \
            #            -np.log10(Z_est))/v_vec)**self.norm)/v
            # fitting ssq of Z
            ssq_Z = self.inv_dat[4]*-np.sum(((np.log10(obs[:,5]) \
                        -np.log10(Z_est*np.sqrt(2)/2))/v_vec)**self.norm)/v
            # retunr sum od probabilities

            return TE_sc + TM_sc + max_Z + ssq_Z #+ det_Z 
        
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
                    if (pars[1]*pars[3]<100. or pars[1]*pars[3]>500.): # constrain correlation between rest and thick (alpha) 
                        return -np.Inf  # if product between rest and thick of second layer is outside range
                    Z_est, rho_ap_est, phi_est = self.MT1D_fwd_3layers(*pars,self.T_obs)
                    # calculate prob without priors
                    prob = prob_likelihood(Z_est, rho_ap_est, phi_est)
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

    def plot_results_mcmc(self, corner_plt = False, walker_plt = True): 
        chain = np.genfromtxt(self.path_results+os.sep+'chain.dat')
        if corner_plt: 
        # show corner plot
            weights = chain[:,-1]
            weights -= np.max(weights)
            weights = np.exp(weights)
            labels = ['thick. 1','thick. 2','rest. 1','rest. 2','rest. hs']
            fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30)
            plt.savefig(self.path_results+os.sep+'corner_plot.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            #plt.close(fig)
        if walker_plt:
            labels = ['thick. 1','thick. 2','rest. 1','rest. 2','rest. hs']
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

        
        if plot_fit: 
            f,ax = plt.subplots(1,1)
            f.set_size_inches(8,4) 
            f.suptitle(self.name, size = textsize)
            ax.set_xlim([np.min(self.T_obs), np.max(self.T_obs)])
            ax.set_xlim([1E-3,1e3])
            ax.set_ylim([1e0,1e3])
            ax.set_xlabel('period [s]', size = textsize)
            ax.set_ylabel('app. res. [Ohm m]', size = textsize)
            #ax.set_title('Apparent Resistivity: MCMC posterior samples', size = textsize)
				
            for par in pars:
                if all(x > 0. for x in par):
                    Z_vec_aux,app_res_vec_aux, phase_vec_aux = \
                        self.MT1D_fwd_3layers(*par[1:6],self.T_obs)
                    ax.loglog(self.T_obs, app_res_vec_aux,'b-', lw = 0.1, alpha=0.2, zorder=0)
            ax.loglog(self.T_obs, app_res_vec_aux,'b-', lw = 0.1, alpha=0.2, zorder=0, label = 'sample')

            #plot observed
            ax.loglog(self.T_obs, self.rho_app_obs[1],'r*', lw = 1.5, alpha=0.7, zorder=0, label = 'obs. XY')
            ax.loglog(self.T_obs, self.rho_app_obs[2],'g*', lw = 1.5, alpha=0.7, zorder=0, label = 'obs. YX')
            ax.legend(loc='lower right', shadow=False, fontsize='small')
            plt.tight_layout()
            plt.savefig(self.path_results+os.sep+'app_res_fit.png', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        if exp_fig == None:
            plt.close(f)
            plt.clf()
        if exp_fig:  # True: return figure
            return f


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



            


            