"""
- Module Wells: functions to deal with wells files
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
"""
# ==============================================================================
#  Imports
# ==============================================================================

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math
import glob
import os
import shutil as sh
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from Maping_functions import*
from misc_functios import*
from scipy.stats import norm
import matplotlib.mlab as mlab
#from io import StringIO

textsize = 15.
pale_orange_col = u'#ff7f0e' 
pale_blue_col = u'#1f77b4' 
pale_red_col = u'#EE6666'

# ==============================================================================
# Wells class
# ==============================================================================

class Wells(object):
    """
    This class is for wells
    ===================== =====================================================
    Methods               Description
    ===================== =====================================================
		                           
	==================== ========================================== ==========
    Attributes            Description                                default
    ===================== ========================================== ==========
	name				    extracted from the name of the edi file
    ref		 			    reference number in the code
	path				    path to the edi file
	
	lat					    latitud	in dd:mm:ss
    lon					    longitud in dd:mm:ss
	lat_dec				    latitud	in decimal
    lon_dec				    longitud in decimal	
	elev					topography (elevation of the station)
	
	no_temp                 well with no temperature data               False
    depth				 	depths for temperature profile
	red_depth				reduced depths for temperature profile
	depth_dev				depths deviation
	temp_prof_true		    temperature profile 

	depth_raw				 	no filtered depths for temperature profile
	red_depth_raw				no filtered reduced depths for temperature profile
	depth_dev_raw				no filtered depths deviation
	temp_prof_true_raw		    no filtered temperature profile 
	
	temp_prof				resample true temperaure profile
	betas					beta value for each layer
	slopes				slopes values for each layer
	
    meb                   Methylene-blue (MeB) data available         False
	meb_prof				methylene-blue (MeB) profiles
	meb_depth				methylene-blue (MeB) depths (samples)
    meb_z1_pars                 distribution parameters z1 in square func. 
                            (representive of top boundary of cc)
                            from mcmc chain results: [a,b,c,d]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)
    meb_z2_pars                 distribution parameters z2 in square func. 
                            (representive of bottom boundary of cc)
                            from mcmc chain results: [a,b,c,d]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)

    layer_mod_MT_names      names of MT stations used to calculate layer model in well
                            ['mtsta01',...'mtsta04'] 
    layer_mod_MT_dist       distance to MT stations defined in layer_mod_MT_names (km)
    z1_pars                 distribution parameters for layer 1 
                            thickness (model parameter) estimated 
                            from layer model of  MT stations : [a,b]
                            a: mean
                            b: standard deviation 
    z2_pars                 distribution parameters for layer 2 
                            thickness (model parameter) estimated 
                            from layer model of  MT stations : [a,b]
                            a: mean
                            b: standard deviation 
    T1_pars                 distribution parameters for temperatures at layer z1 
                            (top of the conductor) sample from the true temperature profile: [a,b]
                            a: mean
                            b: standard deviation 
    T2_pars                 distribution parameters for temperatures at layer z1 
                            (bottom of the conductor) sample from the true temperature profile: [a,b]
                            a: mean
                            b: standard deviation 
    thermal_grad            Vertical thermal gradient inside the conductor, 
                            calc as (T2_pars[0]-T1_pars[0])/z2_pars[0] 
    thermal_cond            Thermal conductivity in the conductor [SI]
    heat_flux_cond          Vertical conductive heat flux through the clay cap [W/m2]
    heat_flux_tot           Vertical total heat flux (conductive + advective) through the clay cap [W/m2]

    litho                   Lithology available (boolean)                 False

    """
    def __init__(self, name, ref):  						
        self.name = name # name: extracted from the name of the file
        self.ref = ref	 # ref: reference number in the code
		## Properties to be fill
		## File location 
        self.path_loc = None 		# path to file with wells locations	
        self.path_temp = None 		# path to file with wells temperatures
        self.path_meb = None 		# path to file with wells MeB content		
		# Position 
        self.lat = None 		# latitud in dd:mm:ss	
        self.lon = None 		# longitud in dd:mm:ss
        self.lat_dec = None 	# latitud in decimal	
        self.lon_dec = None 	# longitud  in decimal
        self.elev = None 		# topography (elevation of the well)
        # Temp data
        self.no_temp = False         # well with no temp data
        self.depth = None		     # Depths for temperature profile
        self.red_depth = None		 # Reduced depths for temperature profile
        self.depth_dev = None		 # Depths deviation
        self.temp_prof_true = None 	 # Temperature profile
        # Temp data raw
        self.depth_raw = None		     # Depths for temperature profile
        self.red_depth_raw = None		 # Reduced depths for temperature profile
        self.depth_dev_raw = None		 # Depths deviation
        self.temp_prof_true_raw = None 	 # Temperature profile  
		# Temperature estimates
        self.path_temp_est = None   # path to temp. profiles estimation results, where sample profiles are (path to file?)
        self.temp_prof_rs = None	# True temperaure profile resample
        self.red_depth_rs = None	# Reduced depths for temperature profile resample
        self.betas = None			# beta value for each layer
        self.slopes = None			# slopes values for each layer
		# MeB profile
        self.meb = False            # Methylene-blue (MeB) data available
        self.meb_prof = None		# methylene-blue (MeB) profile
        self.meb_depth = None		# methylene-blue (MeB) depths (samples)
        self.meb_z1_pars = None     #  
        self.meb_z2_pars = None     #  
        # layer model from MT nearest stations (depths and temperatures)
        self.layer_mod_MT_names = None 
        self.layer_mod_MT_dist  = None
        self.z1_pars = None
        self.z2_pars = None
        self.T1_pars = None
        self.T2_pars = None
        self.thermal_grad = None
        self.thermal_cond = None
        self.heat_flux_cond = None
        self.heat_flux_tot = None
        # lithology 
        self.litho = None
    # ===================== 
    # Methods               
    # =====================

    def plot_meb_curve(self, square = False):
        f,(ax1) = plt.subplots(1,1)
        f.set_size_inches(6,8)
        f.suptitle(self.name, fontsize=22)
        ax1.set_xscale("linear")
        ax1.set_yscale("linear")    
        ax1.plot(self.meb_prof,self.meb_depth,'*-')
        ax1.set_xlim([0, 20])
        ax1.set_ylim([0,2000])
        ax1.set_xlabel('MeB [%]', fontsize=18)
        ax1.set_ylabel('Depth [m]', fontsize=18)
        ax1.grid(True, which='both', linewidth=0.4)
        ax1.invert_yaxis()
        return f

    def read_meb_mcmc_results(self):
        # extract meb mcmc results from file 
        meb_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+self.name+os.sep+"est_par.dat")
        # values for mean a std for normal distribution representing the prior
        z1_mean_prior = meb_mcmc_results[0,1] # mean z1
        z1_std_prior =  meb_mcmc_results[0,2] # std z1
        z2_mean_prior = meb_mcmc_results[1,1] # mean z2
        z2_std_prior =  meb_mcmc_results[1,2] # std z2
        self.meb_z1_pars = [z1_mean_prior, z1_std_prior]
        self.meb_z2_pars = [z2_mean_prior, z2_std_prior]

    def plot_temp_profile(self, rs = None, raw = None):
        f,(ax1) = plt.subplots(1,1)
        f.set_size_inches(6,8)
        #f.suptitle(self.name, fontsize=22, y=1.08)
    
        ax1.set_xscale("linear")
        ax1.set_yscale("linear")    
        if raw: 
            ax1.plot(self.temp_prof_true_raw,self.red_depth_raw,'+', label = 'raw data') # plot true data
        ax1.plot(self.temp_prof_true,self.red_depth,'o', label = 'filt. data') # plot true data
        if rs:
            ax1.plot(self.temp_prof_rs,self.red_depth_rs,'-', label = 'SC interpolation')
        #ax1.plot(temp_aux,depth_aux,'b--', linewidth=0.05)
        #ax1.set_xlim([np.min(periods), np.max(periods)])
        #ax1.set_xlim([0, 340])
        #ax1.set_ylim([0,3000])
        ax1.set_xlabel('Temperature [deg C]', fontsize=18)
        ax1.set_ylabel('Depth [m]', fontsize=18)
        ax1.grid(True, which='both', linewidth=0.4)
        #ax1.invert_yaxis()
        ax1.legend()
        plt.title(self.name, fontsize=22,)
        plt.tight_layout()
        return f

    def temp_prof_est(self, plot_samples = None, ret_fig = None, path_files_out = None, Ns = None):
        """
        Calculate estimated temperature profiles by fitting heat equation by layer. 
        As the boundaries of three layer model are distributions, here a distributions
        of profiles are calculated. 
        Beta parameters (See Bredehoeft, J. D., and I. S. Papadopulos (1965)) are fit
        for each layer (3) layers in a temperature profile. 
        - if plot_samples: True  plot temperature profile samples and save .png in 
            '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name)
        - if ret_fig: True       Return figure generated by plot_samples 
        - Ns: number of samples
        Files to create: 
        - betas.txt
            - betas.txt
            - slopes.txt
            - temp_est_samples.txt
            - Tmax.txt
            - Tmin.txt
            - Test_samples.png (if plot_samples == True)

        Attributes assigned:
        -  Path to files generated:
            self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name

        """
        if Ns is None:
            Ns = 30
        else:
            Ns = Ns
        if path_files_out:
            self.path_temp_est = path_files_out
        else:
            if os.path.isdir( '.'+os.sep+'temp_prof_samples'):
                pass
            else:
                os.mkdir('.'+os.sep+'temp_prof_samples')

            if os.path.isdir('.'+os.sep+'temp_prof_samples'+os.sep+'wells'):
                pass
            else:
                os.mkdir('.'+os.sep+'temp_prof_samples'+os.sep+'wells')

            # directory to save results
            if os.path.isdir( '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name):
                self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name
            else:
                os.mkdir('.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name)
                self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name

        ## number of samples 
        #Ns = 50 (method input)
        ## figure of Test samples
        # f,(ax1,ax2,ax3) = plt.subplots(1,3)
        # f.set_size_inches(12,4)
        # f.suptitle(self.name, fontsize=12, y=.995)
        # f.tight_layout()
        ## text file of Test samples 
        t = open(self.path_temp_est+os.sep+'temp_est_samples.txt', 'w')
        t.write('# sample temperature profiles using z1,z2,T1,T2 and betas for 3 layers\n')
        b = open(self.path_temp_est+os.sep+'betas.txt', 'w')
        b.write('# sample betas for 3 layers (from z1,z2,T1,T2)\n')
        s = open(self.path_temp_est+os.sep+'slopes.txt', 'w')
        s.write('# sample slopes (linear dt/dz) for 3 layers (from z1,z2,T1,T2)\n')
        tmin = open(self.path_temp_est+os.sep+'Tmin.txt', 'w')
        tmin.write('# sample temperatures at depths: 0, z1, z2 \n')
        tmax = open(self.path_temp_est+os.sep+'Tmax.txt', 'w') 
        tmax.write('# sample temperatures at depths: z1, z2, temp_profile[-1] \n')
        zmin = open(self.path_temp_est+os.sep+'Zmin.txt', 'w')
        zmin.write('# sample depths around: 0, z1, z2 \n')
        zmax = open(self.path_temp_est+os.sep+'Zmax.txt', 'w') 
        zmax.write('# # sample depths around: z1, z2  temp_profile[-1] \n')

        
        ## create sample of z1 (boundary 1) and z2 (boundary 2) for well position
        ## z1 and z2 are thicknesses of layers one and two
        z1_z1_not_valid = True
        max_depth = abs(min(self.red_depth_rs) - max(self.red_depth_rs))
        s_z1 = np.array([])
        s_z2 = np.array([])
        # constraint samples by mean diference between z1 and z2
        z1_z2_mean = np.mean([self.z1_pars[0], self.z2_pars[0]])
        #z2_z2_std = np.mean(self.z2_pars[1],self.z2_pars[1])

        for i in range(Ns): 
            z1 = np.abs(np.random.normal(self.z1_pars[0], self.z1_pars[1], 1))+1. # 
            z2 = np.abs(np.random.normal(self.z2_pars[0], self.z2_pars[1], 1))+1. #
            # sample z1_z2_mean to constraint 
            #z1_z2_const =  np.abs(np.random.normal(z1_z2_mean, z2_z2_std, 1))
            z1_z2_const = z1_z2_mean
            while z1_z1_not_valid:
                # condition for samples: sum of z1 (thick1) and z2(thick2) can't be larger than max depth of resample prof. 
                if z1 + z2 < max_depth:
                    z1_z1_not_valid = False
                else: 
                    z1 = np.abs(np.random.normal(self.z1_pars[0], self.z1_pars[1], 1))+1. # 
                    z2 = np.abs(np.random.normal(self.z2_pars[0], self.z2_pars[1], 1))+1.
                if np.mean([z1, z2]) < z1_z2_const: 
                    z1_z1_not_valid = False
                else: 
                    z1 = np.abs(np.random.normal(self.z1_pars[0], self.z1_pars[1], 1))+1. # 
                    z2 = np.abs(np.random.normal(self.z2_pars[0], self.z2_pars[1], 1))+1.
            z1_z1_not_valid = True 
            s_z1 = np.append(s_z1,z1)
            s_z2 = np.append(s_z2,z2)
        if plot_samples:
            f,(ax1) = plt.subplots(1,1)
            f.set_size_inches(6,8)
            ax1.set_xscale("linear")
            ax1.set_yscale("linear")    
            ax1.plot(self.temp_prof_true,self.red_depth,'o', label = 'data') # plot true data
            ax1.plot(self.temp_prof_rs,self.red_depth_rs,'-', label = 'SC interpolation')
            
            # upper boundary (z1 distribution)
            ax1.plot([-5.,300.], [self.elev - self.z1_pars[0], self.elev - self.z1_pars[0]],'y-', alpha=0.5)
            ax1.plot([-5.,300.], [self.elev - self.z1_pars[0] - self.z1_pars[1], self.elev - self.z1_pars[0] - self.z1_pars[1]],'y--', alpha=0.3)
            ax1.plot([-5.,300.], [self.elev - self.z1_pars[0] + self.z1_pars[1], self.elev - self.z1_pars[0] + self.z1_pars[1]],'y--', alpha=0.3)
            # lower boundary (z2 distribution)
            ax1.plot([-5.,300.], [self.elev - (self.z1_pars[0] + self.z2_pars[0]), self.elev - (self.z1_pars[0] + self.z2_pars[0])],'r-', alpha=0.5)
            ax1.plot([-5.,300.], [self.elev - (self.z1_pars[0] + self.z2_pars[0]) - self.z2_pars[1], self.elev - (self.z1_pars[0] + self.z2_pars[0]) - self.z2_pars[1]],'r--', alpha=0.3)
            ax1.plot([-5.,300.], [self.elev - (self.z1_pars[0] + self.z2_pars[0]) + self.z2_pars[1], self.elev - (self.z1_pars[0] + self.z2_pars[0]) + self.z2_pars[1]],'r--', alpha=0.3)
            
            ax1.set_xlabel('Temperature [deg C]', fontsize=18)
            ax1.set_ylabel('Depth [m]', fontsize=18)
            ax1.grid(True, which='both', linewidth=0.4)
            #ax1.invert_yaxis()
            plt.title(self.name, fontsize=22,)

        ## Calculates temp profile for samples 
        count = 0
        Test_list = []
        for z1,z2 in zip(s_z1,s_z2):
            #while (z2 >= 0. and z1>= 0.):
            #    z1 = np.random.normal(self.z1_pars[0], self.z1_pars[1], 1) # 
            #    z2 = np.random.normal(self.z2_pars[0], self.z2_pars[1], 1) #
            count+=1
            # for each sample: 
            # define spatial boundary conditions for heat equation: [z1_min, z2_min, z3_min]
            Zmin = [self.elev, self.elev - z1 , self.elev - (z1+z2)]
            Zmax = [Zmin[1] , Zmin[2], self.red_depth_rs[-1]]
            # Calculate beta and estimated temp profile
            Test, beta, Tmin, Tmax, slopes = T_beta_est(self.temp_prof_rs, self.red_depth_rs, Zmin, Zmax) # 
            Test_list.append(Test)
            for item in Test:
                t.write('{}\t'.format(item))
            for item in beta:
                b.write('{}\t'.format(item))
            for item in slopes:
                s.write('{}\t'.format(item))
            for item in Tmin:
                tmin.write('{}\t'.format(item))
            for item in Tmax:
                tmax.write('{}\t'.format(item))
            for item in Zmin:
                zmin.write('{}\t'.format(item))
            for item in Zmax:
                zmax.write('{}\t'.format(item))
            t.write('\n')
            b.write('\n')
            s.write('\n')
            tmin.write('\n')
            tmax.write('\n')
            s.write('\n')
            zmin.write('\n')
            zmax.write('\n')

        if plot_samples:
            for Tsamp in Test_list: 
                ax1.plot(Tsamp[1:-1],self.red_depth_rs,'g-', alpha=0.2, lw = 1)  
            ax1.plot([],[],'g-', alpha=1.0 ,label = 'sample')
            ax1.legend()
            plt.tight_layout()
            f.savefig("Test_samples.png", bbox_inches='tight')
            shutil.move('Test_samples.png',self.path_temp_est+os.sep+'Test_samples.png')
            if ret_fig:
                return f
            else: 
                plt.close("all")
        t.close()
        b.close()
        s.close()
        tmin.close()
        tmax.close()
        zmin.close()
        zmax.close()

    def read_temp_prof_est_wells(self, path = None, beta_hist_corr = None):
        """
        Read results from temp_prof_est(self) and assign attributes to well object. Results are 
        text files containing samples of beta, Tmin and Tmax, for the three layers in the well. 
        The attributes assigned are the mean and standar deviation of the samples, assuming that 
        they are normally distributed.  

        Attributes assigned:
        self.betas_3l = [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
        self.Tmin_3l = [[mean_Tmin1, std_Tmin1],[mean_Tmin2, std_Tmin2],[mean_Tmin3, std_Tmin3]]
        self.Tmax_3l = [[mean_Tmax1, std_Tmax1],[mean_Tmax2, std_Tmax2],[mean_Tmax3, std_Tmax3]]

        Files generated:
        if beta_hist_corr True:
            betas_hist_corr.png : histograms and correlation between betas. Save in path.
        
        Note:
        Files are located in 
            path -> self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name
        """
        # path to files
        if path:  
            self.path_temp_est = path
        else: 
            self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+self.name
            
        # Beta: calc means and stds for parameters
        # Read files for import samples
        b = np.genfromtxt(self.path_temp_est+os.sep+'betas.txt').T

        mean_beta1, std_beta1 = np.mean(b[0]), np.std(b[0])
        mean_beta2, std_beta2 = np.mean(b[1]), np.std(b[1])
        mean_beta3, std_beta3 = np.mean(b[2]), np.std(b[2])
        self.betas_3l = [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
        
        if beta_hist_corr: 
            f = plt.figure(figsize=(7, 9))
            gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])
            #plt.title('Histograms betas   -   Coorrelation betas', fontsize=14)
            ## First column -> ax1, ax2, ax3: histograms for betas 
            ## beta1
            b1 = b[0]
            ax1 = f.add_subplot(gs[0, 0])
            bins = np.linspace(np.min(b1), np.max(b1), int(np.sqrt(len(b1))))
            h,e = np.histogram(b1, bins, density = True)
            m = 0.5*(e[:-1]+e[1:])
            ax1.bar(e[:-1], h, e[1]-e[0])
            ax1.set_xlabel('beta 1', fontsize=10)
            ax1.set_ylabel('freq.', fontsize=10)
            ax1.grid(True, which='both', linewidth=0.1)
            # plot normal fit 
            (mu, sigma) = norm.fit(b1)
            y = mlab.normpdf(bins, mu, sigma)
            ax1.plot(bins, y, 'r--', linewidth=2)

            ## beta2
            b1 = b[1]
            ax2 = f.add_subplot(gs[1, 0])
            bins = np.linspace(np.min(b1), np.max(b1), int(np.sqrt(len(b1))))
            h,e = np.histogram(b1, bins, density = True)
            m = 0.5*(e[:-1]+e[1:])
            ax2.bar(e[:-1], h, e[1]-e[0])
            ax2.set_xlabel('beta 2', fontsize=10)
            ax2.set_ylabel('freq.', fontsize=10)
            ax2.grid(True, which='both', linewidth=0.1)
            # plot normal fit 
            (mu, sigma) = norm.fit(b1)
            y = mlab.normpdf(bins, mu, sigma)
            ax2.plot(bins, y, 'r--', linewidth=2)

            ## beta3
            b1 = b[2]
            ax3 = f.add_subplot(gs[2, 0])
            bins = np.linspace(np.min(b1), np.max(b1), int(np.sqrt(len(b1))))
            h,e = np.histogram(b1, bins, density = True)
            m = 0.5*(e[:-1]+e[1:])
            ax3.bar(e[:-1], h, e[1]-e[0])
            ax3.set_xlabel('beta 3', fontsize=10)
            ax3.set_ylabel('freq.', fontsize=10)
            ax3.grid(True, which='both', linewidth=0.1)
            # plot normal fit 
            (mu, sigma) = norm.fit(b1)
            y = mlab.normpdf(bins, mu, sigma)
            ax3.plot(bins, y, 'r--', linewidth=2)

            ## Second column -> ax3, ax4, ax5: correlation between betas 
            
            ## Coor. between beta1 and beta2
            ax = f.add_subplot(gs[0, 1])
            ax.plot(b[0], b[1],'*')
            ax.set_xlabel('beta 1', fontsize=10)
            ax.set_ylabel('beta 2', fontsize=10)
            ax.set_xlim([min(b[0])-2,max(b[0])+2])
            ax.set_ylim([min(b[1])-2,max(b[1])+2])
            ax.grid(True, which='both', linewidth=0.1)

            ## Coor. between beta1 and beta3
            ax = f.add_subplot(gs[1, 1])
            ax.plot(b[0], b[2],'*')
            ax.set_xlabel('beta 1', fontsize=10)
            ax.set_ylabel('beta 3', fontsize=10)
            ax.set_xlim([min(b[0])-2,max(b[0])+2])
            ax.set_ylim([min(b[2])-2,max(b[2])+2])
            ax.grid(True, which='both', linewidth=0.1)

            ## Coor. between beta2 and beta3
            ax = f.add_subplot(gs[2, 1])
            ax.plot(b[1], b[2],'*')
            ax.set_xlabel('beta 2', fontsize=10)
            ax.set_ylabel('beta 3', fontsize=10)
            ax.set_xlim([min(b[1])-2,max(b[1])+2])
            ax.set_ylim([min(b[2])-2,max(b[2])+2])
            ax.grid(True, which='both', linewidth=0.1)

            f.tight_layout()
            # save figure 
            plt.savefig(self.path_temp_est+os.sep+'betas_hist_corr.png', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

        # Tmin: calc means and stds for parameters
        # Read files for import samples
        tmin = np.genfromtxt(self.path_temp_est+os.sep+'Tmin.txt').T
        mean_tmin1, std_tmin1 = np.mean(tmin[0]), np.std(tmin[0])
        mean_tmin2, std_tmin2 = np.mean(tmin[1]), np.std(tmin[1])
        mean_tmin3, std_tmin3 = np.mean(tmin[2]), np.std(tmin[2])
        self.Tmin_3l = [[mean_tmin1, std_tmin1],[mean_tmin2, std_tmin2],[mean_tmin3, std_tmin3]]
        
        # Tmax: calc means and stds for parameters
        # Read files for import samples
        tmax = np.genfromtxt(self.path_temp_est+os.sep+'Tmax.txt').T
        mean_tmax1, std_tmax1 = np.mean(tmax[0]), np.std(tmax[0])
        mean_tmax2, std_tmax2 = np.mean(tmax[1]), np.std(tmax[1])
        mean_tmax3, std_tmax3 = np.mean(tmax[2]), np.std(tmax[2])
        self.Tmax_3l = [[mean_tmax1, std_tmax1],[mean_tmax2, std_tmax2],[mean_tmax3, std_tmax3]]

        # 
# ==============================================================================
# Read files
# ==============================================================================

def read_well_location(file):
    infile = open(file, 'r')
    next(infile) # jump first line
    wells_location = []
    next(infile) # jump first line
    for line in infile:
        b = line.split("\t")
        c = [[b[0], float(b[1]),float(b[2]),float(b[3])]]
        wells_location.extend(c)
    infile.close()    
    
    return wells_location
	
def read_well_temperature(file):
    infile = open(file, 'r')
    next(infile) # jump first line
    wells_name = []
    # wells catalog names
    for line in infile:
        name = line.split()[0]
        if name not in wells_name:
        	wells_name.append(name)
    infile.close()
	
    # wells temp profiles 
    temp_aux = []
    depth_aux = []
    depth_red_aux = []

    wl_prof_depth = []
    wl_prof_depth_red = []
    wl_prof_temp = []

    dir_no_depth_red = []  # directory of wells that don't have reduce depth
	
    for well_aux2 in wells_name:
        infile2 = open(file, 'r')
        next(infile2) # jump first line
        for line in infile2:
            if well_aux2 in line:
                line2 = line.split("\t")
                if not line2[2]: 
                    line2[2] = line2[1]#float('NaN') # define empty value as NaN 
                    if well_aux2 not in dir_no_depth_red: # save in directory name of the well
                        dir_no_depth_red.append(well_aux2)
                depth_aux.append(float(line2[1]))
                depth_red_aux.append(float(line2[2]))
                temp_aux.append(float(line2[3]))

        wl_prof_depth.append(depth_aux)
        wl_prof_depth_red.append(depth_red_aux)
        wl_prof_temp.append(temp_aux)
            
        temp_aux = []       # clear variable 
        depth_aux = []      # clear variable 
        depth_red_aux = []  # clear variable 
        infile2.close()

    return wells_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red

def read_well_temperature_date(file): # function to read xlsx file from contact energy
    infile = open(file, 'r')
    next(infile) # jump first line
    wells_name = []
    # wells catalog names
    for line in infile:
        name = line.split()[0]
        if name not in wells_name:
        	wells_name.append(name)
    infile.close()

    # wells temp profiles 
    temp_aux = []
    depth_aux = []
    depth_red_aux = []
    date_aux = []

    wl_prof_depth = []
    wl_prof_depth_red = []
    wl_prof_temp = []
    wl_prof_date = []

    dir_no_depth_red = []  # directory of wells that don't have reduce depth

    for well_aux2 in wells_name:
        infile2 = open(file, 'r')
        next(infile2) # jump first line
        for line in infile2:
            if well_aux2 in line:
                line2 = line.split("\t")
                if not line2[2]: 
                    line2[2] = line2[1]#float('NaN') # define empty value as NaN 
                    if well_aux2 not in dir_no_depth_red: # save in directory name of the well
                        dir_no_depth_red.append(well_aux2)
                depth_aux.append(float(line2[1]))
                depth_red_aux.append(float(line2[2]))
                temp_aux.append(float(line2[3]))
                year = line2[4].split('/')
                try: 
                    date_aux.append(year[2])
                except:
                    date_aux.append('1996')

        wl_prof_depth.append(depth_aux)
        wl_prof_depth_red.append(depth_red_aux)
        wl_prof_temp.append(temp_aux)
        wl_prof_date.append(date_aux)
            
        temp_aux = []       # clear variable 
        depth_aux = []      # clear variable 
        depth_red_aux = []  # clear variable 
        date_aux = []

        infile2.close()

    return wells_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red, wl_prof_date

def read_well_meb(file):
    infile = open(file, 'r')
    next(infile) # jump first line
    wells_name = []
    # wells catalog names
    for line in infile:
        name = line.split()[0]
        if name not in wells_name:
            wells_name.append(name)
    infile.close()

    # wells meb profiles 
    meb_aux = []
    depth_aux = []
    wl_prof_depth = []
    wl_prof_meb = []

    for well_aux2 in wells_name:
        infile2 = open(file, 'r')
        next(infile2) # jump first line
        for line in infile2:
            if well_aux2 in line:
                line2 = line.split("\t")
                depth_aux.append(float(line2[1]))
                meb_aux.append(float(line2[2]))
        wl_prof_depth.append(depth_aux) 
        wl_prof_meb.append(meb_aux)

        meb_aux = []        # clear variable 
        depth_aux = []      # clear variable 
        infile2.close()

    return wells_name, wl_prof_depth, wl_prof_meb
	
# ==============================================================================
# Plots
# ==============================================================================

def plot_Temp_profile(name, depth_aux, temp_aux):
    print(name)
    f,(ax1) = plt.subplots(1,1)
    f.set_size_inches(6,8)
    f.suptitle(name, fontsize=22)
    
    ax1.set_xscale("linear")
    ax1.set_yscale("linear")    
    ax1.plot(temp_aux,depth_aux,'o')
    #ax1.plot(temp_aux,depth_aux,'b--', linewidth=0.05)
    #ax1.set_xlim([np.min(periods), np.max(periods)])
    ax1.set_xlim([0, 340])
    ax1.set_ylim([0,3000])
    ax1.set_xlabel('Temperature [deg C]', fontsize=18)
    ax1.set_ylabel('Depth [m]', fontsize=18)
    ax1.grid(True, which='both', linewidth=0.4)
    ax1.invert_yaxis()
    
    
    return f

# ==============================================================================
# Functions 
# ==============================================================================

def calc_layer_mod_quadrant(station_objects, wells_objects): 
    """
    Funtion that calculates the boundaries of a three layer model in the well position.
    The boundaries are two: z1, thinckness of layer 1; z2, thickness of layer 2. Each boundary
    is a distribution (assume normal), so the parameteres to calculate are the mean and std for 
    each boundarie. 

    First, for each quadrant around the well, the nearest MT is found.
    Second, using the mcmcm results for the station, the pars are calculated as a weigthed 
    average of the nearest stations.
    Third, the results are assigned as attributes to the well objects. 

    Attributes generated:
    wel_obj.layer_mod_MT_names      : list of names of nearest wells with MeB 
                                    ['well 1',... , 'ẃell 4']
    wel_obj.z1_pars               : values of normal dist. for the top boundaries (clay cap)
                                    [z1_mean,z1_std]
    wel_obj.z2_pars               : values of normal dist. for the bottom boundaries (clay cap)
                                    [z2_mean,z2_std]
    .. conventions::
	: z1 and z2 in MT object refer to thickness of two first layers
    : cc clay cap
    : distances in meters
    """
    for wl in wells_objects:
        dist_pre_q1 = []
        dist_pre_q2 = []
        dist_pre_q3 = []
        dist_pre_q4 = []
        #
        name_aux_q1 = [] 
        name_aux_q2 = []
        name_aux_q3 = []
        name_aux_q4 = []
        sta_q1 = []
        sta_q2 = []
        sta_q3 = []
        sta_q4 = []
        for sta_obj in station_objects:
            # search for nearest MTsta to well in quadrant 1 (Q1)
            if (sta_obj.lat_dec > wl.lat_dec  and  sta_obj.lon_dec > wl.lon_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q1:
                    dist_pre_q1 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q1: 
                    name_aux_q1 = sta_obj.name
                    sta_q1 = sta_obj
                    dist_pre_q1 = dist
            # search for nearest MTsta to well in quadrant 2 (Q2)
            if (sta_obj.lat_dec < wl.lat_dec and sta_obj.lat_dec > wl.lat_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q2:
                    dist_pre_q2 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q2: 
                    name_aux_q2 = sta_obj.name
                    sta_q2 = sta_obj
                    dist_pre_q2 = dist
            # search for nearest MTsta to well in quadrant 3 (Q3)
            if (sta_obj.lat_dec  < wl.lat_dec and sta_obj.lat_dec < wl.lat_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q3:
                    dist_pre_q3 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q3: 
                    name_aux_q3 = sta_obj.name
                    sta_q3 = sta_obj
                    dist_pre_q3 = dist
            # search for nearest MTsta to well in quadrant 4 (Q4)
            if (sta_obj.lat_dec >  wl.lat_dec and sta_obj.lat_dec < wl.lat_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q4:
                    dist_pre_q4 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q4: 
                    name_aux_q4 = sta_obj.name
                    sta_q4 = sta_obj
                    dist_pre_q4 = dist

        # save names of nearest station to be used l
        wl.layer_mod_MT_names = [name_aux_q1, name_aux_q2, name_aux_q3, name_aux_q4]
        wl.layer_mod_MT_names = list(filter(None, wl.layer_mod_MT_names))
        near_stas = [sta_q1,sta_q2,sta_q3,sta_q4] #list of objects (wells)
        near_stas = list(filter(None, near_stas ))
        dist_stas = [dist_pre_q1,dist_pre_q2,dist_pre_q3,dist_pre_q4]
        dist_stas = list(filter(None, dist_stas))
        wl.layer_mod_MT_dist = dist_stas

        # Calculate dist. pars. for boundaries of the cc in well
        # dist consist of mean and std for parameter, calculate as weighted(distance) average from nearest stations
        # z1
        z1_mean = np.zeros(len(near_stas))
        z1_std = np.zeros(len(near_stas))
        z2_mean = np.zeros(len(near_stas))
        z2_std = np.zeros(len(near_stas))
        count = 0
        # extract mcmc results from nearest MT stations
        for sta_obj in near_stas:
            # extract MT mcmc results from file of the station
            mcmc_results = np.genfromtxt('.'+os.sep+str('mcmc_inversions')+os.sep+sta_obj.name[:-4]+os.sep+"est_par.dat")
            # values for mean a std for normal  distribution representing the prior
            z1_mean[count] = mcmc_results[0,1] # mean [1] z1 # median [3] z1 
            z1_std[count] =  mcmc_results[0,2] # std z1
            z2_mean[count] = mcmc_results[1,1] # mean [1] z2 # median [3] z1
            z2_std[count] =  mcmc_results[1,2] # std z2
            count+=1
        
        # calculete z1 normal dist. parameters
        z1_mean = np.dot(z1_mean,dist_stas)/np.sum(dist_stas)
        z1_std = np.dot(z1_std,dist_stas)/np.sum(dist_stas)
        # calculete z2 normal dist. parameters
        z2_mean = np.dot(z2_mean,dist_stas)/np.sum(dist_stas)
        z2_std = np.dot(z2_std,dist_stas)/np.sum(dist_stas)
        # assign result to attribute
        wl.z1_pars = [z1_mean,z1_std]
        wl.z2_pars = [z2_mean,z2_std]
        
def Texp2(z,Zmin,Zmax,Tmin,Tmax,beta): 
    """
    Calculate temp. profile based on model 1D heat transfer model (See Bredehoeft, J. D., and I. S. Papadopulos (1965))

    Input:
    - z: depths to calculate
    - Zmin,Zmax,Tmin,Tmax: boundary conditions
    - beta: beta coeficient for the model 

    Output:
    - Test: estimated temperature profile (array)

    """
    Test = (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta)-1) + Tmin
    return Test

def T_beta_est(Tw, z, Zmin, Zmax):
    """
    Calculate estimated temperature profile by fitting heat equation by layer. 
    A beta parameter (See Bredehoeft, J. D., and I. S. Papadopulos (1965)) is fit
    for each layer (3) layers in a temperature profile. 
    
    Inputs: 
    - Tw: temperature profile to fit
    - z : z values for Tw 
    - Zmin : Minimim depths of layers 
        [z1_min, z2_min, z3_min]
    - Zmax : Maximum depths of layers 
        [z1_max, z2_max, z3_max]

    Outputs: 
    - Test: Tempeature profile estimated (by fit)
    - beta: betas fit per layer 
        [beta1, beta1, beta3]
    - Tmin: top temperature boundary conditions for each layer 
        [T1_min, T2_min, T3_min]
    - Tmax: temperature boundary condition for each 
        [T1_max, T2_max, T3_max]
    - slopes : slope calculate between [Zmin,Tmin] Zmax and [Zmax,Tax]
        [sl1, sl2, sl3]
    Example:

    Notes:
    """

    #z = np.asarray(z)			
    # Search for index of ccboud in depth profile
    # Tmin y T max for layer 1
    inds_z = np.where(z == find_nearest(z, Zmin[0])[0])
    inds_z_l1_top = int(inds_z[0][0])
    Tmin_l1 = Tw[inds_z_l1_top]

    inds_z = np.where(z == find_nearest(z, Zmax[0])[0])
    inds_z_l1_bot = int(inds_z[0][0])
    Tmax_l1 = Tw[inds_z_l1_bot]
    # Tmin y Tmax for layer 2
    inds_z = np.where(z == find_nearest(z, Zmin[1])[0])
    inds_z_l2_top = int(inds_z[0][0])
    Tmin_l2 = Tw[inds_z_l2_top]
    
    inds_z = np.where(z == find_nearest(z, Zmax[1])[0])
    inds_z_l2_bot = int(inds_z[0][0])
    Tmax_l2 = Tw[inds_z_l2_bot]
    
    # Tmin y T max for hs
    inds_z = np.where(z == find_nearest(z, Zmin[2])[0])
    inds_z_l3_top = int(inds_z[0][0])
    Tmin_l3 = Tw[inds_z_l3_top]
    inds_z = np.where(z == find_nearest(z, Zmax[2])[0])
    inds_z_l3_bot = int(inds_z[0][0])
    Tmax_l3 = Tw[inds_z_l3_bot]
    
    # T boundary condition 
    Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
    Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]

    # Fit Twell with Texp
    beta_range = np.arange(-30.0, 30.0, 0.5)
    beta_def = -0.5
    # beta range per layer
    beta_range_l1 = np.arange(-30.5, -1.0, 0.5)
    #beta_range_l2 = np.arange(-.5, 1.0, 0.2)
    beta_range_l2 = np.arange(-30., 30.0, 0.2)
    beta_range_l2 = np.arange(-10., 10.0, 0.1)
    #beta_range_l2 = np.arange(-1., .05, 0.05)
    beta_range_l3 = np.arange(1.0, 30., 0.5)


    ########## Layer 1 ##############
    beta_range = beta_range_l1
    beta_def = beta_range_l1[1]
    # flip (reversed) depth and temperature vectors in the layer (to be solved by curvefit)
    zv_aux = z[inds_z_l1_top:inds_z_l1_bot+1]
    Tv_aux = Tw[inds_z_l1_top:inds_z_l1_bot+1]
    zv = np.zeros(len(zv_aux))
    Tv = np.zeros(len(Tv_aux))
    count=0
    for i,j in zip(reversed(zv_aux),reversed(Tv_aux)):
        zv[count] = i
        Tv[count] = j
        count+=1
    # Calculate beta that best fot the true temp profile 
    pars = [zv[0],zv[-1],Tv[0],Tv[-1],beta_def]
    popt, pcov = curve_fit(Texp2, zv, Tv, p0= pars, bounds=([zv[0]-1.,zv[-1]-1.,Tv[0]-1,Tv[-1]-1., beta_range[0]], [zv[0]+1.,zv[-1]+1.,Tv[0]+1.,Tv[-1]+1,beta_range[-1]]))
    beta_opt_l1 = popt[-1]
    Test_l1 = Texp2(zv,zv[0],zv[-1],Tv[0],Tv[-1],beta_opt_l1)

    ########## Layer 2 ##############
    beta_range = beta_range_l2
    beta_def = beta_range_l2[1]
    # flip (reversed) depth and temperature vectors in the layer (to be solved by curvefit)
    zv_aux = z[inds_z_l2_top:inds_z_l2_bot+1]
    Tv_aux = Tw[inds_z_l2_top:inds_z_l2_bot+1]
    zv = np.zeros(len(zv_aux))
    Tv = np.zeros(len(Tv_aux))
    count=0
    for i,j in zip(reversed(zv_aux),reversed(Tv_aux)):
        zv[count] = i
        Tv[count] = j
        count+=1
    # Calculate beta that best fot the true temp profile 
    pars = [zv[0],zv[-1],Tv[0],Tv[-1],beta_def]
    popt, pcov = curve_fit(Texp2, zv, Tv, p0= pars, bounds=([zv[0]-1.,zv[-1]-1.,Tv[0]-1,Tv[-1]-1., beta_range[0]], [zv[0]+1.,zv[-1]+1.,Tv[0]+1.,Tv[-1]+1,beta_range[-1]]))
    beta_opt_l2 = popt[-1]
    Test_l2 = Texp2(zv,zv[0],zv[-1],Tv[0],Tv[-1],beta_opt_l2)

    ########## Layer 3 ##############
    beta_range = beta_range_l3
    beta_def = beta_range_l3[1]
    # flip (reversed) depth and temperature vectors in the layer (to be solved by curvefit)
    zv_aux = z[inds_z_l3_top:inds_z_l3_bot+1]
    Tv_aux = Tw[inds_z_l3_top:inds_z_l3_bot+1]
    zv = np.zeros(len(zv_aux))
    Tv = np.zeros(len(Tv_aux))
    count=0
    for i,j in zip(reversed(zv_aux),reversed(Tv_aux)):
        zv[count] = i
        Tv[count] = j
        count+=1
    # Calculate beta that best fot the true temp profile 
    pars = [zv[0],zv[-1],Tv[0],Tv[-1],beta_def]
    if zv[0] == zv[-1]:
        zv[-1] = zv[-1]+10.
    popt, pcov = curve_fit(Texp2, zv, Tv, p0= pars,\
        bounds=([zv[0]-1.,zv[-1]-1.,Tv[0]-1,Tv[-1]-1., beta_range[0]], \
                [zv[0]+1.,zv[-1]+1.,Tv[0]+1.,Tv[-1]+1,beta_range[-1]]))
    beta_opt_l3 = popt[-1]
    Test_l3 = Texp2(zv,zv[0],zv[-1],Tv[0],Tv[-1],beta_opt_l3)


    # concatenate the estimated curves
    Test = np.concatenate((list(reversed(Test_l1)), list(reversed(Test_l2)),list(reversed(Test_l3))),axis=0)

    #Test = np.concatenate((Test_l3[:,], Test_l2[1:], Test_l1[1:]),axis=0) 
    beta = [beta_opt_l1, beta_opt_l2, beta_opt_l3]
    slopes = [(Tmax_l1-Tmin_l1)/(Zmax[0]-Zmin[0]),(Tmax_l2-Tmin_l2)/(Zmax[1]-Zmin[1]),(Tmax_l3-Tmin_l3)/(Zmax[2]-Zmin[2])]	
    
    return Test, beta, Tmin, Tmax, slopes

            
# ==============================================================================
# Coorrelation between Temp and boundaries of the conductor 
# ==============================================================================

def wl_z1_z2_est_mt(wells_objects, station_objects, slp = None, plot = None, masl = None, \
    plot_temp_prof = None, with_meb = None, with_litho = None, fix_axes = None, litho_abre = None,
    save_txt_z1_z2= None): 
    '''
    Fn. to calculate boundaries of the conductor in wells position based on interpolation
    of MT inversion results. z1 (mean and std) and z2 (mean and std) are calculated and
    assigned to the well object.   
    Figures generated:
    - Temp profile with estimate z1 and z2 from MT, save in well folder.  
    - plain_view of temperature at BC, save in global folder.
    - if save_txt_z1_z2 is True: save z1 and z2 in txt named by string given in variable 
    '''

    if slp is None: 
        slp = 4*10.
    if fix_axes is None:
        fix_axes =  False
    if save_txt_z1_z2:
        wl_z1_z2 = open(save_txt_z1_z2,'w')
        wl_z1_z2.write('wl_name'+','+'lon_dec'+','+'lat_dec'+','+'Z1_mean'+','+'Z1_std'+','+'Z2_mean'+','+'Z2_std'+'\n')
    if with_litho: # import formation, color, description -> create dictionary
        path = '.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+"formation_colors.txt"
        #depths_from, depths_to, lito  = np.genfromtxt(path, \
        #    delimiter=',').T
        form_abr = []
        form_col = []
        form_des = []

        with open(path) as p:
            next(p)
            for line in p:
                line = line.strip('\n')
                currentline = line.split(",")
                form_abr.append(currentline[0])
                form_col.append(currentline[1])
                form_des.append(currentline[2])
        dict_form = dict([(form_abr[i],(form_col[i],form_des[i])) for i in range(len(form_des))])
        # x = dict_form["SPAT"][0]

    # estimate boundary of the conductor at each well position
    for wl in wells_objects: # use every stations available
        # save names of stations to be used 
        near_stas = [sta for sta in station_objects] 
        near_stas = list(filter(None, near_stas))
        dist_stas = [dist_two_points([sta.lon_dec, sta.lat_dec], [wl.lon_dec, wl.lat_dec], type_coord = 'decimal')\
            for sta in station_objects]
        dist_stas = list(filter(None, dist_stas))
        # Calculate values for boundaries of the conductor in well
        #  mean and std for z1 and z2, calculate as weighted(distance) average from MT stations
        ## arrays
        z1_mean_MT = np.zeros(len(near_stas))
        z1_std_MT = np.zeros(len(near_stas))
        z2_mean_MT = np.zeros(len(near_stas))
        z2_std_MT = np.zeros(len(near_stas))
        #
        z1_std_MT_incre = np.zeros(len(near_stas))
        z2_std_MT_incre = np.zeros(len(near_stas))
        count = 0
        # extract mcmc results from MT stations 
        for sta in near_stas:
            # extract mcmc results from file 
            mt_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+"est_par.dat")
            # values for mean a std for normal distribution representing the prior
            z1_mean_MT[count] = mt_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
            z1_std_MT[count] =  mt_mcmc_results[0,2] # std z1
            z2_mean_MT[count] = mt_mcmc_results[1,1] # mean [1] z2 # median [3] z1
            z2_std_MT[count] =  mt_mcmc_results[1,2] # std z2
            # calc. increment in std. in the position of the station
            # std. dev. increases as get farder from the well. It double its values per 2 km.
            z1_std_MT_incre[count] = z1_std_MT[count]  + (dist_stas[count] *slp)
            z2_std_MT_incre[count] = z2_std_MT[count]  + (dist_stas[count] *slp)
            # load pars in well 
            count+=1
        # calculete z1 in well position
        dist_weigth = [1./d**3 for d in dist_stas]
        z1_mean = np.dot(z1_mean_MT,dist_weigth)/np.sum(dist_weigth)
        # std. dev. increases as get farder from the well. It double its values per km.  
        z1_std = np.dot(z1_std_MT_incre,dist_weigth)/np.sum(dist_weigth)
        # calculete z2 normal prior parameters
        # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
        #z2_mean_prior = z2_mean_prior - z1_mean_prior
        #print(z2_mean_prior)
        z2_mean = np.dot(z2_mean_MT,dist_weigth)/np.sum(dist_weigth)
        #z2_mean = z2_mean 
        if z2_mean < 0.:
            raise ValueError
        z2_std = np.dot(z2_std_MT_incre,dist_weigth)/np.sum(dist_weigth)        
        # assign values to well object
        wl.z1_pars = [z1_mean, z1_std]
        wl.z2_pars = [z2_mean, z2_std]
        if save_txt_z1_z2:
            # write results 
            wl_z1_z2.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','
                +str(wl.z1_pars[0])+','+str(wl.z1_pars[1])+','+str(wl.z2_pars[0])+','+str(wl.z2_pars[1])+'\n')
            #   
        if masl:
            z1_mean = sta.elev - z1_mean   # need to import topography (elevation in every point of the grid)
    if save_txt_z1_z2:
        wl_z1_z2.close()
    if plot_temp_prof:
        pp = PdfPages('Temp_prof_conductor_bound_est.pdf') # pdf to plot the meb profiles
        if with_litho:
            pp_litho = PdfPages('Temp_litho_prof_conductor_bound_est.pdf') # pdf to plot litho
        for wl in wells_objects:
            f,(ax1) = plt.subplots(1,1)
            f.set_size_inches(5,7)
            ax1.set_xscale("linear")
            ax1.set_yscale("linear") 
            try: 
                depth_2_zero = [r-wl.elev  for r in wl.red_depth]
                depth_2_zero_rs = [r-wl.elev  for r in wl.red_depth_rs]
                ax1.plot(wl.temp_prof_true, [r for r in depth_2_zero], 'o', label = 'Temperature data') # plot true data
                ax1.plot(wl.temp_prof_rs, [r for r in depth_2_zero_rs], '-')#, label = 'SC interpolation')
                # alpha for surface temp point 
                ax1.plot(wl.temp_prof_true[0], depth_2_zero[0], '.', c = 'w', zorder = 6) # plot true data

            except:
                pass
            # upper boundary (z1 distribution)
            #ax1.plot([-5.,300.], [-wl.z1_pars[0],-wl.z1_pars[0]],'r-', alpha=0.5, label = 'MT upper bound.')
            #ax1.plot([-5.,300.], [-wl.z1_pars[0] - wl.z1_pars[1],-wl.z1_pars[0] - wl.z1_pars[1]],'r--', alpha=0.3)
            #ax1.plot([-5.,300.], [-wl.z1_pars[0] + wl.z1_pars[1],-wl.z1_pars[0] + wl.z1_pars[1]],'r--', alpha=0.3)
            # std 3
            #ax1.fill_between([55.,105.],[-wl.z1_pars[0] + 3*wl.z1_pars[1], -wl.z1_pars[0] + 3*wl.z1_pars[1]], 
            #    [-wl.z1_pars[0] - 3*wl.z1_pars[1], -wl.z1_pars[0] - 3*wl.z1_pars[1]], color = 'r', alpha=0.1)
      
            # std 2
            ax1.fill_between([55.,105.],[-wl.z1_pars[0] + 1.5*wl.z1_pars[1], -wl.z1_pars[0] + 1.5*wl.z1_pars[1]], 
                [-wl.z1_pars[0] - 1.5*wl.z1_pars[1], -wl.z1_pars[0] - 1.5*wl.z1_pars[1]], color = pale_orange_col, alpha=0.3)
      
            # std 1
            ax1.fill_between([55.,105.],[-wl.z1_pars[0] + .5*wl.z1_pars[1], -wl.z1_pars[0] + .5*wl.z1_pars[1]], 
                [-wl.z1_pars[0] - .5*wl.z1_pars[1], -wl.z1_pars[0] - .5*wl.z1_pars[1]], color = pale_orange_col, alpha=0.5, label = 'MT upper boundary')
            
            # mean
            #ax1.plot([55.,105.],[-wl.z1_pars[0], -wl.z1_pars[0]],'r--',  alpha=0.3)
 
            # lower boundary (z2 distribution)
            #ax1.plot([-5.,300.], [-(wl.z1_pars[0] + wl.z2_pars[0]),-(wl.z1_pars[0] + wl.z2_pars[0])],'b-', alpha=0.5, label = 'MT lower bound.')
            #ax1.plot([-5.,300.], [-(wl.z1_pars[0] + wl.z2_pars[0]) - wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) - wl.z2_pars[1]],'b--', alpha=0.3)
            #ax1.plot([-5.,300.], [-(wl.z1_pars[0] + wl.z2_pars[0]) + wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) + wl.z2_pars[1]],'b--', alpha=0.3)
            # sts 3
            #ax1.fill_between([55.,105.], [-(wl.z1_pars[0] + wl.z2_pars[0]) - 3*wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) - 3*wl.z2_pars[1]], 
            #    [-(wl.z1_pars[0] + wl.z2_pars[0]) + 3*wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) + 3*wl.z2_pars[1]], 
            #    color = 'b', alpha=0.1)
            # sts 3: 99%
            ax1.fill_between([55.,105.], [-(wl.z1_pars[0] + wl.z2_pars[0]) - 1.5*wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) - 1.5*wl.z2_pars[1]], 
                [-(wl.z1_pars[0] + wl.z2_pars[0]) + 1.5*wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) + 1.5*wl.z2_pars[1]], 
                color = pale_blue_col, alpha=0.3)
            # sts 1: 33%
            ax1.fill_between([55.,105.], [-(wl.z1_pars[0] + wl.z2_pars[0]) - .5*wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) - .5*wl.z2_pars[1]], 
                [-(wl.z1_pars[0] + wl.z2_pars[0]) + .5*wl.z2_pars[1],-(wl.z1_pars[0] + wl.z2_pars[0]) + .5*wl.z2_pars[1]], 
                color = pale_blue_col, alpha=0.5, label = 'MT lower boundary')
            # mean
            #ax1.plot([55.,105.],[-(wl.z1_pars[0] + wl.z2_pars[0]), -(wl.z1_pars[0] + wl.z2_pars[0])],'b--',  alpha=0.3)

            ax1.set_xlabel('Temperature [deg C]', fontsize=textsize)
            ax1.set_ylabel('Depth [m]', fontsize=textsize)
            ax1.grid(True, which='both', linewidth=0.4)
            #ax1.invert_yaxis()
            plt.title(wl.name, fontsize=textsize,)
            # plot MeB inversion
            if wl.meb:
                if with_meb:
                    meb_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+wl.name+os.sep+"est_par.dat")
                    # values for mean a std for normal distribution representing the prior
                    # assign values to well object
                    wl.meb_z1_pars = [meb_mcmc_results[0,1], meb_mcmc_results[0,2]]
                    wl.meb_z2_pars = [meb_mcmc_results[1,1], meb_mcmc_results[1,2]]
                    
                    # upper boundary (z1 distribution)
                    #ax1.plot([-5.,300.], [-wl.meb_z1_pars[0],-wl.meb_z1_pars[0]],'c-', alpha=0.5, label = 'MeB upper bound.')
                    #ax1.plot([-5.,300.], [-wl.meb_z1_pars[0] - wl.meb_z1_pars[1],-wl.meb_z1_pars[0] - wl.meb_z1_pars[1]],'c--', alpha=0.3)
                    #ax1.plot([-5.,300.], [-wl.meb_z1_pars[0] + wl.meb_z1_pars[1],-wl.meb_z1_pars[0] + wl.meb_z1_pars[1]],'c--', alpha=0.3)
                    # 3 std
                    #ax1.fill_between([0.,50.],[min(0.,-wl.meb_z1_pars[0] + 3*wl.meb_z1_pars[1]),min(0.,-wl.meb_z1_pars[0] + 2*wl.meb_z1_pars[1])],
                    #    [-wl.meb_z1_pars[0] - 3*wl.meb_z1_pars[1],-wl.meb_z1_pars[0] - 3*wl.meb_z1_pars[1]], color = 'c', alpha=0.1)
   
                    # 2 std
                    ax1.fill_between([0.,50.],[min(0.,-wl.meb_z1_pars[0] + 1.5*wl.meb_z1_pars[1]),min(0.,-wl.meb_z1_pars[0] + 1.5*wl.meb_z1_pars[1])],
                        [-wl.meb_z1_pars[0] - 1.5*wl.meb_z1_pars[1],-wl.meb_z1_pars[0] - 1.5*wl.meb_z1_pars[1]], color = 'c', alpha=0.1)
                    # 1 std
                    ax1.fill_between([0.,50.],[min(0.,-wl.meb_z1_pars[0] + .5*wl.meb_z1_pars[1]),min(0.,-wl.meb_z1_pars[0] + .5*wl.meb_z1_pars[1])],
                        [-wl.meb_z1_pars[0] - .5*wl.meb_z1_pars[1],-wl.meb_z1_pars[0] - .5*wl.meb_z1_pars[1]], color = 'c', alpha=0.3, label = 'MeB upper boundary')
                    # mean
                    #ax1.plot([0.,50.],[min(0.,-wl.meb_z1_pars[0]),min(0.,-wl.meb_z1_pars[0])],'c--')
                    #ax1.fill_between([0.,50.],[min(0.,-wl.meb_z1_pars[0] + wl.meb_z1_pars[1]),min(0.,-wl.meb_z1_pars[0] + wl.meb_z1_pars[1])],
                    #    [-wl.meb_z1_pars[0] - wl.meb_z1_pars[1],-wl.meb_z1_pars[0] - wl.meb_z1_pars[1]], color = 'c', alpha=0.3, label = 'MeB upper boundary')
                    
                    # lower boundary (z2 distribution)
                    #ax1.plot([-5.,300.], [-(wl.meb_z2_pars[0]),-(wl.meb_z2_pars[0])],'m-', alpha=0.5, label = 'MeB lower bound.')
                    #ax1.plot([-5.,300.], [-(wl.meb_z2_pars[0]) - wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) - wl.meb_z2_pars[1]],'m--', alpha=0.3)
                    #ax1.plot([-5.,300.], [-(wl.meb_z2_pars[0]) + wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) + wl.meb_z2_pars[1]],'m--', alpha=0.3)
                    # 3 std
                    #ax1.fill_between([0.,50.],[-(wl.meb_z2_pars[0]) + 3*wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) + 3*wl.meb_z2_pars[1]], 
                    #    [-(wl.meb_z2_pars[0]) - 3*wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) - 3*wl.meb_z2_pars[1]], color = 'm', alpha=0.1)
                    # 2 std
                    ax1.fill_between([0.,50.],[-(wl.meb_z2_pars[0]) + 1.5*wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) + 1.5*wl.meb_z2_pars[1]], 
                        [-(wl.meb_z2_pars[0]) - 1.5*wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) - 1.5*wl.meb_z2_pars[1]], color = 'm', alpha=0.1)
                    # 1 std
                    ax1.fill_between([0.,50.],[-(wl.meb_z2_pars[0]) + .5*wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) + .5*wl.meb_z2_pars[1]], 
                        [-(wl.meb_z2_pars[0]) - .5*wl.meb_z2_pars[1],-(wl.meb_z2_pars[0]) - .5*wl.meb_z2_pars[1]], color = 'm', alpha=0.3, label = 'MeB lower boundary')
                    # mean
                    #ax1.plot([0.,50.],[min(0.,-wl.meb_z2_pars[0]),min(0.,-wl.meb_z2_pars[0])],'m--')                    
                    
                    # save pars in .txt: MeB
                    g = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'meb_z1_z2.txt', "w")
                    g.write('# mean_z1(detph to bottom layer 1)\tstd_z1\tmean_z2(depth to bottom layer 2)\tstd_z2\n')
                    g.write("{:4.2f}\t{:4.2f}\t{:4.2f}\t{:4.2f}".format(wl.meb_z1_pars[0],wl.meb_z1_pars[1],wl.meb_z2_pars[0],wl.meb_z2_pars[1]))
                    g.close()               

            if with_litho:
                try:
                    path = '.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+wl.name+os.sep+"lithology.txt"
                    #depths_from, depths_to, lito  = np.genfromtxt(path, \
                    #    delimiter=',').T
                    depths_from = []
                    depths_to = []
                    lito = []
                    with open(path) as p:
                        next(p)
                        for line in p:
                            line = line.strip('\n')
                            currentline = line.split(",")
                            depths_from.append(float(currentline[0]))
                            depths_to.append(float(currentline[1]))
                            lito.append(currentline[2])

                    N = len(depths_to) # number of lithological layers
                    colors = [dict_form[lito[i]][0] for i in range(len(lito))]
                    #colors = ['g','r','orange','r','m','g','r','orange','r','m']
                    #with open(r'data\nzafd.json', 'r') as fp:
                    for i in range(N):
                        ax1.fill_between([-45,-10],[-depths_from[i], -depths_from[i]], [-depths_to[i], -depths_to[i]], color = colors[i])
                        thick = depths_to[i] - depths_from[i]
                        if thick > 25:
                            if litho_abre:
                                ax1.text(-30, -depths_from[i] - thick/2, lito[i], fontsize=8,\
                                    horizontalalignment='center', verticalalignment='center')
                
                    if not litho_abre: # black line in yaxis 0
                        ax1.plot([0,0],[-depths_to[-1], 20], 'k--', linewidth = 1)
                        ax1.text(-25, -depths_to[-1] - 150, 'lithology', fontsize=textsize, rotation = 90,\
                            horizontalalignment='center', verticalalignment='center')
                except:
                    pass


            ax1.legend(loc='lower left', shadow=False, fontsize=textsize, framealpha=1.0)
            # axes lims
            if fix_axes:
                ax1.set_xlim(fix_axes[0])
                ax1.set_ylim(fix_axes[1])
            else:
                ax1.set_ylim([-1700.,20.])
                try:
                    ax1.set_ylim([wl.red_depth[-1]-750.,20.])
                    if wl.name == 'WK317':
                        ax1.set_ylim([-1500.,20.])

                except:
                    pass
            # save image as png
            plt.tight_layout()

            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'Temp_prof_conductor_bound_est.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
            if with_litho:
                try:
                    sh.copyfile('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'Temp_prof_conductor_bound_est.png',
                        '.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+wl.name+os.sep+'Temp_prof_conductor_bound_est.png')
                    pp_litho.savefig(f)
                except:
                    pass
            pp.savefig(f)
            plt.close(f)
            # save pars in .txt: MT
            g = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_z1_z2.txt', "w")
            g.write('# mean_z1(thickness layer 1)\tstd_z1\tmean_z2(thickness layer 2)\tstd_z2\n')
            g.write("{:4.2f}\t{:4.2f}\t{:4.2f}\t{:4.2f}".format(wl.z1_pars[0],wl.z1_pars[1],wl.z2_pars[0],wl.z2_pars[1]))
            g.close()
        pp.close()
        shutil.move('Temp_prof_conductor_bound_est.pdf','.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'Temp_prof_conductor_bound_est.pdf')
        if with_litho:
            pp_litho.close()
            shutil.move('Temp_litho_prof_conductor_bound_est.pdf','.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'Temp_litho_prof_conductor_bound_est.pdf')

def wl_T1_T2_est(wells_objects, hist = None, hist_filt = None, thermal_grad = None, heat_flux = None):
    '''
    Sample temperatures at z1 and z1 ranges to create T1_pars and T2_pars (distribrutions for temperatures at conductor bound.)
    hist_filt = [50, 100], wells with temp lower than a and b are not considered in the histogram for top bound. and bottom bound. 
    '''
    ## load z1 and z2 pars
    for wl in wells_objects:
        aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_z1_z2.txt')
        wl.z1_pars = [aux[0],aux[1]]
        wl.z2_pars = [aux[2],aux[3]]
    if hist: 
        T1_batch = []
        T2_batch = []
    # sample temp values and calc T1 and T2 
    Ns = 1000
    for wl in wells_objects:
            
        try: 
            d2_c = wl.z1_pars[0] + wl.z2_pars[0] + wl.z2_pars[1] # depth to bottom of conductor (from surface)
            d2_w = -1*(wl.red_depth_rs[-1]-wl.elev) # max depth at well (from surface)
            # if depth at the bottom of conductor is deeper than deppest temp value
            if d2_c < d2_w :
                # array of samples
                z1_sam = np.random.normal(wl.z1_pars[0], wl.z1_pars[1], Ns)
                z2_sam = np.random.normal(wl.z2_pars[0], wl.z2_pars[1], Ns)
                # 
                T1_sam = z1_sam*0
                T2_sam = z1_sam*0
                # 
                for i in range(len(T1_sam)):
                    # T1
                    val, idx = find_nearest(wl.red_depth_rs, wl.elev - z1_sam[i])
                    T1_sam[i] = wl.temp_prof_rs[idx]
                    # T2
                    d2_c = z1_sam[i] + z2_sam[i] + wl.z2_pars[1]# depth to bottom of conductor (from surface)
                    d2_w = -1*(wl.red_depth_rs[-1]-wl.elev) # max depth at well (from surface)
                    #if d2_c < d2_w:
                    val, idx = find_nearest(wl.red_depth_rs, wl.elev - d2_c)
                    T2_sam[i] = wl.temp_prof_rs[idx]

                # Assign attributes TX_pars and save in .txt
                wl.T1_pars = [np.mean(T1_sam),np.std(T1_sam)]
                wl.T2_pars = [np.mean(T2_sam),np.std(T2_sam)]

                # Calculate thermal gradient inside the conductor 
                if thermal_grad:
                    # Assign attributes TX_pars and save in .txt
                    dT =  wl.T2_pars[0]-wl.T1_pars[0]
                    dZ = wl.z2_pars[0]
                    wl.thermal_grad = dT/dZ
                    if heat_flux:
                        wl.thermal_cond = 2.5 # this couls be variable
                        wl.heat_flux = wl.thermal_cond*wl.thermal_grad

                if wl.T1_pars[0] > 1000:
                    print(wl.name)
                if wl.T2_pars[0] > 1000:
                    print(wl.name)

            else:
                pass
        except: # case when well does not have temp data 
            wl.no_temp = True
            pass

        if hist:
            if hist_filt:
                if wl.T1_pars[0] > hist_filt[0]:
                    T1_batch.append(wl.T1_pars[0])
                if wl.T2_pars[0] > hist_filt[1]:
                    T2_batch.append(wl.T2_pars[0])
            else:
                T1_batch.append(wl.T1_pars[0])
                T2_batch.append(wl.T2_pars[0])
        
        if not wl.no_temp:

            try:
                # save pars in .txt
                g = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt', "w")
                g.write('# mean_T1(temp at z1)\tstd_T1\tmean_T2(temp at z2)\tstd_T2\n')
                g.write("{:4.2f}\t{:4.2f}\t{:4.2f}\t{:4.2f}".format(wl.T1_pars[0], wl.T1_pars[1], wl.T2_pars[0], wl.T2_pars[1]))
                g.close()

            except:
                #pass
                os.remove('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt')
            try:
                # save pars in .txt
                g = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_TG_TC_HF.txt', "w")
                g.write('# ThermalGradient[C/m]\tThermalConductivity[W/m*C]\tHeatFlux[W/m2]\n')
                g.write("{:4.6f}\t{:4.4f}\t{:4.6f}".format(wl.thermal_grad, wl.thermal_cond, wl.heat_flux))
                g.close()
            except:
                #pass
                os.remove('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_TG_TC_HF.txt')

    if hist: 
        f = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        ax2 = f.add_subplot(gs[0, 0])
        ax3 = f.add_subplot(gs[0, 1])

        # T1
        bins = np.linspace(np.min(T1_batch), np.max(T1_batch), int(np.sqrt(2*len(T1_batch))))
        h,e = np.histogram(T1_batch, bins)
        m = 0.5*(e[:-1]+e[1:])
        ax2.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
        ax2.set_xlabel('$T_1$ [°C]', fontsize=textsize)
        ax2.set_ylabel('freq.', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(T1_batch)
        med = np.median(T1_batch)
        try:
            y = mlab.normpdf(bins, mu, sigma)
        except:
            #y = stats.norm.pdf(bins, mu, sigma)
            pass
        #ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
        #ax2.legend(loc='upper right', shadow=False, fontsize=textsize)
        ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        ax2.plot([med,med],[0,np.max(h)+np.max(h/4)],'y-', label = 'median')
        ax2.legend(loc='upper right', shadow=False, fontsize=textsize, framealpha=1.0)

        # T2
        bins = np.linspace(np.min(T1_batch), np.max(T2_batch), int(np.sqrt(2*len(T2_batch))))
        h,e = np.histogram(T2_batch, bins)
        m = 0.5*(e[:-1]+e[1:])
        ax3.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
        ax3.set_xlabel('$T_2$ [°C]', fontsize=textsize)
        ax3.set_ylabel('freq.', fontsize=textsize)
        ax3.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(T2_batch)
        med = np.median(T2_batch)
        try:
            y = mlab.normpdf(bins, mu, sigma)
        except:
            #y = stats.norm.pdf(bins, mu, sigma)
            pass
        #ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
        #ax3.legend(loc='upper right', shadow=False, fontsize=textsize)
        ax3.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        ax3.plot([med,med],[0,np.max(h)+np.max(h/4)],'r-', label = 'median')
        ax3.legend(loc='upper right', shadow=False, fontsize=textsize, framealpha=1.0)

        f.tight_layout()
        if hist_filt:
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_filt_nwells_'+str(len(T1_batch))+'.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        else:
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(T1_batch))+'.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

def histogram_temp_T1_T2(wells_objects, filt_in_count = None, filt_out_count = None, 
    type_hist = None, multi_samples = None): 
    """
    filt_in_count (or filt_out_count) : file of countour (i.e. WT resisitvity boundary)
    multi samples: several samples per well (amount given by the integer in variable multi_samples)
    just_infield: filter infield wells and plot just them (make )
    type_hist: 'sidebyside' infield and outfield side by side (if countour filt given)
                'infield' just infield (if countour filt given)
                None infield and outfield overlap (if countour filt given)
    """

    if filt_in_count:
        lats, lons = np.genfromtxt(filt_in_count, skip_header=1, delimiter=',').T
        poli_in = [[lons[i],lats[i]] for i in range(len(lats))]
    if filt_out_count:
        lats, lons = np.genfromtxt(filt_out_count, skip_header=1, delimiter=',').T
        poli_out = [[lons[i],lats[i]] for i in range(len(lats))]

    t1_batch = []
    t2_batch = []

    if filt_in_count:
        t1_batch_filt_in = []
        t2_batch_filt_in = []
    if filt_out_count:
        t1_batch_filt_out = []
        t2_batch_filt_out = []
    if type_hist:
        type_hist = type_hist
    else:
        type_hist = None
    ## load pars
    for wl in wells_objects:
        if not multi_samples:
            try:
                if not wl.no_temp:
                    aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt')
                    wl.T1_pars = [aux[0],aux[1]]
                    wl.T2_pars = [aux[2],aux[3]]

                    t1_batch.append(wl.T1_pars[0])
                    t2_batch.append(wl.T2_pars[0])
                    
                    if filt_in_count:
                        # check if station is inside poligon 
                        val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_in)
                        if val:
                            t1_batch_filt_in.append(wl.T1_pars[0])
                            if wl.T2_pars[0]>100.:
                                t2_batch_filt_in.append(wl.T2_pars[0])

                    if filt_out_count:
                        # check if station is inside poligon 
                        val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_out)
                        if not val:
                            t1_batch_filt_out.append(wl.T1_pars[0])
                            t2_batch_filt_out.append(wl.T2_pars[0])
            except:
                pass
        # if multi_samples # generate multiplate samples and add to list
        if  multi_samples:
            try:
                if not wl.no_temp:
                    print('sampling T1 and T1 in well: '+wl.name)
                    # aux lists
                    wl_t1_batch = []
                    wl_t2_batch = []
                    #
                    aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt')
                    wl.T1_pars = [aux[0],aux[1]]
                    wl.T2_pars = [aux[2],aux[3]]

                    Ns = multi_samples
                    for i in range(Ns): 
                        T1 = float(np.abs(np.random.normal(wl.T1_pars[0], wl.T1_pars[1], 1))+1.) # 
                        T2 = float(np.abs(np.random.normal(wl.T2_pars[0], wl.T2_pars[1], 1))+1.) #
                        # add sample to list 
                        t1_batch.append(T1)
                        t2_batch.append(T2)
                        wl_t1_batch.append(T1)
                        wl_t2_batch.append(T2)

                    if filt_in_count:
                        # check if station is inside poligon 
                        val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_in)
                        if val:
                            [t1_batch_filt_in.append(T1) for T1 in wl_t1_batch]
                            if wl.T2_pars[0]>100.:
                                [t2_batch_filt_in.append(T2) for T2 in wl_t2_batch]

                    if filt_out_count:
                        # check if station is inside poligon 
                        val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_out)
                        if not val:
                            [t1_batch_filt_out.append(T1) for T1 in wl_t1_batch]
                            [t2_batch_filt_out.append(T2) for T2 in wl_t2_batch]
            except:
                pass

    if type_hist != 'sidebyside' or type_hist != 'infield':
        # plot histograms 
        f = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(nrows=1, ncols=3)
        ax1 = f.add_subplot(gs[0, 0])
        ax2 = f.add_subplot(gs[0, 1])
        ax_leg= f.add_subplot(gs[0, 2])

        # t1
        bins = np.linspace(np.min(t1_batch), np.max(t1_batch), 2*int(np.sqrt(len(t1_batch))))
        h,e = np.histogram(t1_batch, bins)
        m = 0.5*(e[:-1]+e[1:])
        ax1.bar(e[:-1], h, e[1]-e[0], alpha = .6, edgecolor = 'w',  zorder = 1, color = 'lightsteelblue')
        ax1.set_xlabel('$T_1$ [m]', fontsize=textsize)
        ax1.set_ylabel('frequency', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(t1_batch)
        med = np.median(t1_batch)
        try:
            y = mlab.normpdf(bins, mu, sigma)
        except:
            #y = stats.norm.pdf(bins, mu, sigma)
            pass
        #ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
        #ax2.legend(loc='upper right', shadow=False, fontsize=textsize)
        
        if not filt_in_count:
            ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3)
            ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        if filt_in_count:
            # t1
            bins = np.linspace(np.min(t1_batch_filt_in), np.max(t1_batch_filt_in), 2*int(np.sqrt(len(t1_batch_filt_in))))
            h,e = np.histogram(t1_batch_filt_in, bins)
            m = 0.5*(e[:-1]+e[1:])
            ax1.bar(e[:-1], h, e[1]-e[0], alpha =.8, color = 'PaleVioletRed', edgecolor = 'w', zorder = 3)
            #ax1.legend(loc=None, shadow=False, fontsize=textsize)
            # 
            (mu, sigma) = norm.fit(t1_batch_filt_in)
            med = np.median(t1_batch_filt_in)
            try:
                y = mlab.normpdf(bins, mu, sigma)
            except:
                #y = stats.norm.pdf(bins, mu, sigma)
                pass
            ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3, linewidth=3)
            ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        if filt_out_count:
            # t1
            bins = np.linspace(np.min(t1_batch_filt_out), np.max(t1_batch_filt_out), 2*int(np.sqrt(len(t1_batch_filt_out))))
            h,e = np.histogram(t1_batch_filt_out, bins)
            m = 0.5*(e[:-1]+e[1:])
            ax1.bar(e[:-1], h, e[1]-e[0], alpha =.3, edgecolor = None, color = 'cyan', zorder = 2)
            #ax1.legend(loc=None, shadow=False, fontsize=textsize)

        # t2
        bins = np.linspace(np.min(t2_batch), np.max(t2_batch), 2*int(np.sqrt(len(t2_batch))))
        h,e = np.histogram(t2_batch, bins)
        m = 0.5*(e[:-1]+e[1:])
        ax2.bar(e[:-1], h, e[1]-e[0], alpha = .6, edgecolor = 'w', zorder = 1, color = 'lightsteelblue')
        ax2.set_xlabel('$T_2$ [m]', fontsize=textsize)
        ax2.set_ylabel('frequency', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(t2_batch)
        med = np.median(t2_batch)
        try:
            y = mlab.normpdf(bins, mu, sigma)
        except:
            #y = stats.norm.pdf(bins, mu, sigma)
            pass
        #ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
        #ax3.legend(loc='upper right', shadow=False, fontsize=textsize)
        #ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        #ax2.plot([med,med],[0,np.max(h)],'b-')
        
        if not filt_in_count:
            ax2.plot([med,med],[0,np.max(h)],'b-', zorder = 3)
            ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        if filt_in_count:
            # t2
            bins = np.linspace(np.min(t2_batch_filt_in), np.max(t2_batch_filt_in), 2*int(np.sqrt(len(t2_batch_filt_in))))
            h,e = np.histogram(t2_batch_filt_in, bins)
            m = 0.5*(e[:-1]+e[1:])
            ax2.bar(e[:-1], h, e[1]-e[0], alpha =.8, color = 'PaleVioletRed', edgecolor = 'w', zorder = 3)
            #ax2.legend(loc=None, shadow=False, fontsize=textsize)
            # 
            (mu, sigma) = norm.fit(t2_batch_filt_in)
            med = np.median(t2_batch_filt_in)
            try:
                y = mlab.normpdf(bins, mu, sigma)
            except:
                #y = stats.norm.pdf(bins, mu, sigma)
                pass
            ax2.plot([med,med],[0,np.max(h)],'b-', zorder = 3, linewidth=3)
            ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        if filt_out_count:
            # t2
            bins = np.linspace(np.min(t2_batch_filt_out), np.max(t2_batch_filt_out), 2*int(np.sqrt(len(t2_batch_filt_out))))
            h,e = np.histogram(t2_batch_filt_out, bins)
            m = 0.5*(e[:-1]+e[1:])
            ax2.bar(e[:-1], h, e[1]-e[0], alpha =.3, edgecolor = None, color = 'cyan', zorder = 2)
            #ax2.legend(loc=None, shadow=False, fontsize=textsize)

        # plor legend 
        if filt_in_count and filt_out_count:
            #ax_leg.bar([],[],[], alpha =.9, color = 'darkorange', edgecolor = 'w', label = 'active zone',zorder = 3)
            # active zone
            ax_leg.plot([],[],c = 'PaleVioletRed', linewidth=10,label = r'Infield',  alpha =.8)
            # cooling zone
            ax_leg.plot([],[],c = 'cyan', linewidth=10,label = r'Outfield',  alpha =.3)
            # full array
            ax_leg.plot([],[], c = 'lightsteelblue', linewidth=12,label = r'Full array',  alpha =.6)

        ax_leg.plot([],[],'r-',label = r'median of $T_1$')
        ax_leg.plot([],[],'b-',label = r'median of $T_2$')
        ax_leg.legend(loc='center', shadow=False, fontsize=textsize)#, prop={'size': 18})
        ax_leg.axis('off')

        f.tight_layout()

        if filt_in_count and filt_out_count: 
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'_zones'+'.png', dpi=300, facecolor='w', edgecolor='w', 
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        else:
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

    if type_hist == 'sidebyside':
        # plot histograms 
        f = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(nrows=1, ncols=3)
        ax1 = f.add_subplot(gs[0, 0])
        ax2 = f.add_subplot(gs[0, 1])
        ax_leg= f.add_subplot(gs[0, 2])

        # T1
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        colors = ['orange','blue']
        colors = [u'#ff7f0e', u'#1f77b4']
        x_multi = [t1_batch_filt_in, t1_batch_filt_out]
        ax1.hist(x_multi, n_bins, histtype='bar', color = colors)
        ax1.set_xlabel('$T_1$ [°C]', fontsize=textsize)
        ax1.set_ylabel('frequency', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)

        if filt_in_count:
            (mu, sigma) = norm.fit(t1_batch_filt_in)
            med = np.median(t1_batch_filt_in)
            try:
                y = mlab.normpdf(bins, mu, sigma)
            except:
                #y = stats.norm.pdf(bins, mu, sigma)
                pass
            #ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3, linewidth=3)
            ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        # T2
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        colors = ['orange','blue']
        colors = [u'#ff7f0e', u'#1f77b4']
        x_multi = [t2_batch_filt_in, t2_batch_filt_out]
        ax2.hist(x_multi, n_bins, histtype='bar', color = colors)
        ax2.set_xlabel('$T_2$ [°C]', fontsize=textsize)
        ax2.set_ylabel('frequency', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)

        if filt_in_count:
            (mu, sigma) = norm.fit(t2_batch_filt_in)
            med = np.median(t2_batch_filt_in)
            try:
                y = mlab.normpdf(bins, mu, sigma)
            except:
                #y = stats.norm.pdf(bins, mu, sigma)
                pass
            #ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3, linewidth=3)
            ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        #ax_leg.bar([],[],[], alpha =.9, color = 'darkorange', edgecolor = 'w', label = 'active zone',zorder = 3)
        # active zone
        colors = [u'#ff7f0e', u'#1f77b4']
        ax_leg.plot([],[], c = colors[0], linewidth=7, label = r' Infield', alpha =1.)
        # cooling zone
        ax_leg.plot([],[], c = colors[1], linewidth=7, label = r' Outfield', alpha =1.)

        ax_leg.plot([],[],' ',label = r'med : median of infield')
        ax_leg.plot([],[],' ',label = r'$\mu$ : mean of infield')
        ax_leg.plot([],[],' ',label = r'$\sigma$ : std. dev. of infield')
        #ax_leg.plot([],[],'r--',label = r'median of $z_1$')
        #ax_leg.plot([],[],'b--',label = r'median of $z_2$')

        ax_leg.legend(loc='center', shadow=False, fontsize=textsize)#, prop={'size': 18})
        ax_leg.axis('off')
        f.tight_layout()

        if filt_in_count and filt_out_count: 
            if not multi_samples:
                plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'_zones_side_by_side'+'.png', dpi=300, facecolor='w', edgecolor='w', 
                    orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            if multi_samples:
                plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'_zones_side_by_side_multi_samples'+'.png', dpi=300, facecolor='w', edgecolor='w', 
                    orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

        else:
            if not multi_samples: 
                plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            if multi_samples:
                plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'_multi_samples.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

    if type_hist == 'infield':
        # plot histograms 
        f = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        ax1 = f.add_subplot(gs[0, 0])
        ax2 = f.add_subplot(gs[0, 1])

        # T1
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        if multi_samples:
            n_bins = int(.5*np.sqrt(len(t1_batch_filt_in)))#15
        colors = ['orange','blue']
        colors = [u'#ff7f0e', u'#1f77b4']
        ax1.hist(t1_batch_filt_in, n_bins, histtype='bar', color = colors[0], edgecolor='#E6E6E6')
        ax1.set_xlabel('$T_1$ [°C]', fontsize=textsize)
        ax1.set_ylabel('frequency', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)

        if filt_in_count:
            (mu, sigma) = norm.fit(t1_batch_filt_in)
            med = np.median(t1_batch_filt_in)
            try:
                y = mlab.normpdf(bins, mu, sigma)
            except:
                #y = stats.norm.pdf(bins, mu, sigma)
                pass
            #ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3, linewidth=3)
            ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        # T2
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        if multi_samples:
            n_bins = int(.5*np.sqrt(len(t2_batch_filt_in)))#15
        colors = ['orange','blue']
        colors = [u'#ff7f0e', u'#1f77b4']
        #x_multi = [t2_batch_filt_in, t2_batch_filt_out]
        ax2.hist(t2_batch_filt_in, n_bins, histtype='bar', color = colors[1], edgecolor='#E6E6E6')
        ax2.set_xlabel('$T_2$ [°C]', fontsize=textsize)
        ax2.set_ylabel('frequency', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)

        if filt_in_count:
            (mu, sigma) = norm.fit(t2_batch_filt_in)
            med = np.median(t2_batch_filt_in)
            try:
                y = mlab.normpdf(bins, mu, sigma)
            except:
                #y = stats.norm.pdf(bins, mu, sigma)
                pass
            #ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3, linewidth=3)
            ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        f.tight_layout()

        if filt_in_count: 
            if not multi_samples:
                plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'_infield'+'.png', dpi=300, facecolor='w', edgecolor='w', 
                    orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            if multi_samples:
                plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_nwells_'+str(len(wells_objects))+'_infield_multi_samples'+'.png', dpi=300, facecolor='w', edgecolor='w', 
                    orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)


def histogram_T1_T2_Tgrad_Hflux(wells_objects, bounds = None):
    """
    filt_in_count (or filt_out_count) : file of countour (i.e. WT resisitvity boundary)
    """
    t1_batch = []
    t2_batch = []
    gt_batch = []
    hf_batch = []

    ## load pars
    for wl in wells_objects:
        try:
            if not wl.no_temp:
                aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_z1_z2.txt')
                wl.z1_pars = [aux[0],aux[1]]
                wl.z2_pars = [aux[2],aux[3]]

                aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt')
                wl.T1_pars = [aux[0],aux[1]]
                wl.T2_pars = [aux[2],aux[3]]

                aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_TG_TC_HF.txt')
                wl.thermal_grad = aux[0]
                wl.thermal_cond = aux[1]
                wl.heat_flux = aux[2]

                if bounds:
                    #bounds = [-700,95.] # [depth, temp]
                    if (wl.T2_pars[0]>bounds[1] and -1*(wl.z1_pars[0] + wl.z2_pars[0]) > bounds[0]):
                        t1_batch.append(wl.T1_pars[0])
                        t2_batch.append(wl.T2_pars[0])
                        gt_batch.append(wl.thermal_grad)
                        hf_batch.append(wl.heat_flux)
                else:
                    t1_batch.append(wl.T1_pars[0])
                    t2_batch.append(wl.T2_pars[0])
                    gt_batch.append(wl.thermal_grad)
                    hf_batch.append(wl.heat_flux)
        except:
            pass
            

    if True:
        # plot histograms 
        f = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(nrows=2, ncols=2)
        ax1 = f.add_subplot(gs[0, 0])
        ax2 = f.add_subplot(gs[0, 1])
        ax3 = f.add_subplot(gs[1, 0])
        ax4 = f.add_subplot(gs[1, 1])
        #ax_leg= f.add_subplot(gs[0, 2])
        # draw solid white grid lines
        plt.grid(color='w', linestyle='solid')

        # T1
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        colors = [u'#ff7f0e', u'#1f77b4']
        x_multi = [t1_batch]
        ax1.hist(x_multi, n_bins, histtype='bar', edgecolor='#E6E6E6', color=pale_red_col)
        ax1.set_xlabel('$T_1$ [°C]', fontsize=textsize)
        ax1.set_ylabel('freq.', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)

        (mu, sigma) = norm.fit(t1_batch)
        med = np.median(t1_batch)
        ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        # T2
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        colors = [u'#ff7f0e', u'#1f77b4']
        x_multi = [t2_batch]
        ax2.hist(x_multi, n_bins, histtype='bar', edgecolor='#E6E6E6', color=pale_red_col)
        ax2.set_xlabel('$T_2$ [°C]', fontsize=textsize)
        ax2.set_ylabel('freq.', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)

        (mu, sigma) = norm.fit(t2_batch)
        med = np.median(t2_batch)
        ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)


        # TG: thermal gradient
        # Make a multiple-histogram of data-sets with different length.
        n_bins = 15
        colors = [u'#ff7f0e', u'#1f77b4']
        x_multi = [gt_batch]
        ax3.hist(x_multi, n_bins, histtype='bar', edgecolor='#E6E6E6', color=pale_red_col)
        ax3.set_xlabel('Thermal Gradient [°C/m]', fontsize=textsize)
        ax3.set_ylabel('freq.', fontsize=textsize)
        ax3.grid(True, which='both', linewidth=0.1)

        (mu, sigma) = norm.fit(gt_batch)
        med = np.median(gt_batch)
        ax3.set_title('$med$:{:3.3f}, $\mu$:{:3.3f}, $\sigma$: {:2.2f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        # HF: heat flux
        # Make a multiple-histogram of data-sets with different length.

        n_bins = 15
        colors = [u'#ff7f0e', u'#1f77b4']
        x_multi = [hf_batch]
        ax4.hist(x_multi, n_bins, histtype='bar', edgecolor='#E6E6E6', color=pale_red_col)
        ax4.set_xlabel(r'Heat Flux [W/m$^2$]', fontsize=textsize)
        ax4.set_ylabel('freq.', fontsize=textsize)
        ax4.grid(True, which='both', linewidth=0.1)

        (mu, sigma) = norm.fit(hf_batch)
        med = np.median(hf_batch)
        ax4.set_title('$med$:{:3.3f}, $\mu$:{:3.3f}, $\sigma$: {:2.2f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)

        # #ax_leg.bar([],[],[], alpha =.9, color = 'darkorange', edgecolor = 'w', label = 'active zone',zorder = 3)
        # # active zone
        # colors = [u'#ff7f0e', u'#1f77b4']
        # ax_leg.plot([],[], c = colors[0], linewidth=7, label = r' Infield', alpha =1.)
        # # cooling zone
        # ax_leg.plot([],[], c = colors[1], linewidth=7, label = r' Outfield', alpha =1.)

        # ax_leg.plot([],[],' ',label = r'med : median of infield')
        # ax_leg.plot([],[],' ',label = r'$\mu$ : mean of infield')
        # ax_leg.plot([],[],' ',label = r'$\sigma$ : std. dev. of infield')
        # #ax_leg.plot([],[],'r--',label = r'median of $z_1$')
        # #ax_leg.plot([],[],'b--',label = r'median of $z_2$')

        # ax_leg.legend(loc='center', shadow=False, fontsize=textsize)#, prop={'size': 18})
        # ax_leg.axis('off')
        f.tight_layout()

        plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_T1_T2_TG_HF_nwells_'+str(len(wells_objects))+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

