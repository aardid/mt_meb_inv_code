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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from Maping_functions import*

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
	
	lat					latitud	in dd:mm:ss
    lon					longitud in dd:mm:ss
	lat_dec				latitud	in decimal
    lon_dec				longitud in decimal	
	elev					topography (elevation of the station)
	
	depth				 	depths for temperature profile
	red_depth				reduced depths for temperature profile
	depth_dev				depths deviation
	temp_prof_true		temperature profile 
	
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
    z1_pars                 distribution parameters for layer 1 (normal distribution)
                            thickness (model parameter) calculated 
                            from layer model of nearest MT stations : [a,b]
                            a: mean
                            b: standard deviation 
    z1_pars                 distribution parameters for layer 2 (normal distribution)
                            thickness (model parameter) calculated 
                            from layer model of nearest MT stations : [a,b]
                            a: mean
                            b: standard deviation 

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
        self.depth = None		     # Depths for temperature profile
        self.red_depth = None		 # Reduced depths for temperature profile
        self.depth_dev = None		 # Depths deviation
        self.temp_prof_true = None 	 # Temperature profile 
		# Temperature estimates
        self.path_temp_est = None   # path to temp. profiles estimation results, where sample profiles are (path to file?)
        self.temp_prof_rs = None	# True temperaure profile resample
        self.betas = None			# beta value for each layer
        self.slopes = None			# slopes values for each layer
		# MeB profile
        self.meb = False            # Methylene-blue (MeB) data available
        self.meb_prof = None		# methylene-blue (MeB) profile
        self.meb_depth = None		# methylene-blue (MeB) depths (samples)
        self.meb_z1_pars = None     #  
        self.meb_z2_pars = None     #  
        # layer model from MT nearest stations
        self.layer_mod_MT_names = None 
        self.layer_mod_MT_dist  = None
        self.z1_pars = None
        self.z2_pars = None
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

    def plot_temp_profile(self):
        f,(ax1) = plt.subplots(1,1)
        f.set_size_inches(6,8)
        #f.suptitle(self.name, fontsize=22, y=1.08)
    
        ax1.set_xscale("linear")
        ax1.set_yscale("linear")    
        ax1.plot(self.temp_prof_true,self.red_depth,'o')
        #ax1.plot(temp_aux,depth_aux,'b--', linewidth=0.05)
        #ax1.set_xlim([np.min(periods), np.max(periods)])
        #ax1.set_xlim([0, 340])
        #ax1.set_ylim([0,3000])
        ax1.set_xlabel('Temperature [deg C]', fontsize=18)
        ax1.set_ylabel('Depth [m]', fontsize=18)
        ax1.grid(True, which='both', linewidth=0.4)
        #ax1.invert_yaxis()
        plt.title(self.name, fontsize=22,)
        plt.tight_layout()
        return f

    def temp_prof_est(self):
        """
        Calculate estimated temperature profiles by fitting heat equation by layer. 
        As the boundaries of three layer model are distributions, here a distributions
        of profiles are calculated. 
        Beta parameters (See Bredehoeft, J. D., and I. S. Papadopulos (1965)) are fit
        for each layer (3) layers in a temperature profile. 
        
        Attributes assigned:
        - self.T_est : temperature profile
        - self.beta_3l: array of beta for three layers [b1,b2,b3]
        - self.Tmin_3l
        - self.Tmax_3l
        - self.slop_3l
        """
        # directory to save results
        self.path_temp_est = '.'+os.sep+'temp_prof'+os.sep+'wells'+os.sep+self.name
        if self.path_temp_est: 
            pass
        else:
            os.mkdir(self.path_temp_est)
        ## number of samples 
        Ns = 300
        ## figure of Test samples
        # f,(ax1,ax2,ax3) = plt.subplots(1,3)
        # f.set_size_inches(12,4)
        # f.suptitle(self.name, fontsize=12, y=.995)
        # f.tight_layout()
        ## text file of Test samples 
        #t = open(self.path_temp_est+os.sep+"temp_est_samples.dat", "w")
        ## create sample of z1 (boundary 1) and z2 (boundary 2) for well position
        ## z1 and z2 are thicknesses of layer one ans two
        s_z1 = np.random.normal(self.z1_pars[0], self.z1_pars[1], Ns) # 
        s_z2 = np.random.normal(self.z2_pars[0], self.z2_pars[1], Ns) # 
        for z1,z2 in zip(s_z1,s_z2):
            # for each sample: 
            # define spatial boundary conditions for heat equation: [z1_min, z2_min, z3_min]
            Zmin = [self.elev, self.elev - z1 , self.elev - (z1+z2)]
            Zmax = [Zmin[1] , Zmin[2], self.red_depth[-1]]
            # Calculate beta and estimated temp profile

            #### This function needs to operate with a resample temp. profile version:
            #   - Next step: cubic interpolation of temp. profiles. 
            Test, beta, Tmin, Tmax, slopes = T_beta_est(self.temp_prof_true, self.depth_dev, Zmin, Zmax) # 
            print(beta)
            asdf
            # add output parameters to a text file
             

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
                    line2[2] = float('NaN') # define empty value as NaN 
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
            # values for mean a std for normal distribution representing the prior
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
        
def find_nearest(array, value):
    """
    Find nearest to value in an array
    
    Input:
    - array: numpy array to look in
    - value: value to search
    
    Output:
    - array[idx]: closest element in array to value

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def Texp2(z,Zmax,Zmin,Tmin,Tmax,beta): 
    """
    Calculate temp. profile based on model 1D heat transfer model (See Bredehoeft, J. D., and I. S. Papadopulos (1965))

    Input:
    - z: depths to calculate
    - Zmax,Zmin,Tmin,Tmax: boundary conditions
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
    inds_z = np.where(z == find_nearest(z, Zmin[0]))
    inds_z_l1_top = int(inds_z[0][0])
    Tmin_l1 = Tw[inds_z_l1_top]
    
    inds_z = np.where(z == find_nearest(z, Zmax[0]))
    inds_z_l1_bot = int(inds_z[0][0])
    Tmax_l1 = Tw[inds_z_l1_bot]
    
    # Tmin y T max for layer 2
    inds_z = np.where(z == find_nearest(z, Zmin[1]))
    inds_z_l2_top = int(inds_z[0][0])
    Tmin_l2 = Tw[inds_z_l2_top]
    
    inds_z = np.where(z == find_nearest(z, Zmax[1]))
    inds_z_l2_bot = int(inds_z[0][0])
    Tmax_l2 = Tw[inds_z_l2_bot]
    
    # Tmin y T max for hs
    inds_z = np.where(z == find_nearest(z, Zmin[2]))
    inds_z_l3_top = int(inds_z[0][0])
    Tmin_l3 = Tw[inds_z_l3_top]
    inds_z = np.where(z == find_nearest(z, Zmax[2]))
    inds_z_l3_bot = int(inds_z[0][0])
    Tmax_l3 = Tw[inds_z_l3_bot]
    
    # T boundary condition 
    Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
    Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]

    # Fit Twell with Texp
    beta_range = np.arange(-30.0, 30.0, 0.5)
    beta_def = -2.5
    #print(beta_range)
    
    ### Layer 1
    # Calculate beta that best fot the true temp profile 
    popt, pcov = curve_fit(Texp2, z[inds_z_l1_bot:inds_z_l1_top+1], Tw[inds_z_l1_bot:inds_z_l1_top+1], p0=[Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta_def], bounds=([Zmax[0]-1.,Zmin[0]-1.,Tmin[0]-1,Tmax[0]-1., beta_range[0]], [Zmax[0]+1.,Zmin[0]+1.,Tmin[0]+1.,Tmax[0]+1,beta_range[-1]]))
    
    beta_opt_l1 = popt[-1]
    Test_l1 = Texp2(z[inds_z_l1_bot:inds_z_l1_top+1],Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta_opt_l1)

    ### Layer 2
    # Calculate beta that best fot the true temp profile 
    popt, pcov = curve_fit(Texp2, z[inds_z_l2_bot:inds_z_l2_top+1], Tw[inds_z_l2_bot:inds_z_l2_top+1], p0=[Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta_def], bounds=([Zmax[1]-1.,Zmin[1]-1.,Tmin[1]-1,Tmax[1]-1., beta_range[0]], [Zmax[1]+1.,Zmin[1]+1.,Tmin[1]+1.,Tmax[1]+1,beta_range[-1]]))
    
    beta_opt_l2 = popt[-1]
    Test_l2 = Texp2(z[inds_z_l2_bot:inds_z_l2_top+1],Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta_opt_l2)

    # layer 3
    # Calculate beta that best fot the true temp profile 
    popt, pcov = curve_fit(Texp2, z[inds_z_l3_bot:inds_z_l3_top+1], Tw[inds_z_l3_bot:inds_z_l3_top+1], p0=[Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta_def], bounds=([Zmax[2]-1.,Zmin[2]-1.,Tmin[2]-1,Tmax[2]-1., beta_range[0]], [Zmax[2]+1.,Zmin[2]+1.,Tmin[2]+1.,Tmax[2]+1,beta_range[-1]]))
    
    beta_opt_l3 = popt[-1]
    Test_l3 = Texp2(z[inds_z_l3_bot:inds_z_l3_top+1],Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta_opt_l3)
    
    # concatenate the estimated curves
    Test = np.concatenate((Test_l3[:,], Test_l2[1:], Test_l1[1:]),axis=0) 
    beta = [beta_opt_l1, beta_opt_l2, beta_opt_l3]
    slopes = [(Tmax_l1-Tmin_l1)/(Zmax[0]-Zmin[0]),(Tmax_l2-Tmin_l2)/(Zmax[1]-Zmin[1]),(Tmax_l3-Tmin_l3)/(Zmax[2]-Zmin[2])]		
    
    return Test, beta, Tmin, Tmax, slopes

            
