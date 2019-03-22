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
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================
# Wells class
# ==============================================================================

class Wells(object):
    """
    # This class is for wells
    # ===================== =====================================================
    # Methods               Description
    # ===================== =====================================================
	# 	                           
	# ==================== ========================================== ==========
    # Attributes            Description                                default
    # ===================== ========================================== ==========
	# name				    extracted from the name of the edi file
    # ref		 			reference number in the code
	# path					path to the edi file
	#
	# lat					latitud	in dd:mm:ss
    # lon					longitud in dd:mm:ss
	# lat_dec				latitud	in decimal
    # lon_dec				longitud in decimal	
	# elev					topography (elevation of the station)
	#
	# depth				 	depths for temperature profile
	# red_depth				reduced depths for temperature profile
	# depth_dev				depths deviation
	# temp_prof_true		temperature profile 
	#
	# temp_prof				resample true temperaure profile
	# betas					beta value for each layer
	# slopes				slopes values for each layer
	#
    # meb                   Methylene-blue (MeB) data available         False
	# meb_prof				methylene-blue (MeB) profiles
	# meb_depth				methylene-blue (MeB) depths (samples)
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
        self.temp_prof = None		# True temperaure profile resample
        self.betas = None			# beta value for each layer
        self.slopes = None			# slopes values for each layer
		# MeB profile
        self.meb = False            # Methylene-blue (MeB) data available
        self.meb_prof = None		# methylene-blue (MeB) profile
        self.meb_depth = None		# methylene-blue (MeB) depths (samples)
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
        ax1.set_xlabel('MeB [?]', fontsize=18)
        ax1.set_ylabel('Depth [m]', fontsize=18)
        ax1.grid(True, which='both', linewidth=0.4)
        ax1.invert_yaxis()
        return f


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
