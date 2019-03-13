"""
.. module:: Wairakei/Tauhara MT inversion and Temp. extrapolation
   :synopsis: Forward and inversion ot MT using MCMC 1D constrain inversion.
			  Extrapolation from temperature profiles in wells.
			  Estimate Temperature distribution. 

.. moduleauthor:: Alberto Ardid 
				  University of Auckland 
				  
.. conventions:: 
	:order in impedanze matrix [xx,xy,yx,yy]
	: number of layer 3 (2 layers + half-space)
"""
# ==============================================================================
#  Imports
# ==============================================================================

import numpy as np
import glob
from matplotlib import pyplot as plt
import traceback, os, sys, shutil
from multiprocessing import Pool
from scipy.optimize import curve_fit
import corner, emcee
import time
from lib_MT_station import *
from lib_Well import *
from lib_mcmc_MT_inv import * 
from Maping_functions import coord_dms2dec

textsize = 15.

# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
	
	# (0) Import data and create objects: MT from edi files and wells from spreadsheet files
	if True:
		#### Import data: MT from edi files and wells from spreadsheet files
		## MT
		#path_files = "D:\kk_full\*.edi" 	# Whole array 
		#path_files = "D:\kk_sample\*.edi"  # Sample of stations
		path_files = "D:\kk_1\*.edi"		# One station
		
		## Data paths for personal's pc (uncommend the one to use)
		#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\1_sta\\*.edi"
		#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\sample\\*.edi"
		#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\*.edi"
		
		## Temperature in wells 
		## Office's PC
		path_wells_loc = "D:\Wairakei_Tauhara_data\Temp_wells\well_location_latlon.txt"
		path_wells_temp = "D:\Wairakei_Tauhara_data\Temp_wells\well_depth_redDepth_temp.txt" 
			# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
		## Personal 
		
		## Create a directory of the name of the files of the stations
		pos_ast = path_files.find('*')
		file_dir = glob.glob(path_files)
		
		#### Create station and well objects 
		## Loop over the file directory to collect the data, create station objects and fill them
		station_objects = []   # list to be fill with station objects
		count  = 0
		for file_aux in file_dir:
			file = file_aux[pos_ast:] # name on the file
			
			## 1. read edi file: H contains location and Z the impedanse tensor
			#[H, Z, T, Z_rot] = read_edi('C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\'+file) # Personal
			#[H, Z, T, Z_rot] = read_edi('D:\\kk_full\\'+file) # office

			sta_obj = Station(file, count, path_files)
			sta_obj.read_edi_file() 
			sta_obj.rotate_Z()
			sta_obj.app_res_phase()
			## Create station objects and fill them
			station_objects.append(sta_obj)
		
		## Create wells objects
		# Import wells data:
		wl_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red = \
			read_well_temperature(path_wells_temp)
		# Note: dir_no_depth_red contain a list of wells with no information of reduced depth
		
		## Recover location for wells from path_wells_loc
		wells_location = read_well_location(path_wells_loc)
		# Note: wells_location = [[wl_name1,lat1,lon1,elev1],...] list of arrays
		
		## Loop over the wells to create objects and assing data attributes 
		wells_objects = []   # list to be fill with station objects
		count  = 0
		for wl in wl_name: 
			wl_obj = Wells(wl, count)
			# load data attributes
			wl_obj.depth = wl_prof_depth[count]
			wl_obj.red_depth = wl_prof_depth_red[count]
			wl_obj.temp_prof_true = wl_prof_temp[count]
			# Search for location of the well and add to attributes
			#if wl in wells_location[0][:]:
			
			# add well object to directory of well objects
			wells_objects.append(wl_obj)

		
	# (1) Calculate priors for each station (using two nearest) 

	
	# (2) Run MCMC inversion for each staion, obtaning 1D 3L res. model 
	if True:
		print('Running MCMC inversion')
		start_time = time.time()
		for sta_obj in station_objects: 
			print(sta_obj.name[:-4])
			 
			par_range = [[100,1000],[100,1000],[1e2,5*1e3],[1e-3,2*1e1],[1e2,5*1e3]]
			#mcmc_sta = mcmc_inv(sta_obj)
			mcmc_sta = mcmc_inv(sta_obj, prior='uniform', prior_input=par_range)
			mcmc_sta.inv()
			#print(mcmc_sta.time)
			mcmc_sta.plot_results()
		enlap_time = time.time() - start_time # enlapsed time
		print("Time consumed: {:.1f} s".format(enlap_time))
	# (4) Sample posterior and construct uncertain resistivity distribution
	if True: 
		print('Sampling posterior')
		start_time = time.time()
		for sta_obj in station_objects:
			mcmc_sta.sample_post()
			mcmc_sta.model_pars_est()
	
	# (5) Construct uncertain distribution of temperature 

