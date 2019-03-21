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
#from scipy.optimize import curve_fit
#import corner, emcee
import time
from lib_MT_station import *
from lib_Well import *
from lib_mcmc_MT_inv import * 
from Maping_functions import coord_dms2dec
from matplotlib.backends.backend_pdf import PdfPages

textsize = 15.

# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
	# PC that the code will be be run ('ofiice', 'personalSuse', 'personalWin')
	pc = 'office'
	#pc = 'personalSuse'
	#pc = 'personalWin'
	# (0) Import data and create objects: MT from edi files and wells from spreadsheet files
	if True:
		#### Import data: MT from edi files and wells from spreadsheet files
		## MT
		if pc == 'office':
			#path_files = "D:\kk_full\*.edi" 	# Whole array 
			#path_files = "D:\kk_sample\*.edi"  # Sample of stations
			path_files = "D:\kk_1\*.edi"		# One station
		if pc == 'personalSuse':
			## Data paths for personal's pc SUSE (uncommend the one to use)
			path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\1_sta\\*.edi"
			#path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/EDI_files_sample/*.edi"
			#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\*.edi"
		if pc == 'personalWin':		
			## Data paths for personal's pc WINDOWS (uncommend the one to use)
			path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\1_sta\\*.edi"
			#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\sample\\*.edi"
			#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\*.edi"

		## Temperature in wells 
		if pc == 'office':
		## Office's PC
			path_wells_loc = "D:\Wairakei_Tauhara_data\Temp_wells\well_location_latlon.txt"
			path_wells_temp = "D:\Wairakei_Tauhara_data\Temp_wells\well_depth_redDepth_temp.txt" 
				# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
		## Personal 

		## Temperature in wells 
		if pc == 'office': 
			path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data.txt"
			# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
		
		## Create a directory of the name of the files of the stations
		pos_ast = path_files.find('*')
		file_dir = glob.glob(path_files)

		######################################################################################
		## Create station objects 
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
			count  += 1

		######################################################################################	
		# Create wells objects
		## Import wells data:
		## import temperature profiles
		wl_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red = \
			read_well_temperature(path_wells_temp)
		# Note: dir_no_depth_red contain a list of wells with no information of reduced depth
		
		## import meb profiles
		wl_name_meb, wl_prof_depth_meb, wl_prof_meb = read_well_meb(path_wells_meb)

		## Recover location for wells from path_wells_loc
		wells_location = read_well_location(path_wells_loc)
		# Note: wells_location = [[wl_name1,lat1,lon1,elev1],...] list of arrays
		
		## Loop over the wells to create objects and assing data attributes from temp. files 
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
			count  += 1
	
		## Loop wells_objects (list) to assing data attributes from MeB files 
		# list of wells with MeB (names)
		wells_meb = []
		for wl in wells_objects: 
			if wl.name in wl_name_meb:
				idx = wl_name_meb.index(wl.name)
				wl.meb_prof = wl_prof_meb[idx]
				wl.meb_depth = wl_prof_depth_meb[idx]
				wells_meb.append(wl.name)
		# save to .txt names of wells with MeB content 
		f = open("wells_MeB_list.txt", "w")
		f.write('# Wells with MeB data available\n')
		f.write('# Total: {:}\n'.format(len(wells_meb)))
		for wl in wells_meb:
			f.write(wl+'\n')
		f.close()
		
	# (1) Calculate priors for each station (using two nearest) 

	
	# (2) Run MCMC inversion for each staion, obtaning 1D 3L res. model
	# 	  Sample posterior, construct uncertain resistivity distribution and create result plots 
	if True:
		## create pdf file to save the fit results for the whole inversion 
		pp = PdfPages('fit.pdf')

		start_time = time.time()
		for sta_obj in station_objects: 
			print('({:}/{:}) Running MCMC inversion:\t'.format(sta_obj.ref+1,len(station_objects))+sta_obj.name[:-4])
			## range for the parameters
			par_range = [[1.*1e2,2*1e3],[1.*1e1,1*1e3],[1.*1e2,5.*1e3],[1.*1e-3,1.*1e4],[1.*1e1,1.*1e3]]
			## create object mcmc_inv 
			mcmc_sta = mcmc_inv(sta_obj, prior='uniform', prior_input=par_range)
			## run inversion 
			mcmc_sta.inv()
			## plot results (save in .png)
			mcmc_sta.plot_results_mcmc()
			## sample posterior
			#mcmc_sta.sample_post()
			f = mcmc_sta.sample_post(exp_fig = True) # Figure with fit to be add in pdf (whole station)
			pp.savefig(f)
			plt.close("all")
			## calculate estimate parameters
			mcmc_sta.model_pars_est()
			## sum one to index
		## enlapsed time for the inversion (every station in station_objects)
		enlap_time = time.time() - start_time # enlapsed time
		## print time consumed
		print("Time consumed:\t{:.1f} min".format(enlap_time/60))
		pp.close()
	
	# (3) Construct uncertain distribution of temperature 

