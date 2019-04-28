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
from lib_mcmc_meb import * 
from Maping_functions import *
from misc_functios import *
from matplotlib.backends.backend_pdf import PdfPages

textsize = 15.

# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
	## PC that the code will be be run ('ofiice', 'personalSuse', 'personalWin')
	#pc = 'office'
	pc = 'personalSuse'
	#pc = 'personalWin'
	## Folder to be used (1 edi, sample of edis, full array)
	set_up = True
	mcmc_meb_inv = False
	prior_MT_meb_read = True
	mcmc_MT_inv = True
	prof_2D_MT = True

	# (0) Import data and create objects: MT from edi files and wells from spreadsheet files
	if set_up:
		#### Import data: MT from edi files and wells from spreadsheet files
		#########  MT data
		if pc == 'office': 
			#path_files = "D:\workflow_data\kk_1\*.edi"		# One station
			#path_files = "D:\workflow_data\kk_sample\*.edi"  # Sample of stations
			#path_files = "D:\workflow_data\kk_full\*.edi" 	# Whole array 
			#path_files = "D:\workflow_data\profile_2_ext\*.edi" 	# 2D profile 
			path_files = "D:\workflow_data\profile_WRKNW6\*.edi" 	# 2D profile 
			#path_files = "D:\workflow_data\MT_near_well_WK317\*.edi" 	# Stations near well WK317

		## Data paths for personal's pc SUSE (uncommend the one to use)
		if pc == 'personalSuse':
			#path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/EDI_files_1sta/*.edi" # One station
			#path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/EDI_files_sample/*.edi" # Sample of stations
			#path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/EDI_files/*.edi" # Whole array 
			#path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/profile_2D/*.edi" 	# 2D profile 
			path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/profile_WRKNW6/*.edi" 	# 2D profile 
			#path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/profile_WRKNW6_short/*.edi" 	# 2D profile 

		## Data paths for personal's pc WINDOWS (uncommend the one to use)
		if pc == 'personalWin':
			path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\1_sta\\*.edi"
			#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\sample\\*.edi"
			#path_files = "C:\\Users\\ajara\\Desktop\\EDI_Files_WT\\*.edi"

		####### Temperature in wells data
		if pc == 'office':
			path_wells_loc = "D:\Wairakei_Tauhara_data\Temp_wells\well_location_latlon.txt"
			path_wells_temp = "D:\Wairakei_Tauhara_data\Temp_wells\well_depth_redDepth_temp.txt" 
			# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
		## Personal Suse
		if pc == 'personalSuse':
			path_wells_loc = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/wells_loc.txt"
			path_wells_temp = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/well_depth_redDepth_temp.txt"
		## Personal Windows	
		if pc == 'personalWin':
			path_wells_loc = " "
			path_wells_temp = " "

		####### MeB data in wells 
		## 
		if pc == 'office': 
			path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data.txt"
			#path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data_sample.txt"	
			
			# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
		## Personal Suse
		if pc == 'personalSuse':
			#path_wells_meb = "/home/aardid/Documentos/data/Wairakei_Tauhara/MeB_wells/MeB_data.txt"
			#path_wells_meb = "/home/aardid/Documentos/data/Wairakei_Tauhara/MeB_wells/MeB_data_sample_4.txt"
			path_wells_meb = "/home/aardid/Documentos/data/Wairakei_Tauhara/MeB_wells/MeB_data_prof_WRKNW6.txt"
		## Personal Win
		if pc == 'personalWin':
			path_wells_meb = " "

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
		## Create wells objects
		# # Import wells data:
		wl_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red = \
		 	read_well_temperature(path_wells_temp)
		# # Note: dir_no_depth_red contain a list of wells with no information of reduced depth
	
		# ## Recover location for wells from path_wells_loc
		wells_location = read_well_location(path_wells_loc)
		# # Note: wells_location = [[wl_name1,lat1,lon1,elev1],...] list of arrays

		# ## Recover MeB data for wells from path_wells_meb
		wl_name_meb, wl_prof_depth_meb, wl_prof_meb = read_well_meb(path_wells_meb)

		# ## Loop over the wells to create objects and assing data attributes 
		wells_objects = []   # list to be fill with station objects
		count  = 0
		for wl in wl_name: 
			wl_obj = Wells(wl, count)
			# load data attributes
			wl_obj.depth = wl_prof_depth[count]
			wl_obj.red_depth = wl_prof_depth_red[count]
			wl_obj.temp_prof_true = wl_prof_temp[count]
			## add well object to directory of well objects
			wells_objects.append(wl_obj)
			count  += 1
		
		# Search for location of the well and add to attributes
		for wl in wells_objects:
			for i in range(len(wells_location)): 
				wl_name = wells_location[i][0]
				if wl.name == wl_name: 
					wl.lat_dec = wells_location[i][2]
					wl.lon_dec = wells_location[i][1]
					wl.elev = wells_location[i][3]

		## Loop wells_objects (list) to assing data attributes from MeB files 
		# list of wells with MeB (names)
		wells_meb = []
		count_meb_wl = 0
		for wl in wells_objects: 
			if wl.name in wl_name_meb:
				idx = wl_name_meb.index(wl.name)
				wl.meb = True
				wl.meb_prof = wl_prof_meb[idx]
				wl.meb_depth = wl_prof_depth_meb[idx]
				count_meb_wl+=1
				#wells_meb.append(wl.name)
		## create text file for google earth
		#list_meb_wells = [obj for obj in wells_objects if obj.meb] 
		#for_google_earth(list_meb_wells, name_file = 'meb_wells_google_earth.txt', type_obj = 'well')

		# # save to .txt names of wells with MeB content 
		# f = open("wells_MeB_list.txt", "w") # text file to save names of meb wells 
		# f.write('# Wells with MeB data available\n')
		# f.write('# Total: {:}\n'.format(len(wells_meb)))
		# for wl in wells_meb: # loop over the meb wells (objects)
		# 	f.write(wl+'\n')
		# f.close()
		## plot MeB curves 
		# pp = PdfPages('wells_MeB.pdf') # pdf to plot the meb profiles
		# for wl in wells_objects:
		# 	if wl.meb: 
		# 		f = wl.plot_meb_curve()
		# 		pp.savefig(f)
		# 		plt.close("all")
		# pp.close()
		
	# (1) Run MCMC for MeB priors  
	if mcmc_meb_inv:
		pp = PdfPages('fit.pdf')
		start_time = time.time()
		count = 1
		print("(1) Run MCMC for MeB priors")
		for wl in wells_objects:
			if wl.meb: 
				#if wl.name == 'WK401':
				print(wl.name +  ': {:}/{:}'.format(count, count_meb_wl))
				mcmc_wl = mcmc_meb(wl)
				mcmc_wl.run_mcmc()
				mcmc_wl.plot_results_mcmc()
				f = mcmc_wl.sample_post(exp_fig = True) # Figure with fit to be add in pdf 
				pp.savefig(f)
				plt.close("all")
				## calculate estimate parameters (percentiels)
				mcmc_wl.model_pars_est()
				count += 1
		## enlapsed time for the inversion (every station in station_objects)
		enlap_time = time.time() - start_time # enlapsed time
		## print time consumed
		print("Time consumed:\t{:.1f} min".format(enlap_time/60))
		pp.close()
		# move figure fit to global results folder
		shutil.move('fit.pdf','.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'00_fit.pdf')
	
	# (2) Construct priors for MT stations
	if prior_MT_meb_read:
		# attribute in meb wells for path to mcmc results 
		for wl in wells_objects:
			if wl.meb:
				wl.path_mcmc_meb = '.'+os.sep+str('mcmc_meb')+os.sep+wl.name
		# Calculate prior values for boundaries of the cc in station
		# (prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells)
		# Function assign results as attributes for MT stations in station_objects (list)
		calc_prior_meb_quadrant(station_objects, wells_objects)

	# (2) Run MCMC inversion for each staion, obtaning 1D 3L res. model
	# 	  Sample posterior, construct uncertain resistivity distribution and create result plots 
	if mcmc_MT_inv:
		## create pdf file to save the fit results for the whole inversion 
		pp = PdfPages('fit.pdf')
		start_time = time.time()
		prior_meb = True
		for sta_obj in station_objects: 
			print('({:}/{:}) Running MCMC inversion:\t'.format(sta_obj.ref+1,len(station_objects))+sta_obj.name[:-4])

			## range for the parameters
			par_range = [[.5*1e2,.5*1e3],[1.*1e1,1*1e3],[1.*1e0,1.*1e3],[1.*1e-3,1.*1e3],[1.*1e1,1.*1e3]]
			## create object mcmc_inv 
			#mcmc_sta = mcmc_inv(sta_obj)
  			# inv_dat: weighted data to invert [1,0,1,0,0,0,0]

			mcmc_sta = mcmc_inv(sta_obj, prior='uniform', inv_dat = [1,0,1,0,0,0,0],prior_input=par_range, \
				walk_jump = 10000,prior_meb = prior_meb)
			if prior_meb:
				print("	wells for MeB prior: {} ".format(sta_obj.prior_meb_wl_names))
				#print("	[[z1_mean,z1_std],[z2_mean,z2_std]] = {} \n".format(sta_obj.prior_meb))
				#print("	distances = {} \n".format(sta_obj.prior_meb_wl_dist)) 
	
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

		## enlapsed time for the inversion (every station in station_objects)
		enlap_time = time.time() - start_time # enlapsed time
		## print time consumed
		print("Time consumed:\t{:.1f} min".format(enlap_time/60))
		pp.close()
		# move figure fit to global results folder
		shutil.move('fit.pdf','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'00_fit.pdf')

		## create text file for google earth, containing names of MT stations considered 
		for_google_earth(station_objects, name_file = '00_stations_4_google_earth.txt', type_obj = 'Station')
		shutil.move('00_stations_4_google_earth.txt','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion' \
			+os.sep+'00_stations_4_google_earth.txt')
	
	# (3) Construct uncertain distribution of temperature
	if prof_2D_MT:
		# load mcmc results and assign to attributes of pars to station attributes 
		load_sta_est_par(station_objects)
		# Create figure of unceratain boundaries of the clay cap and move to mcmc_inversions folder
		file_name = 'z1_z2_uncert'
		#plot_2D_uncert_bound_cc(station_objects, pref_orient = 'EW', file_name = file_name)
		plot_2D_uncert_bound_cc_mult_env(station_objects, pref_orient = 'EW', file_name = file_name, 
			width_ref = '60%', prior_meb = wells_objects)#, plot_some_wells = ['WK404'])#,'WK401','WK402'])
		shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+os.sep+file_name+'.png')
				


