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
	: z1 and z2 in MT object refer to thickness of two first layers
    : z1 and z2 in results of MeB mcmc inversion refer to depth of the top and bottom boundaries of CC (second layer)
    : cc clay cap
    : distances in meters
    : MeB methylene blue
    : temperature in celcius
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
from lib_sample_data import*
from Maping_functions import *
from misc_functios import *
from matplotlib.backends.backend_pdf import PdfPages

textsize = 15.
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
	## PC that the code will be be run ('ofiice', 'personalSuse', 'personalWin')
	#pc = 'office'
	#pc = 'personalSuse'
	#pc = 'personalWin'
	pc = 'personalMac'
	# ==============================================================================
	## Set of data to work with 
	full_dataset = True
	# Profiles
	prof_WRKNW6 = False
	prof_WRKNW5 = False
	array_WRKNW5_WRKNW6 = False
	#
	prof_NEMT2 = False
	prof_THNW03 = False
	prof_THNW04 = False
	prof_THNW05 = False
	# Filter has qualitu MT stations
	filter_lowQ_data = True
	# Stations not modeled
	sta_2_re_invert = False
	# ==============================================================================
	## Sections of the code tu run
	set_up = True
	mcmc_meb_inv = False
	prior_MT_meb_read = False
	mcmc_MT_inv = False
	plot_2D_MT = False
	plot_3D_MT = False
	wells_temp_fit = False
	sta_temp_est = False
	files_paraview = True

	# (0) Import data and create objects: MT from edi files and wells from spreadsheet files
	if set_up:
		#### Import data: MT from edi files and wells from spreadsheet files
		#########  MT data
		if pc == 'office': 
			#########  MT data
			path_files = "D:\workflow_data\kk_full\*.edi" 	# Whole array 
			####### Temperature in wells data
			path_wells_loc = "D:\Wairakei_Tauhara_data\Temp_wells\well_location_latlon.txt"
			path_wells_temp = "D:\Wairakei_Tauhara_data\Temp_wells\well_depth_redDepth_temp.txt"
			path_wells_temp_date = "D:\Wairakei_Tauhara_data\Temp_wells\well_depth_redDepth_temp_date.txt" 
			# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
			####### MeB data in wells 
			path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data.txt"
			#path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data_sample.txt"	

		## Data paths for personal's pc SUSE (uncommend the one to use)
		if pc == 'personalSuse':
			#########  MT data
			path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/EDI_Files/*.edi" # Whole array 			
			####### Temperature in wells data
			path_wells_loc = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/wells_loc.txt"
			path_wells_temp = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/well_depth_redDepth_temp_fixTH12_rmTHM24_fixWK404.txt"
			path_wells_temp_date = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/well_depth_redDepth_temp_date.txt"
			####### MeB data in wells 
			path_wells_meb = "/home/aardid/Documentos/data/Wairakei_Tauhara/MeB_wells/MeB_data.txt"
	
		## Data paths for personal's pc SUSE (uncommend the one to use)
		if pc == 'personalMac':
			#########  MT data
			path_files = os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'MT_survey'+os.sep+'EDI_Files'+os.sep+'*.edi'
			# Whole array 			
			####### Temperature in wells data
			path_wells_loc = 		os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_location_latlon.txt'
			path_wells_temp = 		os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_fixTH12_rmTHM24_fixWK404.txt'
			path_wells_temp_date = 	os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_date.txt'
			####### MeB data in wells 
			path_wells_meb = 		os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'MeB_wells'+os.sep+'MeB_data.txt'

		## Create a directory of the name of the files of the stations
		pos_ast = path_files.find('*')
		file_dir = glob.glob(path_files)
		#########################################################################################
		#########################################################################################
		## Create station objects 
		# Defined lists of MT station 
		if full_dataset:
			sta2work = [file_dir[i][pos_ast:-4] for i in range(len(file_dir))]
		if sta_2_re_invert:
			# not modelled correctly: 13
			sta2work = ['WT003a', 'WT005a', 'WT008a',\
				'WT151a','WT213a', 'WT213a', 'WT300a',\
				'WT301a', 'WT081a', 'WT206b','WT209a', 'WT213a', 'WT300a']
			# bad quality data: 19
			#sta2work = ['WT030a', 'WT031a', 'WT032a', 'WT061a',\
			#	 'WT097a', 'WT141a', 'WT150a', 'WT180a', \
			#		 'WT199a', 'WT200a', 'WT202a', 'WT216a', 'WT217a',\
			#		'WT308a', 'WT323a', 'WT327a', 'WT335a', 'WT508a', 'WT014a', ]
		if prof_WRKNW6:
			sta2work = ['WT004a','WT015a','WT048a','WT091a','WT102a','WT111a','WT222a']
			sta2work = ['WT004a','WT015a','WT048a','WT091a','WT111a','WT222a']
			sta2work = ['WT091a','WT102a','WT111a','WT222a']
			#sta2work = ['WT102a']
		if prof_WRKNW5:
			sta2work = ['WT039a','WT024a','WT030a','WT501a','WT502a','WT060a','WT071a','WT068a','WT223a','WT070b','WT107a','WT111a']#,'WT033b']
			#sta2work = ['WT039a','WT024a','WT030a','WT501a','WT502a','WT060a','WT071a','WT223a','WT107a','WT111a']
			#sta2work = ['WT039a','WT024a','WT030a']
			#sta2work = ['WT039a','WT024a','WT030a','WT501a','WT502a','WT060a','WT071a','WT068a','WT223a','WT070b','WT107a','WT111a',\
			#	'WT004a','WT015a','WT048a','WT091a','WT102a','WT111a','WT222a']
		if array_WRKNW5_WRKNW6:
			sta2work = ['WT091a','WT102a','WT111a','WT222a','WT039a','WT024a','WT030a','WT501a','WT502a','WT060a','WT071a','WT068a','WT223a','WT070b','WT107a','WT111a']#,'WT033b']
		if prof_NEMT2:
			sta2work= ['WT108a','WT116a','WT145a','WT153b','WT164a','WT163a','WT183a','WT175a','WT186a','WT195a','WT197a','WT134a']
		# Tauhara profiles
		if prof_THNW03: 
			sta2work= ['WT117b','WT127a','WT132a','WT142a','WT185a']
		if prof_THNW04: 
			sta2work= ['WT140a','WT151a', 'WT177a','WT184a','WT193a','WT197a','WT134a'] # ,'WT160a'
		if prof_THNW05: 
			sta2work= ['WT179a','WT189a', 'WT200a','WT198a','WT202a','WT181a','WT206a','WT014a','WT217a','WT194a'] 
		
		#########################################################################################
		## Loop over the file directory to collect the data, create station objects and fill them
		station_objects = []   # list to be fill with station objects
		count  = 0
		# remove bad quality stations from list 'sta2work' (based on inv_pars.txt)
		if filter_lowQ_data: 
			name_file =  '.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'inv_pars.txt'
			BQ_sta = [x.split()[0][:-4] for x in open(name_file).readlines() if x[0]!='#' and x[-2] is '0']
			sta2work = [x for x in sta2work if not x in BQ_sta]
		#
		for file_aux in file_dir:
			if (file_aux[pos_ast:-4] in sta2work and file_aux[pos_ast:-4] != 'WT067a'):# incomplete station WT067a, no tipper
				file = file_aux[pos_ast:] # name on the file
				sta_obj = Station(file, count, path_files)
				sta_obj.read_edi_file()
				## correction in elevation for discrepancy between elev in wells and MT stations (temporal solition, to check)
				sta_obj.elev = float(sta_obj.elev) - 42.
				##
				sta_obj.rotate_Z()
				sta_obj.app_res_phase()
				#plt.loglog(sta_obj.T,sta_obj.rho_app[1])
				# import PT derotated data 
				#print(sta_obj.Z_xy[0])
				#plt.loglog(sta_obj.T,sta_obj.Z_xy[0])
				sta_obj.read_PT_Z(pc = pc) 
				#print(sta_obj.Z_xy[0])
				#plt.loglog(sta_obj.T,sta_obj.Z_xy[0])
				sta_obj.app_res_phase() # [self.rho_app, self.phase_deg, self.rho_app_er, self.phase_deg_er]
				## Create station objects and fill them
				station_objects.append(sta_obj)
				count  += 1
		#save to .txt names of stations and coord. 
		f = open("MT_sta_latlon.txt", "w") # text file to save names of meb wells 
		f.write('# sta lat lon lat_dec lon_dec elev\n')
		for sta in station_objects: # loop over the meb wells (objects)
			f.write(sta.name+'\t'+str(sta.lat)+'\t'+str(sta.lon)+'\t'+str(sta.lat_dec)+'\t'+str(sta.lon_dec)+'\t'+str(sta.elev)+'\n')
		f.close()

		# plot sounding curves
		# pp = PdfPages('MT_sound_curves.pdf')
		# for sta_obj in station_objects: 
		# 	f = sta_obj.plot_app_res_phase()
		# 	pp.savefig(f)
		# 	plt.close(f)
		# pp.close()
		# if full_dataset:
		# 	shutil.move('MT_sound_curves.pdf','.'+os.sep+'MT_info'+os.sep+'MT_sound_curves_full.pdf')
		# if prof_WRKNW6:
		# 	shutil.move('MT_sound_curves.pdf','.'+os.sep+'MT_info'+os.sep+'MT_sound_curves_prof_WRKNW6.pdf')
		# if prof_WRKNW5:
		# 	shutil.move('MT_sound_curves.pdf','.'+os.sep+'MT_info'+os.sep+'MT_sound_curves_prof_WRKNW5.pdf')

		#########################################################################################
		#########################################################################################
		## Import wells data:
		#wl_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red = \
		# 	read_well_temperature(path_wells_temp_date)
		wl_name, wl_prof_depth, wl_prof_depth_red, wl_prof_temp, dir_no_depth_red, wl_prof_date = \
		 	read_well_temperature_date(path_wells_temp_date)

		# # Note: dir_no_depth_red contain a list of wells with no information of reduced depth
		# ## Recover location for wells from path_wells_loc
		wells_location = read_well_location(path_wells_loc)
		# # Note: wells_location = [[wl_name1,lat1,lon1,elev1],...] list of arrays
		# ## Recover MeB data for wells from path_wells_meb
		wl_name_meb, wl_prof_depth_meb, wl_prof_meb = read_well_meb(path_wells_meb)
		#########################################################################################
		## Create wells objects
		# Defined lists of wells
		if full_dataset:
			wl2work = wl_name
			#wl2work = ['TH01']
		if sta_2_re_invert:
			wl2work = wl_name
		if prof_WRKNW6:
			wl2work = ['TH19','TH08','WK404','WK408','WK224','WK684','WK686'] #WK402
			wl2work = ['TH19','TH08','WK404','WK224','WK684','WK686'] #WK402
		if prof_WRKNW5:
			wl2work = ['WK261','WK262','WK263','WK243','WK267A','WK270','TH19','WK408','WK401', 'WK404'] # 'WK260' 
			#wl2work = ['WK401','TH19', 'WK404'] 
			wl2work = ['WK260','WK261','TH19','WK401','WK267A']#,'WK270']#WK263' ,'WK267A'
			#wl2work = ['WK260']
			#wl2work = ['WK401','TH19','WK261']
		if prof_NEMT2:
			wl2work = ['TH12','TH18','WK315B','WK227','WK314','WK302']
			wl2work = ['WK261']
		if array_WRKNW5_WRKNW6:
			wl2work = ['WK260','WK261','TH19','WK401','WK267A','TH19','TH08','WK404','WK224','WK684','WK686']
		# Tauhara profiles
		if prof_THNW03: 
			wl2work= ['TH13']
		if prof_THNW04: 
			wl2work= ['TH12']
		if prof_THNW05: 
			wl2work= []
		#########################################################################################
		# ## Loop over the wells to create objects and assing data attributes 
		wells_objects = []   # list to be fill with station objects
		count  = 0
		count2 = 0
		for wl in wl_name:
			if wl in wl2work and wl != 'THM24':
				# create well object
				wl_obj = Wells(wl, count)
				# Search for location of the well and add to attributes	
				for i in range(len(wells_location)): 
					wl_name = wells_location[i][0]
					if wl_obj.name == wl_name: 
						wl_obj.lat_dec = wells_location[i][2]
						wl_obj.lon_dec = wells_location[i][1]
						wl_obj.elev = wells_location[i][3] 
				## load data attributes
				## filter the data to the most recent one (well has overlap data cooresponding to reintepretations)
				filter_by_date = True
				filter_by_temp = False
				if filter_by_date:
					#year = max(wl_prof_date[count2]) # last year of interpretation 
					wl_prof_date[count2].sort() # last year of interpretation
					idx_year = [i for i, x in enumerate(wl_prof_date[count2]) if x == wl_prof_date[count2][-1]]  # great
					if len(idx_year) < 2: 
						idx_year = [i for i, x in enumerate(wl_prof_date[count2]) if (x == wl_prof_date[count2][-1] or x == wl_prof_date[count2][-2])]  # great
					# condition for data in part. wells
					if wl == 'WK047':
						idx_year = [0,2,-1]
					if wl == 'WK028':
						idx_year = [-1]
					if wl == 'WK401':
						idx_year = [i for i, x in enumerate(wl_prof_date[count2])]  
					if wl == 'WK045':
						idx_year = [i for i, x in enumerate(wl_prof_date[count2])]  
					if wl == 'WK005':
						wdata = [1,2]
						idx_year = [i for i, x in enumerate(wl_prof_date[count2]) if i not in wdata] 
					if wl == 'TH12':
						idx_year = [i for i, x in enumerate(wl_prof_date[count2]) if x is not '2016']
					if wl == 'WK219':
						idx_year = [i for i, x in enumerate(wl_prof_date[count2]) if i != 5]
					# if wl == 'WK684':
					# 	idx_year = [i for i in idx_year if i != 3]

					wl_obj.depth = [wl_prof_depth[count2][i] for i in idx_year]
					wl_obj.red_depth = [wl_prof_depth_red[count2][i] for i in idx_year]
					wl_obj.temp_prof_true = [wl_prof_temp[count2][i] for i in idx_year]
				
				elif filter_by_temp:
					pass
				else:	
					wl_obj.depth = wl_prof_depth[count2]	
					wl_obj.red_depth = wl_prof_depth_red[count2]
					wl_obj.temp_prof_true = wl_prof_temp[count2]	
				
				wl_obj.depth_raw = wl_prof_depth[count2]	
				wl_obj.red_depth_raw = wl_prof_depth_red[count2]
				wl_obj.temp_prof_true_raw = wl_prof_temp[count2]	

				# check if measure points are too close
				# find indexes of repeat values in red_depth and create vectors with no repetitions (ex. wel WK401)
				wl_obj.red_depth, rep_idx= np.unique(wl_obj.red_depth, return_index = True)
				temp_aux = [wl_obj.temp_prof_true[i] for i in rep_idx]
				wl_obj.temp_prof_true = temp_aux 
				## add a initial point to temp (15°C) profile at 0 depth (elevation of the well)
				if wl_obj.red_depth[-1] != wl_obj.elev:
					wl_obj.red_depth = np.append(wl_obj.red_depth, wl_obj.elev)
					if wl_obj.temp_prof_true[-1] < 10.:
						wl_obj.temp_prof_true = np.append(wl_obj.temp_prof_true, wl_obj.temp_prof_true[-1] - 5.)
					else:
						wl_obj.temp_prof_true = np.append(wl_obj.temp_prof_true, 10.0)
				## sort depth and temp based on depth (from max to min)
				wl_obj.red_depth, wl_obj.temp_prof_true = zip(*sorted(zip(wl_obj.red_depth, wl_obj.temp_prof_true), reverse = True))
				## resample .temp_prof_true and add to attribute prof_NEMT2 .temp_prof_rs
				
				## method of interpolation : Cubic spline interpolation 
				## inverse order: wl_obj.red_depth start at the higuer value (elev)
				xi = np.asarray(wl_obj.red_depth)
				yi = np.asarray(wl_obj.temp_prof_true)
				N_rs = 500 # number of resample points data
				xj = np.linspace(xi[0],xi[-1],N_rs)	
				yj = cubic_spline_interpolation(xi,yi,xj, rev = True)
				# add attributes
				wl_obj.red_depth_rs = xj
				wl_obj.temp_prof_rs = yj
				
				## method of fit: cubic polinomial
				## inverse order: wl_obj.red_depth start at the higuer value (elev)
				xi = np.asarray(wl_obj.red_depth)
				yi = np.asarray(wl_obj.temp_prof_true)
				N_rs = 500 # number of resample points data
				xj = np.linspace(xi[0],xi[-1],N_rs)	
				yj = cubic_spline_interpolation(xi,yi,xj, rev = True)
				# add attributes
				wl_obj.red_depth_rs = xj
				wl_obj.temp_prof_rs = yj

				## add well object to directory of well objects
				wells_objects.append(wl_obj)
				count  += 1
			count2 +=1

		# # plot temp profile for wells
		# pp = PdfPages('wells_temp_prof.pdf')
		# for wl in wells_objects: 
		# 	f = wl.plot_temp_profile(rs = True, raw = True)
		# 	pp.savefig(f)
		# 	plt.close(f)
		# pp.close()
		# if full_dataset:
		# 	shutil.move('wells_temp_prof.pdf','.'+os.sep+'wells_info'+os.sep+'wells_temp_prof_full.pdf')
		# if prof_WRKNW6:
		# 	shutil.move('wells_temp_prof.pdf','.'+os.sep+'wells_info'+os.sep+'wells_temp_prof_WRKNW6.pdf')
		# if prof_WRKNW5:
		# 	shutil.move('wells_temp_prof.pdf','.'+os.sep+'wells_info'+os.sep+'wells_temp_prof_WRKNW5.pdf')

		# # Search for location of the well and add to attributes
		# for wl in wells_objects:
		# 	for i in range(len(wells_location)): 
		# 		wl_name = wells_location[i][0]
		# 		if wl.name == wl_name: 
		# 			wl.lat_dec = wells_location[i][2]
		# 			wl.lon_dec = wells_location[i][1]
		# 			wl.elev = wells_location[i][3]

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

		## Figure of station and well positons on top of satelite image of the field (save in current folder)
		# plot over topography
		if False:
			file_name = 'map_stations_wells'
			ext_file = [175.934859, 176.226398, -38.722805, -38.567571]
			x_lim = [176.0,176.1]
			y_lim = [-38.68,-38.58]
			path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd.jpg'
			map_stations_wells(station_objects, wells_objects, file_name = file_name, format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.8 ,xlim = x_lim, ylim = y_lim, ext_img = ext_file)
			
			# plot over resistivity boundary 
			file_name = 'map_res_bound_stations_wells'
			ext_file = [175.934859, 176.226398, -38.722805, -38.567571]
			x_lim = [176.0,176.1]
			y_lim = [-38.68,-38.58]
			path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_res_map_gearth.jpg'
			map_stations_wells(station_objects, wells_objects, file_name = file_name, format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.8 ,xlim = x_lim, ylim = y_lim, ext_img = ext_file, dash_arrow = False)

		# plot noise distribution
		if False:
			bands  = [[0,.1],[.1,10.],[10,1000.]]
			plot_dist_noise(station_objects,bands)

	# (1) Run MCMC for MeB priors  
	if mcmc_meb_inv:
		pp = PdfPages('fit.pdf')
		start_time = time.time()
		count = 1
		temp_full_list_z1 = []
		temp_full_list_z2 = []
		print("(1) Run MCMC for MeB priors")
		for wl in wells_objects:
			if wl.meb: 
				mes_err = 2.
				## filter meb logs: when values are higher than 5 % => set it to 5 % (in order to avoid control of this points)
				fiter_meb = True
				if fiter_meb:		
						wl.meb_prof = [5. if ele > 5. else ele for ele in wl.meb_prof]
				print(wl.name +  ': {:}/{:}'.format(count, count_meb_wl))
				mcmc_wl = mcmc_meb(wl, norm = 2., scale = 'lin', mes_err = mes_err, walk_jump=3000)
				mcmc_wl.run_mcmc()
				mcmc_wl.plot_results_mcmc()
				#
				f = mcmc_wl.sample_post(exp_fig = True, plot_hist_bounds = True, plot_fit_temp = False, wl_obj = wl, \
					temp_full_list_z1 = temp_full_list_z1, temp_full_list_z2 = temp_full_list_z2) # Figure with fit to be add in pdf pp
				#f = mcmc_wl.sample_post_temp(exp_fig = True) # Figure with fit to be add in pdf 
				pp.savefig(f)
				## calculate estimate parameters (percentiels)
				mcmc_wl.model_pars_est()
				count += 1
	
		## save lists: temp_full_list_z1, temp_full_list_z2
		#with open('corr_z1_z1_temp_glob.txt', 'w') as f:
		#	for f1, f2 in zip(temp_full_list_z1, temp_full_list_z2):
		#		print(f1, f2, file=f)	
		#shutil.move('corr_z1_z1_temp_glob.txt','.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'corr_cc_temp'+os.sep+'corr_z1_z1_temp_glob.txt')
		## enlapsed time for the inversion (every station in station_objects)
		enlap_time = time.time() - start_time # enlapsed time
		## print time consumed
		print("Time consumed:\t{:.1f} min".format(enlap_time/60))
		pp.close()
		# move figure fit to global results folder
		shutil.move('fit.pdf','.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'00_fit.pdf')
		# print histogram of temperatures of z1 and z2 for the whole net
		#g = plot_fit_temp_full(temp_full_list_z1,temp_full_list_z2) 
		#g = hist_z1_z2_temp_full()
		#g.savefig('.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'01_temp_z1_z2_full_net.png')   # save the figure to file
		#plt.close(g)    # close the figure
 
	# (2) Construct priors for MT stations
	if prior_MT_meb_read:
		# attribute in meb wells for path to mcmc results 
		for wl in wells_objects:
			if wl.meb:
				wl.path_mcmc_meb = '.'+os.sep+str('mcmc_meb')+os.sep+wl.name
		# Calculate prior values for boundaries of the cc in station
		# (prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells)
		# Function assign results as attributes for MT stations in station_objects (list)
		calc_prior_meb(station_objects, wells_objects, slp = 3*10., quadrant = False) # calc prior at MT stations position
		# plot surface of prior
		if False:	
			if False: # by Delanuay triangulation 
				path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd.jpg'
				ext_file = [175.934859, 176.226398, -38.722805, -38.567571]
				x_lim = None #[176.0,176.1]
				y_lim = None #[-38.68,-38.58]
				path_q_wells = '.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'corr_cc_temp'+os.sep+'Q_meb_inv_results.txt'
				# Figure: general
				file_name = 'trig_meb_prior_wells_WT'
				triangulation_meb_results(station_objects, wells_objects, path_base_image = path_base_image, xlim = x_lim, ylim = y_lim, ext_img = ext_file,\
					file_name = file_name, format = 'png', filter_wells_Q = path_q_wells)
				# Figure: z1_mean
				file_name = 'trig_meb_prior_wells_WT_z1_mean'
				triangulation_meb_results(station_objects, wells_objects, path_base_image = path_base_image, xlim = x_lim, ylim = y_lim, ext_img = ext_file,\
					file_name = file_name, format = 'png', value = 'z1_mean', vmin=50, vmax=900, filter_wells_Q = path_q_wells)
				# Figure: z2_mean
				file_name = 'trig_meb_prior_wells_WT_z2_mean'
				triangulation_meb_results(station_objects, wells_objects, path_base_image = path_base_image, xlim = x_lim, ylim = y_lim, ext_img = ext_file,\
					file_name = file_name, format = 'png', value = 'z2_mean', vmin=50, vmax=900, filter_wells_Q = path_q_wells)
				# Figure: z1_std
				file_name = 'trig_meb_prior_wells_WT_z1_std'
				triangulation_meb_results(station_objects, wells_objects, path_base_image = path_base_image, xlim = x_lim, ylim = y_lim, ext_img = ext_file,\
					file_name = file_name, format = 'png', value = 'z1_std', filter_wells_Q = path_q_wells, vmin=50, vmax=900)
				# Figure: z2_std
				file_name = 'trig_meb_prior_wells_WT_z2_std'
				triangulation_meb_results(station_objects, wells_objects, path_base_image = path_base_image, xlim = x_lim, ylim = y_lim, ext_img = ext_file,\
					file_name = file_name, format = 'png', value = 'z2_std', filter_wells_Q = path_q_wells, vmin=50, vmax=900)

			if True: # by gridding surface
				# define region to grid
				coords = [175.97,176.178,-38.69,-38.59] # [min lon, max lon, min lat, max lat]
				# fn. for griding and calculate prior => print .txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
				file_name = 'grid_meb_prior'
				path_output = '.'+os.sep+'plain_view_plots'+os.sep+'meb_prior'
				try:
					os.mkdir('.'+os.sep+'plain_view_plots'+os.sep+'meb_prior')
				except:
					pass
				##
				# image background
				path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd.jpg'
				ext_file = [175.934859, 176.226398, -38.722805, -38.567571]
				x_lim = None #[176.0,176.1]
				y_lim = None #[-38.68,-38.58]
				# call function 
				grid_meb_prior(wells_objects, coords = coords, n_points = 100, slp = 4*10., file_name = file_name, path_output = path_output,\
					plot = True, path_base_image = path_base_image, ext_img = ext_file)
				
	# (3) Run MCMC inversion for each staion, obtaning 1D 3 layer res. model
	# 	  Sample posterior, construct uncertain resistivity distribution and create result plots 
	if mcmc_MT_inv:
		## create pdf file to save the fit results for the whole inversion
		pdf_fit = False 
		if pdf_fit:
			pp = PdfPages('fit.pdf')
		start_time_f = time.time()
		prior_meb = True  # if false -> None
		prior_meb_weigth = 1.0
		station_objects.sort(key=lambda x: x.ref, reverse=False)
		# run inversion
		if True:
			for sta_obj in station_objects:
				if sta_obj.ref < 240: # start at 0
				#if sta_obj.name[:-4] != 'WT030a':
					pass
				else: 
					print('({:}/{:}) Running MCMC inversion:\t'.format(sta_obj.ref+1,len(station_objects))+sta_obj.name[:-4])
					## range for the parameters
					par_range = [[.01*1e2,2.*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
					#par_range = [[.01*1e2,.5*1e3],[.5*1e1,.5*1e3],[1.*1e1,1.*1e3],[1.*1e0,1.*1e1],[1.*1e1,1.*1e3]]
					#par_range = [[.01*1e2,2.*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,20.*1e1],[.5*1e1,1.*1e3]]
					# error floor
					error_max_per = [5.,2.5] # [10.,5.]	[20.,10.]			
					## inv_dat: weighted data to invert and range of periods
					## 		inv_dat = [1,1,1,1] # [appres zxy, phase zxy, appres zyx, phase zyx]
					## range_p = [0.001,10.] # range of periods
					if True: # import inversion parameters from file 
						name_file =  '.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'inv_pars.txt'
						inv_pars = [x.split() for x in open(name_file).readlines() if x[0]!='#']
						inv_pars_names = [x[0] for x in inv_pars] 
						idx = inv_pars_names.index(sta_obj.name)
						# load pars
						range_p = [float(inv_pars[idx][1]), float(inv_pars[idx][2])] # range of periods
						if inv_pars[idx][3] is '2':
							inv_dat = [1,1,1,1] # [appres zxy, phase zxy, appres zyx, phase zyx]
						elif inv_pars[idx][3] is '1':
							inv_dat = [0,0,1,1] # [appres zxy, phase zxy, appres zyx, phase zyx]
						elif inv_pars[idx][3] is '0':
							inv_dat = [1,1,0,0] # [appres zxy, phase zxy, appres zyx, phase zyx]
					else:
						# Default values (inv pars)
						range_p = [0.001,10] # range of periods, default values
						inv_dat = [1,1,1,1] # [appres zxy, phase zxy, appres zyx, phase zyx]
						# fitting mode xy or yx: 
						fit_max_mode = False

					# inversion pars. per station (manual)
					if True:
						if sta_obj.name[:-4] == 'WT024a': # station with static shift
							error_max_per = [5.,2.5]
							range_p = [0,100.] # range of periods
						if sta_obj.name[:-4] == 'WT030a': # station with static shift
							range_p = [0,10.] # range of periods
						if sta_obj.name[:-4] == 'WT039a': # station with static shift
							range_p = [0,10.] # range of periods
							error_max_per = [5.,2.5]
						if sta_obj.name[:-4] == 'WT060a': # station with static shift
							range_p = [0.,1.] # range of periods
							par_range = [[.01*1e2,.5*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
							#error_max_per = [20.,10.]
						if sta_obj.name[:-4] == 'WT068a': # station with static shift
							range_p = [0,5.] # range of periods
							error_max_per = [20.,10.]
							#inv_dat = [1,1,0,1]
						if sta_obj.name[:-4] == 'WT070b': # station with static shift
							range_p = [0,5.] # range of periods
						if sta_obj.name[:-4] == 'WT071a': # station with static shift
							range_p = [0,5.] # range of periods
							range_p = [0,10.] # range of periods
							error_max_per = [5.,2.5]
						if sta_obj.name[:-4] == 'WT107a': # station with static shift
							par_range = [[.01*1e2,.5*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
							range_p = [0,5.] # range of periods
							error_max_per = [5.,2.5]
						if sta_obj.name[:-4] == 'WT111a': # station with static shift
							range_p = [0,5.] # range of periods
						if sta_obj.name[:-4] == 'WT223a': # station with static shift
							range_p = [0,10.] # range of periods
							error_max_per = [20.,5.]
						if sta_obj.name[:-4] == 'WT501a': # station with static shift
							range_p = [0,5.] # range of periods
						if sta_obj.name[:-4] == 'WT502a': # station with static shift
							#range_p = [0,5.] # range of periods
							par_range = [[.01*1e2,.5*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
						if sta_obj.name[:-4] == 'WT003a': # station with static shift
							error_max_per = [20.,10.]	
							#inv_dat = [1,0,1,0]
					## print relevant information
					print('range of periods: [{:2.3f}, {:2.2f}] [s]'.format(range_p[0],range_p[1]))
					print('inverted data: '+str(inv_dat))
					## plot noise
					try:
						path_img = 'mcmc_inversions'+os.sep+sta_obj.name[:-4]
						sta_obj.plot_noise(path_img = path_img)
					except:
						pass
					#print('mean noise in app res XY: {:2.2f}'.format(np.mean(sta_obj.rho_app_er[1])))
					#print('mean noise in phase XY: {:2.2f}'.format(np.mean(sta_obj.phase_deg_er[1])))
					###
					error_mean = False
					if error_mean:
						error_max_per = [1.,1.]
					# set number of walkers and walker jumps
					nwalkers = 40
					walk_jump = 3000
					# run inversion
					try: 
						mcmc_sta = mcmc_inv(sta_obj, prior='uniform', inv_dat = inv_dat, prior_input = par_range, \
							walk_jump = walk_jump, nwalkers = nwalkers, prior_meb = prior_meb, prior_meb_weigth = prior_meb_weigth,\
								range_p = range_p, autocor_accpfrac = True, data_error = True, \
									error_max_per=error_max_per, error_mean = error_mean)
					except: # in case that inversion breaks due to low number of independet samples 
						mcmc_sta = mcmc_inv(sta_obj, prior='uniform', inv_dat = inv_dat, prior_input = par_range, \
							walk_jump = walk_jump+1000, nwalkers = nwalkers+10, prior_meb = prior_meb, prior_meb_weigth = prior_meb_weigth,\
								range_p = range_p, autocor_accpfrac = True, data_error = True, \
									error_max_per=error_max_per, error_mean = error_mean)

					if error_max_per:
						## plot noise
						try:
							name_file='noise_appres_phase_error_floor'
							path_img = 'mcmc_inversions'+os.sep+sta_obj.name[:-4]
							sta_obj.plot_noise(path_img = path_img, name_file = name_file)
						except:
							pass
					if prior_meb:
						print("	wells for MeB prior: {} ".format(sta_obj.prior_meb_wl_names))
						#print("	[[z1_mean,z1_std],[z2_mean,z2_std]] = {} \n".format(sta_obj.prior_meb))
						print("	distances = {}".format(sta_obj.prior_meb_wl_dist)) 
						print("	prior [z1_mean, std][z2_mean, std] = {} \n".format(sta_obj.prior_meb)) 
					## run inversion
					mcmc_sta.inv()
					## plot results (save in .png)
					if True: # plot results for full chain 
						mcmc_sta.plot_results_mcmc(chain_file = 'chain.dat', corner_plt = False, walker_plt = True)
						#shutil.move(mcmc_sta.path_results+os.sep+'corner_plot.png', mcmc_sta.path_results+os.sep+'corner_plot_full.png')
						shutil.move(mcmc_sta.path_results+os.sep+'walkers.png', mcmc_sta.path_results+os.sep+'walkers_full.png')
					## sample posterior
					#mcmc_sta.sample_post()
					if pdf_fit:
						f, g = mcmc_sta.sample_post(idt_sam = True, plot_fit = True, exp_fig = True, plot_model = True) # Figure with fit to be add in pdf (whole station)
					else:
						mcmc_sta.sample_post(idt_sam = True, plot_fit = True, exp_fig = False, plot_model = True) # Figure with fit to be add in pdf (whole station)		
					#mcmc_sta.sample_post(idt_sam = True, plot_fit = True, exp_fig = False, plot_model = True) # Figure with fit to be add in pdf (whole station)
					## plot results without burn-in section
					mcmc_sta.plot_results_mcmc(chain_file = 'chain_sample_order.dat', corner_plt = True, walker_plt = False)
					shutil.move(mcmc_sta.path_results+os.sep+'corner_plot.png', mcmc_sta.path_results+os.sep+'corner_plot_burn.png')

					# save figures
					if pdf_fit:
						pp.savefig(g)
						pp.savefig(f)
						plt.close('all')
					#plt.clf()
					## calculate estimate parameters
					mcmc_sta.model_pars_est()
			## enlapsed time for the inversion (every station in station_objects)
			enlap_time_f = time.time() - start_time_f # enlapsed time
			## print time consumed
			print("Time consumed:\t{:.1f} min".format(enlap_time_f/60))
			if pdf_fit:
				pp.close()
				# move figure fit to global results folder
				shutil.move('fit.pdf','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'00_fit.pdf')

	# (4) Plot 2D profile of unceratain boundaries z1 and z2 (results of mcmc MT inversion)
	if plot_2D_MT:
		print('(4) Plot 2D profile of uncertain boundaries z1 and z2 (results of mcmc MT inversion)')
		# quality inversion pars. plot (acceptance ratio and autocorrelation time)
		autocor_accpfrac = False
		# load mcmc results and assign to attributes of pars to station attributes 
		load_sta_est_par(station_objects, autocor_accpfrac = autocor_accpfrac)
		# Create figure of unceratain boundaries of the clay cap and move to mcmc_inversions folder
		file_name = 'z1_z2_uncert'
		#plot_2D_uncert_bound_cc(station_objects, pref_orient = 'EW', file_name = file_name) # width_ref = '30%' '60%' '90%', 
		plot_2D_uncert_bound_cc_mult_env(station_objects, pref_orient = 'EW', file_name = file_name, \
			width_ref = '90%', prior_meb = wells_objects, mask_no_cc = 112.) #, plot_some_wells = ['WK404'])#,'WK401','WK402'])
		shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+file_name+'.png')

		# plot autocorrelation time and acceptance factor 
		if autocor_accpfrac:
			file_name = 'autocor_accpfrac'
			plot_profile_autocor_accpfrac(station_objects, pref_orient = 'EW', file_name = file_name)
			shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+file_name+'.png')

		# plot profile of KL divergence 
		if False:
			file_name = 'KL_div_prof'
			plot_profile_KL_divergence(station_objects, wells_objects, pref_orient = 'EW', file_name = file_name)
			shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+file_name+'.png')
			
		## create text file for google earth, containing names of MT stations considered 
		for_google_earth(station_objects, name_file = '00_stations_4_google_earth.txt', type_obj = 'Station')
		shutil.move('00_stations_4_google_earth.txt','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion' \
			+os.sep+'00_stations_4_google_earth.txt')

		# plot uncertaity in boundary depths
		if True:
			file_name = 'bound_uncert'
			plot_bound_uncert(station_objects, file_name = file_name) #
			shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+file_name+'.png')

	# (4.1) Plot surface of uncertain boundaries z1 and z2 (results of mcmc MT inversion)
	if plot_3D_MT:
		print('(4.1) Plot surface of uncertain boundaries z1 and z2 (results of mcmc MT inversion)')
		if False: # plot plain view with circles 
			##
			ext_file = [175.948466, 176.260520, -38.743590, -38.574484]
			x_lim = [175.948466, 176.260520]
			y_lim = [-38.743590,-38.574484]
			type_plot = 'scatter'
			path_plots = '.'+os.sep+'plain_view_plots'# path to place the outputs 
			##
			# for plot with rest bound background
			path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_res_map_gearth_2.jpg'
			bound2plot = 'top' # top bound
			file_name = 'interface_LRA_'+bound2plot+'_rest_bound'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = True, path_plots = path_plots)
			bound2plot = 'bottom' 
			file_name = 'interface_LRA_'+bound2plot+'_rest_bound'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = True, path_plots = path_plots)
			
			# for plot with topo background
			path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd_2.jpg'
			bound2plot = 'top' # top bound
			file_name = 'interface_LRA_'+bound2plot+'_topo'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = True, path_plots = path_plots)
			bound2plot = 'bottom' 
			file_name = 'interface_LRA_'+bound2plot+'_topo'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = True, path_plots = path_plots)

		if True: # plot plain view with countours
			##
			# define region to grid
			grid = [175.935, 176.255,-38.77,-38.545] # [min lon, max lon, min lat, max lat]
			# fn. for griding and calculate prior => print .txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
			file_name = 'grid_MT_inv'
			path_output = '.'+os.sep+'plain_view_plots'+os.sep+'MT_inv'
			try:
				os.mkdir('.'+os.sep+'plain_view_plots'+os.sep+'MT_inv')
			except:
				pass
			##
			# image background
			#path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd.jpg'
			#ext_file = [175.934859, 176.226398, -38.722805, -38.567571]
			#path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd_2.jpg'
			#ext_file = [175.948466, 176.260520, -38.743590, -38.574484]
			path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd_3.jpg'
			ext_file = [175.781956, 176.408620, -38.802528, -38.528097]
			x_lim = [175.9,176.3]
			y_lim = None #[-38.68,-38.57]
			#path topo 
			path_topo = '.'+os.sep+'base_map_img'+os.sep+'coords_elev'+os.sep+'Topography_zoom_WT_re_sample_vertices_LatLonDec.csv'
			# call function to grid and plot 
			if False:
				grid_MT_inv_rest(station_objects, coords = grid, n_points = 20, slp = 4*10., file_name = file_name, path_output = path_output,\
					plot = True, path_base_image = path_base_image, ext_img = ext_file, xlim = x_lim, masl = False)
			# call function to plot with topo
			if True: 
				file_name = 'topo_MT_inv' 
				try:
					os.mkdir('.'+os.sep+'plain_view_plots'+os.sep+'MT_inv'+os.sep+'Topo')
				except:
					pass
				path_output = '.'+os.sep+'plain_view_plots'+os.sep+'MT_inv'+os.sep+'Topo'
				path_topo = '.'+os.sep+'base_map_img'+os.sep+'coords_elev'+os.sep+'Topography_zoom_WT_re_sample_vertices_LatLonDec.csv'
				topo_MT_inv_rest(station_objects, path_topo, slp = 4*10., file_name = file_name, path_output = path_output, \
					plot = True, path_base_image = path_base_image, ext_img = ext_file, xlim = x_lim, masl = False)

	# (5) Estimated distribution of temperature profile in wells. Calculate 3-layer model in wells and alpha parameter for each well
	if wells_temp_fit: 
		print('(5) Calculating beta in wells and fitting temperature profile')
		## Calculate 3-layer model in wells. Fit temperature profiles and calculate beta for each layer. 
		# Calculate normal dist. pars. [mean, std] for layer boundaries (z1 znd z2) in well position. 
		# Function assign results as attributes for wells in wells_objects (list of objects).
		## Note: to run this section prior_MT_meb_read == True
		calc_layer_mod_quadrant(station_objects, wells_objects)
		## loop over wells to fit temp. profiles ad calc. betas
		## file to save plots of temp prof samples for every well 
		pp = PdfPages('Test_samples.pdf') # pdf to plot the meb profiles
		for wl in wells_objects:
			print('Well: {}'.format(wl.name))
			# calculate Test and beta values 
			f = wl.temp_prof_est(plot_samples = True, ret_fig = True) # method of well object
			pp.savefig(f)
			#Test, beta, Tmin, Tmax, slopes = T_beta_est(well_obj.temp_profile[1], well_obj.temp_profile[0], Zmin, Zmax) # 
		pp.close()
		shutil.move('Test_samples.pdf','.'+os.sep+'temp_prof_samples'+os.sep+'wells'+os.sep+'Test_samples.pdf')

	# (6) Estimated temperature profile in station positions
	if sta_temp_est: 
		print('(6) Estimate Temerature profile in MT stations')
		for wl in wells_objects:
			# read samples of betas and others from wells. Load attributes 
			wl.read_temp_prof_est_wells(beta_hist_corr = True)
		# Calculate betas and other in MT station positions 
		calc_beta_sta_quadrant(station_objects, wells_objects)
		## Construct temperature profiles in MT stations
		## file to save plots of temp prof samples for every well 
		pp = PdfPages('Test_samples.pdf') # pdf to plot the meb profiles
		for sta_obj in station_objects:
			print(sta_obj.name[:-4])
			# read samples of betas and others from wells. Load attributes 
			f = sta_obj.temp_prof_est(plot_samples = True, ret_fig = True, Ns = 1000)
			perc = np.arange(15.,90.,5.)#np.arange(5.,100.,5.) # percentiels to calculate: [5% , 10%, ..., 95%]
			isoth = [50,100,150,200,250]
			sta_obj.uncert_isotherms_percentils(isotherms = isoth, percentiels = perc)
			pp.savefig(f)
			# calc
		pp.close()
		shutil.move('Test_samples.pdf','.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'+os.sep+'Test_samples.pdf')

		# plot 2D profile
		if prof_WRKNW6 or prof_WRKNW5:
			print('(6.1) Printing uncertain isotherms plot')
			# note: isotherms = [] are strings coherent with value given in uncert_isotherms_percentils()
			#isoth = ['50','100','150','200']#,'250']
			isoth = ['50','100','200']
			plot_2D_uncert_isotherms(station_objects, wells_objects, pref_orient = 'EW', file_name = 'isotherm_uncert',\
				percentiels = perc, isotherms = isoth) 

	# (7) Files for Paraview
	if files_paraview: 
		filter_no_cc = True # filter stations with z2 too thin (no inferred claycap)
		# (0) Create folder
		if not os.path.exists('.'+os.sep+str('paraview_files')):
			os.mkdir('.'+os.sep+str('paraview_files')) 
		# (1) print topography file 
		if True:
			f = open('.'+os.sep+str('paraview_files')+os.sep+'topo.csv','w')
			f.write('station, lon_dec, lat_dec, elev\n')
			for sta in station_objects:
				z, l, x, y = project([sta.lon_dec, sta.lat_dec])
				f.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str(sta.elev)+'\n')
			f.close()
		# (2) files with percentils surfaces
		if True:
			# crear archivos de percentiles 
			load_sta_est_par(station_objects)
			# .csv for mean
			f = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_mean.csv','w')
			f.write('station, lon_dec, lat_dec, z1, z2\n')
			# .csv for mean
			f0 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_mean_byrow.csv','w')
			f0.write('station, lon_dec, lat_dec, z\n') # contains z1 and z2 mean as one collumn 
			# .csv for percentils
			f10 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_10.csv','w')
			f10.write('station, lon_dec, lat_dec, z1, z2\n')
			f20 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_20.csv','w')
			f20.write('station, lon_dec, lat_dec, z1, z2\n')
			f40 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_40.csv','w')
			f40.write('station, lon_dec, lat_dec, z1, z2\n')
			f60 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_60.csv','w')
			f60.write('station, lon_dec, lat_dec, z1, z2\n')
			f80 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_80.csv','w')
			f80.write('station, lon_dec, lat_dec, z1, z2\n')
			f90 = open('.'+os.sep+str('paraview_files')+os.sep+'z1_z2_90.csv','w')
			f90.write('station, lon_dec, lat_dec, z1, z2\n')
			for sta in station_objects:
			# filter stations with z2 too thin (no inferred claycap)
				if filter_no_cc:
					#print(sta.name)
					#print(sta.z2_pars[0])
					if sta.z2_pars[0] > 50.:
						# mean
						z, l, x, y = project([sta.lon_dec, sta.lat_dec])
						f.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[0]))+', '+
							str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[0])))+'\n')
						f0.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[0]))+'\n')
						f0.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[0])))+'\n')
						# percentils
						# [mean, std, med, [5%, 10%, 15%, ..., 85%, 90%, 95%]]
						f10.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][1]))+', '+
							str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][1])))+'\n')
						f20.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][3]))+', '+
							str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][3])))+'\n')
						f40.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][7]))+', '+
							str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][7])))+'\n')
						f60.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][-7]))+', '+
							str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][-7])))+'\n')
						f80.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][-3]))+', '+
							str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][-3])))+'\n')
						f90.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][-1]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][-1])))+'\n')
				else:
					# mean
					z, l, x, y = project([sta.lon_dec, sta.lat_dec])
					f.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[0]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[0])))+'\n')
					# percentils
					# [mean, std, med, [5%, 10%, 15%, ..., 85%, 90%, 95%]]
					f10.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][1]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][1])))+'\n')
					f20.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][3]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][3])))+'\n')
					f40.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][7]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][7])))+'\n')
					f60.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][-7]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][-7])))+'\n')
					f80.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][-3]))+', '+
						str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][-3])))+'\n')
					f90.write(str(sta.name[:-4])+', '+str(x)+', '+str(y)+', '+str((sta.elev - sta.z1_pars[3][-1]))+', '+
					str((sta.elev - (sta.z1_pars[0]+sta.z2_pars[3][-1])))+'\n')
			f.close()
			f10.close()
			f20.close()
			f40.close()
			f60.close()
			f80.close()
			f90.close()

#####################################################################################################################################################################
## EXTRAS that use list of objects
	if False:
		# PDF file with figure of inversion misfit (observe data vs. estatimated data)
		if False: 
			if False: # option 1: print appres fit to pdf
				from PIL import Image
				imagelist = []
				for sta_obj in station_objects:
					pngfile = Image.open('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'app_res_fit.png')
					pngfile = pngfile.convert('RGB')
					#pngfile = pngfile.resize(size = (500, 500))
					imagelist.append(pngfile)
				#print(imagelist)
				pngfile.save('.'+os.sep+'mcmc_inversions'+os.sep+'fit.pdf', save_all=True, append_images=[imagelist[1],imagelist[3]])
				# move
				try:
					shutil.move('.'+os.sep+'mcmc_inversions'+os.sep+'fit.pdf', '.'+os.sep+'mcmc_inversions'+os.sep+'01_bad_model'+os.sep+'fit.pdf')
				except:
					os.mkdir( '.'+os.sep+'mcmc_inversions'+os.sep+'01_bad_model')
					shutil.move('.'+os.sep+'mcmc_inversions'+os.sep+'fit.pdf', '.'+os.sep+'mcmc_inversions'+os.sep+'01_bad_model'+os.sep+'fit.pdf')

			# in evaluation
			if True: # option 2: move appres fit to a folder
				try:
					os.mkdir('.'+os.sep+'mcmc_inversions'+os.sep+'01_bad_model')
				except:
					pass
				for sta_obj in station_objects:
					shutil.copy('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'app_res_fit.png', '.'+os.sep+'mcmc_inversions'+os.sep+'01_bad_model'+os.sep+'app_res_fit_'+sta_obj.name[:-4]+'.png')

		# delete chain.dat (text file with the whole markov chains) from station folders
		if False: 
			
			for sta in station_objects:
				try:
					os.remove('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+'chain.dat')
				except:
					pass
			for wl in wells_objects:
				if wl.meb: 
					try:
						os.remove('.'+os.sep+'mcmc_meb'+os.sep+wl.name+os.sep+'chain.dat')
					except:
						pass

		## create text file for google earth, containing names of MT stations considered 
		if False: 
			for_google_earth(station_objects, name_file = '00_stations_4_google_earth.txt', type_obj = 'Station')
			#shutil.move('00_stations_4_google_earth.txt','.'+os.sep+'base_map_img'+os.sep+'00_stations_4_google_earth.txt')
			shutil.move('00_stations_4_google_earth.txt','.'+os.sep+'mcmc_inversions'+os.sep+'02_bad_sta_map'+os.sep+'00_stations_4_google_earth.txt')

		## create file with range of periods to invert for every station
		if False: 
			name_file =  '.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'range_periods_inv.txt'
			range_p_set = open(name_file,'w')
			range_p_set.write('#'+' '+'station_name'+'\t'+'initial_period'+'\t'+'final_period'+'\t'+'mode to model (0:TE, 1:TM, 2:both)'+'\t'+'Quality (0:bad, 1:mid, 2:good)'+'\n')
			range_p_def = [0.001,10.,2,2] # default range of periods for inversion 
			for sta in station_objects:
				range_p_set.write(sta.name+'\t'+str(range_p_def[0])+'\t'+str(range_p_def[1])+'\n')
			range_p_set.close()

		## create list of stations to invert base on changes in file range_periods_inv.txt
		if False: 
			name_file_in =  '.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'inv_pars.txt'
			par_inv_def = [str(0.001),str(10),str(2),str(2)] # default

			sta_re_inv = [x for x in open(name_file_in).readlines() if x[0]!='#']
			sta_re_inv = [x.split() for x in sta_re_inv]
			sta_re_inv = [x[0][:-4] for x in sta_re_inv if \
				x[1] != par_inv_def[0] or \
					x[:][2] != par_inv_def[1] or \
						x[:][3] != par_inv_def[2] or \
							x[:][4] != par_inv_def[3]]  
			print(sta_re_inv)

		if False: # list of stations to re invert (bad quality or wrong modelling)
			name_file_in =  '.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'inv_pars.txt'
			sta_re_inv = [x.split() for x in open(name_file_in).readlines()[1:]]
			sta_re_inv = [x[0][:-4] for x in sta_re_inv if x[4] is '0']
			print(sta_re_inv)

		if True:  # histogram of MT inversion parameters for stations inverted
			z1_batch = []
			z2_batch = []
			r1_batch = []
			r2_batch = []
			r3_batch = []
			## load pars
			for sta in station_objects:
				aux = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+'est_par.dat')
				sta.z1_pars = [aux[0][1],aux[0][2]]
				sta.z2_pars = [aux[1][1],aux[1][2]]
				sta.r1_pars = [aux[2][1],aux[2][2]]
				sta.r2_pars = [aux[3][1],aux[3][2]]
				sta.r3_pars = [aux[4][1],aux[4][2]]
				# add to batch
				if sta.z2_pars[0] > 50.:
					z1_batch.append(sta.z1_pars[0])
					z2_batch.append(sta.z2_pars[0])
					r1_batch.append(sta.r1_pars[0])
					r2_batch.append(sta.r2_pars[0])
					r3_batch.append(sta.r3_pars[0])
			# plot histograms 
			f = plt.figure(figsize=(12, 7))
			gs = gridspec.GridSpec(nrows=2, ncols=3)
			ax1 = f.add_subplot(gs[0, 0])
			ax2 = f.add_subplot(gs[0, 1])
			ax3 = f.add_subplot(gs[1, 0])
			ax4 = f.add_subplot(gs[1, 1])
			ax5 = f.add_subplot(gs[1, 2])
			ax_leg= f.add_subplot(gs[0, 2])

			# z1
			bins = np.linspace(np.min(z1_batch), np.max(z1_batch), int(np.sqrt(len(z1_batch))))
			h,e = np.histogram(z1_batch, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax1.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
			ax1.set_xlabel('$z_1$ [m]', fontsize=textsize)
			ax1.set_ylabel('freq.', fontsize=textsize)
			ax1.grid(True, which='both', linewidth=0.1)
			# plot normal fit 
			(mu, sigma) = norm.fit(z1_batch)
			med = np.median(z1_batch)
			try:
				y = mlab.normpdf(bins, mu, sigma)
			except:
				#y = stats.norm.pdf(bins, mu, sigma)
				pass
			#ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
			#ax2.legend(loc='upper right', shadow=False, fontsize=textsize)
			ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
			ax1.plot([med,med],[0,np.max(h)],'r-')

			# z2
			bins = np.linspace(np.min(z2_batch), np.max(z2_batch), int(np.sqrt(len(z2_batch))))
			h,e = np.histogram(z2_batch, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax2.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
			ax2.set_xlabel('$z_2$ [m]', fontsize=textsize)
			ax2.set_ylabel('freq.', fontsize=textsize)
			ax2.grid(True, which='both', linewidth=0.1)
			# plot normal fit 
			(mu, sigma) = norm.fit(z2_batch)
			med = np.median(z2_batch)
			try:
				y = mlab.normpdf(bins, mu, sigma)
			except:
				#y = stats.norm.pdf(bins, mu, sigma)
				pass
			#ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
			#ax3.legend(loc='upper right', shadow=False, fontsize=textsize)
			ax2.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
			ax2.plot([med,med],[0,np.max(h)],'b-')

			# r1
			bins = np.linspace(np.min(r1_batch), np.max(r1_batch), int(np.sqrt(len(r1_batch))))
			h,e = np.histogram(r1_batch, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax3.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
			ax3.set_xlabel(r'$\rho_1$ [$\Omega m$]', fontsize=textsize)
			ax3.set_ylabel('freq.', fontsize=textsize)
			ax3.grid(True, which='both', linewidth=0.1)
			# plot normal fit 
			(mu, sigma) = norm.fit(r1_batch)
			med = np.median(r1_batch)
			try:
				y = mlab.normpdf(bins, mu, sigma)
			except:
				#y = stats.norm.pdf(bins, mu, sigma)
				pass
			#ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
			#ax3.legend(loc='upper right', shadow=False, fontsize=textsize)
			#ax3.set_title('$med$:{:1.1e}, $\mu$:{:1.1e}, $\sigma$: {:1.1e}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
			ax3.set_title('$\mu$:{:1.1e}, $\sigma$: {:1.1e}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
			ax3.plot([med,med],[0,np.max(h)],'y-')
			
			# r2
			bins = np.linspace(np.min(r2_batch), np.max(r2_batch), int(np.sqrt(len(r2_batch))))
			h,e = np.histogram(r2_batch, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax4.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
			ax4.set_xlabel(r'$\rho_2$ [$\Omega m$]', fontsize=textsize)
			ax4.set_ylabel('freq.', fontsize=textsize)
			ax4.grid(True, which='both', linewidth=0.1)
			# plot normal fit 
			(mu, sigma) = norm.fit(r2_batch)
			med = np.median(r2_batch)
			try:
				y = mlab.normpdf(bins, mu, sigma)
			except:
				#y = stats.norm.pdf(bins, mu, sigma)
				pass
			#ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
			#ax2.legend(loc='upper right', shadow=False, fontsize=textsize)
			ax4.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
			ax4.plot([med,med],[0,np.max(h)],'g-')

			# r3
			bins = np.linspace(np.min(r3_batch), np.max(r3_batch), int(np.sqrt(len(r3_batch))))
			h,e = np.histogram(r3_batch, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax5.bar(e[:-1], h, e[1]-e[0], alpha = .8, edgecolor = 'w')#, label = 'histogram')
			ax5.set_xlabel(r'$\rho_3$ [$\Omega m$]', fontsize=textsize)
			ax5.set_ylabel('freq.', fontsize=textsize)
			ax5.grid(True, which='both', linewidth=0.1)
			# plot normal fit 
			(mu, sigma) = norm.fit(r3_batch)
			med = np.median(r3_batch)
			try:
				y = mlab.normpdf(bins, mu, sigma)
			except:
				#y = stats.norm.pdf(bins, mu, sigma)
				pass
			#ax2.plot(bins, y, 'r--', linewidth=2, label = 'normal fit')
			#ax2.legend(loc='upper right', shadow=False, fontsize=textsize)
			ax5.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
			ax5.plot([med,med],[0,np.max(h)],'m-')

			# plor legend 
			ax_leg.plot([],[],'r-',label = r'median of $z_1$')
			ax_leg.plot([],[],'b-',label = r'median of $z_2$')
			ax_leg.plot([],[],'y-',label = r'median of $\rho_1$')
			ax_leg.plot([],[],'g-',label = r'median of $\rho_2$')
			ax_leg.plot([],[],'m-',label = r'median of $\rho_3$')
			ax_leg.legend(loc='center', shadow=False, fontsize=textsize)
			ax_leg.axis('off')

			f.tight_layout()
			plt.savefig('.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'hist_pars_nsta_'+str(len(station_objects))+'.png', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)





			












		


				


