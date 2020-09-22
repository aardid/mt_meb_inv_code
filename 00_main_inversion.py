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
	pc = 'personalMac'
	# ==============================================================================
	## Set of data to work with 
	full_dataset = True # True always
	# Profiles
	prof_WRKNW6 = False
	prof_WRKNW5 = True
	array_WRKNW5_WRKNW6 = False
	prof_WRK_EW_7 = False # PW_TM_AR
	prof_WRK_SENW_8 = False # KS_OT_AR
	prof_WT_NS_1 = False # KS_OT_AR
	#
	prof_TH_SENW_2 = False # KS_OT_AR
	prof_NEMT2 = False
	prof_THNW03 = False
	prof_THNW04 = False
	prof_THNW05 = False
	#
	# Filter has qualitu MT stations
	filter_lowQ_data_MT = False
	# Filter MeB wells with useless info (for prior)
	filter_useless_MeB_well = True
	## run with quality filter per well
	filter_lowQ_data_well = True  # need to be checked, not working: changing the orther of well obj list
	## re model temp profiles from wells with lateral inflows ()
	temp_prof_remodel_wells = True # re model base on '.'+os.sep+'corr_temp_bc'+os.sep+'RM_temp_prof.txt'
	# Stations not modeled
	sta_2_re_invert = False
	# ==============================================================================
	## Sections of the code tu run
	set_up = True
	mcmc_meb_inv = False
	prior_MT_meb_read = True
	mcmc_MT_inv = True
	plot_2D_MT = True
	plot_3D_MT = False
	wells_temp_fit = False
	sta_temp_est = False
	files_paraview = False

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
            #path_wells_temp_date = 	os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_date.txt'
			path_wells_temp_date = 	os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_date_3_new_wells.txt'
			# Column order: Well	Depth [m]	Interpreted Temperature [deg C]	Reduced Level [m]
			####### MeB data in wells 
			path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data.txt"
			#path_wells_meb = "D:\Wairakei_Tauhara_data\MeB_wells\MeB_data_sample.txt"	

		## Data paths for personal's pc SUSE (uncommend the one to use)
		if pc == 'personalMac':
			#########  MT data
			path_files = os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'MT_survey'+os.sep+'EDI_Files'+os.sep+'*.edi'
			# Whole array 			
			####### Temperature in wells data
			path_wells_loc = 		os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_location_latlon.txt'
			path_wells_temp = 		os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_fixTH12_rmTHM24_fixWK404.txt'
			#path_wells_temp_date = 	os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_date.txt'
			path_wells_temp_date = 	os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'Temp_wells'+os.sep+'well_depth_redDepth_temp_date_3_new_wells.txt'

			####### MeB data in wells 
			path_wells_meb = 		os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'MeB_wells'+os.sep+'MeB_data.txt'

		## Create a directory of the name of the files of the stations
		pos_ast = path_files.find('*')
		file_dir = glob.glob(path_files)

		#########################################################################################
		## Create station objects 
		# Defined lists of MT station 
		if full_dataset:
			sta2work = [file_dir[i][pos_ast:-4] for i in range(len(file_dir))]
			if prof_WRKNW5:
				sta2work = ['WT039a','WT024a','WT030a','WT501a','WT502a','WT060a','WT071a', \
					'WT068a','WT070b','WT223a','WT107a','WT111a']			
				#sta2work = ['WT111a']
			if prof_WRK_EW_7:
				sta2work = ['WT169a','WT008a','WT006a','WT015a','WT023a','WT333a','WT060a', \
					'WT507a','WT103a','WT114a','WT140a','WT153b','WT172a','WT179a'] # 'WT505a','WT079a','WT148a'
			if prof_WRK_SENW_8:
				sta2work = ['WT225a','WT066a','WT329a','WT078a','WT091a','WT107a','WT117b',
					'WT122a','WT130a','WT140a','WT152a','WT153b'] # ,'WT150b'
			if prof_WT_NS_1:
				sta2work = ['WT061a','WT063a','WT069b','WT513a','WT082a','WT098a','WT096a', \
					'WT099a','WT119a','WT125a','WT128a','WT036a','WT308a','WT156a','WT028a',\
						'WT086a','WT055a','WT300a'] # 'WT117b' 
			if prof_TH_SENW_2:
				sta2work = ['WT192a','WT306a','WT149a','WT328a','WT323a','WT199a',
					'WT156a','WT166a','WT168a','WT185a','WT040a','WT313a','WT202a',\
						'WT197a'] # ,'WT150b', 'WT340a', 'WT307a',
		#########################################################################################
		## Loop over the file directory to collect the data, create station objects and fill them
		station_objects = []   # list to be fill with station objects
		count  = 0
		# remove bad quality stations from list 'sta2work' (based on inv_pars.txt)
		if filter_lowQ_data_MT: 
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
				# import PT derotated data 
				sta_obj.read_PT_Z(pc = pc) 
				sta_obj.app_res_phase() # [self.rho_app, self.phase_deg, self.rho_app_er, self.phase_deg_er]
				## Create station objects and fill them
				station_objects.append(sta_obj)
				count  += 1
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
        # name exepctions: 
		for i, wl_meb in enumerate(wl_name_meb):
			if wl_meb in ['WK409','WK123','WK124']: # wells with an A in temp profiles but not A in MeB profiles
				wl_name_meb[i] = wl_meb+'A'
		#########################################################################################
		## Create wells objects
		# Defined lists of wells
		if full_dataset:
			wl2work = wl_name
			# add to wl2work names of well with meb data and no temp
			wls_meb_notemp = []
			for wl_meb in wl_name_meb:
				if wl_meb not in wl_name:
					#wl2work.append(wl_meb)
					wls_meb_notemp.append(wl_meb)
			# list of wells with bad quality temperature, wells with bas quality temp data 
			wls_BQ_temp = []
			name_file =  '.'+os.sep+'corr_temp_bc'+os.sep+'Q_temp_prof.txt'
			BQ_wls = [x.split()[0] for x in open(name_file).readlines() if x[0]!='#' and x[-2] is '0']
			[wls_BQ_temp.append(wl_bq) for wl_bq in BQ_wls] 

		#########################################################################################
		# ## Loop over the wells to create objects and assing data attributes 
		wells_objects = []   # list to be fill with station objects
		count  = 0
		count2 = 0
		for wl in wl_name:
			if wl in wl2work: # and wl != 'THM24':
				# create well object
				wl_obj = Wells(wl, count)
				# Search for location of the well and add to attributes	
				for i in range(len(wells_location)): 
					wl_name = wells_location[i][0]
					if wl_obj.name == wl_name: 
						wl_obj.lat_dec = wells_location[i][2]
						wl_obj.lon_dec = wells_location[i][1]
						wl_obj.elev = wells_location[i][3]
				# check if well have meb data and no temp data 
				#if wl in wls_meb_notemp:
				#    wl_obj.no_temp = True
			
				#   if not wl_obj.no_temp:
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
				if abs(wl_obj.red_depth[-1] - wl_obj.elev) > 10.: #[m]
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
				# check if remodel is need fro temperature profile in the well
				# it removes some points defore the cubic spline interpolation (-> resample profile)
				if temp_prof_remodel_wells:
					# read list_rm = RM_temp_prof.txt and make a copy of red_depth and temp_prof_true
					with open('.'+os.sep+'corr_temp_bc'+os.sep+'RM_temp_prof.txt') as p:
						next(p) # skip header
						for line in p:
							line = line.strip('\n')
							currentline = line.split('\t')
							# check is the wl match with any of the wells in list_rm[0]
							if currentline[0] == wl:
								# if it match -> remove data points from red_depth2 and temp_prof_true2 between list_rm[1] and list_rm[2]
								# depth_from
								val, idx_from = find_nearest(wl_obj.elev - np.asarray(wl_obj.red_depth), float(currentline[1]))
								# depth_to
								val, idx_to = find_nearest(wl_obj.elev - np.asarray(wl_obj.red_depth), float(currentline[2]))
								# aux list red_depth2
								red_depth2 = wl_obj.red_depth[:idx_from] + wl_obj.red_depth[idx_to:] 
								temp_prof_true2 = wl_obj.temp_prof_true[:idx_from] + wl_obj.temp_prof_true[idx_to:] 
								# perform the SCI on red_depth2 and temp_prof_true2
								xi = np.asarray(red_depth2)
								yi = np.asarray(temp_prof_true2)
								N_rs = 500 # number of resample points data
								xj = np.linspace(xi[0],xi[-1],N_rs)	
								yj = cubic_spline_interpolation(xi,yi,xj, rev = True)
								# add attributes
								wl_obj.red_depth_rs = xj
								wl_obj.temp_prof_rs = yj
								# save wl_obj.red_depth_rs and wl_obj.temp_prof_rs
							else:
								pass

				# if wl == 'TH20':
				#     f1 = plt.figure(figsize=[9.5,7.5])
				#     ax1 = plt.axes([0.15,0.15,0.75,0.75]) 
				#     ax1.plot(wl_obj.temp_prof_rs,-1*(wl_obj.elev - wl_obj.red_depth_rs), 'r-')

				#     ax1.plot(wl_obj.temp_prof_true,-1*(wl_obj.elev - np.asarray(wl_obj.red_depth)),'b*')
				#     plt.grid()
				#     plt.show()

				if filter_lowQ_data_well: 
					if wl_obj.name in wls_BQ_temp:
						wl_obj.no_temp = True
				## add well object to directory of well objects
				wells_objects.append(wl_obj)
				count  += 1
				#if not wl_obj.no_temp:
				count2 +=1

		# create well objects of wells with meb data and no temp data
		for wl in wls_meb_notemp:
			# create well object
			wl_obj = Wells(wl, count)
			# Search for location of the well and add to attributes	
			for i in range(len(wells_location)): 
				wl_name = wells_location[i][0]
				if wl_obj.name == wl_name: 
					wl_obj.lat_dec = wells_location[i][2]
					wl_obj.lon_dec = wells_location[i][1]
					wl_obj.elev = wells_location[i][3]
			wells_objects.append(wl_obj)
			count  += 1

		# #if filter_lowQ_data_well: 
		# if True: 
		# 	name_file =  '.'+os.sep+'corr_temp_bc'+os.sep+'Q_temp_prof.txt'
		# 	BQ_wls = [x.split()[0] for x in open(name_file).readlines() if x[0]!='#' and x[-2] is '0']
		# 	wl2work = [x for x in wl2work if not x in BQ_wls]

		## Loop wells_objects (list) to assing data attributes from MeB files 
		# list of wells with MeB (names)
		
		if filter_useless_MeB_well:
			# extract meb mcmc results from file 
			useless_MeB_well = []
			with open('.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'wls_meb_Q.txt') as p:
				next(p) # skip header
				for line in p:
					line = line.strip('\n')
					currentline = line.split(",")
					if int(currentline[3]) == 0:
						useless_MeB_well.append(str(currentline[0]))
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
				try:
					if wl.name in useless_MeB_well:
						wl.meb = False
				except:
					pass	
		## create folder structure
		if True:
			try: 
				os.mkdir('.'+os.sep+'corr_temp_bc'+os.sep+'00_global')
			except:
				pass
			for wl in wells_objects:
				try: 
					os.mkdir('.'+os.sep+'corr_temp_bc'+os.sep+wl.name)
				except:
					pass

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
				if count >= 0:
				#if wl.name == 'WKM15':
					mes_err = 2.
					## filter meb logs: when values are higher than 5 % => set it to 5 % (in order to avoid control of this points)
					fiter_meb = True # fix this per station criteria
					if fiter_meb:
						filt_val= 7.		
						wl.meb_prof = [filt_val if ele > filt_val else ele for ele in wl.meb_prof]
					print(wl.name +  ': {:}/{:}'.format(count, count_meb_wl))
					mcmc_wl = mcmc_meb(wl, norm = 2., scale = 'lin', mes_err = mes_err, walk_jump=3000)
					mcmc_wl.run_mcmc()
					mcmc_wl.plot_results_mcmc()
					#
					if wl.no_temp:
						f = mcmc_wl.sample_post(exp_fig = True, plot_hist_bounds = True, plot_fit_temp = False) # Figure with fit to be add in pdf pp
					else:
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
		calc_prior_meb(station_objects, wells_objects, slp = 4*10., quadrant = False) # calc prior at MT stations position
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
				coords = [175.99,176.178,-38.69,-38.59] # [min lon, max lon, min lat, max lat]
				# fn. for griding and calculate prior => print .txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
				file_name = 'grid_meb_prior'
				path_output = '.'+os.sep+'plain_view_plots'+os.sep+'meb_prior'
				try:
					os.mkdir('.'+os.sep+'plain_view_plots'+os.sep+'meb_prior')
				except:
					pass
				##
				# image background
				# image 1
				path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd.jpg'
				ext_file = [175.934859, 176.226398, -38.722805, -38.567571]
				# image 2
				#path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd_3.jpg'
				#ext_file = [175.781956, 176.408620, -38.802528, -38.528097]
				#
				x_lim = [175.95,176.21]
				y_lim = None #[-38.68,-38.58]
				#x_lim = [175.99,176.21]
				#y_lim = [-38.75,-38.58]
				# call function 
				grid_meb_prior(wells_objects, coords = coords, n_points = 20, slp = 4*10., \
					file_name = file_name, path_output = path_output,plot = True, \
						path_base_image = path_base_image, ext_img = ext_file, \
							xlim = x_lim, ylim = y_lim, cont_plot = True, scat_plot = True)
				
	# (3) Run MCMC inversion for each staion, obtaning 1D 3 layer res. model
	# 	  Sample posterior, construct uncertain resistivity distribution and create result plots 
	if mcmc_MT_inv:
		## create pdf file to save the fit results for the whole inversion
		pdf_fit = False 
		if pdf_fit:
			pp = PdfPages('fit.pdf')
		start_time_f = time.time()
		prior_meb = None  # if False -> None; extra condition inside the loop: if every app res value is > 10 ohm m, prior_meb False
		prior_meb_weigth = 1.0
		station_objects.sort(key=lambda x: x.ref, reverse=False)
		# run inversion
		if True:
			for sta_obj in station_objects:
				if sta_obj.ref < 0: # start at 0
				#if sta_obj.name[:-4] != 'WT130a': #sta2work = ['WT122','WT130a','WT115a']
					pass
				else: 
					print('({:}/{:}) Running MCMC inversion:\t'.format(sta_obj.ref+1,len(station_objects))+sta_obj.name[:-4])
					verbose = True
					## range for the parameters
					par_range = [[.01*1e2,2.*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
					#par_range = [[.01*1e2,.5*1e3],[.5*1e1,.5*1e3],[1.*1e1,1.*1e3],[1.*1e0,1.*1e1],[1.*1e1,1.*1e3]]
					#par_range = [[.01*1e2,2.*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,20.*1e1],[.5*1e1,1.*1e3]]
					# error floor
					error_max_per = [5.,2.5] # [10.,5.]	[20.,10.]			
					## inv_dat: weighted data to invert and range of periods
					## 		inv_dat = [1,1,1,1] # [appres zxy, phase zxy, appres zyx, phase zyx]
					## range_p = [0.001,10.] # range of periods
					if False: # import inversion parameters from file 
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
					####### condition for MeB prior
					# if prior_meb:
					# 	if all([sta_obj.rho_app[1][i] > 10. for i in range(len(sta_obj.rho_app[1])-10)]):
					# 		prior_meb = False
					# 	if all([sta_obj.rho_app[2][i] > 10. for i in range(len(sta_obj.rho_app[2])-10)]):
					# 		prior_meb = False

					# inversion pars. per station (manual)
					if True:
						if sta_obj.name[:-4] == 'WT024a': # station with static shift
							error_max_per = [5.,2.5]
							range_p = [0.005,100.] # range of periods
							prior_meb = True
							# for two layers: 
							#par_range = [[.01*1e2,2.*1e3],[0.*1e1,1.*1e0],[1.*1e1,1.*1e5],[.50*1e1,.51*1e1],[.5*1e1,1.*1e3]]
						if sta_obj.name[:-4] == 'WT039a': # station with static shift
							range_p = [0.001,100.] # range of periods
							error_max_per = [5.,2.5]
							inv_dat = [0,0,1,1]
						if sta_obj.name[:-4] == 'WT030a': # station with static shift
							inv_dat = [1,1,0,0]
							range_p = [0.001,10.] # range of periods
						if sta_obj.name[:-4] == 'WT060a': # station with static shift
							range_p = [0.005,1.] # range of periods
							inv_dat = [1,1,0,0]
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
							range_p = [0.005,3.] # range of periods
							error_max_per = [5.,2.5]
						if sta_obj.name[:-4] == 'WT107a': # station with static shift
							par_range = [[.01*1e2,.5*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
							range_p = [0.001,5.] # range of periods
							error_max_per = [5.,2.5]
						if sta_obj.name[:-4] == 'WT111a': # station with static shift
							range_p = [0.001, 5.] # range of periods
						if sta_obj.name[:-4] == 'WT223a': # station with static shift
							range_p = [0,10.] # range of periods
							error_max_per = [20.,5.]
						if sta_obj.name[:-4] == 'WT501a': # station with static shift
							range_p = [0.005,5.] # range of periods
						if sta_obj.name[:-4] == 'WT502a': # station with static shift
							#range_p = [0,5.] # range of periods
							par_range = [[.01*1e2,.5*1e3],[.5*1e1,1.*1e3],[1.*1e1,1.*1e5],[1.*1e0,.5*1e1],[.5*1e1,1.*1e3]]
							range_p = [0.005,5.] # range of periods
						if sta_obj.name[:-4] == 'WT003a': # station with static shift
							error_max_per = [20.,10.]	
							#inv_dat = [1,0,1,0]

					###### run inversion
					## print relevant information
					if verbose:
						print('range of periods: [{:2.3f}, {:2.2f}] [s]'.format(range_p[0],range_p[1]))
						print('inverted data: '+str(inv_dat))
					## plot noise
					try: 
						mcmc_sta = mcmc_inv(sta_obj, prior='uniform', inv_dat = inv_dat, prior_input = par_range, \
							walk_jump = walk_jump, nwalkers = nwalkers, prior_meb = prior_meb, prior_meb_weigth = prior_meb_weigth,\
								range_p = range_p, autocor_accpfrac = True, data_error = True, \
									error_mean = error_mean, error_max_per=error_max_per)
					except: # in case that inversion breaks due to low number of independet samples 
						try:
							mcmc_sta = mcmc_inv(sta_obj, prior='uniform', inv_dat = inv_dat, prior_input = par_range, \
								walk_jump = walk_jump+1000, nwalkers = nwalkers+10, prior_meb = prior_meb, prior_meb_weigth = prior_meb_weigth,\
									range_p = range_p, autocor_accpfrac = True, data_error = True, \
										error_mean = error_mean, error_max_per=error_max_per)
						except:
							pass

					if error_max_per:
						## plot noise
						try:
							name_file='noise_appres_phase_error_floor'
							path_img = 'mcmc_inversions'+os.sep+sta_obj.name[:-4]
							sta_obj.plot_noise(path_img = path_img, name_file = name_file)
						except:
							pass
					if prior_meb:
						if verbose:
							#print("	wells for MeB prior: {} ".format(sta_obj.prior_meb_wl_names))
							print("Near wells for MeB prior: ")
							for sta in sta_obj.prior_meb_wl_names:
								print(str(sta.name))
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
						mcmc_sta.sample_post(idt_sam = True, plot_fit = True, rms = True, exp_fig = False, plot_model = True) # Figure with fit to be add in pdf (whole station)		
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
					## delete chain.dat
					#os.remove('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+'chain.dat')

			# save rms stations 
			rms_appres_list = []
			rms_phase_list  = []
			rms_file = open('.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'rms_misfit.txt','w')
			rms_file.write('Station RMS misfit for apparent resistivity and phase, based on chi-square misfit (Pearson, 1900)'+'\n')
			for sta_obj in station_objects:
				rms_sta = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'rms_misfit.txt',skip_header=1).T
				rms_file.write(sta_obj.name[:-4]+'\t'+str(np.round(rms_sta[0],2))+'\t'+str(np.round(rms_sta[1],2))+'\n')
				rms_appres_list.append(rms_sta[0])
				rms_phase_list.append(rms_sta[1]) 
			rms_file.write('\n')
			rms_file.write('mean'+'\t'+str(np.mean(rms_sta[0]))+'\t'+str(np.mean(rms_sta[1]))+'\n')
			rms_file.close()

			## enlapsed time for the inversion (every station in station_objects)
			enlap_time_f = time.time() - start_time_f # enlapsed time
			## print time consumed
			print("Time consumed:\t{:.1f} min".format(enlap_time_f/60))
			if pdf_fit:
				pp.close()
				# move figure fit to global results folder
				shutil.move('fit.pdf','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'00_fit.pdf')
		# save fitting plot in folder '01_sta_model'
		try:
			os.mkdir('.'+os.sep+'mcmc_inversions'+os.sep+'01_sta_model')
		except:
			pass
		for sta_obj in station_objects:
			shutil.copy('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'app_res_fit.png', '.'+os.sep+'mcmc_inversions'+os.sep+'01_sta_model'+os.sep+'app_res_fit_'+sta_obj.name[:-4]+'.png')

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
		prior_meb = True
		if prior_meb:
			if prof_WRK_EW_7:
				plot_some_wells = ['WK681','WK122','WK315B','WKM15','WK321'] # 'WK314','WK682','WK123'
			elif prof_WRK_SENW_8:
				plot_some_wells = ['WK402','WK404','WK321','WK315B','WK318'] # 'WK308' ,'WK403'
				#'WK308','WK304','WK317','WK318'
			elif prof_TH_SENW_2: 
				plot_some_wells = ['THM15','THM16','THM14','THM19','TH13','TH18','TH12']
			elif prof_WT_NS_1:
				plot_some_wells = ['THM15','THM21','THM16','THM14','THM17','THM13','WK123','WKM14','WK124','WK122']
			else:
				plot_some_wells = False
		else:
			plot_some_wells = False
		# for plotting lithology 
		litho_plot = True
		if litho_plot:
			if prof_WRK_EW_7:
				plot_litho_wells = ['WK315B','WKM14','WKM15','WK681'] # 'WK315b','WKM14','WKM15','WK681'
				plot_litho_wells = ['WK315B','WKM14','WKM15','WK681','WK263','WK124A','WK317'] # 'WK315b','WKM14','WKM15','WK681','WK318'
			elif prof_WRK_SENW_8:
				plot_litho_wells = ['WK402','WK404','WK317','WK315B'] # 'WK403'
				plot_litho_wells = ['WK402','WK404','WK317','WK315B','WK318'] # 'WK403'
			else: 
				plot_litho_wells = False
		else: 
			plot_litho_wells = False
		if False: # plot resisitvity boundary reference
			if prof_WRK_EW_7:
				# risk, 1984
				path_rest_bound_inter = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'rb_1980_coord_for_WK7.txt' 
				# update, 2015, IN
				#path_rest_bound_inter = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'rb_2015_IN_coord_for_WK7.txt' 
				# update, 2015, OUT
				#path_rest_bound_inter = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'rb_2015_OUT_coord_for_WK7.txt' 

			elif prof_WRK_SENW_8:
				path_rest_bound_inter = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'rb_1980_coord_for_WK8.txt' 
				# update, 2015, IN
				#path_rest_bound_inter = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'rb_2015_IN_coord_for_WK8.txt' 
				# update, 2015, OUT
				#path_rest_bound_inter = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'rb_2015_OUT_coord_for_WK8.txt' 
		else:
			path_rest_bound_inter = False
		# plot temperature contours
		temp_count = True
		if temp_count: 
			if prof_WRK_EW_7:
				temp_iso = [100.,160.,180.,200.]
				temp_count_wells = ['WK650','WK210','WK217','WK052','WK019','WK060','WK048','WK045','WK059','WK301'] # 'WK681','WKM14','WKM15','WK313', 'WK045'
				position_label = None
			elif prof_WRK_SENW_8:
				temp_iso = [100.,140.,180.,200.]
				#temp_count_wells = ['WK402','WK407','WK226','WK310','WK301'] # WK321,WK036 WK133
				temp_count_wells = ['WK402','WK407','WK226','WK310','WK301'] 
				position_label = 'mid'
			else:
				temp_count_wells = False
				temp_iso = False
				position_label = None
		## thickness of second layer to be considerer 'no conductor'
		mask_no_cc = 80.#112.
		##
		pref_orient = 'EW'
		if prof_WT_NS_1:
			pref_orient = 'NS'
		if prof_WRK_EW_7:
			title = 'Uncertain conductor boundaries: profile WK7'
		else:
			title = False
		
		plot_2D_uncert_bound_cc_mult_env(station_objects, pref_orient = pref_orient, file_name = file_name, \
			width_ref = '90%', prior_meb = prior_meb, wells_objects = wells_objects , plot_some_wells = plot_some_wells,\
				 mask_no_cc = mask_no_cc, rest_bound_ref = path_rest_bound_inter, plot_litho_wells = plot_litho_wells,\
					 temp_count_wells = temp_count_wells, temp_iso = temp_iso, position_label = position_label, title = title)
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
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = False, path_plots = path_plots)
			bound2plot = 'bottom' 
			file_name = 'interface_LRA_'+bound2plot+'_rest_bound'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = False, path_plots = path_plots)
			
			# for plot with topo background
			path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd_2.jpg'
			bound2plot = 'top' # top bound
			file_name = 'interface_LRA_'+bound2plot+'_topo'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = False, path_plots = path_plots)
			bound2plot = 'bottom' 
			file_name = 'interface_LRA_'+bound2plot+'_topo'
			plot_surface_cc_count(station_objects, wells_objects, file_name = file_name, bound2plot = bound2plot, type_plot = type_plot,format = 'png', \
				path_base_image = path_base_image, alpha_img = 0.6, ext_img = ext_file, xlim = x_lim, ylim = y_lim, hist_pars = False, path_plots = path_plots)

		if False: # plot plain view with countours
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
			if True:
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

		if True: # plot plain vien as scatter plot (colorbar as depths)
			## load z1 and z2 pars
			# load mcmc results and assign to attributes of pars to station attributes 
			load_sta_est_par(station_objects)
			file_name = 'MT_inv' 
			try:
				os.mkdir('.'+os.sep+'plain_view_plots'+os.sep+'MT_inv'+os.sep+'Scatter')
			except:
				pass
			path_output = '.'+os.sep+'plain_view_plots'+os.sep+'MT_inv'+os.sep+'Scatter'

			# define region to grid
			coords = [175.97,176.200,-38.74,-38.58] # [min lon, max lon, min lat, max lat]
			# fn. for griding and calculate prior => print .txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
			##
			img_back_topo_ge = True
			img_back_rest_bound = False
			# image background: topo 
			if img_back_topo_ge:
				path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_area_gearth_hd_3.jpg'
				ext_file = [175.781956, 176.408620, -38.802528, -38.528097]
			# image background: rest_bound 
			if img_back_rest_bound:
				path_base_image = '.'+os.sep+'base_map_img'+os.sep+'WT_res_map_gearth_2.jpg'
				ext_file = [175.948466, 176.260520, -38.743590, -38.574484]             
			x_lim = [175.9,176.3]
			y_lim = None #[-38.68,-38.57]
			# call function 
			#if False: # contourf plot
			#	print('gridding temp at cond. bound')
			#	grid_temp_conductor_bound(wells_objects, coords = coords, n_points = 100, slp = 5., file_name = file_name, path_output = path_output,\
			#		plot = True, path_base_image = path_base_image, ext_img = ext_file, xlim = x_lim, masl = False)
			# scatter plot of temps at conductor boundaries
			if img_back_topo_ge: # scatter plot
				x_lim = [175.95,176.23]#[175.98,176.22] 
				y_lim = [-38.78,-38.57]
				WK_resbound_line = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat'
				taupo_lake_shoreline= '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'shoreline_TaupoLake.dat'
				scatter_MT_conductor_bound(station_objects, path_output = path_output, alpha_img = 0.6,\
					path_base_image = path_base_image, ext_img = ext_file, xlim = x_lim, ylim = y_lim, \
						WK_resbound_line = WK_resbound_line, taupo_lake_shoreline = taupo_lake_shoreline)

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
					os.mkdir('.'+os.sep+'mcmc_inversions'+os.sep+'01_sta_model')
				except:
					pass
				for sta_obj in station_objects:
					shutil.copy('.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'app_res_fit.png', '.'+os.sep+'mcmc_inversions'+os.sep+'01_sta_model'+os.sep+'app_res_fit_'+sta_obj.name[:-4]+'.png')

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
			# Resistivity Boundary, Risk
			path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat'
			# Resistivity Boundary, Mielke, OUT
			path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
			#histogram_mcmc_MT_inv_results(station_objects, filt_in_count=path_rest_bound_WT, filt_out_count=path_rest_bound_WT, type_hist = 'overlap')
			#histogram_mcmc_MT_inv_results(station_objects, filt_in_count=path_rest_bound_WT, filt_out_count=path_rest_bound_WT, type_hist = 'sidebyside')
			histogram_mcmc_MT_inv_results_multisamples(station_objects, filt_in_count=path_rest_bound_WT)

		if False:   # histogram of MeB inversion parameters for wells 
			path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat'
			#histogram_mcmc_meb_inv_results(wells_objects, filt_in_count=path_rest_bound_WT, filt_out_count=path_rest_bound_WT, type_hist = 'overlap')
			histogram_mcmc_meb_inv_results(wells_objects, filt_in_count=path_rest_bound_WT, filt_out_count=path_rest_bound_WT, type_hist = 'sidebyside')

		if False:   # .dat of latlon for of wells and MT stations 
			# mt
			mt_loc = open('.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_MT_stations.dat','w')
			for sta in station_objects:
				mt_loc.write(str(sta.lon_dec)+','+str(sta.lat_dec)+'\n')
			mt_loc.close()
			# wells
			wl_loc = open('.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls.dat','w')
			wlmeb_loc = open('.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls_meb.dat','w')
			for wl in wells_objects:
				wl_loc.write(str(wl.lon_dec)+','+str(wl.lat_dec)+'\n')
				if wl.meb:
					wlmeb_loc.write(str(wl.lon_dec)+','+str(wl.lat_dec)+'\n')
			wl_loc.close()
			wlmeb_loc.close()

		if False:   # .dat with results meb inversion, mt inversion, and temp estimation at boundaries of conductor 
			# mcmc MeB results 
			if False:
				wl_meb_results = open('.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'wl_meb_results.dat','w')
				wl_meb_results.write('well_name'+','+'lon_dec'+','+'lat_dec'+','+'z1_mean'+','+'z1_std'+','+'z2_mean'+','+'z2_std'+'\n')
				# name lat lon file
				wls_loc = open('.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'wls_meb_loc.txt','w')
				wls_loc.write('well_name'+','+'lon_dec'+','+'lat_dec'+'\n')	
				for wl in wells_objects:
					if wl.meb:
						# extract meb mcmc results from file 
						meb_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+str(wl.name)+os.sep+"est_par.dat")
						# values for mean a std for normal distribution representing the prior
						wl.meb_z1_pars = [meb_mcmc_results[0,1], meb_mcmc_results[0,2]] # mean [1] z1 # median [3] z1 
						wl.meb_z2_pars = [meb_mcmc_results[1,1], meb_mcmc_results[1,2]] # mean [1] z1 # median [3] z1 
						# write results 
						wl_meb_results.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','+str(wl.meb_z1_pars[0])\
							+','+str(wl.meb_z1_pars[1])+','+str(wl.meb_z2_pars[0])+','+str(wl.meb_z2_pars[1])+'\n')
						wls_loc.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+'\n')
				wl_meb_results.close()
				wls_loc.close()
			# mcmc MT results 
			if False:
				sta_mcmc_results = open('.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'mt_sta_results.dat','w')
				sta_mcmc_results.write('sta_name'+','+'lon_dec'+','+'lat_dec'+','+'z1_mean'+','+'z1_std'+','+'z2_mean'+','+'z2_std'+'\n')
				for sta in station_objects:
					# extract meb mcmc results from file 
					mt_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+str(sta.name[:-4])+os.sep+"est_par.dat")
					# values for mean a std for normal distribution representing the prior
					sta.z1_pars = [mt_mcmc_results[0,1], mt_mcmc_results[0,2]] # mean [1] z1 # median [3] z1 
					sta.z2_pars = [mt_mcmc_results[1,1], mt_mcmc_results[1,2]] # mean [1] z1 # median [3] z1 
					# write results 
					sta_mcmc_results.write(str(sta.name[:-4])+','+str(sta.lon_dec)+','+str(sta.lat_dec)+','
						+str(sta.z1_pars[0])+','+str(sta.z1_pars[1])+','+str(sta.z2_pars[0])+','+str(sta.z2_pars[1])+'\n')
				sta_mcmc_results.close()
			# temp at z1 an z2 in wells  and z1 and z2 at well positions
			if True:
				wl_temp_z1_z2 = open('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_T1_T2.dat','w')
				wl_temp_z1_z2.write('wl_name'+','+'lon_dec'+','+'lat_dec'+','+'T1_mean'+','+'T1_std'+','+'T2_mean'+','+'T2_std'+'\n')
				for wl in wells_objects:
					# extract meb mcmc results from file 
					try:
						if not wl.no_temp:
							wl_temp_results = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+str(wl.name)+os.sep+"conductor_T1_T2.txt")
							# values for mean a std for normal distribution representing the prior
							wl.T1_pars = [wl_temp_results[0], wl_temp_results[1]] # mean [1] z1 # median [3] z1 
							wl.T2_pars = [wl_temp_results[2], wl_temp_results[3]] # mean [1] z1 # median [3] z1 
							# write results 
							wl_temp_z1_z2.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','
								+str(wl.T1_pars[0])+','+str(wl.T1_pars[1])+','+str(wl.T2_pars[0])+','+str(wl.T2_pars[1])+'\n')
					except:
						pass
				wl_temp_z1_z2.close()
				
				##### D1 and D2 at well location 
				wl_d1_d2 = open('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2.dat','w')
				wl_d1_d2.write('wl_name'+','+'lon_dec'+','+'lat_dec'+','+'D1_mean(depth to top of cond.)'+','+'D1_std'+','+'D2_mean(depth to bottom of cond.)'+','+'D2_std'+'\n')
				for wl in wells_objects:
					# extract z1 and z2 results from file 
					try:
						wl_z1_z2_results = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+str(wl.name)+os.sep+"conductor_z1_z2.txt", skip_header=1)
						# values for mean a std for normal distribution representing the prior
						wl.z1_pars = [wl_z1_z2_results[0], wl_z1_z2_results[1]] # mean [1] z1 # median [3] z1 
						wl.z2_pars = [wl_z1_z2_results[2], wl_z1_z2_results[3]] # mean [1] z1 # median [3] z1 
						# write results 
						wl_d1_d2.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','
							+str(wl.z1_pars[0])+','+str(wl.z1_pars[1])+','+str(wl.z1_pars[0] + wl.z2_pars[0])+','+str(wl.z2_pars[1])+'\n')
					except:
						pass
				wl_d1_d2.close()

				##### D1, T1 and D2,T2 at well location 
				wl_d1_d2_T1_T2 = open('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat','w')
				wl_d1_d2_T1_T2.write('wl_name'+','+'lon_dec'+','+'lat_dec'+','+'D1_mean(depth to top of cond.)'+','+'D2_mean(depth to bottom of cond.)'+','+'T1_mean'+','+'T2_mean'+'\n')
				for wl in wells_objects:
					# extract z1 and z2 results from file 
					try:
						if not wl.no_temp:
							wl_z1_z2_results = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+str(wl.name)+os.sep+"conductor_z1_z2.txt", skip_header=1)
							wl_temp_results = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+str(wl.name)+os.sep+"conductor_T1_T2.txt")
							# values for mean a std for normal distribution representing the prior
							wl.z1_pars = [wl_z1_z2_results[0], wl_z1_z2_results[1]] # mean [1] z1 # median [3] z1 
							wl.z2_pars = [wl_z1_z2_results[2], wl_z1_z2_results[3]] # mean [1] z1 # median [3] z1 
							wl.T1_pars = [wl_temp_results[0], wl_temp_results[1]] # mean [1] z1 # median [3] z1 
							wl.T2_pars = [wl_temp_results[2], wl_temp_results[3]] # mean [1] z1 # median [3] z1 
							# write results 
							wl_d1_d2_T1_T2.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','+
								str(wl.z1_pars[0])+','+str(wl.z1_pars[0] + wl.z2_pars[0])+','+
								str(wl.T1_pars[0])+','+str(wl.T2_pars[0])+'\n')
					except:
						pass
				wl_d1_d2_T1_T2.close()

				##### Thermal Gradient, Thermal Conductivity and Heat Flux  at well location 
				wl_TG_TC_HF = open('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat','w')
				wl_TG_TC_HF.write('wl_name'+','+'lon_dec'+','+'lat_dec'+','+'TG(Thermal Gradient)[C/m]'+','+'TC(Thermal Conductivity)[W/mC]'+','+'HF(Heat Flux)[W/m2]'+'\n')
				for wl in wells_objects:
					# extract z1 and z2 results from file 
					try:
						if not wl.no_temp:
							wl_TG_TC_HF_results = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+str(wl.name)+os.sep+"conductor_TG_TC_HF.txt", skip_header=1)
							# values for mean a std for normal distribution representing the prior
							wl.thermal_grad = wl_TG_TC_HF_results[0]
							wl.thermal_cond = wl_TG_TC_HF_results[1]
							wl.heat_flux = wl_TG_TC_HF_results[2]
							# write results 
							wl_TG_TC_HF.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','+
								str(round(wl.thermal_grad,2))+','+str(round(wl.thermal_cond,1))+','+str(round(wl.heat_flux,2))+'\n')
					except:
						pass
				wl_TG_TC_HF.close()

		if False:   # .txt with names, lon, lat of wells with lithology
			# lito wells
			path = '.'+os.sep+'wells_info'+os.sep+'well_names_with_litho.txt' # this is filled manually
			wls_lito = []
			with open(path) as p:
				next(p)
				for line in p:
					line = line.strip('\n')
					wls_lito.append(line)
			# file
			wls_with_litho = open('.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+'wls_with_litho.txt','w')
			wls_with_litho.write('well_name'+','+'lon_dec'+','+'lat_dec'+'\n')
			for wl in wells_objects:
				if wl.name in wls_lito:
					wl.litho = True
					# write results 
					wls_with_litho.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+'\n')
			wls_with_litho.close()
			
		if False: 	# .dat of temp at fix depth in location of every well 
			#depth = 500. # [m] from surface 
			masl = 0  
			temp_at_masl = open('.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'temps_at_'+str(masl)+'m_masl.dat','w')
			temp_at_masl.write('well_name'+','+'lon_dec'+','+'lat_dec'+','+'tempC'+'\n')
			i = 0
			for wl in wells_objects:
				if not wl.litho:
					try:
						depth, idx = find_nearest(wl.red_depth_rs, masl)
						temp = wl.temp_prof_rs[idx]
						temp_at_masl.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+','+str(temp)+'\n')
					except:
						i+=1
						pass

			print('wells not considered: '+str(i)+'/'+str(len(wells_objects)))
			temp_at_masl.close()





			












		


				


