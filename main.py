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
	pc = 'office'
	#pc = 'personalSuse'
	#pc = 'personalWin'

	## Set of data to work with 
	full_dataset = False
	prof_WRKNW6 = False
	prof_WRKNW5 = True

	prof_NEMT2 = False
	
	prof_THNW03 = False
	prof_THNW04 = False
	prof_THNW05 = False

	## Sections of the code tu run
	set_up = True
	mcmc_meb_inv = False
	prior_MT_meb_read = True
	mcmc_MT_inv = False
	prof_2D_MT = False
	surf_3D_MT = True
	wells_temp_fit = False
	sta_temp_est = False

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

		## Create a directory of the name of the files of the stations
		pos_ast = path_files.find('*')
		file_dir = glob.glob(path_files)

		#########################################################################################
		#########################################################################################
		## Create station objects 
		# Defined lists of MT station 
		if full_dataset:
			sta2work = [file_dir[i][pos_ast:-4] for i in range(len(file_dir))]
		if prof_WRKNW6:
			sta2work = ['WT004a','WT015a','WT048a','WT091a','WT102a','WT111a','WT222a']
			sta2work = ['WT004a','WT015a','WT048a','WT091a','WT111a','WT222a']
			sta2work = ['WT091a','WT102a','WT111a','WT222a']
			#sta2work = ['WT091a']
		if prof_WRKNW5:
			sta2work = ['WT039a','WT024a','WT030a','WT501a','WT033a','WT502a','WT060a','WT071a','WT068a','WT223a','WT070b','WT107a','WT111a']#,'WT033c']
			#sta2work = ['WT033c']
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
		for file_aux in file_dir:
			if (file_aux[pos_ast:-4] in sta2work and file_aux[pos_ast:-4] != 'WT067a'):# incomplete station WT067a, no tipper
				file = file_aux[pos_ast:] # name on the file
				sta_obj = Station(file, count, path_files)
				sta_obj.read_edi_file() 
				sta_obj.rotate_Z()
				# import PT derotated data 
				sta_obj.read_PT_Z(pc = pc) 
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
		if prof_WRKNW6:
			wl2work = ['TH19','TH08','WK404','WK408','WK224','WK684','WK686'] #WK402
			wl2work = ['TH19','TH08','WK404','WK224','WK684','WK686'] #WK402
		if prof_WRKNW5:
			wl2work = ['WK260','WK261','WK262','WK263','WK243','WK267A','WK270','TH19','WK408','WK401', 'WK404']
			#wl2work = ['WK260'] 
		if prof_NEMT2:
			wl2work = ['TH12','TH18','WK315B','WK227','WK314','WK302']

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
				#if wl.name == 'WK401':
				print(wl.name +  ': {:}/{:}'.format(count, count_meb_wl))
				mcmc_wl = mcmc_meb(wl)
				mcmc_wl.run_mcmc()
				mcmc_wl.plot_results_mcmc()
				#
				f = mcmc_wl.sample_post(exp_fig = False, plot_fit_temp = True, wl_obj = wl, \
					temp_full_list_z1 = temp_full_list_z1, temp_full_list_z2 = temp_full_list_z2) # Figure with fit to be add in pdf pp
				#f = mcmc_wl.sample_post_temp(exp_fig = True) # Figure with fit to be add in pdf 
				pp.savefig(f)
				plt.close("all")
				## calculate estimate parameters (percentiels)
				mcmc_wl.model_pars_est()
				count += 1

		# save lists: temp_full_list_z1, temp_full_list_z2
		with open('corr_z1_z1_temp_glob.txt', 'w') as f:
			for f1, f2 in zip(temp_full_list_z1, temp_full_list_z2):
				print(f1, f2, file=f)	
		shutil.move('corr_z1_z1_temp_glob.txt','.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'corr_cc_temp'+os.sep+'corr_z1_z1_temp_glob.txt')
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
		calc_prior_meb_quadrant(station_objects, wells_objects)

	# (3) Run MCMC inversion for each staion, obtaning 1D 3 layer res. model
	# 	  Sample posterior, construct uncertain resistivity distribution and create result plots 
	if mcmc_MT_inv:
		## create pdf file to save the fit results for the whole inversion 
		pp = PdfPages('fit.pdf')
		start_time = time.time()
		prior_meb = False
		for sta_obj in station_objects:
			if sta_obj.ref <  161:
		#	if False:
				pass
			else: 
				print('({:}/{:}) Running MCMC inversion:\t'.format(sta_obj.ref+1,len(station_objects))+sta_obj.name[:-4])

				## range for the parameters
				par_range = [[.1*1e2,.5*1e3],[1.*1e1,1*1e3],[.1*1e1,1.*1e3],[1.*1e-3,.5*1e1],[.5*1e1,1.*1e3]]
				## create object mcmc_inv 
				#mcmc_sta = mcmc_inv(sta_obj)
				# inv_dat: weighted data to invert [1,1,1,1,0,0,0]
				mcmc_sta = mcmc_inv(sta_obj, prior='uniform', inv_dat = [1,1,1,1,0,0,0], prior_input=par_range, \
					walk_jump = 2000, prior_meb = prior_meb, range_p = [0.,1.0], autocor_accpfrac = True)
				if prior_meb:
					print("	wells for MeB prior: {} ".format(sta_obj.prior_meb_wl_names))
					#print("	[[z1_mean,z1_std],[z2_mean,z2_std]] = {} \n".format(sta_obj.prior_meb))
					print("	distances = {} \n".format(sta_obj.prior_meb_wl_dist)) 
				## run inversion 
				mcmc_sta.inv()
				## plot results (save in .png)
				mcmc_sta.plot_results_mcmc(corner_plt = True, walker_plt = True)
				## sample posterior
				#mcmc_sta.sample_post()
				f = mcmc_sta.sample_post(idt_sam = True, plot_fit = True, exp_fig = True, plot_model = True) # Figure with fit to be add in pdf (whole station)
				#mcmc_sta.sample_post(idt_sam = True, plot_fit = True, exp_fig = False, plot_model = True) # Figure with fit to be add in pdf (whole station)
				pp.savefig(f)
				plt.close(f)
				plt.clf()
				## calculate estimate parameters
				mcmc_sta.model_pars_est()

		## enlapsed time for the inversion (every station in station_objects)
		enlap_time = time.time() - start_time # enlapsed time
		## print time consumed
		print("Time consumed:\t{:.1f} min".format(enlap_time/60))
		pp.close()
		# move figure fit to global results folder
		shutil.move('fit.pdf','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'00_fit.pdf')

	
	# (4) Plot 2D profile of unceratain boundaries z1 and z2 (results of mcmc MT inversion)
	if prof_2D_MT:
		print('(4) Plot 2D profile of unceratain boundaries z1 and z2 (results of mcmc MT inversion)')
		# quality inversion pars. plot (acceptance ratio and autocorrelation time)
		autocor_accpfrac = True
		# load mcmc results and assign to attributes of pars to station attributes 
		load_sta_est_par(station_objects, autocor_accpfrac = autocor_accpfrac)
		# Create figure of unceratain boundaries of the clay cap and move to mcmc_inversions folder
		file_name = 'z1_z2_uncert'
		#plot_2D_uncert_bound_cc(station_objects, pref_orient = 'EW', file_name = file_name) # width_ref = '30%' '60%' '90%', 
		plot_2D_uncert_bound_cc_mult_env(station_objects, pref_orient = 'EW', file_name = file_name, 
			width_ref = '90%', prior_meb = wells_objects)#, plot_some_wells = ['WK404'])#,'WK401','WK402'])
		shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+file_name+'.png')

		# plot autocorrelation time and acceptance factor 
		if autocor_accpfrac:
			file_name = 'autocor_accpfrac'
			plot_profile_autocor_accpfrac(station_objects, pref_orient = 'EW', file_name = file_name)
			shutil.move(file_name+'.png','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+file_name+'.png')

		## create text file for google earth, containing names of MT stations considered 
		for_google_earth(station_objects, name_file = '00_stations_4_google_earth.txt', type_obj = 'Station')
		shutil.move('00_stations_4_google_earth.txt','.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion' \
			+os.sep+'00_stations_4_google_earth.txt')

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
			perc = np.arange(5.,100.,5.) # percentiels to calculate: [5% , 10%, ..., 95%]
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
			isoth = ['50','100','150','200','250']
			plot_2D_uncert_isotherms(station_objects, wells_objects, pref_orient = 'EW', file_name = 'isotherm_uncert',\
				percentiels = perc, isotherms = isoth) 





		


				


