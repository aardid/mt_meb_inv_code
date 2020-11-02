"""
.. module:: Wairakei/Tauhara MT inversion and Temp. extrapolation
   :synopsis: Dimnsional analysis of MT data 

.. moduleauthor:: Alberto Ardid 
				  University of Auckland 
				  
.. conventions:: 
	:order in impedanze matrix [xx,xy,yx,yy]
	: number of layer 3 (2 layers + half-space)
    : distances in meters
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
from lib_sample_data import *
from Maping_functions import *
from misc_functios import *
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.neighbors import KernelDensity

textsize = 15
matplotlib.rcParams.update({'font.size': textsize})
pale_orange_col = u'#ff7f0e' 
pale_blue_col = u'#1f77b4' 
pale_red_col = u'#EE6666'
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
	## PC that the code will be be run ('ofiice', 'personalSuse', 'personalWin')
	pc = 'office'
	#pc = 'personalMac'
	# ==============================================================================
	## Set of data to work with 
	full_dataset = True
	# Profiles
	#
	# Filter has qualitu MT stations
	filter_lowQ_data_MT = True
	# ==============================================================================
	## Sections of the code tu run
	set_up = True
	dif_xy_yx = False
	z_strike = True
	phase_ellipses = True

	# (0) Import data and create objects: MT from edi files and wells from spreadsheet files
	if set_up:
		#### Import data: MT from edi files and wells from spreadsheet files
		#########  MT data
		if pc == 'office': 
			#########  MT data
			path_files = "D:\workflow_data\kk_full\*.edi" 	# Whole array 

		## Data paths for personal's pc SUSE (uncommend the one to use)
		if pc == 'personalMac':
			#########  MT data
			path_files = os.sep+'Users'+os.sep+'macadmin'+os.sep+'Documents'+os.sep+'WT_MT_inv'+os.sep+'data'+os.sep+'Wairakei_Tauhara_data'+os.sep+'MT_survey'+os.sep+'EDI_Files'+os.sep+'*.edi'

		## Create a directory of the name of the files of the stations
		pos_ast = path_files.find('*')
		file_dir = glob.glob(path_files)

		#########################################################################################
		## Create station objects 
		# Defined lists of MT station 
		if full_dataset:
			sta2work = [file_dir[i][pos_ast:-4] for i in range(len(file_dir))]

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
	
	# (1) Diference in apparent resitivity of XY and YX
	if dif_xy_yx:
    	#(0) create plot
		fig, (ax1, ax2) = plt.subplots(2, 1)
		fig.set_size_inches(8,12)
		fig.suptitle(' ')
		# lists to fill
		dif_appres = [] # matrix
		#(1) loop over stations 
		for sta in station_objects:
			ax1.loglog(sta.T, sta.rho_app[1], '-', color = pale_red_col, alpha = .3, linewidth =0.3)
			ax1.loglog(sta.T, sta.rho_app[2], '-', color = pale_blue_col, alpha = .3, linewidth =0.3)

		ax1.loglog([],[], '-', color = pale_red_col, alpha = .5, label = 'Z_xy')
		ax1.loglog([],[], '-', color = pale_blue_col, alpha = .5, label = 'Z_yx')
		#ax1.title('Apparent resistivity: WT survey', fontsize = textsize)
		ax1.set_xlabel('Period [s]',  fontsize = textsize)
		ax1.set_ylabel(r'\rho_\text{app} [\Omega\,s]',  fontsize = textsize)
		ax1.set_xlim([0.0001,1000])
		ax1.set_ylim([0.01,10000])		
		plt.tight_layout()
		plt.show()
		#(1) 

		#(2)

	# (2) z_strike
	
	Zstrikes_array = []
	if z_strike: 
		by_depth = True
		if by_depth:
			Zstrikes_depths_1 = []
			Zstrikes_depths_2 = []
			Zstrikes_depths_3 = []	
			d1_from = 0.  # m
			d1_to =   500.    # m
			d2_from = d1_to  # m
			d2_to =   5000.    # m
			d3_from = d2_to  # m
			d3_to =   50000.    # m
			###
			# Figure
			f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1)
			#f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
			f.set_size_inches(10,14)	

			#(1) loop over stations 
			for sta in station_objects:
				# Z strike between 0 and 90 
				Z_strike_aux = np.zeros(len(sta.Z_strike))
				for i, zs in enumerate(sta.Z_strike):
					if zs < 0.: 
						Z_strike_aux[i] = zs + 180
					else:
						Z_strike_aux[i] = zs
				for i, zs in enumerate(Z_strike_aux):
					if zs > 90.: 
						Z_strike_aux[i] = zs - 90.
					else:
						Z_strike_aux[i] = zs	
				# Zstrike to plot (original or corrected)
				Z_strike_2_plot = sta.Z_strike
				Z_strike_2_plot = Z_strike_aux

				## calc of skin depths 
				# classic simple skin depths
				ave_appres = (sta.rho_app[1]+sta.rho_app[2])/2.
				sd_vec = 500*np.sqrt(sta.T*ave_appres) # m
				# Schmucker skin depth
				if True:
					mu_o =  1.257e-6
					# XY
					zxy = sta.Z_xy[2]
					z_ask_xy = (zxy/(1j*np.pi*2/sta.T)).real
					app_res_xy = sta.rho_app[1]
					phase_de_xy = sta.phase_deg[1]
					phase_ra_xy = ((2*np.pi)/360) * phase_de_xy
					res_ask_xy =  np.zeros(len(z_ask_xy))
					for i in range(len(phase_de_xy)):
						if phase_de_xy[i] >= 45.0:
							res_ask_xy[i] = 2*app_res_xy[i] * np.cos(phase_ra_xy[i])**2
						else: 
							res_ask_xy[i] = app_res_xy[i] / (2*np.sin(phase_ra_xy[i])**2)
					app_res_schm_xy = abs(res_ask_xy - z_ask_xy)  # check is abs() is ok
    				# YX
					zyx = sta.Z_yx[2]
					z_ask_yx = (zyx/(1j*np.pi*2/sta.T)).real
					app_res_yx = sta.rho_app[2]
					phase_de_yx = sta.phase_deg[2]
					phase_ra_yx = ((2*np.pi)/360) * phase_de_yx
					res_ask_yx =  np.zeros(len(z_ask_yx))
					for i in range(len(phase_de_yx)):
						if phase_de_yx[i] >= 45.0:
							res_ask_yx[i] = 2*app_res_yx[i] * np.cos(phase_ra_yx[i])**2
						else: 
							res_ask_yx[i] = app_res_yx[i] / (2*np.sin(phase_ra_yx[i])**2)
					app_res_schm_yx = abs(res_ask_yx - z_ask_yx)  # check is abs() is ok
					## skin depths
					app_res_schm = (app_res_schm_xy + app_res_schm_yx)/2 
					sd_vec = np.sqrt(sta.T*app_res_schm/(2.*np.pi*mu_o)) # skin depth
				##
				for i, sd in enumerate(sd_vec):
					if sd < d1_to: 
						Zstrikes_depths_1.append(Z_strike_2_plot[i])
						ax1.plot(sd/1e3, Z_strike_2_plot[i], color = pale_orange_col, marker = '.')
					elif sd < d2_to:
						Zstrikes_depths_2.append(Z_strike_2_plot[i])
						ax1.plot(sd/1e3, Z_strike_2_plot[i], color = pale_blue_col, marker = '.')
					elif sd < d3_to:
						Zstrikes_depths_3.append(Z_strike_2_plot[i])
						ax1.plot(sd/1e3, Z_strike_2_plot[i], color = pale_red_col, marker = '.')
			ax1.plot([], [], color = pale_orange_col, marker = '.', label = '['+str(d1_from/1e3)+', '+str(d1_to/1e3)+'] km')
			ax1.plot([], [], color = pale_blue_col, marker = '.', label = '['+str(d2_from/1e3)+', '+str(d2_to/1e3)+'] km')
			ax1.plot([], [], color = pale_red_col, marker = '.', label = '['+str(d3_from/1e3)+', '+str(d3_to/1e3)+'] km')
			ax1.set_title(r'(a) $Z_{strike}$ vs skin depth for the MT array', fontsize=textsize, loc='center')
			ax1.set_xlabel(r'depth [km]', fontsize=textsize)
			ax1.set_ylabel(r'$Z_{strike}$ [°]', fontsize=textsize)
			ax1.grid(True, which='both', linewidth=0.1)				

			# HIST depths 1
			bins = np.linspace(np.min(Zstrikes_depths_1), np.max(Zstrikes_depths_1), int(.5*np.sqrt(len(Zstrikes_depths_1))))
			h,e = np.histogram(Zstrikes_depths_1, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax2.bar(e[:-1], h, e[1]-e[0], alpha = .6, edgecolor = 'w',  zorder = 1, color = pale_orange_col)
			ax2.set_xlabel(r'$Z_{strike}$ angle [°]', fontsize=textsize)
			ax2.set_ylabel('samples (n°'+str(len(Zstrikes_depths_1))+')', fontsize=textsize)
			ax2.grid(True, which='both', linewidth=0.1)
			ax2.set_title(r'(b) $Z_{strike}$ for depths: ['+str(d1_from/1e3)+', '+str(d1_to/1e3)+'] km', fontsize=textsize, loc='center')

			# HIST depths 2
			bins = np.linspace(np.min(Zstrikes_depths_2), np.max(Zstrikes_depths_2), int(.75*np.sqrt(len(Zstrikes_depths_2))))
			h,e = np.histogram(Zstrikes_depths_2, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax3.bar(e[:-1], h, e[1]-e[0], alpha = .6, edgecolor = 'w',  zorder = 1, color = pale_blue_col)
			ax3.set_xlabel(r'$Z_{strike}$ angle [°]', fontsize=textsize)
			ax3.set_ylabel('samples (n°'+str(len(Zstrikes_depths_2))+')', fontsize=textsize)
			ax3.grid(True, which='both', linewidth=0.1)	
			ax3.set_title(r'(c) $Z_{strike}$ for depths: ['+str(d2_from/1e3)+', '+str(d2_to/1e3)+'] km', fontsize=textsize, loc='center')

			# HIST depths 3
			bins = np.linspace(np.min(Zstrikes_depths_3), np.max(Zstrikes_depths_3), int(.75*np.sqrt(len(Zstrikes_depths_3))))
			h,e = np.histogram(Zstrikes_depths_3, bins)
			m = 0.5*(e[:-1]+e[1:])
			ax4.bar(e[:-1], h, e[1]-e[0], alpha = .6, edgecolor = 'w',  zorder = 1, color = pale_red_col)
			ax4.set_xlabel(r'$Z_{strike}$ angle [°]', fontsize=textsize)
			ax4.set_ylabel('samples (n°'+str(len(Zstrikes_depths_3))+')', fontsize=textsize)
			ax4.grid(True, which='both', linewidth=0.1)
			ax4.set_title(r'(d) $Z_{strike}$ for depths: ['+str(d3_from/1e3)+', '+str(d3_to/1e3)+'] km', fontsize=textsize, loc='center')
			plt.tight_layout()
			#save 
			plt.savefig('.'+os.sep+'dat_dim_ana'+os.sep+'Z_strike_skin_depth_'+str(d1_from/1e3)+'_'+str(d3_to/1e3)+'_km'+'.png', dpi=100, facecolor='w', edgecolor='w',
				orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)












		


				


