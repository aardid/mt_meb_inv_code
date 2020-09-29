"""
.. module:: map figure for Wairakei results
   :synopsis: plot topography and on top results from: MeB, MT, Temp.

.. moduleauthor:: Alberto Ardid 
				  University of Auckland 
				  
.. conventions:: 
	:
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
import matplotlib.patheffects as PathEffects
from scipy.interpolate import griddata

textsize = 15.
pale_orange_col = u'#ff7f0e' 
pale_blue_col = u'#1f77b4' 
pale_red_col = u'#EE6666'
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
##################### BASEMAP PLOTS
	if False: # base map figures
		base_map = True # plot map with wells and stations # modify .png ouput
		##
		zones = False # add injection and extraction zones
		wells_loc = False # add wells (plot temp, meb and litho wells)
		wells_meb_loc = False
		wells_litho_loc = False
		stations_loc = False # add mt stations
		mt_2d_profiles = True
		temp_fix_depth = False # temperature at fix depth (def 0 masl)
		######## Just one of the following can be 'True'
		meb_results = False # add scatter MeB results # modify .png ouput
		mt_results = True # add scatter MT results # modify .png ouput
		temp_results = False # add scatter Temp results  # modify .png ouput
		temp_grad = False # add scatter temperature gradient inside the conductor # modify .png ouput
		temp_hflux = False # add scatter conductive heatflux  # modify .png ouput
		temp_hflux_tot = False # add scatter conductive total heatflux (cond+adv)  # modify .png ouput
		z1_vs_temp_z1_basemap = False # z1 from MT vs temp at z1  # modify .png ouput
		d2_vs_temp_d2_basemap = False # d2 (z1+z2) from MT vs temp at d2  # modify .png ouput
		temp_grad_HF_basemap = False # temperature gradient and heat flux # modify .png ouput

		########
		# plot map with wells and stations
		if base_map:
			path_topo = '.'+os.sep+'base_map_img'+os.sep+'coords_elev'+os.sep+'Topography_25_m_vertices_LatLonDec.csv'#Topography_zoom_WT_re_sample_vertices_LatLonDec.csv'
			path_lake_shoreline = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'shoreline_TaupoLake.dat'
			path_faults = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'nzafd.json'
			# DC. Risk, 1984
			path_rest_bound = ['.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat', 
								'.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_RK_50ohmm.dat']
			# Update. Mielke, 2015
			path_rest_bound = ['.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_IN_Mielke.txt', 
								'.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt',
								'.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_RK_50ohmm.dat']
			path_powerlines = glob.glob('.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'powerlines'+os.sep+'*.dat')
			#
			x_lim = [175.95,176.23]#[175.98,176.22] 
			y_lim = [-38.79,-38.57]
			if z1_vs_temp_z1_basemap or d2_vs_temp_d2_basemap or temp_grad_HF_basemap:
				two_cbar = True
			else:
				two_cbar = False
			# base figure
			f, ax, topo_cb = base_map_region(path_topo = path_topo,  xlim = x_lim, ylim = y_lim,path_rest_bound = path_rest_bound,
				path_lake_shoreline = path_lake_shoreline, path_faults = False, path_powerlines = False, two_cbar = two_cbar)
		# add injection and extraction zones
		if zones:
			with_squares = False
			c_label = 'k'
			# Otupu
			if True:
				path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Otupu_inj.dat'
				lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'b:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)+0.01, np.min(lats)-0.005, 'Otupu', color='b', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else: 
					txt = plt.text(np.mean(lons), np.mean(lats), 'Otupu', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			# Pohipi west
			if True:
				path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'pohipi_west_inj.dat'
				lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'b:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)-0.005, np.min(lats)-0.005, 'Pohipi west', color='b', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.min(lons), np.mean(lats), 'Pohipi \n west', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])					
			# Te Mihi
			if True:
				path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'TeMihi2_prod.dat'
				lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'r:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.max(lons), np.max(lats)-0.001, 'Te Mihi', color='r', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.min(lons), np.mean(lats), 'Te Mihi', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			# West Bore
			if False:
				path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'WestBoreField_prod.dat'
				lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'r:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)-0.01, np.min(lats)-0.005, 'West Bore', color='r', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.mean(lons), np.mean(lats), 'West \n Bore', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])					
			# Karapiti South 
			if True:
				path_karapiti_s = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'karapiti_south_inj.dat'
				lats, lons = np.genfromtxt(path_karapiti_s, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'r:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)-0.005, np.min(lats)-0.005, 'Karapiti South', color='r', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.min(lons), np.mean(lats), 'Karapiti \n South', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

			# Aratiatia flats 
			if True:
				path_aratiatia = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Aratiatia_flats_inj_d.dat'
				lats, lons = np.genfromtxt(path_aratiatia, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'g:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)+0.005, np.max(lats)+0.005, 'Aratiatia Flats', color='g', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.mean(lons), np.max(lats), 'Aratiatia \n Flats', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			# Te Huka  
			if True:
				path_aratiatia = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Te_Huka_prod.dat'
				lats, lons = np.genfromtxt(path_aratiatia, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'r:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)+0.000, np.min(lats)-0.005, 'Te Huka', color='r', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.min(lons)-0.002, np.mean(lats), 'Te Huka', color=c_label, size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			# Mount Tauhara  
			if False:
				path_aratiatia = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'mount_tauhara.dat'
				lats, lons = np.genfromtxt(path_aratiatia, skip_header=1, delimiter=',').T
				if with_squares:
					plt.plot(lons, lats, 'g:' ,linewidth= 3, zorder = 7)
					# print name 
					txt = plt.text(np.min(lons)+0.005, np.min(lats)+0.005, 'Mount Tauhara', color='b', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				else:
					txt = plt.text(np.min(lons), np.mean(lats), 'Mount Tauhara', color='b', size=textsize, zorder = 9)
					txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			# for legend 
			if with_squares:
				plt.plot([],[], 'b:' ,linewidth= 3, label = 'Injection zone', zorder = 7)
				plt.plot([],[], 'r:' ,linewidth= 3, label = 'Production zone', zorder = 7)
				plt.plot([],[], 'g:' ,linewidth= 3, label = 'Decommissioned zone', zorder = 7)
			# add labels for reservoirs
			txt = plt.text(176.13, -38.732, 'Wairakei-Tauhara', color='darkorange', size=textsize, zorder = 7)
			txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			txt = plt.text(176.18, -38.645, 'Rotokawa', color='darkorange', size=textsize, zorder = 7)
			txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
			# add label for Mount Tauhara
			txt = plt.text(176.15-0.01, -38.70, 'Mount\n Tauhara', color='b', size=textsize, zorder = 7)
			txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

		######## DATA
		# add mt stations
		if stations_loc:
			path_mt_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_MT_stations.dat'
			lons, lats = np.genfromtxt(path_mt_locs, delimiter=',').T
			#plt.plot(lons, lats, marker = 7, markerfacecolor='m', markeredgecolor='w', zorder = 2, markersize=10)
			plt.plot(lons, lats, 'v', markerfacecolor='m', markeredgecolor='w', zorder = 2, markersize=10)
			plt.plot([], [], 'v' , marker = 7, c = 'm', zorder = 2, label = 'MT station', markersize=10)
		# add wells
		if wells_loc or wells_meb_loc or wells_litho_loc:
			path_wl_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls.dat'
			lons, lats = np.genfromtxt(path_wl_locs, delimiter=',').T
			if wells_loc:
				plt.plot(lons, lats, 's' , zorder = 2, markersize=6,  markerfacecolor='b', markeredgecolor='white')
				plt.plot([], [], 's' , c = 'b', zorder = 2, label = 'Well', markersize=8)
			if wells_meb_loc:
				# meb wells 
				path_wlmeb_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls_meb.dat'
				lons, lats = np.genfromtxt(path_wlmeb_locs, delimiter=',').T
				plt.plot(lons, lats, 's' , zorder = 8, markersize=6, markerfacecolor='white', markeredgecolor='b')
				#plt.plot(lons, lats, 's' , c = 'white', zorder = 8, markersize=6, markerfacecolor='none')
				plt.plot([], [], 's' , zorder = 2, label = 'Well with MeB data', markersize=8, markerfacecolor='white', markeredgecolor='b')
			if wells_litho_loc:
				# lito wells
				path = '.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+'wls_with_litho.txt'
				names, lons, lats = np.genfromtxt(path, delimiter=',').T
				#plt.plot(lons, lats, 's' , c = 'w', zorder = 2, markersize=6)
				#plt.plot(lons, lats, 's' , c = 'lime', zorder = 2, markersize=6, markerfacecolor='none')
				plt.plot(lons, lats, 's' , zorder = 8, markersize=6, markerfacecolor='lime', markeredgecolor='b')
				plt.plot([], [], 's' , zorder = 2, label = 'Well with lithology', markersize=8, markerfacecolor='lime', markeredgecolor='b')
				#plt.plot([], [], 's' , c = 'lime', zorder = 2, label = 'Well with lithology', markersize=8, markerfacecolor='none')
		# add MT profiles as lines
		if mt_2d_profiles:
			path_MT_profiles_coords = ['.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'prof_PW_TM_WB_AR_7.txt' 
				]#, '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'prof_KS_OT_AR_8.txt']#,
				#'.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'prof_SENW_TH2.txt']
			names = ['WK7','WK8']#,'TH2']
			colors = ['brown',u'#1f77b4', u'#ff7f0e']#, u'#2ca02c']
			for i, path in enumerate(path_MT_profiles_coords): 
				lats, lons = np.genfromtxt(path, delimiter=',').T
				plt.plot(lons, lats, '--' , c = colors[i], zorder = 7,linewidth = 3)
				#plt.plot([], [], '--' , c = colors[i], zorder = 0, label = names[i])
				## print name 
				#txt = plt.text(lons[1]-0.01, lats[1]-0.005, names[i], color=colors[i], size=textsize, zorder = 7)
				#txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
				txt = plt.text(lons[1], lats[1], 'A', color=colors[i], size=textsize, zorder = 7)
				txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

				txt = plt.text(lons[-1], lats[-1], "A'", color=colors[i], size=textsize, zorder = 7)
				txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
		# add temp. at fix depth
		if temp_fix_depth: 
			## countour plot of temp at fix depth
			#path_temps = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'temps_at_400m_masl.dat'
			#path_temps = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'temps_at_300m_masl.dat'
			#path_temps = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'temps_at_200m_masl.dat'
			#path_temps = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'temps_at_100m_masl.dat'
			path_temps = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'temps_at_0m_masl.dat'
			names, lons, lats, temp = np.genfromtxt(path_temps, skip_header=1, delimiter=',').T
			#plt.plot(lons, lats, '.')
			if False: # tricontour
				levels = [50,100,150,200,250,300]
				CS = ax.tricontour(lons, lats, temp, levels=levels, linewidths=1.0, colors='m', alpha = 0.8)
				ax.clabel(CS, CS.levels, inline=True, fontsize=8)	
			if True: #grid and contour
				# grid_x, grid_y = np.mgrid[min(lons):max(lons):100j, 0:1:200j]
				n_points = 100
				x = np.linspace(min(lons), max(lons), n_points) # long
				y = np.linspace(min(lats), max(lats), n_points) # lat
				grid_x, grid_y = np.meshgrid(x, y)
				points = [[lon,lat] for lon,lat in zip(lons,lats)]
				grid_z2 = griddata(points, temp, (grid_x, grid_y), method='cubic')
				# plot
				levels = [100,200,300]
				CS = ax.contour(grid_x, grid_y, grid_z2, levels=levels, linewidths=1.0, colors='m', alpha = 0.8)
				ax.clabel(CS, CS.levels, inline=True, fontsize=8)
		##################################################
		######## RESULTS 
		# add scatter MeB results 
		if meb_results: # 
			z1_mean = []
			z1_std = []
			z2_mean = []
			z2_std = []
			lon_stas = []
			lat_stas = []
			count = 0
			# import results 
			meb_result = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'wl_meb_results.dat', delimiter = ',', skip_header=1).T
			# columns: well_name[0],lon_dec[1],lat_dec[2],z1_mean[3],z1_std[4],z2_mean[5],z2_std[6]
			lon_stas = meb_result[1]
			lat_stas = meb_result[2]
			# array to plot
			if True: # z1 mean
				z1_mean = meb_result[3]
				array = z1_mean
				name = 'z1_mean'
				levels = np.arange(150,400,25)#(200,576,25)  # for mean z1
			if True: # z2 mean
				z2_mean = meb_result[5]
				array = z2_mean
				name = 'z2_mean'
				levels = np.arange(200,800,50)#(100,2000,25)  # for mean z1
			# scatter plot
			size = 200*np.ones(len(array))
			vmin = min(levels)
			vmax = max(levels)
			normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			cmap = 'winter'
			scatter_meb = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', cmap = cmap, \
				norm = normalize, zorder = 5)#, label = 'MeB inversion: z1 mean')#alpha = 0.5)
			ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
				zorder = 5, label = 'MeB inversion: z1 mean')#alpha = 0.5)
			#
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_meb_'+name+'.png'
		# add scatter MT results
		if mt_results: # 
			z1_mean = []
			z1_std = []
			z2_mean = []
			z2_std = []
			lon_stas = []
			lat_stas = []
			count = 0
			# import results 
			mt_result = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'mt_sta_results.dat', delimiter = ',', skip_header=1).T
			# columns: well_name[0],lon_dec[1],lat_dec[2],z1_mean[3],z1_std[4],z2_mean[5],z2_std[6]
			lon_stas = mt_result[1]
			lat_stas = mt_result[2]
			# array to plot
			if True: # z1_mean
				z1_mean = mt_result[3]
				name = 'z1_mean'
				array = z1_mean
				levels = np.arange(50,450,25)  # for mean z1
				title = 'Depth to the top of the conductor'
			if False: # z2_mean
				z2_mean = mt_result[5]
				name = 'z2_mean'
				array = z2_mean
				levels = np.arange(200,650,25)  # for mean z10
				title = 'Conductor Thickness'
			if False: # z2_std
				z2_std = mt_result[6]
				name = 'z2_std'
				array = z2_std
				array = 3*z2_std # 3*sigma contains ~99% of probability
				levels = np.arange(50,500,25)  # for mean z10
				title = 'Uncertainty in the Bottom of the Conductor depth'
			if False: # d2_mean
				z1_mean = mt_result[3]
				z2_mean = mt_result[5]
				name = 'd2_mean'
				array = z1_mean + z2_mean
				levels = np.arange(200,1000,25)  # for mean z1
				title = 'Depth to the Bottom of the Conductor'
			# scatter plot
			size = 200*np.ones(len(array))
			vmin = min(levels)
			vmax = max(levels)
			normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			scatter_MT = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', cmap = 'winter', \
				norm = normalize, zorder = 5)#, label = 'MT inversion: z1 mean')#alpha = 0.5)
			# ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
			# 	zorder = 5, label = 'MT inversion: '+name+', '+r'$\mu$: '+str(np.median(array))+', '+r'$\sigma$: '+str(np.round(np.std(array),1)))#alpha = 0.5)
			if name == 'z1_mean':
				ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
					zorder = 5, label = r'$z_1$: Depth to the top of the conductor at MT station')
			if name == 'z2_mean':
				ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
					zorder = 5, label = 'Thickness of the Conductor at MT station')
			if True:
				ax.set_title(title, size = textsize)
			# absence of CC (no_cc)
			#for i, z2 in enumerate(mt_result[5]):
			#	if (z2 < 75.):
			#		plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=28, zorder = 6) 
			#		plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=12, zorder = 6)
			for i, (z1,z2) in enumerate(zip(mt_result[3],mt_result[5])):
				d2 = z2+z1
				if (z2 < d2/5):
					plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=28, zorder = 6) 
					plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=12, zorder = 6)
			# label for infered absence of clay cap
			ax.plot([],[],'bx',markersize=12, \
				zorder = 6, label = 'Absence of conductor inferred at MT station')#alpha = 0.5)
			# plot profiles 
			if False:
				path_profs= ['.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'prof_PW_TM_WB_AR_7.txt',
					'.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'prof_KS_OT_AR_8.txt',
					'.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'mt_prof'+os.sep+'prof_SENW_TH2.txt']
				names = ['WK_7','WK_8','TH_2']
				for i,path in enumerate(path_profs):
					lats, lons = np.genfromtxt(path, skip_header=1, delimiter=',').T
					plt.plot(lons, lats, 'm--' ,linewidth = 2, zorder = 7)
					txt = plt.text(lons[0]-0.02, lats[0]-0.001, names[i], color='m', size=textsize, zorder = 7)
					txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
				plt.plot([],[], 'm--' ,linewidth = 2, label = 'MT profile', zorder = 7)
			#
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_MT_'+name+'.png'
		# add scatter Temp results 
		if temp_results: 
			T1_mean = []
			T1_std = []
			T2_mean = []
			T2_std = []
			lon_stas = []
			lat_stas = []
			count = 0
			# import results 
			temp_result = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_T1_T2.dat', delimiter = ',', skip_header=1).T
			# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
			lon_stas = temp_result[1]
			lat_stas = temp_result[2]
			# array to plot
			if False: # T1
				T1_mean = temp_result[3]
				array = T1_mean
				name = 'T1_mean'
			if True:  # T2
				T2_mean = temp_result[5]
				array = T2_mean
				name = 'T2_mean'
			# scatter plot
			size = 200*np.ones(len(array))
			# levels = np.arange(200,576,25)  # for mean z1
			## vmin = min(levels)
			# vmax = max(levels)
			#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			scatter_temp = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', 
				cmap = 'YlOrRd', zorder = 5)#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
			ax.scatter([],[], s = size, c = 'r', edgecolors = 'k', \
				zorder = 5, label = r'$T_2$: Wells temperature at the bottom of the conductor')#alpha = 0.5)
			#
			title = 'Depth to the Top of the Conductor'
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_temp_'+name+'.png'
		# add scatter geothermal gradient 
		if temp_grad:
			gg_mean = []
			lon_stas = []
			lat_stas = []
			count = 0
			# import results 
			gg_result = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat', delimiter = ',', skip_header=1).T
			# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
			lon_stas = gg_result[1]
			lat_stas = gg_result[2]
			array = gg_result[3] # array to plot: geothermal gradient 
			array = 1.e3*array # from m to km
			# array to plot
			name = 'geothermal_gradient'
			# scatter plot
			size = 200*np.ones(len(array))
			# levels = np.arange(200,576,25)  # for mean z1
			## vmin = min(levels)
			# vmax = max(levels)
			#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			scatter_temp = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', 
				cmap = 'spring_r', zorder = 5)#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
			ax.scatter([],[], s = size, c = 'pink', edgecolors = 'k', \
				zorder = 5, label = 'Temperature gradient at well location')#alpha = 0.5)
			# Need to
			ax.set_title('Geothermal gradient inside the conductor', size = textsize)
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_'+name+'.png'
		# add scatter Heat Flux 
		if temp_hflux or temp_hflux_tot:
			scatter_p = False # scatter plot in well locations. Total power calc as mean * area 
			contour_p = True  # contour plot. Total power calc as sum of grid points
			cmap_hf = 'GnBu'
			# if both 'True', both are plot 

			if scatter_p: # scatter plot in well locations. Total power calc as mean * area 
				HF_mean = []
				lon_stas = []
				lat_stas = []
				count = 0
				# import results 
				wl_lon_lat_TG_TC_HF = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat', delimiter = ',', skip_header=1).T
				# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
				lon_stas = wl_lon_lat_TG_TC_HF[1]
				lat_stas = wl_lon_lat_TG_TC_HF[2]
				# array to plot
				if True:  # HF
					HF_mean = wl_lon_lat_TG_TC_HF[5]
					array = HF_mean
					name = 'Heat_Flux'
				# scatter plot
				size = 100*np.ones(len(array))
				# levels = np.arange(200,576,25)  # for mean z1
				## vmin = min(levels)
				# vmax = max(levels)
				#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
				#cmap = 'GnBu'
				scatter_hf = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', 
					cmap = 'Oranges', zorder = 5)#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
				ax.scatter([],[], s = size, c = 'orange', edgecolors = 'k', \
					zorder = 5, label = 'Heat Flux at well locations')#alpha = 0.5)
				# Estimate heat flux for the whole field 
				def PolyArea(x,y):
					'''
					Calculate area define by polygon(x,y)
					'''
					return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
				path_area = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
				lats, lons = np.genfromtxt(path_area, skip_header=1, delimiter=',').T
				conv_to_meters = 111000. # 111 km
				area_m2 = PolyArea(lons*conv_to_meters,lats*conv_to_meters)
				area_km2 = area_m2/1.e6 # 171

				# filter wells: inside area
				# check if station is inside poligon 
				HF_inside = []
				poli_in = [[lons[i],lats[i]] for i in range(len(lats))]
				for i in range(len(wl_lon_lat_TG_TC_HF[0])):			
					val = ray_tracing_method(wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], poli_in)
					if val:
						HF_inside.append(wl_lon_lat_TG_TC_HF[5][i])
				HF_area_m2 = np.mean(HF_inside) * area_m2
				std_HF_area_m2 = np.std(HF_inside) * area_m2
				ax.set_title('Wairakei-Tauhara estimated POWER: '+str(round(HF_area_m2/1.e6,2))+' ± '+str(round(std_HF_area_m2/1.e6,2))+' [MW]', size = textsize)
				#ax.set_title('Estimated total POWER: '+str(round(HF_area_m2/1.e6,2))+' [MW]', size = textsize)
				#
				file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_'+name+'.png'

			if contour_p: # distribibution plot. Total power calc as sum of grid points
				HF_mean = []
				# import results 
				if temp_hflux:
					wl_lon_lat_TG_TC_HF = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat', delimiter = ',', skip_header=1).T
				if temp_hflux_tot:
					wl_lon_lat_TG_TC_HF = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HFC_HFT.dat', delimiter = ',', skip_header=1).T
				
				# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
				if temp_hflux:
					name = 'Heat_Flux_grid'	
				if temp_hflux_tot:
					name = 'Heat_Flux_Total_grid'
				lon_wls = wl_lon_lat_TG_TC_HF[1]
				lat_wls = wl_lon_lat_TG_TC_HF[2]
				# resistivity boundary
				path_area = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
				lats_rb, lons_rb  = np.genfromtxt(path_area, skip_header=1, delimiter=',').T
				poli_in = [[lons_rb[i],lats_rb[i]] for i in range(len(lats_rb))]
				## sampling the grid
				if False: # (testing) distribution from dist_3_bound vs HF
					# loop over wells pt1
					wl_dist_2_bound = []
					wl_hf = []
					for i in range(len(wl_lon_lat_TG_TC_HF[0])):
						dist_min = 1.e8
						lon_wl = wl_lon_lat_TG_TC_HF[1][i]
						lat_wl = wl_lon_lat_TG_TC_HF[2][i]
						# check if station is inside poligon
						val = ray_tracing_method(lon_wl,lat_wl, poli_in)
						if val: # wl is inside restbound
							# loop over restbound pt2 
							for j in range(len(lats_rb)):
								lon_rb = lons_rb[j]
								lat_rb = lats_rb[j]
								# calc distance between pt1 and pt2
								dist_aux = dist_two_points([lon_wl, lat_wl], [lon_rb, lat_rb], type_coord = 'decimal')
								# check if distance in min for the well
								if dist_aux < dist_min:
									dist_min = dist_aux
							# add dist_min list
							if dist_min != 1.e8:
								wl_dist_2_bound.append(dist_min)
								if temp_hflux:
									wl_hf.append(wl_lon_lat_TG_TC_HF[5][i])
								if temp_hflux_tot:
									wl_hf.append(wl_lon_lat_TG_TC_HF[6][i])
					#
					f1 = plt.figure(figsize=[7.5,5.5])
					ax1 = plt.axes([0.18,0.25,0.70,0.50]) 
					# plot tempT2 vs depthD2(at well location)
					# import 
					ax1.plot(wl_dist_2_bound, wl_hf, 'r*', markersize = 8,zorder = 3)
					ax1.set_ylim([0,2])
					ax1.set_xlabel('Distance to rest boundary [km]', size = textsize)
					ax1.set_ylabel('Heat Flux [W/m2]', size = textsize)
					f1.tight_layout()
					# save figure
					#plt.show()
					f1.savefig('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'dist2bound_vs_hf.png', \
						dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
					plt.close(f1)
				if True: # distribution from constant boundaries HF. distance weigth. 
					# weigth to bounds
					w_bound = True
					#### tune pars
					frac_grid = 100 # distance in meters per point in the grid 
					# array of fix pm2 values at resistivity boundary locations 
					frac = 20 # fraction of points from rb to be taken
					bound_pw2 =0.5
					if temp_hflux_tot:
						if w_bound:
							frac = 1#25 # fraction of points from rb to be taken
						else:
							frac = 25 # fraction of points from rb to be taken
						bound_pw2 =0.5#0.5
					####
					pm2_rb = [[lons_rb[i*frac], lats_rb[i*frac], bound_pw2] for i in range(int(len(lons_rb)/frac))]
					# 0.64 is HF at a well located in the boundary WK650
					# array of pm2 at well locatins
					if temp_hflux:
						pm2_wls = [[wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], wl_lon_lat_TG_TC_HF[5][i]] 
							for i in range(len(wl_lon_lat_TG_TC_HF[5]))]
					if temp_hflux_tot:
						if w_bound:
							n = 0
							pm2_wls = []
							while n < 20: # more weigth in interpolation to real points in well location (virtual points in the boundary)
								pm2_wls_aux = [[wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], wl_lon_lat_TG_TC_HF[6][i]] 
									for i in range(len(wl_lon_lat_TG_TC_HF[5]))]
								pm2_wls = pm2_wls + pm2_wls_aux
								n+=1
						else:
							pm2_wls_aux = [[wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], wl_lon_lat_TG_TC_HF[6][i]] 
								for i in range(len(wl_lon_lat_TG_TC_HF[5]))]
					del(pm2_wls_aux)
					# array of both
					pm2 = pm2_wls.copy()
					pm2 = pm2 + pm2_rb
					# grid surface
					n_points_x = int(111000*(max(lons_rb) - min(lons_rb)))
					n_points_x = int(n_points_x/frac_grid)
					n_points_y = int(111000*(max(lats_rb) - min(lats_rb)))
					n_points_y = int(n_points_y/frac_grid)
					#
					x = np.linspace(min(lons_rb), max(lons_rb), n_points_x) # long
					y = np.linspace(min(lats_rb), max(lats_rb), n_points_y) # lat
					X, Y = np.meshgrid(x, y)
					pm2_grid = X.copy()*0.
					pm2_aux = [pm2_wl[2] for pm2_wl in pm2]
					pm2_grid_list = []
					# calculate pm2 at grid location
					for j,lat in enumerate(y):
						for i,lon in enumerate(x):
							if True: #for wl in pm2_wls:  # pm2_wls : [[lon, lat, HF], [...], ...]
								# distances between point in grid and points
								dist_wls = [dist_two_points([wl[0], wl[1]], [lon, lat], type_coord = 'decimal')\
									for wl in pm2]
								dist_wls = list(filter(None, dist_wls))
								#
								if temp_hflux:
									dist_weigth = [1./d**2 for d in dist_wls]
								if temp_hflux_tot:
									dist_weigth = [1./d**2 for d in dist_wls]
								pm2_grid[j][i] = np.dot(pm2_aux,dist_weigth)/np.sum(dist_weigth)
								pm2_grid_list.append([lon,lat,pm2_grid[j][i]])
					# filter points inside rest bound
					lat_filt = []
					lon_filt = []
					pm2_filt_rb = []
					pm2_grid_filt_rb = []
					for wl in pm2_grid_list:
						# check if station is inside poligon
						val = ray_tracing_method(wl[0], wl[1], poli_in)
						if val: #val:
							# for grid plot
							pm2_grid_filt_rb.append(wl)
							# for means
							pm2_filt_rb.append(wl[2])
							# for scatter plot 
							lon_filt.append(wl[0])
							lat_filt.append(wl[1])

					# Calculate mean power (per m2) inside the field
					#aux = np.mean(pm2_grid_filt_rb, axis=0)
					hf_mean_m2 = np.mean(pm2_filt_rb)
					# Calculate power for the whole field 
					hf_full = sum(pm2_filt_rb*(frac_grid**2))
					# print power for the whole field
					try: # add error base on senstivity to HF at bound 
						if temp_hflux_tot:
							resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf_total.txt', delimiter = ',', skip_header=1).T
						else:
							resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf.txt', delimiter = ',', skip_header=1).T
						d_power = abs(power[-1] - power[0])/2
						print('Wairakei-Tauhara estimated POWER: '+str(round(hf_full/1.e6,2))+' ± '+str(round(d_power,2))+' [MW]')#, size = textsize)
					except:
						print('Wairakei-Tauhara estimated POWER: '+str(round(hf_full/1.e6,2))+' [MW]')#, size = textsize)

				# plottoing the grid
				if False: # plot grid with contourf
					name = 'Heat_Flux_Grid'
					levels = np.arange(bound_pw2,2.1,0.25)
					levels = np.arange(0.5,1.51,0.25)
					#cmap = plt.get_cmap('GnBu')
						#ax.set_aspect('equal')
					cf = ax.contourf(X,Y,pm2_grid,levels = levels,cmap=cmap_hf, alpha=.9, antialiased=True)
					try: # add error base on senstivity to HF at bound 
						resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf.txt', delimiter = ',', skip_header=1).T
						d_power = abs(power[-1] - power[0])/2
						ax.set_title('Wairakei-Tauhara estimated POWER: '+str(int(hf_full/1.e6))+' ± '+str(int(d_power))+' [MW]', size = textsize)
					except:
						ax.set_title('Wairakei-Tauhara estimated POWER: '+str(int(hf_full/1.e6))+' [MW]', size = textsize)

					#ax.set_title('Estimated total POWER: '+str(round(HF_area_m2/1.e6,2))+' [MW]', size = textsize)
					#
					file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_'+name+'.png'				
				if True: # plot integral as scatter 
					# scatter plot
					vmin = None#0. # bound_pw2
					vmax = None#max(pm2_filt_rb)-.2 
					if temp_hflux_tot: 
						vmin = 0.5#0. # bound_pw2
						vmax = 3.#max(pm2_filt_rb)-.2 
					size = (frac_grid/3)*np.ones(len(pm2_filt_rb))
					cf = ax.scatter(lon_filt,lat_filt, s = size, c = pm2_filt_rb, alpha=0.9, edgecolors = None, \
						vmin = vmin, vmax = vmax, cmap = cmap_hf, zorder = 4)#
					#cf = ax.scatter(lon_filt,lat_filt, s = size, c = pm2_filt_rb, alpha=0.8, edgecolors = None, \
					#	vmin = vmin, vmax = vmax, cmap = cmap_hf, zorder = 0)#

					if True: # plot wells
						path_wl_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls.dat'
						lons, lats = np.genfromtxt(path_wl_locs, delimiter=',').T
						plt.plot(lons, lats, 's' , c = 'gray', zorder = 4, markersize=3)
						plt.plot([], [], 's' , c = 'gray', zorder = 0, label = 'Well', markersize=8)
					try: # add error base on senstivity to HF at bound 
						if temp_hflux_tot:
							resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf_total.txt', delimiter = ',', skip_header=1).T
						else:
							resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf.txt', delimiter = ',', skip_header=1).T
						d_power = abs(power[-1] - power[0])/2
						d_power = abs(power[-1] - power[0])/2
						ax.set_title('Wairakei-Tauhara estimated POWER: '+str(round(hf_full/1.e6,1))+' ± '+str(round(d_power,1))+' [MW]', size = textsize)
					except:
						ax.set_title('Wairakei-Tauhara estimated POWER: '+str(int(hf_full/1.e6))+' [MW]', size = textsize)

					# plot resistivity boundary on top the scatter plot
					plt.plot(lons_rb, lats_rb, color = 'orange' ,linewidth = 2, zorder = 4)
					# plot well locations
					if False:
						path_wl_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls.dat'
						lons, lats = np.genfromtxt(path_wl_locs, delimiter=',').T
						plt.plot(lons, lats, '.' , c = 'gray', zorder = 5, markersize=4)
						plt.plot([], [], '.' , c = 'gray', zorder = 0, label = 'Well', markersize=8)

					if False: # plot scatter of HF in wells 
						if temp_hflux: 
							HF_mean = wl_lon_lat_TG_TC_HF[5]
						if temp_hflux_tot: 
							HF_mean = wl_lon_lat_TG_TC_HF[6]
						array = HF_mean
						# scatter plot
						size = 100*np.ones(len(array))
						# levels = np.arange(200,576,25)  # for mean z1
						## vmin = min(levels)
						# vmax = max(levels)
						#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
						#cmap = 'GnBu'
						scatter_hf = ax.scatter(lon_wls,lat_wls, s = size, c = array, edgecolors = 'gray', 
							vmin = vmin, vmax = vmax, cmap = cmap_hf, zorder = 5)#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
						ax.scatter([],[], s = size, c = 'white', edgecolors = 'gray', \
							zorder = 0, label = 'Heat Flux at well locations')#alpha = 0.5)

					file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_'+name+'.png'
							#
		# add scatter of z1 and temp at z1 (double colorbar)
		if z1_vs_temp_z1_basemap: # 
			if True: # plot z1
				z1_mean = []
				z1_std = []
				lon_stas = []
				lat_stas = []
				count = 0
				# import results 
				mt_result = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'mt_sta_results.dat', delimiter = ',', skip_header=1).T
				# columns: well_name[0],lon_dec[1],lat_dec[2],z1_mean[3],z1_std[4],z2_mean[5],z2_std[6]
				lon_stas = mt_result[1]
				lat_stas = mt_result[2]
				# array to plot
				if True: # z1_mean
					z1_mean = mt_result[3]
					array = z1_mean
					levels = np.arange(200,476,25)  # for mean z1
				# scatter plot
				size = 50*np.ones(len(array))
				vmin = min(levels)
				vmax = max(levels)
				normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
				scatter_MT = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', cmap = 'winter', \
					norm = normalize, zorder = 5)#, label = 'MT inversion: z1 mean')#alpha = 0.5)
				ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
					zorder = 5, label = 'MT inversion: z1 mean')#alpha = 0.5)
				# absence of CC (no_cc)
				#for i, z2 in enumerate(mt_result[5]):
				#	if (z2 < 75.):
				#		plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=28, zorder = 6) 
				#		plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=12, zorder = 6)
				for i, (z1,z2) in enumerate(zip(mt_result[3],mt_result[5])):
					d2 = z2+z1
					if (z2 < d2/4):
						plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=size[0], zorder = 6) 
						plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=size[0], zorder = 6)
			if True: # plot temp at z1
				T1_mean = []
				T1_std = []
				lon_stas = []
				lat_stas = []
				count = 0
				# import results 
				temp_result = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_T1_T2.dat', delimiter = ',', skip_header=1).T
				# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
				lon_stas = temp_result[1]
				lat_stas = temp_result[2]
				# array to plot
				if True: # T1
					T1_mean = temp_result[3]
					array = T1_mean
				# scatter plot
				size = 50*np.ones(len(array))
				# levels = np.arange(200,576,25)  # for mean z1
				## vmin = min(levels)
				# vmax = max(levels)
				#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
				scatter_temp = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', 
					cmap = 'YlOrRd', zorder = 5, marker = 's')#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
				ax.scatter([],[], s = size, c = 'r', edgecolors = 'k', marker = 's', \
					zorder = 5, label = 'Well temperature at: z1 mean')#alpha = 0.5)
				#		################### colorbar to plot
					#
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'base_map_'+'d1_mt_vs_T1'+'.png'
		# add scatter of z1 and temp at z1 (double colorbar)
		if d2_vs_temp_d2_basemap: # 
			if True: # plot d2
				z2_mean = []
				z2_std = []
				lon_stas = []
				lat_stas = []
				count = 0
				# import results 
				mt_result = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+'00_global_inversion'+os.sep+'mt_sta_results.dat', delimiter = ',', skip_header=1).T
				# columns: well_name[0],lon_dec[1],lat_dec[2],z1_mean[3],z1_std[4],z2_mean[5],z2_std[6]
				lon_stas = mt_result[1]
				lat_stas = mt_result[2]
				# array to plot
				if True: # z2_mean
					z1_mean = mt_result[3]
					z2_mean = mt_result[5]
					array = z1_mean + z2_mean
					levels = np.arange(400,700,25)  # for mean z1
				# scatter plot
				size = 50*np.ones(len(array))
				vmin = min(levels)
				vmax = max(levels)
				normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
				scatter_MT = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', cmap = 'winter', \
					norm = normalize, zorder = 5)#, label = 'MT inversion: z1 mean')#alpha = 0.5)
				ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
					zorder = 5, label = 'MT inversion: z2 mean')#alpha = 0.5)
				# absence of CC (no_cc)
				#for i, z2 in enumerate(mt_result[5]):
				#	if (z2 < 75.):
				#		plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=28, zorder = 6) 
				#		plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=12, zorder = 6)
				for i, (z1,z2) in enumerate(zip(mt_result[3],mt_result[5])):
					d2 = z2+z1
					if (z2 < d2/4):
						plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=12, zorder = 6) 
						plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=6, zorder = 6)
			if True: # plot temp at d2
				T1_mean = []
				T1_std = []
				lon_stas = []
				lat_stas = []
				count = 0
				# import results 
				temp_result = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_T1_T2.dat', delimiter = ',', skip_header=1).T
				# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
				lon_stas = temp_result[1]
				lat_stas = temp_result[2]
				# array to plot
				if True: # T2
					T2_mean = temp_result[5]
					array = T2_mean
				# scatter plot
				size = 50*np.ones(len(array))
				# levels = np.arange(200,576,25)  # for mean z1
				## vmin = min(levels)
				# vmax = max(levels)
				#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
				scatter_temp = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', 
					cmap = 'YlOrRd', zorder = 5, marker = 's')#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
				ax.scatter([],[], s = size, c = 'r', edgecolors = 'k', marker = 's', \
					zorder = 5, label = 'Well temperature at: z2 mean')#alpha = 0.5)
				#		################### colorbar to plot
					#
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'base_map_'+'d2_mt_vs_T2'+'.png'		
		# add scatter of geogradient and background of heat flux
		if temp_grad_HF_basemap:
			## 
			HF_tot = True # plot total heat fluc (adv + cond)
			##
			name = 'gg_hf'
			cmap_hf = 'GnBu'	
			# geothermal gradient 
			gg_mean = []
			lon_stas = []
			lat_stas = []
			count = 0
			# import results 
			gg_result = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat', delimiter = ',', skip_header=1).T
			if HF_tot: 
				gg_result = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HFC_HFT.dat', delimiter = ',', skip_header=1).T
			# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
			lon_stas = gg_result[1]
			lat_stas = gg_result[2]
			array = gg_result[3] # array to plot: geothermal gradient 
			array = 1.e3*array # from m to km
			# array to plot
			# scatter plot
			size = 200*np.ones(len(array))
			# levels = np.arange(200,576,25)  # for mean z1
			## vmin = min(levels)
			# vmax = max(levels)
			#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			scatter_temp = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', 
				cmap = 'spring_r', zorder = 5)#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
			ax.scatter([],[], s = size, c = 'pink', edgecolors = 'k', \
				zorder = 5, label = 'Temperature gradient at well location')#alpha = 0.5)
			# Need to
			ax.set_title('Geothermal gradient inside the conductor', size = textsize)

			############
			# add in the background heat flux extrapolation 

			HF_mean = []
			# import results 
			wl_lon_lat_TG_TC_HF = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat', delimiter = ',', skip_header=1).T
			if HF_tot:
				wl_lon_lat_TG_TC_HF = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HFC_HFT.dat', delimiter = ',', skip_header=1).T

			# columns: well_name[0],lon_dec[1],lat_dec[2],T1_mean[3],T1_std[4],T2_mean[5],T2_std[6]
			
			lon_wls = wl_lon_lat_TG_TC_HF[1]
			lat_wls = wl_lon_lat_TG_TC_HF[2]
			# resistivity boundary
			path_area = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
			# Wairakei boundary 
			#path_area = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Wairakei_countour_path.txt'
			# Tauhara boundary 
			#path_area = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Tauhara_countour_path.txt'
			#
			lats_rb, lons_rb  = np.genfromtxt(path_area, skip_header=1, delimiter=',').T
			poli_in = [[lons_rb[i],lats_rb[i]] for i in range(len(lats_rb))]

			if True: # distribution from constant boundaries HF. distance weigth. 
				w_bound = True # calc integral with weigths on real (wells) and virtual (boundaries) HF points
				#### tune pars
				frac_grid = 100 # distance in meters per point in the grid 
				# array of fix pm2 values at resistivity boundary locations 
				frac = 20 # fraction of points from rb to be taken
				if HF_tot:
					frac = 10
					if w_bound:
						frac = 1 
				bound_pw2 = 0.75 # 0.64 is HF at a well located in the boundary WK650
				####
				pm2_rb = [[lons_rb[i*frac], lats_rb[i*frac], bound_pw2] for i in range(int(len(lons_rb)/frac))]
				# 0.64 is HF at a well located in the boundary WK650
				# array of pm2 at well locatins
				pm2_wls = [[wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], wl_lon_lat_TG_TC_HF[5][i]] 
					for i in range(len(wl_lon_lat_TG_TC_HF[5]))]
				if HF_tot:
					if w_bound:
						n = 0
						pm2_wls = []
						while n < 20: # more weigth in interpolation to real points in well location (virtual points in the boundary)
							pm2_wls_aux = [[wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], wl_lon_lat_TG_TC_HF[6][i]] 
								for i in range(len(wl_lon_lat_TG_TC_HF[5]))]
							pm2_wls = pm2_wls + pm2_wls_aux
							n+=1
					else: 
						pm2_wls = [[wl_lon_lat_TG_TC_HF[1][i], wl_lon_lat_TG_TC_HF[2][i], wl_lon_lat_TG_TC_HF[6][i]] 
							for i in range(len(wl_lon_lat_TG_TC_HF[6]))]
				# array of both
				pm2 = pm2_wls.copy()
				pm2 = pm2 + pm2_rb
				# grid surface
				n_points_x = int(111000*(max(lons_rb) - min(lons_rb)))
				n_points_x = int(n_points_x/frac_grid)
				n_points_y = int(111000*(max(lats_rb) - min(lats_rb)))
				n_points_y = int(n_points_y/frac_grid)
				#
				x = np.linspace(min(lons_rb), max(lons_rb), n_points_x) # long
				y = np.linspace(min(lats_rb), max(lats_rb), n_points_y) # lat
				X, Y = np.meshgrid(x, y)
				pm2_grid = X.copy()*0.
				pm2_aux = [pm2_wl[2] for pm2_wl in pm2]
				pm2_grid_list = []
				# calculate pm2 at grid location
				for j,lat in enumerate(y):
					for i,lon in enumerate(x):
						if True: #for wl in pm2_wls:  # pm2_wls : [[lon, lat, HF], [...], ...]
							# distances between point in grid and points
							dist_wls = [dist_two_points([wl[0], wl[1]], [lon, lat], type_coord = 'decimal')\
								for wl in pm2]
							dist_wls = list(filter(None, dist_wls))
							#
							dist_weigth = [1./d**2 for d in dist_wls]
							#dist_weigth = [1./d**1.5 for d in dist_wls]
							pm2_grid[j][i] = np.dot(pm2_aux,dist_weigth)/np.sum(dist_weigth)
							pm2_grid_list.append([lon,lat,pm2_grid[j][i]])
				# filter points inside rest bound
				lat_filt = []
				lon_filt = []
				pm2_filt_rb = []
				pm2_grid_filt_rb = []
				for wl in pm2_grid_list:
					# check if station is inside poligon
					val = ray_tracing_method(wl[0], wl[1], poli_in)
					if val: #val:
						# for grid plot
						pm2_grid_filt_rb.append(wl)
						# for means
						pm2_filt_rb.append(wl[2])
						# for scatter plot 
						lon_filt.append(wl[0])
						lat_filt.append(wl[1])

				# Calculate mean power (per m2) inside the field
				#aux = np.mean(pm2_grid_filt_rb, axis=0)
				hf_mean_m2 = np.mean(pm2_filt_rb)
				# Calculate power for the whole field 
				hf_full = sum(pm2_filt_rb*(frac_grid**2))
				# print power for the whole field
				try: # add error base on senstivity to HF at bound 
					resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf.txt', delimiter = ',', skip_header=1).T
					d_power = abs(power[-1] - power[0])/2
					print('Wairakei-Tauhara estimated POWER: '+str(round(hf_full/1.e6,2))+' ± '+str(round(d_power,2))+' [MW]')#, size = textsize)
				except:
					print('Wairakei-Tauhara estimated POWER: '+str(round(hf_full/1.e6,2))+' [MW]')#, size = textsize)

			# plottoing the grid
			if True: # plot integral as scatter 
				# scatter plot
				vmin = None#0. # bound_pw2
				vmax = None#max(pm2_filt_rb)-.2
				if HF_tot: 
					vmin = 0.#0. # bound_pw2
					vmax = 3.#max(pm2_filt_rb)-.2 
				size = (frac_grid/3)*np.ones(len(pm2_filt_rb))
				cf = ax.scatter(lon_filt,lat_filt, s = size, c = pm2_filt_rb, edgecolors = None, \
					vmin = vmin, vmax = vmax, cmap = cmap_hf, zorder = 4)#
				#try: # add error base on senstivity to HF at bound 
				#	resboundHF, power = np.genfromtxt('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'heat_flux_power'+os.sep+'senst_2_boundRB_hf.txt', delimiter = ',', skip_header=1).T
				#	d_power = abs(power[-1] - power[0])/2
				#	ax.set_title('Heat flux through the clay cap: '+str(round(hf_full/1.e6,1))+' ± '+str(round(d_power,1))+' [MW]', size = textsize)
				#except:
				#	ax.set_title('Heat flux through the clay cap: '+str(int(hf_full/1.e6))+' [MW]', size = textsize)
				#ax.set_title('Heat flux through the clay cap: '+str(round(hf_full/1.e6,1))+' ± '+str(15.)+' [MW]', size = textsize)
				ax.set_title('Heat flux through the clay cap: '+str(int(hf_full/1.e6)+2)+' ± '+str(21)+' [MW]', size = textsize)

				# plot resistivity boundary on top the scatter plot
				plt.plot(lons_rb, lats_rb, color = 'orange' ,linewidth = 2, zorder = 4)

				if True: # plot scatter of HF in wells 
					HF_mean = wl_lon_lat_TG_TC_HF[5]
					array = HF_mean
					# scatter plot
					size = 100*np.ones(len(array))
					# levels = np.arange(200,576,25)  # for mean z1
					## vmin = min(levels)
					# vmax = max(levels)
					#normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
					#cmap = 'GnBu'
					
					scatter_hf = ax.scatter(lon_wls,lat_wls, s = size, c = array, edgecolors = 'gray', 
						vmin = vmin, vmax = vmax, cmap = cmap_hf, zorder = 4)#, label = 'Well temperature at: z2 mean')#alpha = 0.5)
					#ax.scatter([],[], s = size, c = 'white', edgecolors = 'gray', \
					#	zorder = 0, label = 'Heat Flux at well locations')#alpha = 0.5)

			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_'+name+'.png'
						#
		####################################################
		## add colorbar 
		if base_map: # topo 
			if not (meb_results or mt_results or temp_results \
					or z1_vs_temp_z1_basemap or d2_vs_temp_d2_basemap \
						or temp_hflux or temp_hflux_tot or temp_grad or temp_grad_HF_basemap):
				f.colorbar(topo_cb, ax=ax, label ='Elevation [m] (m.a.s.l.)')
				file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_topo.png'
		if meb_results: # meb
			f.colorbar(scatter_meb, ax=ax, label ='Depth [m]')
		if mt_results: # MT
			f.colorbar(scatter_MT, ax=ax, label ='[m]')			
		if temp_results: # Temp
			f.colorbar(scatter_temp, ax=ax, label ='Temperature [°C]')
		if temp_grad: # Temp
			f.colorbar(scatter_temp, ax=ax, label ='Temperature gradient [°C/km]')
		if temp_hflux or temp_hflux_tot: # Temp
			if scatter_p:
				f.colorbar(scatter_hf, ax=ax, label ='Heat Flux [W/m2]')
			if contour_p:
				f.colorbar(cf, ax=ax, label ='Heat Flux [W/m2]')
		if z1_vs_temp_z1_basemap:
			f.colorbar(scatter_MT, ax=ax, label ='Depth [m]')
			f.colorbar(scatter_temp, ax=ax, label ='Temperature [°C]')
		if d2_vs_temp_d2_basemap:
			f.colorbar(scatter_MT, ax=ax, label ='Depth [m]')
			f.colorbar(scatter_temp, ax=ax, label ='Temperature [°C]')
		if temp_grad_HF_basemap:
			f.colorbar(scatter_temp, ax=ax, label ='Temperature gradient [°C/km]')
			f.colorbar(scatter_hf, ax=ax, label =r'Heat flux [W/m$^2$]')
		#
		ax.legend(loc=3, prop={'size': textsize}, fancybox=True, framealpha=0.5)
		f.tight_layout()
		# save figure
		#plt.show()
		f.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
			orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	

#################### SCATTER VERSUS PLOT
	if True: # 
		### SCATTER PLOTS
		d12_vs_temp_d12_scatter = True
		d12_vs_hf_scatter = False
		###
		if d12_vs_temp_d12_scatter:
			# D1 vs T1: Depth vs Temperature
			if False:
				f1 = plt.figure(figsize=[9.5,11.5])
				ax1 = plt.axes([0.18,0.25,0.70,0.50]) 
				# plot tempT2 vs depthD2(at well location)
				# import 
				name_D1_D2_T1_T2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat', delimiter = ',', skip_header=1).T
				ax1.plot(name_D1_D2_T1_T2[5],-1*name_D1_D2_T1_T2[3],'r*', markersize = 8,zorder = 3)
				if False: # plot interpretations
					# line for depth division 
					ax1.plot([min(name_D1_D2_T1_T2[5]),max(name_D1_D2_T1_T2[5])],[-250.,-250.],'b--', alpha = 0.5, linewidth=2.0, zorder = 1)
					# line for temp division 
					ax1.plot([50.,50.],[min(-1*name_D1_D2_T1_T2[3]),max(-1*name_D1_D2_T1_T2[3])],'r--', alpha = 0.5, linewidth=2.0, zorder = 1)

					#for i in range(len(name_D1_D2_T1_T2[4])):
					#	if (name_D1_D2_T1_T2[4][i]>150. and name_D1_D2_T1_T2[2][i]>600.):
					#		print(i) # TH11, TH12

				ax1.set_ylim([-1200,0])
				ax1.set_xlim([0,300])

				ax1.set_xlabel('Temperature [°C]', size = textsize)
				ax1.set_ylabel('Depth [m]', size = textsize)
				ax1.set_title('Temp. vs. Depth: TOP of the Conductor', size = textsize)
				ax1.grid(linestyle='-', linewidth=.1, zorder=0)
				ax1.tick_params(labelsize=textsize)
				file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'scatter_d1_vs_T1'+'.png'
				f1.savefig(file_name_aux, dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	

			# D2 vs T2: Depth vs Temperature
			if False:
				interpretation = True
				if interpretation:
					bounds = [-600,50.] # [depth, temp]
					bounds = [-700,95.] # [depth, temp]
				f1 = plt.figure(figsize=[9.5,11.5])
				ax1 = plt.axes([0.18,0.25,0.70,0.50]) 
				# plot tempT2 vs depthD2(at well location)
				### files
				wls_infield = open('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'wls_infield.txt','w')
				wls_infield.write('well_name'+','+'lon_dec'+','+'lat_dec'+'\n')
				wls_peripheral = open('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'wls_peripheral.txt','w')
				wls_peripheral.write('well_name'+','+'lon_dec'+','+'lat_dec'+'\n')
				wls_outfield = open('.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'wls_outfield.txt','w')
				wls_outfield.write('well_name'+','+'lon_dec'+','+'lat_dec'+'\n')

				### import 
				name_D1_D2_T1_T2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat', delimiter = ',', skip_header=1).T
				if interpretation: # plot interpretations
					for i in range(len(name_D1_D2_T1_T2[6])):
						if name_D1_D2_T1_T2[6][i] > bounds[1]:
							if -name_D1_D2_T1_T2[4][i] > bounds[0]: # infield
								ax1.plot(name_D1_D2_T1_T2[6][i],-1*name_D1_D2_T1_T2[4][i],'*', c='r', markersize = 8, zorder = 3)
								wls_infield.write(str(name_D1_D2_T1_T2[0][i])+','+str(name_D1_D2_T1_T2[1][i])+','+str(name_D1_D2_T1_T2[2][i])+'\n')
							if -name_D1_D2_T1_T2[4][i] < bounds[0]:
								ax1.plot(name_D1_D2_T1_T2[6][i],-1*name_D1_D2_T1_T2[4][i],'*',c= u'#ff7f0e',markersize = 8, zorder = 3)
								wls_peripheral.write(str(name_D1_D2_T1_T2[0][i])+','+str(name_D1_D2_T1_T2[1][i])+','+str(name_D1_D2_T1_T2[2][i])+'\n')
						if name_D1_D2_T1_T2[6][i] < bounds[1]:
							ax1.plot(name_D1_D2_T1_T2[6][i],-1*name_D1_D2_T1_T2[4][i],'*',c=u'#1f77b4',markersize = 8, zorder = 3)
							wls_outfield.write(str(name_D1_D2_T1_T2[0][i])+','+str(name_D1_D2_T1_T2[1][i])+','+str(name_D1_D2_T1_T2[2][i])+'\n')
				else:
					ax1.plot(name_D1_D2_T1_T2[6],-1*name_D1_D2_T1_T2[4],'b*',markersize = 8, zorder = 3)
				
				wls_infield.close()
				wls_peripheral.close()
				wls_outfield.close()

				if interpretation: # plot interpretations
					# line for depth division 
					#ax1.plot([min(name_D1_D2_T1_T2[6]),max(name_D1_D2_T1_T2[6])],[-500.,-500.],'b--', alpha = 0.5, linewidth=2.0, zorder = 1)
					ax1.plot([0,max(name_D1_D2_T1_T2[6])],[bounds[0],bounds[0]],'c--', alpha = 0.5, linewidth=2.0, zorder = 1)
					# line for temp division 
					#ax1.plot([90.,90.],[min(-1*name_D1_D2_T1_T2[2]),max(-1*name_D1_D2_T1_T2[2])],'r--', alpha = 0.5, linewidth=2.0, zorder = 1)
					ax1.plot([bounds[1],bounds[1]],[0.,min(-1*name_D1_D2_T1_T2[4])],'c--', alpha = 0.5, linewidth=2.0, zorder = 1)
					ax1.set_xlabel('Temperature [°C]', size = textsize)
					####

				ax1.set_ylim([-1200,0])
				ax1.set_xlim([0,300])

				ax1.set_xlabel('Temperature [°C]', size = textsize)
				ax1.set_ylabel('Depth [m]', size = textsize)
				ax1.set_title('Temp. vs. Depth: BOTTOM of the Conductor', size = textsize)
				ax1.grid(linestyle='-', linewidth=.1, zorder=0)
				ax1.tick_params(labelsize=textsize)
				file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'scatter_d2_vs_T2'+'.png'
				f1.savefig(file_name_aux, dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	

			# D1,2 vs T1,2: Depth vs Temperature
			if True:

				if False: # Points
					f2 = plt.figure(figsize=[9.5,11.5])
					ax2 = plt.axes([0.18,0.25,0.70,0.50]) 
					# plot tempT2 vs depthD2(at well location)
					# import 
					name_D1_D2_T1_T2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat', delimiter = ',', skip_header=1).T
					#ax2.plot(name_D1_D2_T1_T2[5],-1*name_D1_D2_T1_T2[3], marker = 'o', c = u'#1f77b4', label = 'Top' ,zorder = 3)
					#ax2.plot(name_D1_D2_T1_T2[6],-1*name_D1_D2_T1_T2[4], marker = 'o', c = u'#ff7f0e', label = 'Bottom',zorder = 3)
					ax2.plot(name_D1_D2_T1_T2[5],-1*name_D1_D2_T1_T2[3], 'r*', label = 'Top' ,zorder = 3)
					ax2.plot(name_D1_D2_T1_T2[6],-1*name_D1_D2_T1_T2[4], 'b*', label = 'Bottom',zorder = 3)
			
					if False: # plot interpretations
						# line for depth division 

						ax1.plot([min(name_D1_D2_T1_T2[6]),max(name_D1_D2_T1_T2[6])],[-500.,-500.],'b--', alpha = 0.5, linewidth=2.0, zorder = 1)
						#ax1.plot([min(name_D1_D2_T1_T2[4]),max(name_D1_D2_T1_T2[4])],[-600.,-600.],'b--', alpha = 0.5, linewidth=2.0, zorder = 1)
						# line for temp division 
						#ax1.plot([90.,90.],[min(-1*name_D1_D2_T1_T2[2]),max(-1*name_D1_D2_T1_T2[2])],'r--', alpha = 0.5, linewidth=2.0, zorder = 1)
						ax1.plot([160.,160.],[min(-1*name_D1_D2_T1_T2[4]),max(-1*name_D1_D2_T1_T2[4])],'r--', alpha = 0.5, linewidth=2.0, zorder = 1)
						#for i in range(len(name_D1_D2_T1_T2[4])):
						#	if (name_D1_D2_T1_T2[4][i]>150. and name_D1_D2_T1_T2[2][i]>600.):
						#		print(i) # TH11, TH12
					
					ax1.set_ylim([-1200,0])
					ax1.set_xlim([0,300])

					ax2.legend()
					ax2.set_xlabel('Temperature [°C]', size = textsize)
					ax2.set_ylabel('Depth [m]', size = textsize)
					ax2.set_title('Temp. vs. Depth: TOP and BOTTOM of the Conductor', size = textsize)
					ax2.grid(linestyle='-', linewidth=.1, zorder=0)
					ax2.tick_params(labelsize=textsize)
					file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'scatter_d12_vs_T12'+'.png'
					f2.savefig(file_name_aux, dpi=300, facecolor='w', edgecolor='w',
						orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	

				if True: # Geothermal gradient 
					# filter by depth and temp bounds = [-700,95.]
					if True: 
						f2 = plt.figure(figsize=[13,10])
						ax2 = plt.axes([0.1,0.15,0.75,0.75]) 
						bounds = [-700,95.] # [depth, temp]
						# plot tempT2 vs depthD2(at well location)
						# import 
						name_D1_D2_T1_T2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat', delimiter = ',', skip_header=1).T
						#ax2.plot(name_D1_D2_T1_T2[5],-1*name_D1_D2_T1_T2[3], marker = 'o', c = u'#1f77b4', label = 'Top' ,zorder = 3)
						#ax2.plot(name_D1_D2_T1_T2[6],-1*name_D1_D2_T1_T2[4], marker = 'o', c = u'#ff7f0e', label = 'Bottom',zorder = 3)
						inf_grad = []
						for i in range(len(name_D1_D2_T1_T2[3])):
							if -name_D1_D2_T1_T2[4][i] > bounds[0] and name_D1_D2_T1_T2[6][i] > bounds[1]: # infiel, condition on depth and temp
								ax2.plot([name_D1_D2_T1_T2[5][i], name_D1_D2_T1_T2[6][i]],[-1*name_D1_D2_T1_T2[3][i], -1*name_D1_D2_T1_T2[4][i]],
									c = pale_orange_col ,alpha = 0.5,zorder = 2)
								inf_grad.append([(name_D1_D2_T1_T2[6][i]-name_D1_D2_T1_T2[5][i])/(name_D1_D2_T1_T2[4][i]-name_D1_D2_T1_T2[3][i])])
							else:
								ax2.plot([name_D1_D2_T1_T2[5][i], name_D1_D2_T1_T2[6][i]],[-1*name_D1_D2_T1_T2[3][i], -1*name_D1_D2_T1_T2[4][i]],
									c = pale_blue_col ,alpha = 0.5,zorder = 2)
							
							ax2.plot(name_D1_D2_T1_T2[6][i],-1*name_D1_D2_T1_T2[4][i],'o', c='b', markersize = 5, zorder = 3)
							ax2.plot(name_D1_D2_T1_T2[5][i],-1*name_D1_D2_T1_T2[3][i],'o', c='r', markersize = 5, zorder = 3)

						# for legend 
						ax2.plot([],[],c = pale_orange_col , label = r'Infield gradient: '+str(round(np.mean(inf_grad)*1e3,1))+' ± '+str(round(np.std(inf_grad)*1e3,1))+' [°C/km]' ,zorder = 3)
						ax2.plot([],[],c = pale_blue_col , label = 'Outfield gradient' ,zorder = 3)
						ax2.plot([],[],'*', c='r', markersize = 8, label = 'Top Boundary',zorder = 3)
						ax2.plot([],[],'*', c='b', markersize = 8, label = 'Bottom Boundary', zorder = 3)
						# reference lines
						#ax2.plot([0,max(name_D1_D2_T1_T2[6])],[bounds[0],bounds[0]],'c--', alpha = 0.5, linewidth=2.0, zorder = 1)
						#ax2.plot([bounds[1],bounds[1]],[0.,min(-1*name_D1_D2_T1_T2[4])],'c--', alpha = 0.5, linewidth=2.0, zorder = 1)

						ax2.set_ylim([-1300,50])
						ax2.set_xlim([0,280])

						ax2.legend(loc = 4)
						ax2.set_xlabel('Temperature [°C]', size = textsize)
						ax2.set_ylabel('Depth [m]', size = textsize)
						ax2.set_title('Geothermal gradient inside the Conductor', size = textsize)
						ax2.grid(linestyle='-', linewidth=.1, zorder=0)
						ax2.tick_params(labelsize=textsize)
						file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'gradient_d12_vs_T12'+'.png'
						f2.savefig(file_name_aux, dpi=300, facecolor='w', edgecolor='w',
							orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
					
					if True: # filter by resisitvity boundary
						with_hist = True
						if with_hist:
							# plot histograms 
							f2 = plt.figure(figsize=(13, 5.5))#(10, 4.5))#(15, 7))
							gs = gridspec.GridSpec(nrows=1, ncols=2)
							ax2 = f2.add_subplot(gs[0, 0])
							ax1 = f2.add_subplot(gs[0, 1])
							#ax_leg= f.add_subplot(gs[0, 2])
						else:
							f2 = plt.figure(figsize=[11,11])
							ax2 = plt.axes([0.1,0.15,0.75,0.75]) 
						# plot tempT2 vs depthD2(at well location)
						# import 
						name_D1_D2_T1_T2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat', delimiter = ',', skip_header=1).T
						## Resistivity Boundary, Mielke
						path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_IN_Mielke.txt'
        				# Wairakei boundary 
						#path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Wairakei_countour_path.txt'
						# Tauhara boundary 
						#path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Tauhara_countour_path.txt'
						#
						lats, lons = np.genfromtxt(path_rest_bound_WT, skip_header=1, delimiter=',').T
						poli_in = [[lons[i],lats[i]] for i in range(len(lats))]
						inf_grad = []
						for i in range(len(name_D1_D2_T1_T2[3])):
							# check if station is inside poligon 
							val = ray_tracing_method(name_D1_D2_T1_T2[1][i], name_D1_D2_T1_T2[2][i], poli_in)
							if val:
								ax2.plot([name_D1_D2_T1_T2[5][i], name_D1_D2_T1_T2[6][i]],[-1*name_D1_D2_T1_T2[3][i], -1*name_D1_D2_T1_T2[4][i]],
									c = pale_orange_col ,alpha = 0.5,zorder = 2)
								ax2.plot(name_D1_D2_T1_T2[6][i],-1*name_D1_D2_T1_T2[4][i],'o', c='b', markersize = 5, zorder = 3)
								ax2.plot(name_D1_D2_T1_T2[5][i],-1*name_D1_D2_T1_T2[3][i],'o', c='r', markersize = 5, zorder = 3)
								inf_grad.append([(name_D1_D2_T1_T2[6][i]-name_D1_D2_T1_T2[5][i])/(name_D1_D2_T1_T2[4][i]-name_D1_D2_T1_T2[3][i])])
								#if (name_D1_D2_T1_T2[6][i]-name_D1_D2_T1_T2[5][i])/(name_D1_D2_T1_T2[4][i]-name_D1_D2_T1_T2[3][i]) < 0.100:
								#	print(i)
								#	print(name_D1_D2_T1_T2[6][i])
							else:
								pass
								#ax2.plot([name_D1_D2_T1_T2[5][i], name_D1_D2_T1_T2[6][i]],[-1*name_D1_D2_T1_T2[3][i], -1*name_D1_D2_T1_T2[4][i]],
								#	c = pale_blue_col ,alpha = 0.5,zorder = 2)
							
							#ax2.plot(name_D1_D2_T1_T2[6][i],-1*name_D1_D2_T1_T2[4][i],'o', c='b', markersize = 5, zorder = 3)
							#ax2.plot(name_D1_D2_T1_T2[5][i],-1*name_D1_D2_T1_T2[3][i],'o', c='r', markersize = 5, zorder = 3)

						# for legend 
						if not with_hist:
							ax2.plot([],[],c = pale_orange_col , label = r'Infield gradient: '+str(round(np.median(inf_grad)*1e3,1))+' ± '+str(round(np.std(inf_grad)*1e3,1))+' [°C/km]' ,zorder = 3)
						if with_hist:
							ax2.plot([],[],c = pale_orange_col , label = r'Infield gradient' , zorder = 3)

						#ax2.plot([],[],c = pale_blue_col , label = 'Outfield gradient' ,zorder = 3)
						ax2.plot([],[],'*', c='r', markersize = 8, label = 'Top Boundary',zorder = 3)
						ax2.plot([],[],'*', c='b', markersize = 8, label = 'Bottom Boundary', zorder = 3)
						# reference lines
						#ax2.plot([0,max(name_D1_D2_T1_T2[6])],[bounds[0],bounds[0]],'c--', alpha = 0.5, linewidth=2.0, zorder = 1)
						#ax2.plot([bounds[1],bounds[1]],[0.,min(-1*name_D1_D2_T1_T2[4])],'c--', alpha = 0.5, linewidth=2.0, zorder = 1)

						ax2.set_ylim([-1200,50])
						ax2.set_xlim([0,280])

						ax2.legend(loc = 4)
						ax2.set_xlabel('temperature [°C]', size = textsize)
						ax2.set_ylabel('depth [m]', size = textsize)
						ax2.set_title('Clay cap temperature gradient', size = textsize, color = 'gray')
						ax2.grid(linestyle='-', linewidth=.1, zorder=0)
						ax2.tick_params(labelsize=textsize)
						file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'gradient_d12_vs_T12_infield'+'.png'

						# histogram of geothermal gradient

						if with_hist:
							# histogram for geothermal gradient
							inf_grad = np.asarray(inf_grad)*1.e3 # from meters to km
							bins = np.linspace(np.min(inf_grad), np.max(inf_grad), 2*int(np.sqrt(len(inf_grad))))
							h,e = np.histogram(inf_grad, bins)
							m = 0.5*(e[:-1]+e[1:])
							ax1.bar(e[:-1], h, e[1]-e[0], alpha =.8, color = pale_orange_col, edgecolor = 'w', zorder = 3)
							#ax1.legend(loc=None, shadow=False, fontsize=textsize)
							# 
							(mu, sigma) = norm.fit(inf_grad)
							med = np.median(inf_grad)
							try:
								y = mlab.normpdf(bins, mu, sigma)
							except:
								#y = stats.norm.pdf(bins, mu, sigma)
								pass
							ax1.plot([med,med],[0,np.max(h)],'r-', zorder = 3, linewidth=3)
							#ax1.set_title('$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
							p5,p50,p95 = np.percentile(inf_grad, [5,50,95])
							ax1.set_title('{:3.0f}'.format(p50)+'$^{+'+'{:3.0f}'.format(p95-p50)+'}_{-'+'{:3.0f}'.format(p50-p5)+'}$', fontsize = textsize, color='gray')
							ax1.set_xlabel('temperature gradient [°C/km]', fontsize=textsize)
							ax1.set_ylabel('frequency', fontsize=textsize)
							ax1.grid(True, which='both', linewidth=0.1)
							file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'gradient_d12_vs_T12_histogram_infield'+'.png'
						f2.tight_layout()
						f2.savefig(file_name_aux, dpi=300, facecolor='w', edgecolor='w',
							orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	


		if d12_vs_hf_scatter:

			f1 = plt.figure(figsize=[9.5,7.5])
			ax1 = plt.axes([0.15,0.15,0.75,0.75]) 
			### import 
			name_lon_lat_D1_D2_T1_T2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_D1_D2_T1_T2.dat', delimiter = ',', skip_header=1).T
			name_lon_lat_TG_TC_HF = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HF.dat', delimiter = ',', skip_header=1).T

			ax1.plot(name_lon_lat_D1_D2_T1_T2[3],name_lon_lat_TG_TC_HF[5],'r*',markersize = 8, zorder = 3, label = 'top of the conductor')
			ax1.plot(name_lon_lat_D1_D2_T1_T2[4],name_lon_lat_TG_TC_HF[5],'b*',markersize = 8, zorder = 3, label = 'bottom of the conductor')


			#ax1.set_ylim([-1200,0])
			#ax1.set_xlim([0,300])
			ax1.legend(loc=1, prop={'size': textsize})
			ax1.set_xlabel('Depth [m]', size = textsize)
			ax1.set_ylabel('Heat Flux [W/m2]', size = textsize)
			ax1.set_title('Depth vs. Heat Flux', size = textsize)
			ax1.grid(linestyle='-', linewidth=.1, zorder=0)
			ax1.tick_params(labelsize=textsize)
			file_name_aux = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'versus_plots'+os.sep+'scatter_d12_vs_HF'+'.png'
			f1.savefig(file_name_aux, dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
			plt.close(f1)