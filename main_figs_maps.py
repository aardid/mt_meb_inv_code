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

textsize = 15.
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
	base_map = True # plot map with wells and stations
	zones = True # add injection and extraction zones
	wells_loc = False # add wells
	stations_loc = False # add mt stations
	########
	meb_results = False # add scatter MeB results 
	mt_results = False # add scatter MT results
	temp_results = True # add scatter Temp results 
	########
	# plot map with wells and stations
	if base_map:
		path_topo = '.'+os.sep+'base_map_img'+os.sep+'coords_elev'+os.sep+'Topography_25_m_vertices_LatLonDec.csv'#Topography_zoom_WT_re_sample_vertices_LatLonDec.csv'
		path_lake_shoreline = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'shoreline_TaupoLake.dat'
		path_faults = '.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'nzafd.json'
		path_rest_bound = ['.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat', 
							'.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_RK_50ohmm.dat']
		path_powerlines = glob.glob('.'+os.sep+'base_map_img'+os.sep+'extras'+os.sep+'powerlines'+os.sep+'*.dat')
		#
		x_lim = [175.95,176.23]#[175.98,176.22] 
		y_lim = [-38.79,-38.57]
		# base figure
		f, ax, topo_cb = base_map_region(path_topo = path_topo,  xlim = x_lim, ylim = y_lim,path_rest_bound = path_rest_bound,
			path_lake_shoreline = path_lake_shoreline, path_faults = path_faults, path_powerlines = path_powerlines)
	# add injection and extraction zones
	if zones:
		# Otupu
		if True:
			path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Otupu_inj.dat'
			lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
			plt.plot(lons, lats, 'b:' ,linewidth= 1, zorder = 7)
			# print name 
			plt.text(np.min(lons), np.max(lats), 'Otupu', color='b', size=textsize, zorder = 7)
		# Pohipi west
		if True:
			path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'pohipi_west_inj.dat'
			lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
			plt.plot(lons, lats, 'b:' ,linewidth= 1, zorder = 7)
			# print name 
			plt.text(np.min(lons), np.max(lats), 'Pohipi west', color='b', size=textsize, zorder = 7)
		# Te Mihi
		if True:
			path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'TeMihi2_prod.dat'
			lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
			plt.plot(lons, lats, 'r--' ,linewidth= 1, zorder = 7)
			# print name 
			plt.text(np.min(lons), np.max(lats), 'Te Mihi', color='r', size=textsize, zorder = 7)
		# West Bore
		if True:
			path_otupu = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'WestBoreField_prod.dat'
			lats, lons = np.genfromtxt(path_otupu, skip_header=1, delimiter=',').T
			plt.plot(lons, lats, 'r--' ,linewidth= 1, zorder = 7)
			# print name 
			plt.text(np.min(lons), np.max(lats), 'West Bore', color='r', size=textsize, zorder = 7)
		# for legend 
		plt.plot([],[], 'b:' ,linewidth= 1, label = 'Injection zone', zorder = 7)
		plt.plot([],[], 'r--' ,linewidth= 1, label = 'Production zone', zorder = 7)
		# add labels for reservoirs
		plt.text(176.13, -38.732, 'Wairakei-Tauhara', color='orange', size=textsize, zorder = 7)
		plt.text(176.18, -38.645, 'Rotokawa', color='orange', size=textsize, zorder = 7)
	# add wells
	if wells_loc:
		path_wl_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls.dat'
		lons, lats = np.genfromtxt(path_wl_locs, delimiter=',').T
		plt.plot(lons, lats, 's' , c = 'gray', zorder = 2, markersize=6)
		plt.plot([], [], 's' , c = 'gray', zorder = 2, label = 'Well', markersize=8)
		# meb wells 
		path_wlmeb_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_wls_meb.dat'
		lons, lats = np.genfromtxt(path_wlmeb_locs, delimiter=',').T
		plt.plot(lons, lats, 's' , c = 'w', zorder = 2, markersize=6)
		plt.plot(lons, lats, 's' , c = 'gray', zorder = 2, markersize=6, markerfacecolor='none')
		plt.plot([], [], 's' , c = 'gray', zorder = 2, label = 'Well with MeB data', markersize=8, markerfacecolor='none')
	# add mt stations
	if stations_loc:
		path_mt_locs = '.'+os.sep+'base_map_img'+os.sep+'location_mt_wells'+os.sep+'location_MT_stations.dat'
		lons, lats = np.genfromtxt(path_mt_locs, delimiter=',').T
		plt.plot(lons, lats, 'x' , c = 'm', zorder = 2, markersize=6)
		plt.plot([], [], 'x' , c = 'm', zorder = 2, label = 'MT station', markersize=8)
	####################################
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
		z1_mean = meb_result[3]
		array = z1_mean
		name = 'z1_mean'
		levels = np.arange(200,576,25)  # for mean z1
		# scatter plot
		size = 200*np.ones(len(array))
		vmin = min(levels)
		vmax = max(levels)
		normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
		scatter_meb = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', cmap = 'winter', \
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
		z1_mean = mt_result[3]
		name = 'z1_mean'
		array = z1_mean
		levels = np.arange(200,576,25)  # for mean z1
		# scatter plot
		size = 200*np.ones(len(array))
		vmin = min(levels)
		vmax = max(levels)
		normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
		scatter_MT = ax.scatter(lon_stas,lat_stas, s = size, c = array, edgecolors = 'k', cmap = 'winter', \
			norm = normalize, zorder = 5)#, label = 'MT inversion: z1 mean')#alpha = 0.5)
		ax.scatter([],[], s = size, c = 'b', edgecolors = 'k', \
			zorder = 5, label = 'MT inversion: z1 mean')#alpha = 0.5)
		# absence of CC (no_cc)
		for i, z2 in enumerate(mt_result[5]):
			if (z2 < 50.):
				plt.plot(mt_result[1][i], mt_result[2][i],'w.', markersize=28, zorder = 6) 
				plt.plot(mt_result[1][i], mt_result[2][i],'bx', markersize=12, zorder = 6)
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
			zorder = 5, label = 'Well temperature at: z2 mean')#alpha = 0.5)
		#
		file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_temp_'+name+'.png'
	################### colorbar to plot
	# topo 
	if base_map: # topo 
		if not (meb_results or mt_results or temp_results):
			f.colorbar(topo_cb, ax=ax, label ='Elevation [m] (m.a.s.l.)')
			file_name = '.'+os.sep+'base_map_img'+os.sep+'figures'+os.sep+'base_map_topo.png'
	if meb_results: # meb
		f.colorbar(scatter_meb, ax=ax, label ='Depth [m]')
	if mt_results: # MT
		f.colorbar(scatter_MT, ax=ax, label ='Depth [m]')			
	if temp_results: # Temp
		f.colorbar(scatter_temp, ax=ax, label ='Temperature [°C]')
	#
	ax.legend(loc=3, prop={'size': textsize})
	plt.tight_layout()
	# save figure
	plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
		orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	