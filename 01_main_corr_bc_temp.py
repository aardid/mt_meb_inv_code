"""
.. module:: Correlation between CC and temperature data
   :synopsis: CC is imported from mcmc inversion of MeB data 


.. moduleauthor:: Alberto Ardid 
				  University of Auckland 
				  
.. conventions:: 
	: number of layer 3 (2 layers + half-space)
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
from io import StringIO
from matplotlib.backends.backend_pdf import PdfPages
# ==============================================================================

if __name__ == "__main__":
    ## PC that the code will be be run ('ofiice', 'personalSuse', 'personalWin')
    #pc = 'office'
    pc = 'personalMac'

    ## Set of MT data to work with 
    full_dataset = True
    # Filter has qualitu MT stations
    filter_lowQ_data_MT = True
    ## run with quality filter per well
    filter_lowQ_data_well = False  # need to be checked, not working
    ## Sections of the code tu run
    set_up = True 
    calc_cond_bound = False
    calc_cond_bound_temps = True
    plot_temp_bc = False

    # (0) Import data and create objects: wells from spreadsheet files
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
            # add to wl2work names of well with meb data and no temp
            wls_meb_notemp = []
            for wl_meb in wl_name_meb:
                if wl_meb not in wl_name:
                    #wl2work.append(wl_meb)
                    wls_meb_notemp.append(wl_meb)
            #
            #wl_name = wl_name_meb # just for inverting meb data with no temp data
            #wl2work = ['WK317']

		# remove bad quality wells from list 'wl2work' (based on Q_temp_prof.txt)
        if filter_lowQ_data_well: 
            name_file =  '.'+os.sep+'corr_temp_bc'+os.sep+'Q_temp_prof.txt'
            BQ_sta = [x.split()[0] for x in open(name_file).readlines() if x[0]!='#' and x[-2] is '0']
            wl2work = [x for x in wl2work if not x in BQ_sta]

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
                    wl_obj.no_temp = True
            wells_objects.append(wl_obj)
            count  += 1

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

    # (1) Calc. z1 and z2 in well positions 
    if calc_cond_bound:
        print('(1) Calc. z1 and z2 in well positions\n')
        ## Estimate z1 and z2 at wells positions from MT inversion
        wl_z1_z2_est_mt(wells_objects, station_objects, slp = 5., plot_temp_prof = True, 
            with_meb = True, with_litho = True)
        # remove objects from list when BC is deeper than max depth of temp. profiles
        if False:
            idx2rmv = [] 
            for i, wl in enumerate(wells_objects):
                if wl.red_depth_rs[-1]+50 > wl.elev - (wl.z1_pars[0] + wl.z1_pars[1]):
                    idx2rmv.append(i)
            wells_objects_aux = [wl for i, wl in enumerate(wells_objects) if i not in idx2rmv]
            wells_objects = wells_objects_aux

    # (2) Calc. T1 and T2 at well positions 
    if calc_cond_bound_temps: 
        print('(2) Calc. T1 and T2 at well positions\n ')
        # Sample temperatures at z1 and z1 ranges to create T1_pars and T2_pars (distribrutions for temperatures at conductor bound.)
        # wl_T1_T2_est(wells_objects, hist = False, hist_filt = [0,0])
        wl_T1_T2_est(wells_objects)
        # histogram filtering by area: inside and outside (reservoir)
        path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat'
        histogram_temp_T1_T2(wells_objects, filt_in_count=path_rest_bound_WT, filt_out_count=path_rest_bound_WT, type_hist = 'sidebyside')

    # (2) grid surface and plot temperature at conductor boundaries
    if plot_temp_bc: 
        ## load z1 and z2 pars
        for wl in wells_objects:
            aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt')
            wl.T1_pars = [aux[0],aux[1]]
            wl.T2_pars = [aux[2],aux[3]]
        # define region to grid
        coords = [175.97,176.200,-38.74,-38.58] # [min lon, max lon, min lat, max lat]
        # fn. for griding and calculate prior => print .txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
        file_name = 'grid_temp_bc' # txt file with grid values
        path_output = '.'+os.sep+'corr_temp_bc'+os.sep+'00_global'
        ##
        img_back_topo_ge = False
        img_back_rest_bound = True
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
        if False: # contourf plot
            print('gridding temp at cond. bound')
            grid_temp_conductor_bound(wells_objects, coords = coords, n_points = 100, slp = 5., file_name = file_name, path_output = path_output,\
                plot = True, path_base_image = path_base_image, ext_img = ext_file, xlim = x_lim, masl = False)
        # scatter plot of temps at conductor boundaries
        if True: # scatter plot
            x_lim = [175.99,176.21]
            y_lim = [-38.75,-38.58]
            scatter_temp_conductor_bound(wells_objects,  path_output = path_output, alpha_img = 0.6,\
                path_base_image = path_base_image, ext_img = ext_file, xlim = x_lim, ylim = y_lim)



     