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
    # Filter MeB wells with useless info (for prior)
    filter_useless_MeB_well = True
    ## run with quality filter per well
    filter_lowQ_data_well = True  # need to be checked, not working: changing the orther of well obj list
    ## re model temp profiles from wells with lateral inflows ()
    temp_prof_remodel_wells = True # re model base on '.'+os.sep+'corr_temp_bc'+os.sep+'RM_temp_prof.txt'
    ## Sections of the code tu run
    set_up = True 
    calc_cond_bound = False
    calc_cond_bound_temps = False
    plot_temp_bc = False
    #
    HF_by_beta = True
    stat_HF_by_beta = True

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
            # sta2work = 	['WT312a','WT060a','WT190a','WT186a','WT011a','WT046a','WT336a','WT182a',
            # 			'WT194a','WT177a','WT198a','WT015a','WT306a','WT078a','WT326a',
            # 			'WT167a','WT188a','WT184a','WT312a','WT312a','WT312a','WT312a']

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

    # (1) Calc. z1 and z2 in well positions 
    if calc_cond_bound:
        print('(1) Calc. z1 and z2 in well positions\n')
        ## Estimate z1 and z2 at wells positions from MT inversion
        # folder estructure
        if os.path.isdir('.'+os.sep+'wells_info'+os.sep+'estimated'):
            pass
        else:
            os.mkdir('.'+os.sep+'wells_info'+os.sep+'estimated')
        # txt to save z1 and z2 in well positions
        file_name = '.'+os.sep+'wells_info'+os.sep+'estimated'+os.sep+'z1_z2_at_wells_location.txt'
        #
        fix_axes = [[-60,300],[-1700,40]] # axes for [temp_min,temp_ax][depth_min (eg: -800), depth_max (eg: -100)]
        wl_z1_z2_est_mt(wells_objects, station_objects, slp = 5., plot_temp_prof = True, 
            with_meb = True, with_litho = True, fix_axes = False, litho_abre = True, save_txt_z1_z2 = file_name)

    # (2) Calc. T1 and T2 at well positions 
    if calc_cond_bound_temps: 
        print('(2) Calc. T1 and T2 at well positions\n ')

        # Sample temperatures at z1 and z1 ranges to create T1_pars and T2_pars (distribrutions for temperatures at conductor bound.)
        # wl_T1_T2_est(wells_objects, hist = False, hist_filt = [0,0])
        wl_T1_T2_est(wells_objects, thermal_grad = True, heat_flux = True)
        # histogram filtering by area: inside and outside (reservoir)
        # Resistivity Boundary, Risk
        path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_WK_50ohmm.dat'
        # Resistivity Boundary, Mielke
        path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
        # Wairakei boundary
        #path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Wairakei_countour_path.txt'
        # Tauhara boundary
        #path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'Tauhara_countour_path.txt'
        # remove from well with no temp from list of well objects 
        wells_objects_aux = [wl for wl in wells_objects if not wl.no_temp]
        wells_objects = wells_objects_aux
        # remove objects from list when BC is deeper than max depth of temp. profiles
        if True: # if true, turn calc_cond_bound to True
            idx2rmv = [] 
            for i, wl in enumerate(wells_objects):
                # load z1 and z2 pars
                #wl.z1_pars[0], wl.z1_pars[1], wl.z2_pars[0], wl.z2_pars[1] = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_z1_z2.txt')
                if (wl.red_depth_rs[-1]+50 > wl.elev - (wl.z1_pars[0] + wl.z1_pars[1])):
                    idx2rmv.append(i)
            wells_objects_aux = [wl for i, wl in enumerate(wells_objects) if i not in idx2rmv]
            wells_objects = wells_objects_aux

        # histogram of T1 and T2, filtering by resisitivity boundary
        Ns = 50 # number of samples per well (if None, one sample per well)
        type_hist = 'infield'#'sidebyside'# None
        histogram_temp_T1_T2(wells_objects, filt_in_count=path_rest_bound_WT, filt_out_count=path_rest_bound_WT, 
            type_hist = type_hist, multi_samples = Ns)
        # histogram of T1, T2, thermal_grad and heat flux, filtering by bounds of temp and depth
        bounds = [-700,95.] # [depth, temp]
        histogram_T1_T2_Tgrad_Hflux(wells_objects, bounds = bounds)

    # (3) grid surface and plot temperature at conductor boundaries
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

    ####################################################
    # Estimate heat flux through the clay cap (by beta method)
    # (1) Sample the posterior to obtained boundaries for top and bottom of conductor at the location of a well.
    # (2) Interpolate temperatures boundary locations.
    # (3) Construct best fit exponential heat transfer model (from Bredeheoft and Papadopoulos) inside this layer and obtain the beta parameter.
    # (4) Computer average proportion of the conductive heat flux across the layer per my derivation in the attached. When this number is close to 1, it justifies an assertion that heat transport is dominantly conductive (but does give a nice breakdown of the split).
    # (5) Compute the average conductive heat flux across the layer as int_0^L(K*dT/dz)dz / L, where T is a known analytical function (I can derive this expression for you if you like).
    # (6) Resample the posterior multiple times for each well.
    # (7) Conduct for all wells. 
    
    adv_all_negative = True # np.abs(beta) : advection possitve or negative
    if HF_by_beta:
        #
        wl_TG_TC_HFC_HFT = open('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'wls_conductor_TG_TC_HFC_HFT.dat','w')
        wl_TG_TC_HFC_HFT.write('wl_name'+','+'lon_dec'+','+'lat_dec'+','+'TG(Thermal Gradient)[C/m]'
            +','+'TC(Thermal Conductivity)[W/mC]'+','+'Cond. HF(Heat Flux)[W/m2]'+'Tot. HF(Heat Flux)[W/m2]'+'\n')
        #
        pdf_temp_samlples = False # when sampling well temp profiles (by beta), save all temp profiles in pdf
        # check if directory to save temp samples exist
        if os.path.isdir('.'+os.sep+'corr_temp_bc'+os.sep+'temp_prof_samples'):
            pass
        else:
            os.mkdir('.'+os.sep+'corr_temp_bc'+os.sep+'temp_prof_samples')

        if pdf_temp_samlples: # 
            pp = PdfPages('.'+os.sep+'corr_temp_bc'+os.sep+'temp'+os.sep+'temp_prof_samples.pdf')
        ## load pars
        for i, wl in enumerate(wells_objects):
            # (1) Sample the posterior to obtained boundaries for top and bottom of conductor at the location of a well.
            # wl_z1_z2_est_mt(wells_objects, station_objects, slp = 5.)
            aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_z1_z2.txt')
            wl.z1_pars = [aux[0],aux[1]]
            wl.z2_pars = [aux[2],aux[3]]

            # (2) Interpolate temperatures boundary locations.
            try: # some wells don'r have temp values at z2
                if not wl.no_temp:
                    aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'conductor_T1_T2.txt')
                    wl.T1_pars = [aux[0],aux[1]]
                    wl.T2_pars = [aux[2],aux[3]]
            except:
                wl.T1_pars = [False,False]
                wl.T2_pars = [False,False]
            
            # (3) Construct best fit exponential heat transfer model (from Bredeheoft and Papadopoulos) inside this layer and obtain the beta parameter.
            # calculate Test and beta values 
            #if wl.T2_pars[0]: # if T2 has value
            if True: # time consuming, run once. sample beta values are save in path_files_out
                try:
                    print('sampling betas in well: '+wl.name+' '+str(i+1)+'/'+str(len(wells_objects)))
                    path_files_out = '.'+os.sep+'corr_temp_bc'+os.sep+wl.name
                    # this function calc beta reference positive up (+ up)
                    wl.temp_prof_est(plot_samples = True, ret_fig = None, 
                        path_files_out = path_files_out, Ns = 50) # method of well object
                    if pdf_temp_samlples:
                        g = wl.temp_prof_est(plot_samples = True, ret_fig = None, 
                            path_files_out = path_files_out, Ns = 50) # method of well object
                        pp.savefig(g)
                        g.close()
                    else: 
                        # save temp profile png in well folder 
                        gen_temp_sample = True
                        wl.temp_prof_est(plot_samples = gen_temp_sample, ret_fig = None, 
                            path_files_out = path_files_out, Ns = 50) # method of well object
                        if gen_temp_sample: # save in folder
                            file_name = '.'+os.sep+'corr_temp_bc'+os.sep+'temp_prof_samples'+os.sep+wl.name+'_temp_sample.png'
                            # copy from well folder to temp_prof_samples folder 
                            shutil.copy('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'Test_samples.png'\
                                , file_name)
                except:
                    pass
            # (4) Compute average proportion of the conductive heat flux across the layer per my derivation in the attached. 
            #     When this number is close to 1, it justifies an assertion that heat transport is dominantly conductive 
            #     (but does give a nice breakdown of the split).
            #     * reference system positive looking down (same as Bredehoeft, J. D., and I. S. Papadopulos (1965))
            try:
                if not wl.no_temp:
                    # import samples of z1, z2, t1, t2 and beta
                    dsam_z1, dsam_z2, dsam_botl3 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'Zmax.txt').T
                    tsam_z1, tsam_z2, tsam_botl3 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'Tmax.txt').T
                    bsam_l1, bsam_l2, bsam_l3 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'betas.txt').T
                    # calc average proportion of  conductive heat flux across layer 2 (clay cap)
                    ave_prop_list = []
                    tg_list = [] # thermal gradient 
                    hf_list = [] # conductive heatflux 
                    hf_tot_list = [] # total heatflux  
                    for z1,z2,t1,t2,beta in zip(dsam_z1,dsam_z2,tsam_z1,tsam_z2,bsam_l2): 
                        #### reference system: positive down 
                        beta = -1*beta # change to ref. positive down (+ down)
                        ####
                        # average proportion
                        #if adv_all_negative: # np.abs(beta) : advection possitve or negative 
                        #    ave_prop = -1.*(t2-t1)/(-1.*(np.abs(beta))*t1-(t2-t1))
                        #else:
                        #    ave_prop = -1.*(t2-t1)/(beta*t1-(t2-t1))
                        #
                        if abs(beta) > .001:
                            ave_prop = (1/beta)*-1*(t2-t1)/(t1-(t2-t1)/(np.exp(beta)-1))   
                        else:
                            ave_prop = -1.*(t2-t1)/(beta*t1-(t2-t1))   
                        ave_prop_list.append(ave_prop)
                        
                        k = 2.5 # thermal conductivity [W/mC]
                        L = abs(z2-z1) # thickness
                        dT = t2-t1 
                        dz = z2-z1 # this is negative 
                        # (5.A) Compute heat flux (conductive) using beta 
                        # HFs_l2 = -k * dT / dz   #
                        HFs_l2 = k * dT / L   #
                        # (5.B) Compute total heat flux (conductive + advective) using beta
                        #if beta < 0.:
                        #    beta = 0. 
                        #dT = t2 - t1
                        #L = abs(L)
                        #if beta > .2:
                        #    HFs_tot_l2 = -k*beta/L * (dT/(np.exp(beta)-1) - t1)   ### need to be checked
                        #else:
                        #    HFs_tot_l2 = -k*beta/L * (dT/(beta) - t1)   ### need to be checked
                        if abs(beta) > .001:
                            HFs_tot_l2 = k*(beta/L * (dT/(np.exp(beta)-1) - t1))   
                        else:
                            HFs_tot_l2 = k*(dT/L - beta*t1/L)  
                        # add to lists
                        tg_list.append(dT/L)
                        hf_list.append(HFs_l2)
                        hf_tot_list.append(HFs_tot_l2)
                    #print(np.mean(hf_list), np.median(hf_list), np.std(hf_list)) 
                    # save samples average prop. in txt    
                    ave_prop = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'prop_cond_adv_cc_samples.txt', 'w')
                    ave_prop.write('# samples of ratio (proportion) of conduction / (conduction + advection) heat transport in the second layer (clay cap)\n')
                    for item in ave_prop_list:
                        ave_prop.write('{}\n'.format(item))
                    ave_prop.close()
                    # save samples heat flux (conductive) in txt    
                    hf_samp = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'hf_samples.txt', 'w')
                    hf_samp.write('# samples of total heat flux [W/m2] (conduction) heat transport in the second layer (clay cap)\n')
                    for item in hf_list:
                        hf_samp.write('{}\n'.format(item))
                    hf_samp.close()
                    # save samples heat flux (conductive + advective) in txt   
                    hf_tot_samp = open('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'hf_tot_samples.txt', 'w')
                    hf_tot_samp.write('# samples of total heat flux [W/m2] (conduction + advection) heat transport in the second layer (clay cap)\n')
                    for item in hf_tot_list:
                        hf_tot_samp.write('{}\n'.format(item))
                    hf_tot_samp.close()
                
                    # Save to file with mean values to plot with latlon
                    wl.thermal_grad = [np.mean(tg_list),np.std(tg_list)]
                    wl.thermal_cond = k
                    wl.heat_flux_cond = [np.mean(hf_list),np.std(hf_list)]
                    wl.heat_flux_tot = [np.mean(hf_tot_list),np.std(hf_tot_list)]
                    # write results 
                    #if wl.name == 'TH06':
                    #    print('hola')
                    if wl.heat_flux_tot[0] > wl.heat_flux_cond[0]: # if adventive heatflux is positive (ok)
                        wl_TG_TC_HFC_HFT.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+',' \
                            +str(round(wl.thermal_grad[0],2))+','+str(round(wl.thermal_cond,2))+',' \
                                +str(round(wl.heat_flux_cond[0],2))+','+str(round(wl.heat_flux_tot[0],2))+'\n')
                    else: # if adventive heatflux is negative, 
                            # replave total heat flux with conducitve heatflux 
                        wl_TG_TC_HFC_HFT.write(str(wl.name)+','+str(wl.lon_dec)+','+str(wl.lat_dec)+',' \
                            +str(round(wl.thermal_grad[0],2))+','+str(round(wl.thermal_cond,2))+',' \
                                +str(round(wl.heat_flux_cond[0],2))+','+str(round(wl.heat_flux_cond[0],2))+'\n')
            except:
                pass
        if pdf_temp_samlples:
            pp.close()
        wl_TG_TC_HFC_HFT.close()

    if stat_HF_by_beta: 
        if True: # histogram of betas and average prop across the clay cap
            betas_total = []
            ave_prop_total = []
            count_wells = 0
            for i, wl in enumerate(wells_objects):
                # check if well is infield
                # Resistivity Boundary, Mielke
                path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
                lats, lons = np.genfromtxt(path_rest_bound_WT, skip_header=1, delimiter=',').T
                poli_in = [[lons[i],lats[i]] for i in range(len(lats))]
                # check if well is infield (bool)
                val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_in)
                if val:
                    try:
                        # import betas and add to list
                        bl1,bl2,bl3 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'betas.txt').T
                        #[betas_total.append(item) for item in bl2] # positive up
                        [betas_total.append(-1*item) for item in bl2] # positive down
                        # import ave_prop and add to list
                        aux = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'prop_cond_adv_cc_samples.txt').T
                        #[ave_prop_total.append(item) for item in aux if np.abs(item)<=10.]
                        [ave_prop_total.append(item) for item in aux if (np.abs(item)<=1. and item>=0.)]
                        #[ave_prop_total.append(item) for item in aux if (item>=0.)]
                        del(aux,bl1,bl2,bl3)
                        count_wells+=1
                    except:
                        pass
            # plot histograms
            f = plt.figure(figsize=(15, 7))
            gs = gridspec.GridSpec(nrows=1, ncols=2)
            ax1 = f.add_subplot(gs[0, 0])
            ax2 = f.add_subplot(gs[0, 1])
            #ax_leg= f.add_subplot(gs[0, 2])
            colors = ['orange','blue']
            colors = [u'#ff7f0e', u'#1f77b4']
            
            ## betas
            #x_multi = [betas_total, t1_batch_filt_out]
            n_bins = int(2*np.sqrt(len(betas_total)))#15
            ax1.hist(betas_total, n_bins, histtype='bar', color = colors[0],edgecolor='#E6E6E6')
            ax1.set_xlabel(r'$\beta$', fontsize=textsize)
            ax1.set_ylabel('frequency', fontsize=textsize)
            ax1.set_xlim([-3,3])
            ax1.grid(True, which='both', linewidth=0.1)
            (mu, sigma) = norm.fit(betas_total)
            med = np.median(betas_total)
            ax1.set_title(r'$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
            
            ## average proportion
            #x_multi = [betas_total, t1_batch_filt_out]
            n_bins = int(2*np.sqrt(len(betas_total)))#15
            n_bins = 36
            ax2.hist(ave_prop_total, n_bins, histtype='bar', color = colors[1],edgecolor='#E6E6E6')
            ax2.set_xlabel(r'$\gamma$', fontsize=textsize)
            ax2.set_ylabel('frequency', fontsize=textsize)
            #if adv_all_negative:
            #    pass
                # ax2.set_xlim([0,1])
            #else:
            #    ax2.set_xlim([-1,3])
            #ax2.set_xlim([-.1,1.1])
            ax2.grid(True, which='both', linewidth=0.1)
            (mu, sigma) = norm.fit(ave_prop_total)
            med = np.median(ave_prop_total)
            ax2.set_title(r'$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
            #ax2.set_title(r'$med$:{:3.1f}, $per. 50$%: {:2.2f}, $per. 90$%: {:2.2f}'.format(med,np.percentile(ave_prop_total, 50, axis=0),
            #    np.percentile(ave_prop_total, 90, axis=0)), fontsize = textsize, color='gray')#, y=0.8)
            # legend
            #ax_leg.legend(loc='center', shadow=False, fontsize=textsize)#, prop={'size': 18})
            #ax_leg.axis('off')
            f.tight_layout()
            # save PNG
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'betas_ave_prop_nwells_'+str(count_wells)+'.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)  
        
        if True: # histogram of total heat flux (conduction + advection) through the clay cap  
            
            ###########################
            # conductive heat flux
            hf_total_cond = []
            count_wells = 0
            for i, wl in enumerate(wells_objects):
                # check if well is infield
                # Resistivity Boundary, Mielke
                path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
                lats, lons = np.genfromtxt(path_rest_bound_WT, skip_header=1, delimiter=',').T
                poli_in = [[lons[i],lats[i]] for i in range(len(lats))]
                # check if well is infield (bool)
                val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_in)
                if val:
                    try:
                        # import betas and add to list
                        hf_l2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'hf_samples.txt').T
                        [hf_total_cond.append(item) for item in hf_l2 if item>0.]
                        del(hf_l2)
                        count_wells+=1
                    except:
                        pass
            # plot histograms
            f = plt.figure(figsize=(8, 7)) # 15, 7
            gs = gridspec.GridSpec(nrows=1, ncols=1)
            ax1 = f.add_subplot(gs[0, 0])
            #ax2 = f.add_subplot(gs[0, 1])
            #ax_leg= f.add_subplot(gs[0, 2])
            colors = ['orange','blue']
            colors = [u'#ff7f0e', u'#1f77b4']
            #x_multi = [betas_total, t1_batch_filt_out]
            n_bins = int(1.5*np.sqrt(len(hf_total_cond)))#15
            ax1.hist(hf_total_cond, n_bins, histtype='bar', color = colors[1],edgecolor='#E6E6E6')
            ax1.set_xlabel(r'Heat flux [W/m$^2$]', fontsize=textsize)
            ax1.set_ylabel('frequency', fontsize=textsize)
            ax1.set_xlim([0,3])
            ax1.grid(True, which='both', linewidth=0.1)
            (mu, sigma) = norm.fit(hf_total_cond)
            med = np.median(hf_total_cond)
            ax1.set_title(r'$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
            # legend
            #ax_leg.legend(loc='center', shadow=False, fontsize=textsize)#, prop={'size': 18})
            #ax_leg.axis('off')
            f.tight_layout()
            # save PNG
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_Heat_flux_nwells_'+str(count_wells)+'_multi_samples.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
            
            ###########################
            # TOTAL HEAT FLUX
            hf_total = []
            count_wells = 0
            for i, wl in enumerate(wells_objects):
                # check if well is infield
                # Resistivity Boundary, Mielke
                path_rest_bound_WT = '.'+os.sep+'base_map_img'+os.sep+'shorelines_reservoirlines'+os.sep+'rest_bound_OUT_Mielke.txt'
                lats, lons = np.genfromtxt(path_rest_bound_WT, skip_header=1, delimiter=',').T
                poli_in = [[lons[i],lats[i]] for i in range(len(lats))]
                # check if well is infield (bool)
                val = ray_tracing_method(wl.lon_dec, wl.lat_dec, poli_in)
                if val:
                    try:
                        # import betas and add to list
                        hf_l2 = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+'hf_tot_samples.txt').T
                        [hf_total.append(item) for item in hf_l2 if item>0.]
                        del(hf_l2)
                        count_wells+=1
                    except:
                        pass
            # plot histograms
            f = plt.figure(figsize=(8, 7))
            gs = gridspec.GridSpec(nrows=1, ncols=1)
            ax1 = f.add_subplot(gs[0, 0])
            #ax2 = f.add_subplot(gs[0, 1])
            #ax_leg= f.add_subplot(gs[0, 2])
            colors = ['orange','blue']
            colors = [u'#ff7f0e', u'#1f77b4']
            
            #x_multi = [betas_total, t1_batch_filt_out]
            n_bins = int(2*np.sqrt(len(hf_total)))#15
            ax1.hist(hf_total, n_bins, histtype='bar', color = colors[1],edgecolor='#E6E6E6')
            ax1.set_xlabel(r'Heat flux [W/m$^2$]', fontsize=textsize)
            ax1.set_ylabel('frequency', fontsize=textsize)
            ax1.set_xlim([0,10])
            ax1.grid(True, which='both', linewidth=0.1)
            (mu, sigma) = norm.fit(hf_total)
            med = np.median(hf_total)
            #ax1.set_title(r'$med$:{:3.1f}, $\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(med,mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
            ax1.set_title(r'$med$:{:3.1f}, $per. 70$%: {:2.2f}, $per. 90$%: {:2.2f}'.format(med,np.percentile(hf_total, 70, axis=0),
                np.percentile(hf_total, 90, axis=0)), fontsize = textsize, color='gray')
            # legend
            #ax_leg.legend(loc='center', shadow=False, fontsize=textsize)#, prop={'size': 18})
            #ax_leg.axis('off')
            f.tight_layout()
            # save PNG
            plt.savefig('.'+os.sep+'corr_temp_bc'+os.sep+'00_global'+os.sep+'hist_Heat_flux_total_nwells_'+str(count_wells)+'_multi_samples.png', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)