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
    pc = 'office'
    #pc = 'personalSuse'
    #pc = 'personalWin'

    ## Set of data to work with 
    full_dataset = True
    WK_Tmihi = False
    WK_NE_in = False
    WK_NE_out = False
    WK_S = False
    WK_NW_in = False
    WK_NW_out = False
    TH_in = False
    TH_out = False
    TH = False

    ## run with quality filter per well
    q_filt = True

    ## name output png file
    name_file = 'gen_corr_cc_temp.png'
    #name_file = 'W_S_corr_cc_temp_q.png'

    ## Sections of the code tu run
    set_up = True
    gen_trend = True 

    # (0) Import data and create objects: wells from spreadsheet files
    if set_up:
        #### Import data: MT from edi files and wells from spreadsheet files
        #########  MT data
        if pc == 'office': 
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
            ####### Temperature in wells data
            path_wells_loc = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/wells_loc.txt"
            path_wells_temp = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/well_depth_redDepth_temp_fixTH12_rmTHM24_fixWK404.txt"
            path_wells_temp_date = "/home/aardid/Documentos/data/Wairakei_Tauhara/Temp_wells/well_depth_redDepth_temp_date.txt"
            ####### MeB data in wells 
            path_wells_meb = "/home/aardid/Documentos/data/Wairakei_Tauhara/MeB_wells/MeB_data.txt"

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
        if WK_Tmihi :
            wl2work = ['WK122']
        if WK_NE_in :
            wl2work = ['WK301','WK305','WK321','WK308','WK304','WK309','WK310','WK317','WK307','WK306']
        if WK_NE_out :
            wl2work = ['WK314','WK315B']
        if WK_S: 
            wl2work = ['WK401','WK404','WK408','WK409A','WK402']
        if WK_NW_in :
            wl2work = ['WK270','WK267A','WK243','WK261','WK262','WK260','WK263']
        if WK_NW_out :
            wl2work = ['WK685','WK686','WK684','WK681','WK682','WK683']
        if TH_in :
            wl2work = ['TH19','TH13']
        if TH_out :
            wl2work = ['TH18','TH12']
        if TH :
            wl2work = ['TH18','TH12','TH19','TH13']

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

    # (1.1) Correlation temperature in z1 and z2 positions calc from meb inversion and temperature
    if gen_trend: 
        ## build histograms for temperatures at z1 and at z2. 
        ## the weigth of each well on histograms depends on the quality factor 
        # open quality ref file (corr_z1_z1_temp_glob.txt)
        path = '.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'\
            +os.sep+'corr_cc_temp'+os.sep+'Q_meb_inv_results.txt'
        q_wl_names, q_vec = np.genfromtxt(path).T

        # list to be used for histograms
        temp_full_list_z1 = []
        temp_full_list_z2 = []
        i = 0
        n_wells = 0
        # loop over wells: fill list temp_full_list_z1 and _z2
        for wl in wells_objects:
            if wl.meb: 
                if q_filt:
                    ## check quality of well
                    # index of wl in q_wl_names
                    #idx = np.where(q_wl_names == wl.name)
                    # quality factor of the well 
                    q_wl = q_vec[i]
                    i+=1
                    # add q_vec to list depending on the quality
                    if q_wl == 0:
                        pass
                    elif q_wl == 1:  
                        # open corr_z1_z2_temp.txt
                        path = '.'+os.sep+'mcmc_meb'+os.sep+wl.name+os.sep+'corr_z1_z2_temp.txt'
                        z1_vec, temp_z1_vec, z2_vec, temp_z2_vec = np.genfromtxt(path).T
                        for t1,t2 in zip(temp_z1_vec,temp_z2_vec):
                            temp_full_list_z1.append(t1)
                            temp_full_list_z2.append(t2)
                        # t1 = np.mean(temp_z1_vec) 
                        # t2 = np.mean(temp_z2_vec) 
                        # temp_full_list_z1.append(t1)
                        # temp_full_list_z2.append(t2)
                            
                    elif q_wl == 2:  
                        # open corr_z1_z2_temp.txt
                        path = '.'+os.sep+'mcmc_meb'+os.sep+wl.name+os.sep+'corr_z1_z2_temp.txt'
                        z1_vec, temp_z1_vec, z2_vec, temp_z2_vec = np.genfromtxt(path).T
                        n = 4 # number of append in list (weigth in hist)
                        for t1,t2 in zip(temp_z1_vec,temp_z2_vec):
                            for j in range(n):
                                temp_full_list_z1.append(t1)
                                temp_full_list_z2.append(t2)
                        # t1 = np.mean(temp_z1_vec) 
                        # t2 = np.mean(temp_z2_vec) 
                        # for j in range(n):
                        #     temp_full_list_z1.append(t1)
                        #     temp_full_list_z2.append(t2)
                    n_wells += 1
                else:
                    # open corr_z1_z2_temp.txt
                    path = '.'+os.sep+'mcmc_meb'+os.sep+wl.name+os.sep+'corr_z1_z2_temp.txt'
                    z1_vec, temp_z1_vec, z2_vec, temp_z2_vec = np.genfromtxt(path).T
                    for t1,t2 in zip(temp_z1_vec,temp_z2_vec):
                        temp_full_list_z1.append(t1)
                        temp_full_list_z2.append(t2)
                    n_wells += 1

        print('\n')
        print('Number of wells considered: {:}'.format(n_wells))
        # build histograms and plot
        # create a figure  of temp of z1 and z2 for the full net
        g,(ax1,ax2) = plt.subplots(1,2)
        g.set_size_inches(15,7)
        g.suptitle('Temps. for top and bottom boundaries of CC (MeB data)', fontsize=20)
        bins = np.linspace(0,251,10)
        # z1
        #bins = np.linspace(np.min(temp_full_list_z1), np.max(temp_full_list_z1), 1.5*int(np.sqrt(len(temp_full_list_z1))))
        h,e = np.histogram(temp_full_list_z1, bins, density = True )
        m = 0.5*(e[:-1]+e[1:])
        ax1.bar(e[:-1], h, e[1]-e[0])
        ax1.set_xlabel('Temp z1 [°C]', fontsize=10)
        ax1.set_ylabel('freq.', fontsize=10)
        ax1.grid(True, which='both', linewidth=0.1)
        # z2
        #bins = np.linspace(np.min(temp_full_list_z2), np.max(temp_full_list_z2), 1.5*int(np.sqrt(len(temp_full_list_z2))))
        h,e = np.histogram(temp_full_list_z2, bins, density = True)
        m = 0.5*(e[:-1]+e[1:])
        ax2.bar(e[:-1], h, e[1]-e[0])
        ax2.set_xlabel('Temp z2 [°C]', fontsize=10)
        ax2.set_ylabel('freq.', fontsize=10)
        ax2.grid(True, which='both', linewidth=0.1)
        # save figure
        path = '.'+os.sep+'mcmc_meb'+os.sep+'00_global_inversion'+os.sep+'corr_cc_temp'
        plt.savefig(path+os.sep+name_file, dpi=300, facecolor='w', edgecolor='w',\
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        plt.close("all")  





                


