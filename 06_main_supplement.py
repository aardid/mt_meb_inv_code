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
from fpdf import FPDF


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
    sup_1 = True

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
            #sta2work = ['WT102a']
        if prof_WRKNW5:
            sta2work = ['WT039a','WT024a','WT030a','WT501a','WT502a','WT060a','WT071a','WT068a','WT223a','WT070b','WT107a','WT111a']#,'WT033b']
            #sta2work = ['WT039a','WT024a','WT030a']
            #sta2work = ['WT111a']
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
                ## correction in elevation for discrepancy between elev in wells and MT stations (temporal solition, to check)
                sta_obj.elev = float(sta_obj.elev) - 42.
                ##
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
            wl2work = ['WK261','WK262','WK263','WK243','WK267A','WK270','TH19','WK408','WK401', 'WK404'] # 'WK260' 
            #wl2work = ['WK401','TH19', 'WK404'] 
            wl2work = ['WK260','WK261','TH19','WK401','WK267A','WK270']#WK263' ,'WK267A'
            #wl2work = ['WK401']
        if prof_NEMT2:
            wl2work = ['TH12','TH18','WK315B','WK227','WK314','WK302']
            wl2work = ['WK261']

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

    if sup_1:
        # open pdf 
        pp = PdfPages('Sup_Mat_1.pdf')
        count = 1
        for sta_obj in station_objects: 
            # import figure 1: apparent rest and phase 
            #if True:
            f = plt.figure(figsize=[20,20])
            ax = f.add_subplot(111)
            path_base_image = '.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'app_res_fit.png'
            img=mpimg.imread(path_base_image)
            ax.imshow(img)#, interpolation='none')
            ax.axis('off')
            # add caption
            txt="Figure S-"+str(count)+". Observed and estimated data for the two MT station "+sta_obj.name[:-4]+". \
                \nUpper and lower panels of both figures show apparent resistivity and phase observed data points for \
                \nthe non-diagonal components of the impedance tensor $Z_{xy}$ (red '*') and $Z_{yx}$ (green '*'). \
                \nBlue lines show the estimated data generated by forwarding a set of models. \
                \nThe samples are sampled from the posterior distribution result from the MT inversion. \
                \nEach sample corresponds to a combination of parameters of the three-layer model consistent with the observed data."
            # center text
            f.text(.5, .01, txt, ha='center', fontsize=textsize+5)
            # resize the figure to match the aspect ratio of the Axes    
            #f.set_size_inches(7, 8, forward=True)
            #plt.tight_layout()
            pp.savefig(f)
            plt.close()
            count += 1

            # import figure 2: model and pars uncrt

            # f = plt.figure(figsize=[20,20])
            # ax = f.add_subplot(111)
            # path_base_image = '.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'model_samples.png'
            # img=mpimg.imread(path_base_image)
            # ax.imshow(img)#, interpolation='none')
            # ax.axis('off')
            # # add caption
            # txt="Figure S-"+str(count)+". Upper panel shows a set of models that best fit observed data (cyan lines) \n \
            #     for MT station WT111a. Lower panels show histograms of occurrence for the three-layer model parameters: \n \
            #     $z_1$ (first layer thickness), $z_2$ (second layer thickness), $ρ_1$ (first layer resistivity), \n \
            #     $ρ_2$(second layer resistivity) and ρ_3 (third layer resistivity).  \n \
            #     Red lines over histograms are the best normal fit. Dashed color lines show the mean estimated for each parameter.\n \
            #     Mean and standard deviation values are shown at the top of each histogram plot."
            # # center text
            # f.text(.5, .01, txt, ha='center', fontsize=textsize+5)
            # # resize the figure to match the aspect ratio of the Axes    
            # #f.set_size_inches(7, 8, forward=True)
            # #plt.tight_layout()
            # pp.savefig(f)
            # plt.close()
            # count += 1
        pp.close()

        # second option 
        pdf = FPDF()
        for sta_obj in station_objects:
            pdf.add_page()
            path_base_image = '.'+os.sep+'mcmc_inversions'+os.sep+sta_obj.name[:-4]+os.sep+'app_res_fit.png'
            list($x1, $y1) = getimagesize(path_base_image)
            pdf.image(path_base_image)
            break
        pdf.output("yourfile.pdf", "F")

