"""
.. module:: Mt inversion validtion by ModEM comparition
   :synopsis: 

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
from lib_modEM import *
from matplotlib.backends.backend_pdf import PdfPages


textsize = 15.

# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
    ## PC that the code will be be run ('ofiice', 'personalSuse', 'personalWin')
    #pc = 'office'
    pc = 'personalSuse'
    #pc = 'personalWin'

    ## Set of data to work with 
    full_dataset = False
    prof_WRKNW6 = False
    prof_WRKNW5 = True
    prof_NEMT2 = False

    ## Sections of the code tu run
    set_up = False
    import_MT_stations = False
    import_results_MT_modem = False
    validation_MT_mcmc_inv = False 

    if set_up: 
        if pc == 'office': 
        #### ModEM results path
            if prof_WRKNW5: 
                path_modem = 'modEM_inv'
        if pc == 'personalSuse': 
        #### ModEM results path
            if prof_WRKNW5: 
                path_modem = 'modEM_inv'

    if import_MT_stations: 
        if pc == 'office': 
            #########  MT data
            path_files = "D:\workflow_data\kk_full\*.edi" 	# Whole array
        if pc == 'personalSuse':
            #########  MT data
            path_files = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/EDI_Files/*.edi" # Whole array 			
        ## Create a directory of the name of the files of the stations
        pos_ast = path_files.find('*')
        file_dir = glob.glob(path_files)
        if prof_WRKNW5:
            sta2work = ['WT039a','WT024a','WT030a','WT501a','WT033a','WT502a','WT060a','WT071a','WT068a','WT223a','WT070b','WT107a','WT111a']
            sta2work = ['WT223a']

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
                #sta_obj.app_res_phase() # [self.rho_app, self.phase_deg, self.rho_app_er, self.phase_deg_er]
                ## Create station objects and fill them
                station_objects.append(sta_obj)
                count  += 1

    if import_results_MT_modem: 
        # create modem object 
        modem = Modem(work_dir = path_modem)
        # load impedance (forward from final rho model)
        fls = glob(modem.work_dir + os.sep+'*.dat')
        # select newest
        newest = max(fls, key = lambda x: os.path.getctime(x))
        modem.read_data(newest, extract_origin = True) # origin: self.x0, y0, z0
        #print(modem.x0)
        #print(modem.y0)
        #print(modem.z0)

        # list of .rho (rho models)
        fls = glob(modem.work_dir + os.sep+'*.rho')
        # select newest
        newest = max(fls, key = lambda x: os.path.getctime(x))
        modem.read_input(newest)
        #print(modem.dim)
        # plot 2D model from results
        #modem.plot_rho2D(modem.work_dir+os.sep+'rho_inv.pdf', xlim = [-6., 6.], ylim = [-3.,0], gridlines = False, clim = [1e0,1.e4])
    
    if validation_MT_mcmc_inv:
        # (1) Extract res. profile in MT station positions from modem results (3D model)
        sta_coord = np.genfromtxt('MT_sta_latlon.txt')

        stas = sta_coord[:,0]
        lat = sta_coord[:,3] 
        lon = sta_coord[:,4] 
        ## depth vector for interpolation 
        #z_intp = np.linspace(0,2000,1000)
        
        sta = 'WT502a'
        #sta = 'WT223a'
        #sta = 'WT111a' 
        #sta = 'WT024a'
        #sta = 'WT070b'
        #sta = 'WT107a'

        if sta == 'WT502a':
            x_intp = 1008.640
            y_intp = -578.359  
            z0 = -487.523 

        if sta == 'WT223a':
            x_intp = -1881.360
            y_intp = 1733.641  
            z0 = -380.028 

        if sta == 'WT111a':
            x_intp = -3615.360
            y_intp = 3178.641 
            z0 = -380.028

        if sta == 'WT024a':
            x_intp = 2453.640
            y_intp = -2312.359  
            z0 = -487.523 

        if sta == 'WT070b':
            x_intp = -1303.360
            y_intp = 1155.641
            z0 = -487.523 

        if sta == 'WT107a':
            x_intp = -2459.360
            y_intp = 2600.641
            z0 = -380.028

        # WT502a => pos: 4
        # WT111a => pos: 11
        #x_intp = (modem.x0 - lat[4])*151111. # convert to meters
        #y_intp = (modem.y0 - lon[4])*111111. # convert to meters
        #print(x_intp)
        #print(y_intp)

        #modem.intp_1D_prof(2000,-2300)
        #res_intp = modem.intp_1D_prof(x_intp, y_intp) # Vector containing interpolate values at depths given in self.z
        res_intp = modem.intp_1D_prof(-1*x_intp, y_intp) # Vector containing interpolate values at depths given in self.z

        ## compare results from 3D inversion and MCMC
        # import results from station 'sta'
        # /home/aardid/Documentos/PhD_19/workspace/wt_inv_code_rep/mcmc_inversions
        sta_mcmc = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta+os.sep+'est_par.dat')
        z1_mcmc = sta_mcmc[0,1:4] # mean, std, median 
        z2_mcmc = sta_mcmc[1,1:4] # mean, std, median
        r2_mcmc = sta_mcmc[3,1:4] # mean, std, median # resistivity second layer 

        ## plot profile from 3D inversion and compare
        f = modem.plot_comp_mcmc(res_intp, z0, z1_mcmc, z2_mcmc, r2_mcmc)
        # check if station folder exist 
        if not os.path.exists('.'+os.sep+'modEM_inv'+os.sep+sta):
            os.makedirs('.'+os.sep+'modEM_inv'+os.sep+sta)        
        # save figure in station folder 
        f.savefig('.'+os.sep+'modEM_inv'+os.sep+sta+os.sep+'comp_3Dinv_mcmcinv.png')   # save the figure to file
        plt.close(f)    # close the figure

        ## plot boundaries uncertainty from 3D inversion 
        f, g = modem.plot_uncert_comp(res_intp, z0, z1_mcmc, z2_mcmc, r2_mcmc)
        f.savefig('.'+os.sep+'modEM_inv'+os.sep+sta+os.sep+'prof_intp.png')   # save the figure to file
        plt.close(f)    # close the figure
        g.savefig('.'+os.sep+'modEM_inv'+os.sep+sta+os.sep+'comp_cc_bound.png')   # save the figure to file
        plt.close(f)    # close the figure

    if True: # test using model in validation_MT_mcmc_inv
        # (1) Extract res. profile in MT station positions from modem results (3D model)
        # read MT_sta_latlon.txt file and extract name, lat and lon (decimal)
        sta_coord = np.genfromtxt('MT_sta_latlon.txt')
        lat = sta_coord[:,3] 
        lon = sta_coord[:,4] 
        elev = -1*sta_coord[:,5] # -1* to change it to possitive downwards
        t = open('MT_sta_latlon.txt','r')
        next(t)
        f1 = t.readlines()
        stas_name = []
        count = 0
        for x in f1:
            stas_name.append(x[0:6])
        t.close()
        
        # read coordinates in the grid (modEM)
        sta_coord_grid = np.genfromtxt('.'+os.sep+'modEM_inv'+os.sep+'sta_lat_lot_x_y_x_grid.txt')
        x_grid = sta_coord_grid[:,3] 
        y_grid = sta_coord_grid[:,4] 
        z_grid = sta_coord_grid[:,5] 
        t = open('.'+os.sep+'modEM_inv'+os.sep+'sta_lat_lot_x_y_x_grid.txt','r')
        next(t)
        f1 = t.readlines()
        stas_name_grid = []
        count = 0
        for x in f1:
            stas_name_grid.append(x[0:6])
        t.close()

        # import resistivity model from 'inversion_model_xyzrho'
        model, model_lat, model_lon, model_z, model_rho =  read_modem_model_column('.'+os.sep+'modEM_inv'+os.sep+'inversion_model_xyzrho')     

        # loop over the stations to 
        # extract 1D interpolated profile in 'stas_name' from 3D model
        pp = PdfPages('.'+os.sep+'modEM_inv'+os.sep+'comp_modEM_mcmc_meb.pdf')
        plot_meb = True

        for i in range(len(stas_name)): 
            # check is folder exist to save results for each station 
            if not os.path.exists('.'+os.sep+'modEM_inv'+os.sep+stas_name[i]):
                os.makedirs('.'+os.sep+'modEM_inv'+os.sep+stas_name[i])     
            # surface position 
            x_surf = lon[i]
            y_surf = lat[i]
            # interpolate and save the profile in .png
            res_z, z_vec, f = intp_1D_prof_from_model(model, x_surf, y_surf, method = None, dz = None ,fig = True, name = stas_name[i])
            #plt.show()
            f.savefig('.'+os.sep+'modEM_inv'+os.sep+stas_name[i]+os.sep+'prof_intp.png')   # save the figure to file
            plt.close(f)    # close the figure
            ## figure comparing modEM results vs mcmc results 
            # import results from station 'sta'
            # /home/aardid/Documentos/PhD_19/workspace/wt_inv_code_rep/mcmc_inversions
            sta_mcmc = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+stas_name[i]+os.sep+'est_par.dat')
            z1_mcmc = sta_mcmc[0,1:4] # mean, std, median 
            z2_mcmc = sta_mcmc[1,1:4] # mean, std, median
            r2_mcmc = sta_mcmc[3,1:4] # mean, std, median # resistivity second layer 
            # elevation of the station in the grid: z0_grid
            count = 0
            for sta in stas_name_grid: 
                if sta == stas_name[i]: 
                    z0_grid = z_grid[count]
                count+=1    
            # create figure
            f = plt.figure(figsize=[6.5,6.5])
            ## plot profile 3d inv
            ax = plt.axes()
            # note: z_vec is + downward 
            #ax.plot(np.log10(res_z), z_vec -  elev[i] ,'m-', label='profile from 3D inv.')
            ax.plot(np.log10(res_z), z_vec -  z0_grid ,'m-', label='profile from 3D inv.')
            plt.ylim([0.,1200])
            plt.xlim([-1,4])
            plt.gca().invert_yaxis()
            ## plot mcmc inv 
            # top boundary
            ax.errorbar(np.log10(r2_mcmc[0]),z1_mcmc[0],z1_mcmc[1],np.log10(r2_mcmc[1]),'r*', label = 'top bound. CC mcmc')
            # bottom boundary
            ax.errorbar(np.log10(r2_mcmc[0]),(z1_mcmc[0] + z2_mcmc[0]),z2_mcmc[1],np.log10(r2_mcmc[1]),'b*', label = 'bottom bound. CC mcmc')
            ax.set_xlabel('Resistivity [Ohm m]')
            ax.set_ylabel('Depth [m]')
            ax.set_title(stas_name[i]+': 3Dinv profile vs. MCMC CC boundaries')
            ax.grid(alpha = 0.3)
            if plot_meb: 
                # plot for selected stations 
                if stas_name[i] == 'WT111a':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'TH19'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    # top boundary
                    ax.errorbar(np.log10(5.),z1_meb[0],z1_meb[1], 0.,'g*', label = 'MeB: top bound. CC')
                    # bottom boundary
                    ax.errorbar(np.log10(5.),z2_meb[0],z2_meb[1], 0.,'c*', label = 'MeB: bottom bound. CC')
                if stas_name[i] == 'WT502a':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'WK243'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    # top boundary
                    ax.errorbar(np.log10(5.),z1_meb[0],z1_meb[1], 0.,'g*', label = 'MeB: top bound. CC')
                    # bottom boundary
                    ax.errorbar(np.log10(5.),z2_meb[0],z2_meb[1], 0.,'c*', label = 'MeB: bottom bound. CC')
                if stas_name[i] == 'WT033c':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'WK267A'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    # top boundary
                    ax.errorbar(np.log10(5.),z1_meb[0],z1_meb[1], 0.,'g*', label = 'MeB: top bound. CC')
                    # bottom boundary
                    ax.errorbar(np.log10(5.),z2_meb[0],z2_meb[1], 0.,'c*', label = 'MeB: bottom bound. CC')
                if stas_name[i] == 'WT501':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'WK270'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    # top boundary
                    ax.errorbar(np.log10(5.),z1_meb[0],z1_meb[1], 0.,'g*', label = 'MeB: top bound. CC')
                    # bottom boundary
                    ax.errorbar(np.log10(5.),z2_meb[0],z2_meb[1], 0.,'c*', label = 'MeB: bottom bound. CC')

            ax.legend(loc = 3)
            plt.tight_layout()
            #plt.show()
            f.savefig('.'+os.sep+'modEM_inv'+os.sep+stas_name[i]+os.sep+'comp_modEM_mcmc_1.png')   # save the figure to file
            pp.savefig(f)
            plt.close(f)    # close the figure
        
        pp.close()


        









         
















