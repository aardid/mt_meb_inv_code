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
    set_up = True
    import_MT_stations = False
    import_results_MT_modem = True
    validation_MT_mcmc_inv = True 

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
        modem = modEM(work_dir = path_modem)
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










