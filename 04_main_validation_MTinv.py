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
import random
from lib_MT_station import *
from lib_Well import *
from lib_mcmc_MT_inv import *
from lib_mcmc_meb import * 
from lib_sample_data import*
from Maping_functions import *
from misc_functios import *
from lib_modEM import *
from matplotlib.backends.backend_pdf import PdfPages

#import matplotlib
#matplotlib.rcParams.update({'font.size': 12})

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

    ## Sections of the code tu run
    set_up = False
    import_MT_stations = False
    import_results_MT_modem = False
    validation_MT_mcmc_inv = False 
    
    fig_comp_1D_models = True
    fig_comp_data_misfit = True

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

    if fig_comp_1D_models: # Figures: 1D resistivity model comparition between 3D ModEM and 1D MCMC inversions result (See Figure 10b-d, main body)
        # (1) Extract res. profile in MT station positions from modem results (3D model)
        # read MT_sta_latlon.txt file and extract name, lat and lon (decimal)
        sta_coord = np.genfromtxt('MT_sta_latlon_prof_WRKNW5.txt')
        lat = sta_coord[:,3] 
        lon = sta_coord[:,4] 
        elev = -1*sta_coord[:,5] # -1* to change it to possitive downwards
        t = open('MT_sta_latlon_prof_WRKNW5.txt','r')
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
        #stas_name = ['WT111a']
        ###########################################
        for i in range(len(stas_name)): 
            print(stas_name[i])
            # check is folder exist to save results for each station 
            if not os.path.exists('.'+os.sep+'modEM_inv'+os.sep+stas_name[i]):
                os.makedirs('.'+os.sep+'modEM_inv'+os.sep+stas_name[i])     
            # surface position 
            x_surf = lon[i]
            y_surf = lat[i]
            # interpolate and save the profile in .png
            res_z, z_vec, f = intp_1D_prof_from_model(model, x_surf, y_surf, method = None, dz = None ,fig = True, name = stas_name[i], format_res = 'LOGE')
            #print(z_vec[:50])
            #print(res_z[:50])
            
            if False: # format rho file LOGE
                res_z = np.log(res_z)
                #res_z = np.exp(res_z)
                print(res_z[:50])
            #plt.show()

            f.savefig('.'+os.sep+'modEM_inv'+os.sep+stas_name[i]+os.sep+'prof_intp.png')   # save the figure to file
            plt.close(f)    # close the figure
            ## figure comparing modEM results vs mcmc results 
            # import results from station 'sta'
            # /home/aardid/Documentos/PhD_19/workspace/wt_inv_code_rep/mcmc_inversions
            sta_mcmc = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+stas_name[i]+os.sep+'est_par.dat')
            z1_mcmc = sta_mcmc[0,1:4] # mean, std, median 
            z2_mcmc = sta_mcmc[1,1:4] # mean, std, median
            r1_mcmc = sta_mcmc[2,1:4] # mean, std, median # resistivity second layer 
            r2_mcmc = sta_mcmc[3,1:4] # mean, std, median # resistivity second layer
            r3_mcmc = sta_mcmc[4,1:4] # mean, std, median # resistivity second layer
            # elevation of the station in the grid: z0_grid
            count = 0
            for sta in stas_name_grid: 
                if sta == stas_name[i]: 
                    z0_grid = z_grid[count]
                count+=1    
            # create figure
            f = plt.figure(figsize=[6.5,5.5])
            ## plot profile 3d inv
            ax = plt.axes()
            # note: z_vec is + downward 
            #ax.plot(np.log10(res_z), z_vec -  elev[i] ,'m-', label='profile from 3D inv.')
            #ax.plot(np.log10(res_z), z_vec -  z0_grid ,'r-', label='1D profile from 3D deterministic inversion', linewidth=2.0)
            ax.plot(res_z, z_vec -  z0_grid ,'r-', label='3D ModEM inversion', linewidth=2.0)


            #plt.xlim([0,10])
            ## plot mcmc inv
            ms=12 
            ax.set_xscale('log')

            ax.set_xlabel(r'$\rho$ [$\Omega$ m]', size = textsize)
            ax.set_ylabel('Depth [m]', size = textsize)
            ax.set_title('MT station: '+stas_name[i], size = textsize)#+': 3D det. inv. vs. 1D sto. inv.', size = textsize)
            ax.grid(alpha = 0.3)

            ## plot resistivity model from mcmc results 
            # top boundary
            #ax.errorbar(r2_mcmc[0],z1_mcmc[0],z1_mcmc[1],r2_mcmc[1],'r*', label = 'MT stochastic: top boundary LRA',  ms=ms)
            # bottom boundary
            #ax.errorbar(r2_mcmc[0],(z1_mcmc[0] + z2_mcmc[0]),z2_mcmc[1],r2_mcmc[1],'b*', label = 'MT stochastic: bottom boundary LRA',  ms=ms)
            plot_model = True
            if plot_model:
                def square_fn(pars, x_axis):
                    """
                    Calcule y axis for square function over x_axis
                    with corners given by pars, starting at y_base. 
                    pars = [x1,x2,y0,y1,y2] 
                    y_base = 0 (default)
                    """
                    # vector to fill
                    y_axis = np.zeros(len(x_axis))
                    # pars = [x1,x2,y1]
                    # find indexs in x axis
                    idx1 = np.argmin(abs(x_axis-pars[0]))
                    idx2 = np.argmin(abs(x_axis-pars[1]))
                    # fill y axis and return 
                    y_axis[0:idx1] = pars[2]
                    y_axis[idx1:idx2+1] = pars[3]
                    y_axis[idx2+1:] = pars[4]
                    return y_axis
                # depths to plot
                z_model = np.arange(0.,1500.,.5)
                # model 
                #for par in pars_order:
                    #if all(x > 0. for x in par):
                
                if True: # sample from normal distribution  
                    Ns = 50
                    mu_z1, sigma_z1 = z1_mcmc[0], z1_mcmc[1]/2 # mean and standard deviation
                    z1 = np.random.normal(mu_z1, sigma_z1, Ns)
                    mu, sigma = z2_mcmc[0], z2_mcmc[1] # mean and standard deviation
                    z2 = np.random.normal(mu, sigma, Ns)
                    mu, sigma = r1_mcmc[0], r1_mcmc[1] # mean and standard deviation
                    r1 = np.random.normal(mu, sigma/2, Ns)
                    mu, sigma = r2_mcmc[0], r2_mcmc[1] # mean and standard deviation
                    r2 = np.random.normal(mu, sigma, Ns)
                    mu, sigma = r3_mcmc[0], r3_mcmc[1] # mean and standard deviation
                    r3 = np.random.normal(mu, sigma/2, Ns)
                    for j in range(Ns):
                        sq_prof_est = square_fn([z1[j],mu_z1+z2[j],r1[j],r2[j],r3[j]], x_axis=z_model)
                        ax.semilogx(sq_prof_est,z_model,'b-', lw = 1.0, alpha=0.2, zorder=0)
                    #ax.semilogx(sq_prof_est,z_model,'b-', lw = 1.0, alpha=0.2, zorder=0, label= '1D profile from 1D stochastic inversion')

                if True: # sample from true samples 
                    chain = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+stas_name[i]+os.sep+'chain_sample_order.dat')
                    N_ind_s = len(chain[:,0])
                    Ns = N_ind_s
                    for j in range(Ns):
                        sam = random.randint(0,N_ind_s-1)
                        sq_prof_est = square_fn([chain[sam,2],chain[sam,2]+chain[sam,3],chain[sam,4],chain[sam,5],chain[sam,6]], x_axis=z_model)
                        ax.semilogx(sq_prof_est,z_model,'b-', lw = .5, alpha=0.1, zorder=0)
                    ax.semilogx(sq_prof_est,z_model,'b-', lw = 1., alpha=0.5, zorder=0, label= '1D Bayesian inversion')


            if plot_meb: 
                # plot for selected stations 
                ms=12
                if stas_name[i] == 'WT111a':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'TH19'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    ## top boundary
                    #ax.errorbar(1.0,z1_meb[0],1.5*z1_meb[1], 1000.,'g-', label = 'MeB well WK401: top boundary clay cap',  ms=ms)
                    ## bottom boundary
                    #ax.errorbar(1.0,z2_meb[0],1.5*z2_meb[1], 1000.,'c-', label = 'MeB well WK401: bottom boundary clay cap',  ms=ms)
                    ## thick lines 
                    # top boundary
                    x_axis = np.linspace(0,100000, 100)
                    ax.fill_between(x_axis, z1_meb[0] - 1.5*z1_meb[1], z1_meb[0] + z1_meb[1],  alpha=.2, edgecolor='g', facecolor='g', label = 'MeB inversion TH19: top boundary')
                    ax.fill_between(x_axis, z2_meb[0] - z2_meb[1], z2_meb[0] + z2_meb[1],  alpha=.2, edgecolor='c', facecolor='c', label = 'MeB inversion TH19: bottom boundary')
                if stas_name[i] == 'WT107a':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'WK401'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    ## top boundary
                    #ax.errorbar(1.0,z1_meb[0],1.5*z1_meb[1], 1000.,'g-', label = 'MeB well WK401: top boundary clay cap',  ms=ms)
                    ## bottom boundary
                    #ax.errorbar(1.0,z2_meb[0],1.5*z2_meb[1], 1000.,'c-', label = 'MeB well WK401: bottom boundary clay cap',  ms=ms)
                    ## thick lines 
                    # top boundary
                    x_axis = np.linspace(0,100000, 100)
                    ax.fill_between(x_axis, z1_meb[0] - 1.5*z1_meb[1], z1_meb[0] + 1.5*z1_meb[1],  alpha=.2, edgecolor='g', facecolor='g', label = 'MeB inversion WK401: top boundary')
                    ax.fill_between(x_axis, z2_meb[0] - 1.5*z2_meb[1], z2_meb[0] + 1.5*z2_meb[1],  alpha=.2, edgecolor='c', facecolor='c', label = 'MeB inversion WK401: bottom boundary')
                if stas_name[i] == 'WT033c':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'WK267A'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    ## top boundary
                    #ax.errorbar(1.0,z1_meb[0],1.5*z1_meb[1], 1000.,'g-', label = 'MeB well WK401: top boundary clay cap',  ms=ms)
                    ## bottom boundary
                    #ax.errorbar(1.0,z2_meb[0],1.5*z2_meb[1], 1000.,'c-', label = 'MeB well WK401: bottom boundary clay cap',  ms=ms)
                    ## thick lines 
                    # top boundary
                    x_axis = np.linspace(0,100000, 100)
                    ax.fill_between(x_axis, z1_meb[0] - 1.5*z1_meb[1], z1_meb[0] + 1.5*z1_meb[1],  alpha=.2, edgecolor='g', facecolor='g', label = 'MeB inversion WK267A: top boundary')
                    ax.fill_between(x_axis, z2_meb[0] - 1.5*z2_meb[1], z2_meb[0] + 1.5*z2_meb[1],  alpha=.2, edgecolor='c', facecolor='c', label = 'MeB inversion WK267A: bottom boundary')
                if stas_name[i] == 'WT501a':
                    # import meb results 
                    # note: z1 and z2 in MeB are depths (not thicknesess)
                    sta_meb = np.genfromtxt('.'+os.sep+'mcmc_meb'+os.sep+'WK267A'+os.sep+'est_par.dat')
                    z1_meb = sta_meb[0,1:4] # mean, std, median 
                    z2_meb = sta_meb[1,1:4] # mean, std, median
                    ## top boundary
                    #ax.errorbar(1.0,z1_meb[0],1.5*z1_meb[1], 1000.,'g-', label = 'MeB well WK401: top boundary clay cap',  ms=ms)
                    ## bottom boundary
                    #ax.errorbar(1.0,z2_meb[0],1.5*z2_meb[1], 1000.,'c-', label = 'MeB well WK401: bottom boundary clay cap',  ms=ms)
                    ## thick lines 
                    # top boundary
                    x_axis = np.linspace(0,100000, 100)
                    ax.fill_between(x_axis, z1_meb[0] - z1_meb[1], z1_meb[0] + z1_meb[1],  alpha=.2, edgecolor='g', facecolor='g', label = 'MeB inversion WK267A: top boundary')
                    ax.fill_between(x_axis, z2_meb[0] - z2_meb[1], z2_meb[0] + z2_meb[1],  alpha=.2, edgecolor='c', facecolor='c', label = 'MeB inversion WK267A: bottom boundary')
            plt.ylim([0.,(z1_mcmc[0] + z2_mcmc[0] + 1000)])
            plt.xlim([.1,1e5])
            plt.gca().invert_yaxis()
            ax.legend(loc = 3, prop={'size': textsize})
            ax.tick_params(labelsize=textsize)
            plt.tight_layout()
            #plt.show()
            f.savefig('.'+os.sep+'modEM_inv'+os.sep+stas_name[i]+os.sep+'comp_modEM_mcmc_'+stas_name[i]+'.png',
                dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)   # save the figure to file
            f.savefig('.'+os.sep+'mcmc_inversions'+os.sep+stas_name[i]+os.sep+'comp_modEM_mcmc_'+stas_name[i]+'.png',
                dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)   # save the figure to file   
            pp.savefig(f)
            plt.close(f)    # close the figure
        
        pp.close()

    if fig_comp_data_misfit: # Figure: observed data vs estimated data from 3DModEM and 1DMCMCMT
        
        pass














