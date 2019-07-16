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
    pc = 'office'
    #pc = 'personalSuse'
    #pc = 'personalWin'

    ## Set of data to work with 
    full_dataset = False
    prof_WRKNW6 = False
    prof_WRKNW5 = True
    prof_NEMT2 = False

    ## Sections of the code tu run
    set_up = True
    import_results_modem = True
    import_results_MTmcmcinv = False

    if set_up: 
        if pc == 'office': 
        #### ModEM results path
            if prof_WRKNW5: 
                path_modem = 'modEM_inv'

    if import_results_modem: 
        # create modem object 
        modem = modEM(work_dir = path_modem)
        # load impedance (forward from final rho model)
        #modem.read_input(modem.work_dir+os.sep+'*.dat')
        # list of .rho (rho models)
        fls = glob(modem.work_dir + os.sep+'*.rho')
        # select newest
        newest = max(fls, key = lambda x: os.path.getctime(x))
        modem.read_input(newest)
        #print(modem.dim)
        # plot result
        print(modem.y)
        modem.plot_rho2D(modem.work_dir+os.sep+'rho_inv.pdf', xlim = [-6., 6.], ylim = [-3.,0], gridlines = False, clim = [1e0,1.e4])


        





