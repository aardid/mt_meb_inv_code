"""
- Module Mapping: functions to contruct maps
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
"""

# ==============================================================================
#  Imports
# ==============================================================================

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import glob
import os
import shutil
from lib_sample_data import*
textsize = 15.

# ==============================================================================
# Coordinates
# ==============================================================================
def coord_dms2dec(H):
    lat = H[2]
    lon = H[3]
    
    lat = H[2]
    pos1 = lat.find(':')
    lat_degree = float(lat[:pos1])
    posaux = lat[pos1+1:].find(':')
    pos2 = pos1 + 1 + posaux 
    lat_minute = float(lat[pos1+1:pos2])
    lat_second = float(lat[pos2+1:])
    
    pos1 = lon.find(':')
    lon_degree = float(lon[:pos1])
    posaux = lon[pos1+1:].find(':')
    pos2 = pos1 + 1 + posaux 
    lon_minute = float(lon[pos1+1:pos2])
    lon_second = float(lon[pos2+1:])
    
    lat_dec = -1*round((abs(lat_degree) + ((lat_minute * 60.0) + lat_second) / 3600.0) * 1000000.0) / 1000000.0
    lon_dec = round((abs(lon_degree) + ((lon_minute * 60.0) + lon_second) / 3600.0) * 1000000.0) / 1000000.0

    return lat_dec, lon_dec

def for_google_earth(list, name_file = 'for_google_earth.txt', type_obj = None):
    # received a list of objects that posses name, lat, lon, elev as attributes
    # save a .txt to be open un googleearth
    locations4gearth= open(name_file,'w')
    if type_obj is None: 
        locations4gearth.write('object '+'\t'+'longitud'+'\t'+'latitud '+'\t'+'elevation'+'\n')
    else:
        locations4gearth.write(type_obj+'\t'+'longitud'+'\t'+'latitud '+'\t'+'elevation'+'\n')
    for obj in list: 
        # for google earth: change coordenate system (to decimals)
        lat_dec = str(obj.lat_dec)
        lon_dec = str(obj.lon_dec)
        elev = str(obj.elev)
        locations4gearth.write(obj.name[:-4]+'\t'+lon_dec+'\t'+lat_dec+'\t'+elev+'\n')
    locations4gearth.close()

def dist_two_points(coord1, coord2, type_coord = 'decimal'): 
    # coord = [lon, lat]
    #Rp = 6.357e3 # Ecuator: radius of earth in km  
    #Re = 6.378e3 # Pole: radius of earth in km 
    R = 6373. # around 39° latitud
    lon1, lat1 = [radians(coord1[0]),radians(coord1[1])]
    lon2, lat2 = [radians(coord2[0]),radians(coord2[1])]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2.))**2. + cos(lat1) * cos(lat2) * (sin(dlon/2.))**2.
    c = 2. * atan2(np.sqrt(a), np.sqrt(1.-a))
    return R * c # (where R is the radius of the Earth)

## test for dist_two_points function
# from Maping_functions import *
# coord1 = [-77.037852, 38.898556]
# coord2 = [-77.043934, 38.897147]
# d = dist_two_points(coord1, coord2)
# assert (d - 0.549) < 0.01 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

# ==============================================================================
# 2D profiles 
# ==============================================================================

def plot_2D_uncert_bound_cc(sta_objects, pref_orient = 'EW', file_name = 'z1_z2_uncert'): 
    ## sta_objects: list of station objects
    ## sort list by longitud (min to max - East to West)
    sta_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    
    x_axis = np.zeros(len(sta_objects))
    topo = np.zeros(len(sta_objects))
    z1_med = np.zeros(len(sta_objects))
    z1_per5 = np.zeros(len(sta_objects))
    z1_per95 = np.zeros(len(sta_objects))
    z2_med = np.zeros(len(sta_objects))
    z2_per5 = np.zeros(len(sta_objects))
    z2_per95 = np.zeros(len(sta_objects))


    i = 0
    for sta in sta_objects:
        coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
        coord2 = [sta.lon_dec, sta.lat_dec]
        ## calculate distances from first station to the others, save in array
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'decimal')
        # z1_pars                 distribution parameters for layer 1 
        #                         thickness (model parameter) calculated 
        #                         from mcmc chain results: [a,b,c,d,e]
        #                         a: mean
        #                         b: standard deviation 
        #                         c: median
        #                         d: percentile 5 (%)
        #                         e: percentile 95 (%)
        ## vectors for plotting 
        topo[i] = sta.elev
        z1_med[i] = topo[i] - sta.z1_pars[2]
        z1_per5[i] = topo[i] - sta.z1_pars[3][0]
        z1_per95[i] = topo[i] - sta.z1_pars[3][-1]
        z2_med[i] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[2])
        z2_per5[i] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[3][0])
        z2_per95[i] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[3][-1])


    ## plot envelopes 5% and 95% for cc boundaries
    f = plt.figure(figsize=[7.5,5.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])

    # plot using meadian a percentile
    ax.plot(x_axis, topo,'g-')
    ax.plot(x_axis, z2_med,'bo-', label='bottom')
    ax.fill_between(x_axis, z2_per5, z2_per95,  alpha=.5, edgecolor='b', facecolor='b')
    ax.plot(x_axis, z1_med,'ro-', label='top')
    ax.fill_between(x_axis, z1_per5, z1_per95,  alpha=.5, edgecolor='r', facecolor='r')

    i = 0
    for sta in sta_objects:
            ax.text(x_axis[i], topo[i]+300., sta.name[:-4], rotation=90, size=8) 
            i+=1
    #plt.gca().invert_yaxis() #invert axis y
    #ax.set_xlim([ystation[0]/1.e3, ystation[-1]/1.e3])
    #ax.set_xlim([-0.2,1.2])
    
    ax.set_ylim([-1.0e3,1.0e3])
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('depth [m]', size = textsize)
    ax.set_title('Clay cap boundaries depth  ', size = textsize)
    ax.legend(loc=4, prop={'size': 10})	
    
    #plt.savefig('z1_z2_uncert.pdf', dpi=300, facecolor='w', edgecolor='w',
    #    orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    plt.close(f)
    plt.clf()

    
def plot_2D_uncert_bound_cc_mult_env(sta_objects, pref_orient = 'EW', file_name = 'z1_z2_uncert', width_ref = None, prior_meb = None, plot_some_wells = None): 
    """
    width_ref: width of percetil of reference as dotted line centered at 50%. ex: '60%'
    prior_meb: full list of wells objects
    plot_some_wells: list of names of wells with MeB data to be plotted in the profile. ex: ['TH13','TH19']

    """
    # check for correct width plot reference input 
    if width_ref is None:
        width_plot = False
    else: 
        width_plot = True
    if [width_ref != '30%' or width_ref != '60%' or width_ref != '90%']:
        assert 'invalid width_ref: 30%, 60%, 90%'
    if prior_meb is None:
        prior_meb = False
    else: 
        wells_objects = prior_meb
    ## sta_objects: list of station objects
    ## sort list by longitud (min to max - East to West)
    sta_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    # vectors to be fill and plot 
    x_axis = np.zeros(len(sta_objects))
    topo = np.zeros(len(sta_objects))
    z1_med = np.zeros(len(sta_objects))
    z2_med = np.zeros(len(sta_objects))
    # percetil matrix
    s = (len(sta_objects),len(sta_objects[0].z1_pars[3])) # n° of stations x n° of percentils to plot
    z1_per = np.zeros(s)
    z2_per = np.zeros(s)
    # vector for negligeble stations in plot (based on second layer)
    stns_negli = np.zeros(len(sta_objects))

    i = 0
    for sta in sta_objects:
        coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
        coord2 = [sta.lon_dec, sta.lat_dec]
        ## calculate distances from first station to the others, save in array
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'decimal')
        ## vectors for plotting 
        topo[i] = sta.elev
        z1_med[i] = topo[i] - sta.z1_pars[2]
        z2_med[i] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[2])
        # fill percentils 
        for j in range(len(sta.z1_pars[3])): # i station, j percentil
            z1_per[i][j] = topo[i] - sta.z1_pars[3][j]
            z2_per[i][j] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[3][j])
        # criteria for being negligible
        if abs(sta.z1_pars[0] - sta.z2_pars[0]) < 20.: 
            stns_negli[i] = True
        else:
            stns_negli[i] = False
        i+=1

    # mask sections were second layer is negligible
    
    ## plot envelopes 5% and 95% for cc boundaries
    f = plt.figure(figsize=[7.5,5.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    # plot meadian and topo
    ax.plot(x_axis, topo,'g-')
    ax.plot(x_axis, z2_med,'b.-', label='bottom')
    ax.plot(x_axis, z1_med,'r.-', label='top')
    # plot percentils
    n_env = 9 # len(sta.z1_pars[3])/2 +1
    #for i in range(len(sta_objects)):
    for j in range(n_env):
        z1_inf = []
        z1_sup = []
        z2_inf = []
        z2_sup = []
        for i in range(len(sta_objects)):
            z1_sup.append(z1_per[i][j])
            z1_inf.append(z1_per[i][-j-1])
            z2_sup.append(z2_per[i][j])
            z2_inf.append(z2_per[i][-j-1])
        ax.fill_between(x_axis, z1_sup, z1_inf,  alpha=.05*(j+1), facecolor='r', edgecolor='r')
        ax.fill_between(x_axis, z2_sup, z2_inf,  alpha=.05*(j+1), facecolor='b', edgecolor='b')
    
    if width_plot: 
        ## plot 5% and 95% percetils as dotted lines 
        if width_ref == '90%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,0],'r--',linewidth=.5, alpha=.5)
            ax.text(x_axis[0]-.4,z1_per[0,0]+25,'5%',size = 8., color = 'r')
            ax.plot(x_axis, z1_per[:,-1],'r--',linewidth=.5, alpha=.5)
            ax.text(x_axis[0]-.4,z1_per[0,-1]-25,'95%',size = 8.,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,0],'b--',linewidth=.5, alpha=.5)
            ax.text(x_axis[-1]+.1,z2_per[-1,0]+0,'5%',size = 8., color = 'b')
            ax.plot(x_axis, z2_per[:,-1],'b--',linewidth=.5, alpha=.5)
            ax.text(x_axis[-1]+.1,z2_per[-1,-1]-50,'95%',size = 8.,color = 'b')

        ## plot 20% and 80% percetils as dotted lines
        if width_ref == '60%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,3],'r--',linewidth=.5, alpha=.2)
            ax.text(x_axis[0]-.8,z1_per[0,3]+25,'20%',size = 8., color = 'r')
            ax.plot(x_axis, z1_per[:,-4],'r--',linewidth=.5, alpha=.2)
            ax.text(x_axis[0]-.8,z1_per[0,-4]-25,'80%',size = 8.,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,3],'b--',linewidth=.5, alpha=.2)
            ax.text(x_axis[-1],z2_per[-1,3],'20%',size = 8., color = 'b')
            ax.plot(x_axis, z2_per[:,-4],'b--',linewidth=.5, alpha=.2)
            ax.text(x_axis[-1],z2_per[-1,-4],'80%',size = 8.,color = 'b')

        ## plot 45% and 65% percetils as dotted lines
        if width_ref == '30%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,8],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,8]+10,'45%',size = 8., color = 'r')
            ax.plot(x_axis, z1_per[:,12],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,12]+10,'65%',size = 8.,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,8],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,8],'20%',size = 8., color = 'b')
            ax.plot(x_axis, z2_per[:,12],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,12],'80%',size = 8.,color = 'b')
    # plot station names    
    i = 0
    for sta in sta_objects:
            ax.text(x_axis[i], topo[i]+400., sta.name[:-4], rotation=90, size=6, bbox=dict(facecolor='red', alpha=0.1)) 
            i+=1

    if prior_meb:
        if plot_some_wells is None: 
            plot_some_wells = False
        wl_names = []
        wls_obj = []
        # collect nearest wells names employed in every station for MeB priors
        if plot_some_wells: # for well names given 
            wl_names = plot_some_wells
        else: # for every well considered in priors
            for sta in sta_objects:
                aux_names = sta.prior_meb_wl_names
                for wln in aux_names: 
                    if wln in wl_names:
                        pass
                    else: 
                        wl_names.append(wln)
        for wl in wells_objects: 
            for wln in wl_names: 
                if wl.name == wln: 
                    wls_obj.append(wl)

        ## wls_obj: list of wells used for MeB prior in profile stations 
        ## sort list by longitud (min to max - East to West)
        wls_obj.sort(key=lambda x: x.lon_dec, reverse=False)
        # vectors to be fill and plot 
        x_axis_wl = np.zeros(len(wls_obj))
        topo_wl = np.zeros(len(wls_obj))
        i = 0
        for wl in wls_obj:
            coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
            coord2 = [wl.lon_dec, wl.lat_dec]
            ## calculate distances from first well to the others, save in array
            x_axis_wl[i] = dist_two_points(coord1, coord2, type_coord = 'decimal')
            ## vectors for plotting 
            topo_wl[i] = wl.elev
            i+=1

        i = 0
        for wl in wls_obj: 
            # plot well names 
            ax.text(x_axis_wl[i], topo_wl[i]-0.9e3, wl.name, rotation=90, size=6, bbox=dict(facecolor='blue', alpha=0.1)) 
            # import and plot MeB mcmc result
            wl.read_meb_mcmc_results()
            ## vectors for plotting 
            top = wl.elev # elevation os the well
            x = x_axis_wl[i]
            # plot top bound. C
            y = top - wl.meb_z1_pars[0] # [z1_mean_prior, z1_std_prior]
            e = wl.meb_z1_pars[1] # [z1_mean_prior, z1_std_prior]
            ax.errorbar(x, y, e, color='lime',linestyle='--',zorder=3, marker='_')
            # plot bottom bound. CC
            y = top - wl.meb_z2_pars[0] # [z1_mean_prior, z1_std_prior]
            e = 2.*wl.meb_z2_pars[1] # [z1_mean_prior, z1_std_prior] # 2 time std (66%)
            ax.errorbar(x, y, e, color='cyan', linestyle='--',zorder=3, marker='_')
            i+=1

    ax.set_xlim([x_axis[0]-1, x_axis[-1]+1])
    if prior_meb:
        ax.set_ylim([-1.0e3, max(topo)+600.])
    else:
        ax.set_ylim([-1.0e3, max(topo)+600.])
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('depth [m]', size = textsize)
    ax.set_title('Clay cap boundaries depth  ', size = textsize)
    ax.legend(loc=4, prop={'size': 8})	
    #ax.grid(True)
    #(color='r', linestyle='-', linewidth=2)
    ax.grid(color='c', linestyle='-', linewidth=.1)
    
    #plt.savefig('z1_z2_uncert.pdf', dpi=300, facecolor='w', edgecolor='w',
    #    orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    plt.close(f)
    plt.clf()

def plot_profile_autocor_accpfrac(sta_objects, pref_orient = 'EW', file_name = None): 
    """
    """
    ## sta_objects: list of station objects
    ## sort list by longitud (min to max - East to West)
    sta_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    # vectors to be fill and plot 
    x_axis = np.zeros(len(sta_objects))
    #topo = np.zeros(len(sta_objects))
    af_med = np.zeros(len(sta_objects)) # acceptance fraction mean for sta.
    af_std = np.zeros(len(sta_objects)) # acceptance fraction std. for sta.
    act_med = np.zeros(len(sta_objects)) # autocorrelation time mean for sta.
    act_std = np.zeros(len(sta_objects)) # autocorrelation time std. for sta.

    i = 0
    for sta in sta_objects:
        # load acceptance fraction and autocorrelation time for station
        coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
        coord2 = [sta.lon_dec, sta.lat_dec]
        ## calculate distances from first station to the others, save in array
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'decimal')
        ## vectors for plotting 
        #topo[i] = sta.elev
        af_med[i] = sta.af_mcmcinv[0] 
        af_std[i] = sta.af_mcmcinv[1] 
        act_med[i] = sta.act_mcmcinv[0] 
        act_std[i] = sta.act_mcmcinv[1] 
        i+=1

    ## plot af and act for profile 
    f = plt.figure(figsize=[7.5,5.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    # plot meadian and topo
    #ax.plot(x_axis, topo,'g-')
    color = 'r'
    ax.errorbar(x_axis,af_med, yerr= af_std, fmt='-o', color = color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0.,1.0])
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'b'
    ax2.errorbar(x_axis, act_med, yerr= act_std, fmt='-o', color = color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.,1.2e3])
    ## plot station names and number of independent samples    
    i = 0
    for sta in sta_objects:
        # load samples information 
        samp_inf = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+'samples_info.txt')
        n_samples = int(samp_inf[0])
        ax2.text(x_axis[i], 1100., sta.name[:-4]+': '+str(n_samples)+' samples', rotation=90, size=6, bbox=dict(facecolor='red', alpha=0.1)) 
        i+=1

    #ax.set_xlim([x_axis[0]-1, x_axis[-1]+1])
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('acceptance ratio', size = textsize)
    ax2.set_ylabel('autocorrelate time', size = textsize)
    ax.set_title('Quality parameters of mcmc inversion', size = textsize)
    #ax.legend(loc=4, prop={'size': 8})	
    ax.grid(True)
    #(color='r', linestyle='-', linewidth=2)
    ax.grid(color='c', linestyle='-', linewidth=.1)
    
    #plt.savefig('z1_z2_uncert.pdf', dpi=300, facecolor='w', edgecolor='w',
    #    orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    plt.close(f)
    plt.clf()

def plot_2D_uncert_isotherms(sta_objects, wells_objects, pref_orient = 'EW', file_name = None, \
     width_ref = None, isotherms = None, percentiels = None, sc_intp = False): 
    """
    Plot profile on uncertain isotherms.
    Save figure in temperature folder 'temp_prof_samples'

    Inputs
    ------
    sta_objects: list of MT station objects
    wells_objects: list of well objects
    file_name: name of file to be saved

    Notes
    -----
    figure is save by default in path = '.'+os.sep+'temp_prof_samples'
    """
    ###
    if file_name is None: 
        file_name = 'isotherm_uncert'
    if isotherms is None:
        isotherms = ['150','210']
    if percentiels is None:
        percentiels = np.arange(5.,100.,5.) 
    ## sort list by longitud (min to max - East to West)
    sta_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    # vectors to be fill and plot 
    x_axis = np.zeros(len(sta_objects))
    topo = np.zeros(len(sta_objects))

    # (List Comprehensions) list of n_isotherms x n_stations x n_percentiels
    iso_plot_matrix = [[[0. for k in range(len(percentiels))] for j in range(len(sta_objects))] for i in range(len(isotherms))]

    # loop over the stations, to create the lines of percentils 
    i=0
    for sta in sta_objects:
        # coord of station
        coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
        coord2 = [sta.lon_dec, sta.lat_dec]
        ## calculate distances from first station to the others, save in array
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'decimal')
        ## vectors for plotting 
        topo[i] = sta.elev
        # load percentiels
        sta_perc = np.genfromtxt(sta.path_temp_est+os.sep+'isotherms_percectils.txt')
        # dictionary of percentils for each isotherm; key: isotherm, values: z of percentiels 
        sta_isoth_perc = dict([(str(int(sta_perc[i,0])), sta_perc[i,1:]) for i in range(len(sta_perc[:,0]))])

        # populate iso_plot_matrix to be plot outside for loop 
        for j in range(len(isotherms)): 
            iso_plot_matrix[j][i][:] = sta_isoth_perc[isotherms[j]] 
        i+=1

    ## plot envelopes of percentils with fill_between
    # create figure
    f = plt.figure(figsize=[9.5,7.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    # plot meadian and topo
    ax.plot(x_axis, topo,'g-')
    # plot station names 
    i=0
    for sta in sta_objects:
            ax.text(x_axis[i], topo[i]+400., sta.name[:-4], rotation=90, size=8, bbox=dict(facecolor='red', alpha=0.1)) 
            i+=1
    # colrs for isotherms
    if len(isotherms) == 5: 
        iso_col = ['m','b','g','y','r']
    if len(isotherms) == 2: 
        iso_col = ['b','r']
    if len(isotherms) == 3: 
        iso_col = ['b','y','r']
    if len(isotherms) == 4: 
        iso_col = ['m','b','y','r']
    ## plot envelopes, per each isotherm 
    for i in range(len(isotherms)):
        # plor per envelope
        n = len(percentiels) # define how many envelopes to construct
        for j in range(int(n/2)+1):
            env_up = [] 
            env_low = []
            for k in range(len(sta_objects)): 
                # fill upper ad lower lines for envelopes 
                # note : Zpercentils are depth from surface 0 elevation -> for plotting, they need to refer to the topo (sta.elev)
                env_up.append(topo[k] - iso_plot_matrix[i][k][j])
                env_low.append(topo[k] - iso_plot_matrix[i][k][(n-1)-j])
                if env_up[-1] == env_low[-1]: 
                    env_up[-1] = -2000.
                    env_low[-1] = -2000.
            # plot envelope (j) for isotherm (i) 
            if sc_intp: 
                pass
                xi = np.asarray(x_axis)
                yi_up = np.asarray(env_up)
                yi_low = np.asarray(env_low)
                N_rs = 50 # number of resample points data
                xj = np.linspace(xi[0],xi[-1],N_rs)	
                yj_up = cubic_spline_interpolation(xi,yi_up,xj)
                yj_low = cubic_spline_interpolation(xi,yi_low,xj)
                if j == (int(n/2)):
                    ax.fill_between(xj, yj_up, yj_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i], label = isotherms[i]+'℃')
                else:
                    ax.fill_between(xj, yj_up, yj_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i])
            else:
                if j == (int(n/2)):
                    ax.fill_between(x_axis, env_up, env_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i], label = isotherms[i]+'℃')
                # elif j == 0:# or j == 1: # do not plot the first envelop for each isothem (5% and 95%) and (10% and 90%)
                #     pass
                else:
                    ax.fill_between(x_axis, env_up, env_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i])

    # labels for figure
    ax.set_ylim([-0.5e3, max(topo)+600.])
    ax.set_xlim([x_axis[0]-1, x_axis[-1]+2.0])
    ax.legend(loc=5, prop={'size': 10})	
    ax.grid(color='c', linestyle='-', linewidth=.1)
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('depth [m]', size = textsize)
    ax.set_title('Uncertain isotherms', size = textsize)
    # save figure
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    shutil.move(file_name+'.png', '.'+os.sep+'temp_prof_samples'+os.sep+file_name+'.png')
    plt.close(f)
    plt.clf()






       


    