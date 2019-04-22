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
        i+=1

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
        i+=1
    
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
            ax.plot(x_axis, z1_per[:,0],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,0]+10,'5%',size = 8., color = 'r')
            ax.plot(x_axis, z1_per[:,-1],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,-1]+10,'95%',size = 8.,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,0],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,0],'5%',size = 8., color = 'b')
            ax.plot(x_axis, z2_per[:,-1],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,-1],'95%',size = 8.,color = 'b')

        ## plot 20% and 80% percetils as dotted lines
        if width_ref == '60%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,3],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,3]+10,'20%',size = 8., color = 'r')
            ax.plot(x_axis, z1_per[:,-4],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,-4]+10,'80%',size = 8.,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,3],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,3],'20%',size = 8., color = 'b')
            ax.plot(x_axis, z2_per[:,-4],'b--',linewidth=.5)
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
            ax.text(x_axis_wl[i], topo_wl[i]-800., wl.name, rotation=90, size=6, bbox=dict(facecolor='blue', alpha=0.1)) 
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
            e = wl.meb_z2_pars[1] # [z1_mean_prior, z1_std_prior]
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
