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
from math import sin, cos, sqrt, atan2, radians, isnan
import glob
import os
import shutil
from lib_sample_data import*
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.cm as cm
from scipy.spatial import Delaunay
import functools
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.image as mpimg
from scipy.stats import norm
import matplotlib.mlab as mlab
textsize = 15.

import matplotlib
matplotlib.rcParams.update({'font.size': textsize})

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

def dist_two_points(coord1, coord2, type_coord = None): 
    '''
    Calculate distance boetween two points 
    '''
    if type_coord == 'decimal':
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
    if type_coord == 'linear':
       # coord = [x, y]
        d = np.sqrt((coord2[1]-coord1[1])**2 + (coord2[0]-coord1[0])**2)
        return d 

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

    plt.clf()
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

    
def plot_2D_uncert_bound_cc_mult_env(sta_objects, type_coord = None, unit_dist = None, pref_orient = 'EW', sort = None, \
    file_name = None, format_fig = None, width_ref = None, prior_meb = None, plot_some_wells = None, xlim = None, ylim = None, \
    center_cero = None, km = None, export_fig = None, no_plot_clay_infered = None, mask_no_cc = None): 
    """
    width_ref: width of percetil of reference as dotted line centered at 50%. ex: '60%'
    prior_meb: full list of wells objects
    plot_some_wells: list of names of wells with MeB data to be plotted in the profile. ex: ['TH13','TH19']
    type_coord: 'decimal' for lat/lon in decimal or 'linear' for x,y cartesian coord. 
    unit_dist: just for linear. 'km' for kilometers and 'm' for meters. If 'm', x_axis is converted to 'km' for plotting.
    mask_no_cc: mask section between two stations where no consuctive layer is found. The value of mask_no_cc indicates the criteria:
        if the thickness of the second layer in meters is less than 'mask_no_cc', consider it as no conductive layer. 
    """
    if type_coord is None: 
        type_coord = 'decimal' # 'linear'
    else:
        type_coord = type_coord 
    if unit_dist is None: 
        unit_dist = 'km'
    if file_name is None: 
        file_name = 'z1_z2_uncert'
    if format_fig is None: 
        format_fig = 'png'
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
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = type_coord)
        ## vectors for plotting 
        topo[i] = sta.elev
        z1_med[i] = topo[i] - sta.z1_pars[2]
        z2_med[i] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[2])
        # fill percentils 
        for j in range(len(sta.z1_pars[3])): # i station, j percentil
            z1_per[i][j] = topo[i] - sta.z1_pars[3][j]
            z2_per[i][j] = topo[i] - (sta.z1_pars[2] + sta.z2_pars[3][j])
        if mask_no_cc: 
            # criteria for being negligible
            if abs(sta.z2_pars[0]) < mask_no_cc: 
                stns_negli[i] = True
            else:
                stns_negli[i] = False
        i+=1

    if center_cero: 
        mid_val = (abs(x_axis[0] - x_axis[-1]))/2
        x_axis = x_axis - mid_val

    if unit_dist is 'm': # transfor data to km
        x_axis = x_axis/1e3
    
    ## plot envelopes 5% and 95% for cc boundaries
    f = plt.figure(figsize=[9.5,6.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    # plot meadian and topo
    ax.plot(x_axis, topo,'g-', label='Topography')
    ax.plot(x_axis, z1_med,'r.-', label='$z_1$ estimated MT', zorder = 0)
    ax.plot(x_axis, z2_med,'b.-', label='$z_2$ estimated MT', zorder = 0)
    if no_plot_clay_infered:
        pass
    else: # plot orange section between means of z1 and z2 (indicating clay cap)
        ax.fill_between(x_axis, z2_med, z1_med,  alpha=.3, facecolor='orange', edgecolor='orange', label='Inferred clay')
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
            #ax.plot(x_axis, z1_per[:,0],'r--',linewidth=.5, alpha=.5)
            ax.text(x_axis[0]-.4,z1_per[0,0]+25,'5%',size = 8., color = 'r')
            #ax.plot(x_axis, z1_per[:,-1],'r--',linewidth=.5, alpha=.5)
            ax.text(x_axis[0]-.4,z1_per[0,-1]-25,'95%',size = 8.,color = 'r')
            # bottom boundary
            #ax.plot(x_axis, z2_per[:,0],'b--',linewidth=.5, alpha=.5)
            ax.text(x_axis[-1]+.1,z2_per[-1,0]+0,'5%',size = 8., color = 'b')
            #ax.plot(x_axis, z2_per[:,-1],'b--',linewidth=.5, alpha=.5)
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

    # mask sections were second layer is negligible
    if mask_no_cc: 
        for k in range(len(sta_objects)-1):
            if stns_negli[k]:# and stns_negli[k+1]: 
                x_mask = [x_axis[k], x_axis[k+1]]
                #y_mask = [topo[k]-10, -2000.]
                y_mask_top = [z1_per[k,0]+25,z1_per[k+1,0]+25]
                y_mask_bottom = [z2_per[k,-1]-50,z2_per[k+1,-1]-50]
                ax.fill_between(x_mask, y_mask_top, y_mask_bottom,  alpha=.98, facecolor='w', zorder=3)

    # plot station names    
    i = 0
    for sta in sta_objects:
            ax.text(x_axis[i], topo[i]+400., sta.name[:-4], rotation=90, size=8, bbox=dict(facecolor='red', alpha=0.1), ) 
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
            # near distance from well to stations in the profile
            dist_wl_prof = []
            for sta in sta_objects: 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta.lon_dec, sta.lat_dec], type_coord = 'decimal')
                if not dist_wl_prof:
                    dist_wl_prof = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_wl_prof: 
                    dist_wl_prof = dist

            # plot well names and distance to the profile (near station) 
            ax.text(x_axis_wl[i], topo_wl[i]-.9e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=7, bbox=dict(facecolor='blue', alpha=0.1)) 
            # import and plot MeB mcmc result
            wl.read_meb_mcmc_results()
            ## vectors for plotting 
            top = wl.elev # elevation os the well
            x = x_axis_wl[i]
            # plot top bound. C
            y1 = top - wl.meb_z1_pars[0] # [z1_mean_prior, z1_std_prior]
            e1 = wl.meb_z1_pars[1] # [z1_mean_prior, z1_std_prior]
            ax.errorbar(x, y1, e1, color='cyan',linestyle='-',zorder=3, marker='_')
            # plot bottom bound. CC
            y2 = top - wl.meb_z2_pars[0] # [z1_mean_prior, z1_std_prior]
            e2 = 2.*wl.meb_z2_pars[1] # [z1_mean_prior, z1_std_prior] # 2 time std (66%)
            ax.errorbar(x, y2, e2, color='m', linestyle='-',zorder=2, marker='_')
            i+=1
        ax.errorbar(x, y1, e1, color='cyan',linestyle='-',zorder=3, marker='_', label = '$z_1$ estimated MeB')
        ax.errorbar(x, y2, e2, color='m', linestyle='-',zorder=3, marker='_', label = '$z_2$ estimated MeB')

    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    else:
        ax.set_xlim([x_axis[0]-2, x_axis[-1]+1])
    if prior_meb:
        if ylim:
            ax.set_ylim([ylim[0], ylim[1]])
        else:
            ax.set_ylim([-1.0e3, max(topo)+600.])
    else:
        if ylim:
            ax.set_ylim([ylim[0], ylim[1]])
        else:
            ax.set_ylim([-1.0e3, max(topo)+600.])

    #plt.xticks(np.linspace(0,10,10))
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('z [m]', size = textsize)
    ax.set_title('LRA uncertain boundaries', size = textsize)

    #ax.grid(True)
    #(color='r', linestyle='-', linewidth=2)
    ax.grid(color='c', linestyle='-', linewidth=.1, zorder=4)
    #ax.tick_params(labelsize=textsize)

    #plt.grid(True)
    
    if export_fig:
        return f, ax

    ax.legend(loc=3, prop={'size': 10})	
    #plt.savefig('z1_z2_uncert.pdf', dpi=300, facecolor='w', edgecolor='w',
    #    orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
    plt.savefig(file_name+'.'+format_fig, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format=format_fig,transparent=True, bbox_inches=None, pad_inches=.1)	
    plt.close(f)
    plt.clf()

def plot_profile_autocor_accpfrac(sta_objects, pref_orient = 'EW', file_name = None, center_cero = None, unit_dist = None): 
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
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'linear')*100
        ## vectors for plotting 
        #topo[i] = sta.elev
        af_med[i] = sta.af_mcmcinv[0] 
        af_std[i] = sta.af_mcmcinv[1] 
        act_med[i] = sta.act_mcmcinv[0] 
        act_std[i] = sta.act_mcmcinv[1] 
        i+=1
    
    if center_cero: 
        mid_val = (abs(x_axis[0] - x_axis[-1]))/2
        x_axis = x_axis - mid_val

    if unit_dist is 'm': # transfor data to km
        x_axis = x_axis/1e3

    ## plot af and act for profile 
    plt.clf()
    f = plt.figure(figsize=[7.5,5.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    # plot meadian and topo
    #ax.plot(x_axis, topo,'g-')
    color = 'r'
    ax.errorbar(x_axis,af_med, yerr= af_std, fmt='-o', color = color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0.,1.3])
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'b'
    ax2.errorbar(x_axis, act_med, yerr= act_std, fmt='-o', color = color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.,1.2e3])
    ## plot station names and number of independent samples    
    i = 0
    for sta in sta_objects:
        # load samples information 
        # if station comes from edi file (.edi)
        if sta.name[-4:] == '.edi': 
            samp_inf = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+'samples_info.txt')
        else: 
            samp_inf = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name+os.sep+'samples_info.txt')
        n_samples = int(samp_inf[0])
        ax2.text(x_axis[i], 1100., sta.name[:-4]+': '+str(n_samples)+' samples', rotation=90, size=8, bbox=dict(facecolor='red', alpha=0.1)) 
        i+=1

    #ax.set_xlim([x_axis[0]-1, x_axis[-1]+1])
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('acceptance ratio', size = textsize)
    ax2.set_ylabel('autocorrelate time', size = textsize)
    ax.set_title('Quality parameters of MT inversion', size = textsize)
    #ax.legend(loc=4, prop={'size': 8})	
    ax.grid(True)
    #(color='r', linestyle='-', linewidth=2)
    ax.grid(color='c', linestyle='-', linewidth=.1)
    
    ax.tick_params(labelsize=textsize-2)
    #plt.yticks([0.,.2,.4,.6,.8,1.])
    ax2.tick_params(labelsize=textsize-2)


    #plt.savefig('z1_z2_uncert.pdf', dpi=300, facecolor='w', edgecolor='w',
    #    orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	

    plt.close(f)
    plt.clf()

def plot_profile_KL_divergence(sta_objects, wells_objects, pref_orient = 'EW', file_name = None, center_cero = None, unit_dist = None): 
    """
    plot KL divergence values (posterior vs meb prior) for parameters z1 and z2.  
    """
    ## sta_objects: list of station objects
    ## sort list by longitud (min to max - East to West)
    sta_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    # vectors to be fill and plot 
    x_axis = np.zeros(len(sta_objects))
    #topo = np.zeros(len(sta_objects))
    KL_z1 = np.zeros(len(sta_objects)) # Kl for z1 for sta.
    KL_z2 = np.zeros(len(sta_objects)) # Kl for z2 for sta.

    i = 0
    for sta in sta_objects:
        # load acceptance fraction and autocorrelation time for station
        coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
        coord2 = [sta.lon_dec, sta.lat_dec]
        ## calculate distances from first station to the others, save in array
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'linear')*100.
        ## vectors for plotting 
        #topo[i] = sta.elev
        if sta.name[-4:] == '.edi': 
            KL = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+'KL_value.txt')
        else: 
            KL = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name+os.sep+'KL_value.txt')
        # z1
        if isnan(KL[0]): # condition if KL value is nan. Default '0'
            KL_z1[i] = 0.
        else: 
            KL_z1[i] = KL[0]
        # z2
        if isnan(KL[1]): # condition if KL value is nan. Default '0'
            KL_z2[i] = 0.
        else: 
            KL_z2[i] = KL[1]
        i+=1
    
    if center_cero: 
        mid_val = (abs(x_axis[0] - x_axis[-1]))/2
        x_axis = x_axis - mid_val

    if unit_dist is 'm': # transfor data to km
        x_axis = x_axis/1e3

    ## plot af and act for profile 
    plt.clf()
    f = plt.figure(figsize=[7.5,5.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    # plot meadian and topo
    #ax.plot(x_axis, topo,'g-')
    ax.plot(x_axis,KL_z1, '-o', color = 'r', label = '$z_1$')
    ax.plot(x_axis,KL_z2, '-o', color = 'b', label = '$z_2$')
    #ax.tick_params(axis='y', labelcolor='r')
    
    ## plot station names  
    i = 0
    for sta in sta_objects:
        ax.text(x_axis[i], np.max(KL_z1) + 4., sta.name[:-4], rotation=90, size=textsize-5, bbox=dict(facecolor='red', alpha=0.1)) 
        i+=1

    # plot wells names

    #ax.set_xlim([x_axis[0]-1, x_axis[-1]+1])
    ax.set_ylim([-1,  np.max(KL_z1) + 5])
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('[]', size = textsize)
    ax.set_title('KL divergence', size = textsize)
    ax.legend(prop={'size': textsize-2}, fancybox=True, framealpha=0.5, loc ='center right')	
    ax.grid(True)
    #(color='r', linestyle='-', linewidth=2)
    #plt.yticks(np.arange(0,10,2))
    ax.grid(color='c', linestyle='-', linewidth=.1)
    ax.tick_params(labelsize=textsize)

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

def triangulation_meb_results(station_objects, well_objects, path_base_image = None, ext_img = None, xlim = None, ylim = None, \
     file_name = None, format = None ): 

    lon_stas = []
    lat_stas = []
    for sta in station_objects:
        lon_stas.append(sta.lon_dec)
        lat_stas.append(sta.lat_dec)
    lon_stas = np.asarray(lon_stas)
    lat_stas = np.asarray(lat_stas)
    
    # (0) Define grid:
    # borders given by well objects

    # define dominio
    lon_dom = []
    lat_dom = []
    z1_dom_mean = []   
    points = []
    values = []
    count=0
    for wl in well_objects:
        if wl.meb:
            # load well pars from meb inversion 
            meb_mcmc_results = np.genfromtxt(wl.path_mcmc_meb+os.sep+"est_par.dat")
            wl.meb_z1_pars = meb_mcmc_results[0,1:]
            wl.meb_z2_pars = meb_mcmc_results[1,1:]
            # fill dom vectors 
            #lon_dom.append(wl.lon_dec)
            #lat_dom.append(wl.lat_dec)
            #z1_dom_mean.append(wl.meb_z1_pars[0]) # means of z1 
            points.append([wl.lon_dec,wl.lat_dec])
            values.append(wl.meb_z1_pars[0])
            #z2_dom_mean.append(wl.meb_z2_pars[0]) # means of z1 
            #z1_dom_std.append(wl.meb_z1_pars[1]) # stds of z2 
            #z2_dom_std.append(wl.meb_z2_pars[1]) # stds of z2 

    f = plt.figure(figsize=[9.5,7.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    if path_base_image:
        img=mpimg.imread(path_base_image)
    else:
        raise 'no path base image'
    if ext_img:
        ext = ext_img
    else:
        raise 'no external (bound) values for image image'
    ax.imshow(img, extent = ext)
    if xlim is None:
        ax.set_xlim(ext[:2])
    else: 
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim(ext[-2:])
    else: 
        ax.set_ylim(ylim)
    
    points = np.asarray(points)
    tri = Delaunay(points)
    #print(tri.simplices)
    #print(points[tri.simplices])
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy(), linewidth=.8)
    plt.plot(points[:,0], points[:,1], 'o', label = 'MeB well', ms = 2)
    plt.plot(lon_stas, lat_stas, '*', label = 'MT sta.', ms = 2)

    ax.legend(loc=1, prop={'size': 6})	
    ax.set_xlabel('latitud [°]', size = textsize)
    ax.set_ylabel('longitud [°]', size = textsize)
    ax.set_title('Triangulation of MeB wells', size = textsize)

    # save figure
    if file_name is None:
        file_name = 'Trig_meb_wells_WRKNW5'
    if format is None: 
        format = 'png'

    # save figure
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    shutil.move(file_name+'.png', '.'+os.sep+'base_map_img'+os.sep+file_name+'.png')
    plt.close(f)
    plt.clf()

def map_stations_wells(station_objects, wells_objects, file_name = None, format = None, \
    path_base_image = None, alpha_img = None, ext_img = None, xlim = None, ylim = None, dash_arrow = None):
    
    if file_name is None:
        f_name = 'map_stations_wells'
    if format is None:
        format = 'png'
    if path_base_image is None:
        path_base_image = '.'
    if ext_img is None: 
        raise 'ext_img not given'
    else:
        ext = ext_img
    if alpha_img is None:
        alpha_img = 1.0
    else:
        alpha_img = alpha_img

    # sort stations and wells by longitud (for plotting)
    station_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    wells_objects.sort(key=lambda x: x.lat_dec, reverse=True)

    lon_stas = []
    lat_stas = []
    for sta in station_objects:
        lon_stas.append(sta.lon_dec)
        lat_stas.append(sta.lat_dec)
    lon_stas = np.asarray(lon_stas)
    lat_stas = np.asarray(lat_stas)

    lon_wls = []
    lat_wls = []
    for wl in wells_objects:
        if wl.meb:
            lon_wls.append(wl.lon_dec)
            lat_wls.append(wl.lat_dec)
    lon_wls = np.asarray(lon_wls)
    lat_wls = np.asarray(lat_wls)

    f = plt.figure(figsize=[12.5,10.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    img=mpimg.imread(path_base_image)
    ax.imshow(img, extent = ext, alpha = alpha_img)

    # stations
    plt.plot(lon_stas, lat_stas, 'b*', label = 'MT station', ms = 9, zorder=2, markeredgecolor= 'w')
    for i, sta in enumerate(station_objects):
        #plt.text(sta.lon_dec, sta.lat_dec, str(i), color = 'w', fontsize=textsize, ha = 'center', va = 'center',zorder=3)
        ax.annotate(sta.name[:-4], xy=(sta.lon_dec, sta.lat_dec),  xycoords='data',
            xytext=(1.1, 1.0 - i/12), textcoords='axes fraction',
            size=textsize,arrowprops=dict(arrowstyle = '-', facecolor='b', edgecolor = 'b'),
            horizontalalignment='left', verticalalignment='top',
            bbox=dict(boxstyle="Round", fc="w"), zorder=1)
    
    # wells
    plt.plot(lon_wls, lat_wls, 'r*', label = 'MeB well', ms = 9, zorder=1, markeredgecolor= 'w')
    import string
    alpha = list(string.ascii_lowercase)
    for i, wl in enumerate(wells_objects):
        #plt.text(wl.lon_dec, wl.lat_dec, alpha[i], color = 'k', fontsize=textsize, ha = 'center', va = 'center')
        ax.annotate(wl.name, xy=(wl.lon_dec, wl.lat_dec),  xycoords='data', #alpha[i]
            xytext=(0.25, 0.6 - i/10), textcoords='axes fraction',
            size=textsize,arrowprops=dict(arrowstyle = '-', facecolor='r', edgecolor = 'r'),
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle="Round", fc="w"), zorder=1)
        if dash_arrow:
            ax.annotate(wl.name, xy=(wl.lon_dec, wl.lat_dec),  xycoords='data', #alpha[i]
                xytext=(0.25, 0.6 - i/10), textcoords='axes fraction',
                size=textsize,arrowprops=dict(arrowstyle = '-', ls= 'dashed'),
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle="Round", fc="w"), zorder=1)
    
    
    ax.legend(loc=1, prop={'size': textsize})	
    ax.set_xlabel('latitude [°]', size = textsize)
    ax.set_ylabel('longitude [°]', size = textsize)
    ax.tick_params(labelsize=textsize)
    #ax.set_title('MT', size = textsize)
    if xlim is None: 
        ax.set_xlim(ext[:2])
    else:
        ax.set_xlim(xlim)
    if ylim is None: 
        ax.set_ylim(ext[-2:])
    else:
        ax.set_ylim(ylim)

    #plt.show()
    # save figure
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    shutil.move(file_name+'.png', '.'+os.sep+'base_map_img'+os.sep+file_name+'.png')

    plt.clf()

def plot_surface_cc_count(station_objects, wells_objects, bound2plot = None, type_plot = None, file_name = None, \
    format = 'png', path_base_image = None, alpha_img = None, ext_img = None, xlim = None, ylim = None, \
        hist_pars = None, path_plots = None): 

    if type_plot is None:
        type_plot = 'scatter'
    else:
        type_plot = type_plot
    if file_name is None:
        f_name = 'map_stations_wells'
    if format is None:
        format = 'png'
    if path_base_image is None:
        path_base_image = '.'
    if ext_img is None: 
        pass
    else:
        ext = ext_img
        if alpha_img is None:
            alpha_img = 1.0
        else:
            alpha_img = alpha_img
    if hist_pars is None:
        hist_pars = False
    else: 
        hist_pars = hist_pars
    if path_plots is None: 
        path_plots = '.'+os.sep+'plain_view_plots'
    else:
        path_plots = path_plots
    # sort stations and wells by longitud (for plotting)
    station_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    wells_objects.sort(key=lambda x: x.lat_dec, reverse=True)
    ## import result for stations of z1 and z2 and create lists to plot
    # values = []
    lon_stas = []
    lat_stas = []
    z1_list_mean = []
    z2_list_mean = []
    r1_list_mean = []
    r2_list_mean = []
    r3_list_mean = []
    z1_list_std = []
    z2_list_std = []
    r1_list_std = []
    r2_list_std = []
    r3_list_std = []
    for sta in station_objects:
        # load well pars from meb inversion 
        mcmc_inv_results = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+"est_par.dat")
        sta.z1_pars = mcmc_inv_results[0,1:]
        sta.z2_pars = mcmc_inv_results[1,1:]
        sta.r1_pars = mcmc_inv_results[2,1:]
        sta.r2_pars = mcmc_inv_results[3,1:]
        sta.r3_pars = mcmc_inv_results[4,1:]
        # fill lists
        lon_stas.append(sta.lon_dec)
        lat_stas.append(sta.lat_dec)
        z1_list_mean.append(sta.z1_pars[0])
        z2_list_mean.append(sta.z2_pars[0]+sta.z1_pars[0])
        r1_list_mean.append(sta.r1_pars[0])
        r2_list_mean.append(sta.r2_pars[0])
        r3_list_mean.append(sta.r3_pars[0])
        z1_list_std.append(sta.z1_pars[1])
        z2_list_std.append(sta.z2_pars[1])
        r1_list_std.append(sta.r1_pars[1])
        r2_list_std.append(sta.r2_pars[1])
        r2_list_std.append(sta.r3_pars[1])    
    #################
    # griddata to plot as surface
    if False:
        points = [[i,j] for i,j in zip(lon_stas,lat_stas)]
        points = np.asarray(points)
        values = np.asarray(data)

        grid_x = np.linspace(min(lon_stas), max(lon_stas), 100)
        grid_y = np.linspace(min(lat_stas), max(lat_stas), 100)
        grid_xs, grid_ys = np.meshgrid(grid_x, grid_y)
        xi = [[i,j] for i,j in zip(grid_x,grid_y)]
        xi = np.asarray(xi)
        grid_z0 = griddata(points, values, xi, method='linear')

        xs, ys = np.meshgrid(grid_x, grid_y)
        data = grid_z0
        #mesh data to plot as lines
        dataMesh = np.empty_like(xs)
        for i, j, d in zip(grid_x, grid_y, data):
            dataMesh[np.where(grid_x==i), np.where(grid_y==j)] = d
        #fig, ax = plt.subplots()
        #CS = ax.contourf(xs, ys, dataMesh)# 5, alpha = .5, cmap='plasma')

        ax = plt.axes(projection='3d')
        ax.plot_surface(xs, ys, dataMesh, rstride=1, cstride=1,
        cmap='viridis', edgecolor='none')
        plt.show()
    #################
    # plot contours
    if False: 
        # mesh list of coord. and data
        xs, ys = np.meshgrid(lon_stas, lat_stas)
        data = z2_list_mean
        data_std = z2_list_std
        # mesh data to plot as lines
        dataMesh = np.empty_like(xs)
        for i, j, d in zip(lon_stas, lat_stas, data):
            dataMesh[lon_stas.index(i), lat_stas.index(j)] = d
        
        fig, ax = plt.subplots()
        levels = np.arange(min(data), max(data), 5)
        #levels = [50,200]#,200,400]
        #CS = ax.contour(xs, ys, dataMesh, alpha = .5)
        CS = ax.contour(xs, ys, dataMesh, 5, alpha = .5, cmap='plasma')
        #plt.clabel(CS, inline=True, fontsize=8)
        ax.plot(lon_stas,lat_stas,'*')
        #ax.clabel(CS, inline=1, fontsize=10)
    #######################
    # scatter plot with size of depending on depth
    if type_plot == 'scatter': 
        # data to plot 
        if bound2plot == 'top':
            data = z1_list_mean
            data_std = z1_list_std
        if bound2plot == 'bottom':
            data = z2_list_mean
            data_std = z2_list_std
        # figure
        fig, ax = plt.subplots(figsize=(15,12))
        # plot base image (if given)
        if ext_img:
            img=mpimg.imread(path_base_image)
            ax.imshow(img, extent = ext, alpha = alpha_img)    
        alphas = np.zeros(len(lon_stas))
        #mx_data_std, mn_data_std = max(data_std), min(data_std) 
        for i in range(len(alphas)):
            alphas[i] = 1. - data_std[i] / max(data_std)

        rgba_colors = np.zeros((len(lon_stas),4))
        # for red the first column needs to be one
        rgba_colors[:,0] = .5
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas
        # size 
        size = abs(max(data) - data)/1
        if bound2plot == 'top': 
            scatter = ax.scatter(lon_stas,lat_stas, s = size, color = rgba_colors)#alpha = 0.5)
        if bound2plot == 'bottom': 
            scatter = ax.scatter(lon_stas,lat_stas, s = size/3, color = rgba_colors)#alpha = 0.5)
        # absence of CC
        for sta in station_objects:
            if sta.z2_pars[0] < 50.:
                plt.plot(sta.lon_dec, sta.lat_dec,'w.', markersize=28) 
                plt.plot(sta.lon_dec, sta.lat_dec,'bx', markersize=12) 
        # Legend 
        if bound2plot == 'top':
            rgba_color = [.5,0,0,1.]
            l1 = plt.scatter([],[], s=400,  color = rgba_color, edgecolors='none')
            l2 = plt.scatter([],[], s=200, color = rgba_color, edgecolors='none')
            l3 = plt.scatter([],[], s=100, color = rgba_color, edgecolors='none')
            l4 = plt.scatter([],[], s=50, color = rgba_color, edgecolors='none')

            labels = [str(int(abs(max(data)- 400)))+' meters', str(int(abs(max(data)- 200)))+' meters',\
                str(int(abs(max(data)- 100)))+' meters', str(int(abs(max(data)- 50)))+' meters']
        
        if bound2plot == 'bottom':
            l1 = plt.scatter([],[], s=800,  color = rgba_colors[0], edgecolors='none')
            l2 = plt.scatter([],[], s=400, color = rgba_colors[0], edgecolors='none')
            l3 = plt.scatter([],[], s=200, color = rgba_colors[0], edgecolors='none')
            l4 = plt.scatter([],[], s=100, color = rgba_colors[0], edgecolors='none')

            labels = [str(int(abs(max(data)-800)))+' meters', str(int(abs(max(data)- 400)))+' meters',\
                str(int(abs(max(data)- 200)))+' meters', str(int(abs(max(data)- 100)))+' meters']

        #leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=textsize,
        #    handlelength=2, loc = 8, borderpad = 1.8, handletextpad=1, \
        #        scatterpoints = 1)#title='Depth to '+bound2plot+' boundary'
        leg = plt.legend([l1, l2, l3, l4], labels, frameon=True, fontsize=textsize,
            handlelength=2, loc = 3, borderpad = 1.8, handletextpad=1, \
                scatterpoints = 1, title='Depth to '+bound2plot+' boundary', \
                    title_fontsize = textsize)
        #plt.setp(legend.get_title(),fontsize=textsize)     
        # limits
        if xlim is None:
            ax.set_xlim(ext[:2])
        else: 
            ax.set_xlim(xlim)
        if ylim is None:
            ax.set_ylim(ext[-2:])
        else: 
            ax.set_ylim(ylim)
        # labels
        #ax.legend(loc=1, prop={'size': 6})	
        ax.set_xlabel('Latitude [°]', size = textsize)
        ax.set_ylabel('Longitude [°]', size = textsize)
        ax.set_title('Depth to inferred clay cap '+bound2plot+' boundary', size = textsize)
        
        ## legend sizes (depth)
        #handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
        #legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
        ## legend uncertainty (color)
        #legend1 = ax.legend(*scatter.legend_elements(),
        #    loc="lower left", title="Uncertainty")
        #ax.add_artist(legend1)

        plt.tight_layout()
        #plt.show()
        #asdf
        # save figure
        if file_name is None:
            file_name = 'interface_LRA_'+bound2plot
        if format is None: 
            format = 'png'

        # save figure
        plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
        shutil.move(file_name+'.png', '.'+os.sep+path_plots+os.sep+file_name+'.png')
        plt.clf()
    #################
    # histograms of parameteres for all the stations considered
    if hist_pars:

        ## plot histograms of ocurrence of each paramater 
        f = plt.figure(figsize=(10, 11))
        #f.suptitle('Model: '+self.name, size = textsize)
        gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

        # plot histograms for pars
        ax1 = f.add_subplot(gs[0, 0]) # z1
        ax2 = f.add_subplot(gs[1, 0]) # z2
        ax3 = f.add_subplot(gs[0, 1]) # r1
        ax4 = f.add_subplot(gs[1, 1]) # r2
        ax5 = f.add_subplot(gs[2, 1]) # r3
        #f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
        #f.set_size_inches(12,2)
        #f.set_size_inches(10,2)  
        texts = textsize
        #f.suptitle(self.name, size = textsize)
        
        # z1
        z1 = z1_list_mean
        bins = np.linspace(np.min(z1), np.max(z1), int(np.sqrt(len(z1))))
        h,e = np.histogram(z1, bins, density = True)
        m = 0.5*(e[:-1]+e[1:])
        ax1.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
        ax1.set_xlabel('$z_1$ [m]', size = texts)
        ax1.set_ylabel('freq.', size = texts)
        #ax1.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(z1)
        y = mlab.normpdf(bins, mu, sigma)
        ax1.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        #ax1.plot(bins, y, 'r-', linewidth=1, label = 'normal fit') #$\mu$:{:3.1f},$\sigma$:{:2.1f}'.format(mu,sigma))
        ax1.plot([mu,mu],[0,max(y)],'r--', label = '$\mu$ of  $z_1$', linewidth=1.0)

        # z2
        z2 = z2_list_mean
        bins = np.linspace(np.min(z2), np.max(z2), int(np.sqrt(len(z2))))
        h,e = np.histogram(z2, bins, density = True)
        m = 0.5*(e[:-1]+e[1:])
        ax2.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
        ax2.set_xlabel('$z_2$ [m]', size = texts)
        ax2.set_ylabel('freq.', size = texts)
        #ax1.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu2, sigma) = norm.fit(z2)
        y = mlab.normpdf(bins, mu2, sigma)
        ax2.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu2,sigma), fontsize = textsize, color='gray')#, y=0.8)
        #ax2.plot(bins, y, 'r-', linewidth=1, label = 'normal fit')
        ax2.plot([mu2,mu2],[0,max(y)],'b--', label = '$\mu$ of  $z_2$', linewidth=1.0)
        # axis for main plot 

        # r1
        r1 = r1_list_mean
        bins = np.linspace(np.min(r1), np.max(r1), int(np.sqrt(len(r1))))
        h,e = np.histogram(r1, bins, density = True)
        m = 0.5*(e[:-1]+e[1:])
        ax3.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
        ax3.set_xlabel(r'$\rho_1$ [$\Omega$ m]', size = texts)
        ax3.set_ylabel('freq.', size = texts)
        #ax1.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(r1)
        y = mlab.normpdf(bins, mu, sigma)
        ax3.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        #ax3.plot(bins, y, 'r-', linewidth=1)
        ax3.plot([mu,mu],[0,max(y)],'g--', label = r'$\mu$ of  $\rho_1$', linewidth=1.0)

        # r2
        r2 = r2_list_mean
        bins = np.linspace(np.min(r2), np.max(r2), int(np.sqrt(len(r2))))
        h,e = np.histogram(r2, bins, density = True)
        m = 0.5*(e[:-1]+e[1:])
        ax4.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
        ax4.set_xlabel(r'$\rho_2$ [$\Omega$ m]', size = texts)
        ax4.set_ylabel('freq.', size = texts)
        #ax1.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(r2)
        y = mlab.normpdf(bins, mu, sigma)
        ax4.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        #ax4.plot(bins, y, 'r-', linewidth=1)
        ax4.plot([mu,mu],[0,max(y)],'m--', label = r'$\mu$ of  $\rho_2$', linewidth=1.0)

        # r3
        r3 = r3_list_mean
        bins = np.linspace(np.min(r3), np.max(r3), int(np.sqrt(len(r3))))
        h,e = np.histogram(r3, bins, density = True)
        m = 0.5*(e[:-1]+e[1:])
        ax5.bar(e[:-1], h, e[1]-e[0], alpha = 0.5)
        ax5.set_xlabel(r'$\rho_3$ [$\Omega$ m]', size = texts)
        ax5.set_ylabel('freq.', size = texts)
        #ax1.grid(True, which='both', linewidth=0.1)
        # plot normal fit 
        (mu, sigma) = norm.fit(r3)
        y = mlab.normpdf(bins, mu, sigma)
        ax5.set_title('$\mu$:{:3.1f}, $\sigma$: {:2.1f}'.format(mu,sigma), fontsize = textsize, color='gray')#, y=0.8)
        #ax5.plot(bins, y, 'r-', linewidth=1)
        ax5.plot([mu,mu],[0,max(y)],'y--', label = r'$\mu$ of  $\rho_3$', linewidth=1.5)

        ### layout figure
        ax.tick_params(labelsize=textsize)
        ax1.tick_params(labelsize=textsize)
        ax2.tick_params(labelsize=textsize)
        ax3.tick_params(labelsize=textsize)
        ax4.tick_params(labelsize=textsize)
        ax5.tick_params(labelsize=textsize)

        #ax1.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
        #ax2.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
        #ax3.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
        #ax4.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
        #ax5.legend(fontsize=textsize, fancybox=True, framealpha=0.5)
        plt.tight_layout()

        # legend subplot 
        ax6 = f.add_subplot(gs[2, 0])
        #ax6.plot([],[],'c-', label = 'model samples')
        ax6.plot([],[],'r--', label = '$\mu$ of  $z_1$')
        ax6.plot([],[],'b--', label =  '$\mu$ of  $z_2$')
        ax6.plot([],[],'g--', label =  r'$\mu$ of  $\rho_1$')
        ax6.plot([],[],'m--', label =  r'$\mu$ of  $\rho_2$')
        ax6.plot([],[],'y--', label =  r'$\mu$ of  $\rho_3$')
        ax6.plot([],[],'r-', label = 'normal fit    ')
        ax6.axis('off')
        ax6.legend(loc = 'upper right', fontsize=textsize, fancybox=True, framealpha=1.)

        if format is None: 
            format = 'png'

        # save figure
        file_name = 'hist_parameters'
        plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
        shutil.move(file_name+'.png', '.'+os.sep+path_plots+os.sep+file_name+'.png')
        plt.clf()