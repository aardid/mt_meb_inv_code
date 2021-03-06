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
import os, json
import shutil
from lib_sample_data import*
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.cm as cm
from scipy.spatial import Delaunay
import functools
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, LinearNDInterpolator, interp2d
from matplotlib import cm
import matplotlib.image as mpimg
from scipy.stats import norm
import matplotlib.mlab as mlab
import pyproj
from misc_functios import *
import matplotlib.patheffects as PathEffects

_projections = {}
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
        R = 6373. # around 39?? latitud
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

# for converting lat-lon to UTM

def zone(coordinates):
    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32
    if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35
        return 37
    return int((coordinates[0] + 180) / 6) + 1

def letter(coordinates):
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]

def project(coordinates):
    z = zone(coordinates)
    l = letter(coordinates)
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    x, y = _projections[z](coordinates[0], coordinates[1])
    if y < 0:
        y += 10000000
    return z, l, x, y

def unproject(z, l, x, y):
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    if l < 'N':
        y -= 10000000
    lng, lat = _projections[z](x, y, inverse=True)
    return (lng, lat)

def utmToLatLng(zone, easting, northing, northernHemisphere=True):
    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996

    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))

    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0

    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)

    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0

    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi

    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi

    if not northernHemisphere:
        latitude = -latitude

    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return (latitude, longitude)

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

    
def plot_2D_uncert_bound_cc_mult_env(sta_objects, type_coord = None, unit_dist = None, pref_orient = None, sort = None, \
    file_name = None, format_fig = None, width_ref = None, prior_meb = None, wells_objects = None, plot_some_wells = None, xlim = None, ylim = None, \
    center_cero = None, km = None, export_fig = None, no_plot_clay_infered = None, mask_no_cc = None, \
    rest_bound_ref = None, plot_litho_wells = None, temp_count_wells = None, temp_iso = None,\
        position_label = None, title = None): 
    """
    width_ref: width of percetil of reference as dotted line centered at 50%. ex: '60%'
    prior_meb: full list of wells objects
    plot_some_wells: list of names of wells with MeB data to be plotted in the profile. ex: ['TH13','TH19']
    type_coord: 'decimal' for lat/lon in decimal or 'linear' for x,y cartesian coord. 
    unit_dist: just for linear. 'km' for kilometers and 'm' for meters. If 'm', x_axis is converted to 'km' for plotting.
    mask_no_cc: mask section between two stations where no consuctive layer is found. The value of mask_no_cc indicates the criteria:
        if the thickness of the second layer in meters is less than 'mask_no_cc', consider it as no conductive layer. 
    """
    textsize = 10
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
        width_plot = False
    if [width_ref != '30%' or width_ref != '60%' or width_ref != '90%']:
        assert 'invalid width_ref: 30%, 60%, 90%'
    if prior_meb is None:
        prior_meb = False
    #else: 
    #    wells_objects = prior_meb
    if plot_some_wells is None: 
        plot_some_wells = False
    if plot_litho_wells is None:
        plot_litho_wells = []

    if pref_orient: 
        pref_orient = pref_orient
    else:    
        pref_orient = 'EW'

    ## sta_objects: list of station objects
    ## sort list by longitud (min to max - East to West)
    if pref_orient == 'EW':
        sta_objects.sort(key=lambda x: x.lon_dec, reverse=False)
    if pref_orient == 'NS':
        sta_objects.sort(key=lambda x: x.lat_dec, reverse=True)
    #
    if temp_iso is None:
        temp_iso = [200.]
    # vectors to be fill and plot 
    x_axis = np.zeros(len(sta_objects))
    topo = np.zeros(len(sta_objects))
    z1_med = np.zeros(len(sta_objects))
    z2_med = np.zeros(len(sta_objects))
    # percetil matrix
    s = (len(sta_objects),len(sta_objects[0].z1_pars[3])) # n?? of stations x n?? of percentils to plot
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
            #if abs(sta.z2_pars[0]) < mask_no_cc: 
            #    stns_negli[i] = True
            d2 = sta.z2_pars[0]+sta.z1_pars[0]
            if sta.z2_pars[0] < d2/5.:
                stns_negli[i] = True
            else:
                stns_negli[i] = False
                # criteria for being negligible
                #if abs((sta.z1_pars[0]+sta.z1_pars[1]) - (sta.z1_pars[0]+sta.z2_pars[0]-sta.z2_pars[1])) < mask_no_cc/2: 
                #    stns_negli[i] = True
                #else:
                #    stns_negli[i] = False
            if sta.name[:-4] in ['WT192a','WT306a']:
                stns_negli[i] = True
        i+=1

    if center_cero: 
        mid_val = (abs(x_axis[0] - x_axis[-1]))/2
        x_axis = x_axis - mid_val

    if unit_dist is 'm': # transfor data to km
        x_axis = x_axis/1e3
    
    ## plot envelopes 5% and 95% for cc boundaries
    f = plt.figure(figsize=[18.5,9.5])
    ax = plt.axes([0.18,0.25,0.70,0.50])
    
    if prior_meb:
        f = plt.figure(figsize=[11.5,9.5])
        ax = plt.axes([0.18,0.25,0.70,0.50]) 
    if plot_litho_wells:
        f = plt.figure(figsize=[18.5,9.5])
        #f = plt.figure(figsize=[8.5,5.5])
        ax = plt.axes([0.1,0.1,0.75,0.75]) # fig.add_axes((left, bottom, width, height))

    # plot meadian and topo
    ax.plot(x_axis, topo,'g-', label='Topography')
    if not plot_litho_wells:
        ax.plot(x_axis, z1_med,'r.-', label='$z_1$ estimated MT', zorder = 1)
        ax.plot(x_axis, z2_med,'b.-', label='$z_2$ estimated MT', zorder = 1)
    else:
        ax.fill_between([], [],color = u'#ff7f0e', label='MT top of the conductor', zorder = 1)
        ax.fill_between([], [],color = u'#1f77b4', label='MT bottom of the conductor', zorder = 1)        
    
    if no_plot_clay_infered:
        pass
    else: # plot orange section between means of z1 and z2 (indicating clay cap)
        if not plot_litho_wells:
            ax.fill_between(x_axis, z2_med, z1_med,  alpha=.1, facecolor='orange', edgecolor='orange', label='Inferred clay')
        else:
            ax.fill_between(x_axis, z2_med, z1_med,  alpha=.1, facecolor='orange', edgecolor='orange')#, label='Inferred clay')

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
        if not plot_litho_wells:    
            ax.fill_between(x_axis, z1_sup, z1_inf,  alpha=.05*(j+1), facecolor='r', edgecolor='r', zorder = 2)
            ax.fill_between(x_axis, z2_sup, z2_inf,  alpha=.05*(j+1), facecolor='b', edgecolor='b', zorder = 2)
        else:
            ax.fill_between(x_axis, z1_sup, z1_inf,  alpha=.05*(j+1), facecolor=u'#ff7f0e', edgecolor=u'#ff7f0e', zorder = 2)
            ax.fill_between(x_axis, z2_sup, z2_inf,  alpha=.05*(j+1), facecolor=u'#1f77b4', edgecolor=u'#1f77b4', zorder = 2)


    if width_plot: 
        ## plot 5% and 95% percetils as dotted lines 
        if width_ref == '90%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,0],'r--',linewidth=.5, alpha=.5)
           # ax.text(x_axis[0]-.4,z1_per[0,0]+25,'5%',size = textsize, color = 'r')
            #ax.plot(x_axis, z1_per[:,-1],'r--',linewidth=.5, alpha=.5)
            ax.text(x_axis[0]-.4,z1_per[0,-1]-25,'95%',size = textsize,color = 'r')
            # bottom boundary
            #ax.plot(x_axis, z2_per[:,0],'b--',linewidth=.5, alpha=.5)
            ax.text(x_axis[-1]+.1,z2_per[-1,0]+0,'5%',size = textsize, color = 'b')
            #ax.plot(x_axis, z2_per[:,-1],'b--',linewidth=.5, alpha=.5)
            ax.text(x_axis[-1]+.1,z2_per[-1,-1]-50,'95%',size = textsize,color = 'b')

        ## plot 20% and 80% percetils as dotted lines
        if width_ref == '60%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,3],'r--',linewidth=.5, alpha=.2)
            ax.text(x_axis[0]-.8,z1_per[0,3]+25,'20%',size = textsize, color = 'r')
            ax.plot(x_axis, z1_per[:,-4],'r--',linewidth=.5, alpha=.2)
            ax.text(x_axis[0]-.8,z1_per[0,-4]-25,'80%',size = textsize,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,3],'b--',linewidth=.5, alpha=.2)
            ax.text(x_axis[-1],z2_per[-1,3],'20%',size = textsize, color = 'b')
            ax.plot(x_axis, z2_per[:,-4],'b--',linewidth=.5, alpha=.2)
            ax.text(x_axis[-1],z2_per[-1,-4],'80%',size = textsize,color = 'b')

        ## plot 45% and 65% percetils as dotted lines
        if width_ref == '30%': 
            # top boundary
            ax.plot(x_axis, z1_per[:,8],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,8]+10,'45%',size = textsize, color = 'r')
            ax.plot(x_axis, z1_per[:,12],'r--',linewidth=.5)
            ax.text(x_axis[0],z1_per[0,12]+10,'65%',size = textsize,color = 'r')
            # bottom boundary
            ax.plot(x_axis, z2_per[:,8],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,8],'20%',size =textsize, color = 'b')
            ax.plot(x_axis, z2_per[:,12],'b--',linewidth=.5)
            ax.text(x_axis[-1],z2_per[-1,12],'80%',size = textsize,color = 'b')

    # mask sections were second layer is negligible
    if mask_no_cc:
        i = 0
        for k in range(len(sta_objects)):
            if stns_negli[k]:# and stns_negli[k+1]: 
				## from the station to the previous
                if k>0:
                    x_mask = [x_axis[k-1], x_axis[k]]
                    #y_mask = [topo[k]-10, -2000.]
                    y_mask_top = [z1_per[k-1,0]+50,z1_per[k,0]+50]
                    y_mask_bottom = [z2_per[k-1,-1]-300,z2_per[k,-1]-300]
                    ax.fill_between(x_mask, y_mask_top, y_mask_bottom,  alpha=.98, facecolor='w', zorder=2)
                ## from the station to the next one
                if k<len(sta_objects)-1:
                    x_mask = [x_axis[k], x_axis[k+1]]
                    #y_mask = [topo[k]-10, -2000.]
                    y_mask_top = [z1_per[k,0]+50,z1_per[k+1,0]+50]
                    y_mask_bottom = [z2_per[k,-1]-100,z2_per[k+1,-1]-100]
                    ax.fill_between(x_mask, y_mask_top, y_mask_bottom,  alpha=.98, facecolor='w', zorder=2)
                if not prior_meb:
                    ax.text(x_axis[k], topo[k]-600., 'No conductor', rotation=90, size=textsize, bbox=dict(facecolor='white', edgecolor='white', alpha=1.0), zorder=3 , label = 'No conductor') 
                    ax.text(x_axis[k], topo[k]-600., 'No conductor', rotation=90, size=textsize, bbox=dict(facecolor='grey', edgecolor='grey', alpha=0.1), zorder=4 , label = 'No conductor') 
                else:
                    ax.text(x_axis[k], topo[k]-800., 'No conductor', rotation=90, size=textsize, bbox=dict(facecolor='white', edgecolor='white', alpha=1.0), zorder=3 , label = 'No conductor') 
                    ax.text(x_axis[k], topo[k]-800., 'No conductor', rotation=90, size=textsize, bbox=dict(facecolor='grey', edgecolor='grey', alpha=0.1), zorder=4 , label = 'No conductor') 
                ax.plot([x_axis[k],x_axis[k]], [topo[k]-100,topo[k]-3000],'w-',  linewidth=10, zorder=3, alpha=.9)

                i+=1
        #ax.text(x_axis[i-1], topo[i-1]-200., '  ', rotation=90, size=textsize-2, bbox=dict(facecolor='black', alpha=0.01), label = 'No conductor',  zorder=0 ) 
    # plot station names    
    i = 0
    for sta in sta_objects:
            ax.text(x_axis[i], topo[i]+100., sta.name[:-4], rotation=90, size=textsize, bbox=dict(facecolor='red', alpha=0.1), ) 
            i+=1  
    # plot prior meb 
    if prior_meb:
        wl_names = []
        wls_obj = []
        # collect nearest wells names employed in every station for MeB priors
        if plot_some_wells: # for well names given 
            wl_names = [wl for wl in wells_objects if wl.name in plot_some_wells]

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
                if wl.name == wln.name: 
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
            if plot_litho_wells is False: 
                if wl.name == 'WK267A': 
                    ax.text(x_axis_wl[i]-.2, topo_wl[i]-1.0e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
                elif wl.name == 'WK403':
                    ax.text(x_axis_wl[i]+.2, topo_wl[i]-(wl.meb_z2_pars[0]+2*wl.meb_z2_pars[1])-.9e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
                elif wl.name == 'WK318':
                    ax.text(x_axis_wl[i]+.2, topo_wl[i]-(wl.meb_z2_pars[0]+2*wl.meb_z2_pars[1])-.9e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
                else: 
                    ax.text(x_axis_wl[i], topo_wl[i]-(wl.meb_z2_pars[0]+2*wl.meb_z2_pars[1])-.9e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
            elif wl.name not in plot_litho_wells:
                if wl.name == 'WK318':
                    ax.text(x_axis_wl[i]-.2, topo_wl[i]-(wl.meb_z2_pars[0]+2*wl.meb_z2_pars[1])-.9e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
                else:
                    ax.text(x_axis_wl[i], topo_wl[i]-(wl.meb_z2_pars[0]+2*wl.meb_z2_pars[1])-.9e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
            else:
                pass
            # import and plot MeB mcmc result
            wl.read_meb_mcmc_results()
            ## vectors for plotting 
            top = wl.elev # elevation os the well
            x = x_axis_wl[i]
            # plot top bound. C
            y1 = top - wl.meb_z1_pars[0] # [z1_mean_prior, z1_std_prior]
            e1 = wl.meb_z1_pars[1] # [z1_mean_prior, z1_std_prior]
            ax.errorbar(x+.25, y1, e1, color='cyan',linestyle='-',zorder=3, marker='_')
            # plot bottom bound. CC
            y2 = top - wl.meb_z2_pars[0] # [z1_mean_prior, z1_std_prior]
            e2 = wl.meb_z2_pars[1] # [z1_mean_prior, z1_std_prior] # 2 time std (66%)
            ax.errorbar(x+.25, y2, e2, color='m', linestyle='-',zorder=2, marker='_')
            i+=1
        ax.errorbar(x+.25, y1, e1, color='cyan',linestyle='-',zorder=3, marker='_', label = '$z_1$ estimated MeB')
        ax.errorbar(x+.25, y2, e2, color='m', linestyle='-',zorder=3, marker='_', label = '$z_2$ estimated MeB')

    # plot resistivity boundary interception location
    if rest_bound_ref:
        # import one or two coord points
        lats_rb, lons_rb = np.genfromtxt(rest_bound_ref, skip_header=1, delimiter=',').T
        # resample topo
        N_rs = 100 # number of resample points data
        xj = np.linspace(x_axis[0],x_axis[-1],N_rs)	
        topo_rs = cubic_spline_interpolation(x_axis,topo,xj)
        for lon,lat in zip(lons_rb, lats_rb):
            coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
            coord2 = [lon,lat]
            ## calculate distances from first station to the others, save in array
            x_aux = dist_two_points(coord1, coord2, type_coord = type_coord)
            # find index of x_aux in topo resample
            arr, idx = find_nearest(xj, x_aux)
            #
            plt.plot([x_aux,x_aux],[topo_rs[idx] - 300.,topo_rs[idx]], linestyle = '--',color = 'orange' ,alpha = 0.7, linewidth = 3, zorder = 5)
            plt.plot([x_aux],[topo_rs[idx] - 300.], marker = '_',color = 'orange' ,alpha = 0.7, markersize = 7, zorder = 5)       
        plt.plot([],[], color = 'orange' ,linewidth = 2, linestyle = '--', alpha = 0.7,label = 'DC resistivity boundary', zorder = 0) 

    # plot litholology from wells in list 'plot_litho_wells'
    if plot_litho_wells: 
        wl_names = []
        wls_obj = []
        # collect nearest wells names employed in every station for MeB priors
        wl_names = [wl for wl in wells_objects if wl.name in plot_litho_wells]

        for wl in wells_objects: 
            for wln in wl_names: 
                if wl.name == wln.name: 
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

        # import lithology references: dictionary
        if True: # import formation, color, description -> create dictionary
            path = '.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+"formation_colors.txt"
            #depths_from, depths_to, lito  = np.genfromtxt(path, \
            #    delimiter=',').T
            form_abr = []
            form_col = []
            form_des = []

            with open(path) as p:
                next(p)
                for line in p:
                    line = line.strip('\n')
                    currentline = line.split(",")
                    form_abr.append(currentline[0])
                    form_col.append(currentline[1])
                    form_des.append(currentline[2])
            # dictionary. key: abreviation of formation. Values: (color, description)
            dict_form = dict([(form_abr[i],(form_col[i],form_des[i])) for i in range(len(form_des))])
            # for legend
            if True:
                # forms to not plot in legend
                not_leg_forms = ['KR2A_RHY', 'RHY_A','RHY_B','RHY_C','HFF','WRFM1','WRFX','WRF3_4']
                ax.fill_between([],[], color = 'w', alpha = 0.1, label = ' ', zorder = 0)
                ax.fill_between([],[], color = 'w', alpha = 0.1,  label = 'LITHOLOGY:', zorder = 0)
                for k in dict_form.keys():
                    if k in not_leg_forms:
                        pass
                    else:
                        # abreviation 
                        #ax.fill_between([],[], color = dict_form[k][0], alpha = 0.8, label = '   '+k, zorder = 0)
                        # description 
                        ax.fill_between([],[], color = dict_form[k][0], alpha = 0.8, label = ' '+dict_form[k][1], zorder = 0)

        i = 0
        for i, wl in enumerate(wls_obj):
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
            if True:#try:
                path = '.'+os.sep+'base_map_img'+os.sep+'wells_lithology'+os.sep+wl.name+os.sep+"lithology.txt"
                #depths_from, depths_to, lito  = np.genfromtxt(path, \
                #    delimiter=',').T
                depths_from = []
                depths_to = []
                lito = []
                with open(path) as p:
                    next(p)
                    for line in p:
                        line = line.strip('\n')
                        currentline = line.split(",")
                        depths_from.append(float(currentline[0]))
                        depths_to.append(float(currentline[1]))
                        lito.append(currentline[2])

                N = len(depths_to) # number of lithological layers
                colors = [dict_form[lito[k]][0] for k in range(len(lito))]
                #colors = ['g','r','orange','r','m','g','r','orange','r','m']
                #with open(r'data\nzafd.json', 'r') as fp:
                for j in range(N):
                    if wl.name == 'WK317':
                        ax.fill_between([x_axis_wl[i]+.0,x_axis_wl[i]+.4],[wl.elev -depths_from[j], wl.elev -depths_from[j]], [wl.elev -depths_to[j],wl.elev -depths_to[j]], \
                            color = colors[j], alpha = 0.8, zorder = 5)
                    elif wl.name == 'WK318':
                        ax.fill_between([x_axis_wl[i]+-.1,x_axis_wl[i]+.3],[wl.elev -depths_from[j], wl.elev -depths_from[j]], [wl.elev -depths_to[j],wl.elev -depths_to[j]], \
                            color = colors[j], alpha = 0.8, zorder = 5)
                        #ax.fill_between([x_axis_wl[i]-.3,x_axis_wl[i]-.1],[wl.elev -depths_from[j], wl.elev -depths_from[j]], [wl.elev -depths_to[j],wl.elev -depths_to[j]], \
                        #    color = colors[j], alpha = 0.8, zorder = 5)
                    else:
                        ax.fill_between([x_axis_wl[i]-.2,x_axis_wl[i]+.2],[wl.elev -depths_from[j], wl.elev -depths_from[j]], [wl.elev -depths_to[j],wl.elev -depths_to[j]], \
                            color = colors[j], alpha = 0.8, zorder = 5)
                    #thick = depths_to[j] - depths_from[j]
                    #if thick > 25:
                    #    ax.text(x_axis_wl[i], wl.elev - (depths_from[j] - thick/2), lito[j], fontsize=6,\
                    #        horizontalalignment='center', verticalalignment='center', zorder = 5)
            #except:
            #    pass
            if wl.name == 'WK124A':
                ax.text(x_axis_wl[i]-.25, topo_wl[i]-depths_to[-1]-.6e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
            elif wl.name == 'WK317':
                ax.text(x_axis_wl[i]+.20, topo_wl[i]-depths_to[-1]-.6e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
            elif wl.name == 'WK318':
                ax.text(x_axis_wl[i]+.10, topo_wl[i]-depths_to[-1]-.6e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 
            else:
                ax.text(x_axis_wl[i], topo_wl[i]-depths_to[-1]-.6e3, wl.name+': '+str(round(dist_wl_prof,1))+' km', rotation=90, size=textsize, bbox=dict(facecolor='blue', alpha=0.1)) 

    if temp_count_wells:
        wl_names = []
        wls_obj = []
        # collect nearest wells names employed in every station for MeB priors
        wl_names = [wl for wl in wells_objects if wl.name in temp_count_wells]

        for wl in wells_objects: 
            for wln in wl_names: 
                if wl.name == wln.name: 
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

        # vectors for cubic spline interpolation. cubic_spline_interpolation(xi,yi,xj, rev = None): 
        xi = x_axis_wl
        xi = []
        #xj = np.linspace(xi[0],xi[-1],100)
        for iso in temp_iso:# loop over isotherms
            yi = [] # vector to be filled with depths od fix temps.
            xi = [] # vector to be filled with x_axis_wl locations
            for i, wl in enumerate(wls_obj): # loop over well for CSI
                val, idx = find_nearest(wl.temp_prof_rs, iso)
                if abs(val - iso) < 10.:
                    yi.append(wl.red_depth_rs[idx])
                    xi.append(x_axis_wl[i])
                    #ax.text(xi[-1],yi[-1],wl.name)
            xj = np.linspace(xi[0],xi[-1],100)
            yj = cubic_spline_interpolation(xi,yi,xj, rev = None)
            # plot
            ax.plot(xj,yj, linestyle = '-', c = 'grey', alpha = .3)
            #ax.plot(xi,yi, marker = '*', c = 'orange', alpha = .5)
            if position_label is 'mid':
                txt = ax.text(xi[int(len(xi)/2)-1] - .1, yi[int(len(xi)/2)-1] - 50., str(int(iso))+'??C', color='grey', size=textsize, alpha = .3, zorder = 7)
            else:
                txt = ax.text(xi[0]-.6, yi[0], str(int(iso))+'??C', color='grey', size=textsize, alpha = .3, zorder = 7)
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
    
    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    else:
        ax.set_xlim([x_axis[0]-1, x_axis[-1]+1])
        if prior_meb:
            ax.set_xlim([x_axis[0]-2, x_axis[-1]+1])
        if plot_litho_wells: 
            ax.set_xlim([x_axis[0]-8.0, x_axis[-1]+1])
    # ylim
    if prior_meb:
        if ylim:
            ax.set_ylim([ylim[0], ylim[1]])
        else:
            ax.set_ylim([-2.0e3, max(topo)+700.])
            if plot_litho_wells: 
                ax.set_ylim([-1.7e3, max(topo)+400.])
    else:
        if ylim:
            ax.set_ylim([ylim[0], ylim[1]])
        else:
            ax.set_ylim([-1.0e3, max(topo)+600.])
            if plot_litho_wells: 
                ax.set_ylim([-1.7e3, max(topo)+400.])
    
    #plt.xticks(np.linspace(0,10,10))
    ax.set_xlabel('y [km]', size = textsize)
    ax.set_ylabel('z [m]', size = textsize)
    if title:
        ax.set_title(title, size = textsize)

    #ax.grid(True)
    #(color='r', linestyle='-', linewidth=2)
    ax.grid(color='c', linestyle='-', linewidth=.1, zorder=4)
    ax.tick_params(labelsize=textsize)

    #plt.grid(True)
    if export_fig:
        return f, ax

    if plot_litho_wells:
        ax.legend(loc=6, prop={'size': textsize})
    else:
        ax.legend(loc=3, prop={'size': textsize})

    #ax.set_ylim([-1.0e3, max(topo)+400.])
    #ax.legend('off')

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
    #f = plt.figure(figsize=[9.5,7.5])
    f = plt.figure(figsize=[10.5,5.0])
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
                    ax.fill_between(xj, yj_up, yj_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i], label = isotherms[i]+'???')
                else:
                    ax.fill_between(xj, yj_up, yj_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i])
            else:
                if j == (int(n/2)):
                    ax.fill_between(x_axis, env_up, env_low,  alpha=.05*(j+1), facecolor=iso_col[i], edgecolor=iso_col[i], label = isotherms[i]+'???')
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

def grid_MT_inv_rest(station_objects, coords, n_points = None,  slp = None, file_name = None, plot = None, 
        path_output = None, path_base_image = None, ext_img = None, xlim = None, 
            ylim = None, just_plot = None, masl = None):
    """
    fn for griding and calculate meb prior in grid (define by coords) points 
    output: file_name.txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
    """   
    if file_name is None:
        file_name = 'grid_MT_inv'
    if slp is None: 
        slp = 4*10.
    if n_points is None:
        n_points = 100
    if path_output is None: 
        path_output = '.'
    
    # vectors for coordinates    
    x = np.linspace(coords[0], coords[1], n_points) # long
    y = np.linspace(coords[2], coords[3], n_points) # lat
    X, Y = np.meshgrid(x, y)
    Z_z1_mean = X*0.
    Z_z1_std = X*0
    Z_z2_mean = X*0
    Z_z2_std = X*0
    Z_z1_plus_z2_mean = X*0
    Z_z1_plus_z2_std = X*0
    # calculate MeB prior at each position 
    f = open(file_name+'.txt', "w")
    f.write("# lat\tlon\tmean_z1\tstd_z1\tmean_z2\tstd_z2\n")
    for j,lat in enumerate(y):
        for i,lon in enumerate(x):
            if True: # use every stations available
                # save names of nearest wells to be used for prior
                near_stas = [sta for sta in station_objects] #list of objects (wells)
                near_stas = list(filter(None, near_stas))
                dist_stas = [dist_two_points([sta.lon_dec, sta.lat_dec], [lon, lat], type_coord = 'decimal')\
                    for sta in station_objects]
                dist_stas = list(filter(None, dist_stas))
                # Calculate prior values for boundaries of the cc in station
                # prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
                # z1
                z1_mean_MT = np.zeros(len(near_stas))
                z1_std_MT = np.zeros(len(near_stas))
                z2_mean_MT = np.zeros(len(near_stas))
                z2_std_MT = np.zeros(len(near_stas))
                #
                z1_std_MT_incre = np.zeros(len(near_stas))
                z2_std_MT_incre = np.zeros(len(near_stas))
                count = 0
                # extract meb mcmc results from nearest wells 
                for sta in near_stas:
                    # extract meb mcmc results from file 
                    mt_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+"est_par.dat")
                    # values for mean a std for normal distribution representing the prior
                    z1_mean_MT[count] = mt_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
                    z1_std_MT[count] =  mt_mcmc_results[0,2] # std z1
                    z2_mean_MT[count] = mt_mcmc_results[1,1] # mean [1] z2 # median [3] z1
                    z2_std_MT[count] =  mt_mcmc_results[1,2] # std z2
                    # calc. increment in std. in the position of the station
                    # std. dev. increases as get farder from the well. It double its values per 2 km.
                    z1_std_MT_incre[count] = z1_std_MT[count]  + (dist_stas[count] *slp)
                    z2_std_MT_incre[count] = z2_std_MT[count]  + (dist_stas[count] *slp)
                    # load pars in well 
                    count+=1
                # calculete z1 normal prior parameters
                dist_weigth = [1./d for d in dist_stas]
                z1_mean = np.dot(z1_mean_MT,dist_weigth)/np.sum(dist_weigth)
                # std. dev. increases as get farder from the well. It double its values per km.  
                z1_std = np.dot(z1_std_MT_incre,dist_weigth)/np.sum(dist_weigth)
                # calculete z2 normal prior parameters
                # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
                #z2_mean_prior = z2_mean_prior - z1_mean_prior
                #print(z2_mean_prior)
                z2_mean = np.dot(z2_mean_MT,dist_weigth)/np.sum(dist_weigth)
                #z2_mean = z2_mean 
                if z2_mean < 0.:
                    raise ValueError
                z2_std = np.dot(z2_std_MT_incre,dist_weigth)/np.sum(dist_weigth)        

                if masl:
                    z1_mean = sta.elev - z1_mean   # need to import topography (elevation in every point of the grid)
            
            # write values in .txt
            f.write("{:4.4f}\t{:4.4f}\t{:4.2f}\t{:4.2f}\t{:4.2f}\t{:4.2f}\n".\
                format(lon,lat,z1_mean,z1_std,z2_mean,z2_std))
            #
            Z_z1_mean[j][i] = z1_mean
            Z_z1_std[j][i] = z1_std
            Z_z2_mean[j][i] = z2_mean
            Z_z2_std[j][i] = z2_std
            Z_z1_plus_z2_mean[j][i] = z2_mean + z1_mean
            Z_z1_plus_z2_std[j][i] = (z1_std + z2_std) / 2

    f.close()
    if masl:
        shutil.move('.'+os.sep+file_name+'.txt', path_output+os.sep+file_name+'_masl.txt')
    else:
        shutil.move('.'+os.sep+file_name+'.txt', path_output+os.sep+file_name+'.txt')

    if plot:
        ## 
        def plot_2Darray_contourf(array, name, levels = None, xlim = None, masl = None):
            if levels is None:
                levels = np.arange(0,501,25)

            f = plt.figure(figsize=[12.5,10.5])
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
            
            cmap = plt.get_cmap('winter')
            ax.set_aspect('equal')

            cf = ax.contourf(X,Y,array,levels = levels,cmap=cmap, alpha=.9, antialiased=True)
            f.colorbar(cf, ax=ax, label ='[m]')

            for sta in station_objects:
                ax.plot(sta.lon_dec,sta.lat_dec,'.k')
                coord_aux = [sta.lon_dec, sta.lat_dec]
            ax.plot(coord_aux,'.k', label = 'MT sta')
            
            f.tight_layout()
            ax.set_xlabel('latitud [??]', size = textsize)
            ax.set_ylabel('longitud [??]', size = textsize)
            if masl:
                ax.set_title(name+' m.a.s.l', size = textsize)
            else:
                ax.set_title(name, size = textsize)
            ax.legend(loc=1, prop={'size': textsize})
            # save figure
            file_name = name+'_MT_inv_contourf.png'
            plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
            if masl:
                shutil.move(file_name, path_output+os.sep+name+'_MT_inv_contourf_masl.png')
            else:
                shutil.move(file_name, path_output+os.sep+file_name)
            plt.clf()
        
        # plot countours with level
        if masl:
            levels = np.arange(0,301,25) # for mean z1
        else:
            levels = np.arange(100,401,25) # for mean z1
        plot_2Darray_contourf(Z_z1_mean, name = 'z1 mean', levels = levels, xlim = xlim, masl = masl)
        levels = np.arange(75,526,25) # for std z1
        plot_2Darray_contourf(Z_z1_std, name = 'z1 std', levels = levels, xlim = xlim)
        ###
        if masl:
            levels = np.arange(-400,551,25) # for mean z2
        else:
            levels = np.arange(225,551,25) # for mean z2
        plot_2Darray_contourf(Z_z2_mean, name = 'z2 mean', levels = levels, xlim = xlim)
        levels = np.arange(175,501,25) # for std z2
        plot_2Darray_contourf(Z_z2_std, name = 'z2 std', levels = levels, xlim = xlim)
        ###
        levels = np.arange(450,901,25) # for mean z1+z2
        plot_2Darray_contourf(Z_z1_plus_z2_mean, name = 'z1+z2 mean', levels = levels, xlim = xlim, masl = masl)
        levels = np.arange(150,501,25) # for std z1+z2
        plot_2Darray_contourf(Z_z1_plus_z2_std, name = 'z1+z2 std', levels = levels, xlim = xlim)
        ###

def topo_MT_inv_rest(station_objects, topo_path, slp = None, file_name = None, path_output = None,\
	plot = None, path_base_image = None, ext_img = None, xlim = None, ylim = None, masl = None): 
    """
    fn to plot the plainview MT results using a grid the topography 
    """  
    if file_name is None:
        file_name = 'topo_MT_inv'
    if slp is None: 
        slp = 4*10.
    if path_output is None: 
        path_output = '.'

    # import topography and create list of coords. pairs 
    topo = np.genfromtxt(topo_path, delimiter = ',', skip_header = 1)
    Lat = [t[0] for t in topo]
    Lon = [t[1] for t in topo]
    Elev = [t[2] for t in topo]
    
    if False: # plot topography 
        f = plt.figure(figsize=[12.5,10.5])
        ax = plt.axes([0.1,0.1,0.9,0.7]) # [left, bottom, width, height]
        # background img
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
        # plot topo 
        cmap = plt.get_cmap('cividis')
        ax.set_aspect('equal')
        cf =  ax.tricontourf(Lon, Lat, Elev, cmap=cmap)
        f.colorbar(cf, ax=ax, label ='[m]')
        ax.plot(Lon, Lat,'.')
        f.tight_layout()
        ax.set_xlabel('latitud [??]', size = textsize)
        ax.set_ylabel('longitud [??]', size = textsize)
        ax.set_title('Topography Wairakei-Tauhara (500 m)', size = textsize)
        # save figure
        file_name = 'Topo_WT.png'
        plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
        shutil.move(file_name, path_output+os.sep+file_name)
        plt.clf()

    # vectors to plot    
    Z_z1_mean = []
    Z_z1_std = []
    Z_z2_mean = []
    Z_z2_std = []
    Z_z1_plus_z2_mean = []
    Z_z1_plus_z2_std = []

    # ----------
    for i in range(len(Lat)):
        # save names of nearest wells to be used for prior
        near_stas = [sta for sta in station_objects] #list of objects (wells)
        near_stas = list(filter(None, near_stas))
        dist_stas = [dist_two_points([sta.lon_dec, sta.lat_dec], [Lon[i], Lat[i]], type_coord = 'decimal')\
            for sta in station_objects]
        dist_stas = list(filter(None, dist_stas))      

        # Calculate z1 and z1 at topo positions or boundaries of the cc in station
        # z1
        z1_mean_MT = np.zeros(len(near_stas))
        z1_std_MT = np.zeros(len(near_stas))
        z2_mean_MT = np.zeros(len(near_stas))
        z2_std_MT = np.zeros(len(near_stas))
        #
        z1_std_MT_incre = np.zeros(len(near_stas))
        z2_std_MT_incre = np.zeros(len(near_stas))
        count = 0
        # extract MT mcmc results from nearest wells (z1 and z2)
        for sta in near_stas:
            # extract mt results from file 
            mt_mcmc_results = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+sta.name[:-4]+os.sep+"est_par.dat")
            # values for mean a std for normal distribution representing the prior
            z1_mean_MT[count] = mt_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
            z1_std_MT[count] =  mt_mcmc_results[0,2] # std z1
            z2_mean_MT[count] = mt_mcmc_results[1,1] # mean [1] z2 # median [3] z1
            z2_std_MT[count] =  mt_mcmc_results[1,2] # std z2
            # calc. increment in std. in the position of the station
            # std. dev. increases as get farder from the well. It double its values per 2 km.
            z1_std_MT_incre[count] = z1_std_MT[count]  + (dist_stas[count] *slp)
            z2_std_MT_incre[count] = z2_std_MT[count]  + (dist_stas[count] *slp)
            # load pars in well 
            count+=1
            # calculete z1 normal prior parameters
            dist_weigth = [1./d for d in dist_stas]
            z1_mean = np.dot(z1_mean_MT,dist_weigth)/np.sum(dist_weigth)
            # std. dev. increases as get farder from the well. It double its values per km.  
            z1_std = np.dot(z1_std_MT_incre,dist_weigth)/np.sum(dist_weigth)
            # calculete z2 normal prior parameters
            # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
            #z2_mean_prior = z2_mean_prior - z1_mean_prior
            #print(z2_mean_prior)
            z2_mean = np.dot(z2_mean_MT,dist_weigth)/np.sum(dist_weigth)
            #z2_mean = z2_mean 
            if z2_mean < 0.:
                raise ValueError
            z2_std = np.dot(z2_std_MT_incre,dist_weigth)/np.sum(dist_weigth)        

        #
        Z_z1_mean.append(z1_mean)
        Z_z1_std.append(z1_std)
        Z_z2_mean.append(z2_mean)
        Z_z2_std.append(z2_std)
        Z_z1_plus_z2_mean.append(z2_mean + z1_mean)
        Z_z1_plus_z2_std.append((z1_std + z2_std) / 2)

    if plot:
        ## 
        def plot_2Darray_tricontourf(array, name, levels = None, xlim = None, ylim = None, masl = None):
            f = plt.figure(figsize=[12.5,10.5])
            ax = plt.axes([0.1,0.1,0.9,0.7]) # [left, bottom, width, height]
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
            
            if masl:
                array = [Elev[i] - array[i] for i in range(len(Elev))]

            # plot topo 
            if masl:
                cmap = plt.get_cmap('winter_r')
            else: 
                cmap = plt.get_cmap('winter')
            ax.set_aspect('equal')
            cf =  ax.tricontourf(Lon, Lat, array, cmap=cmap, levels = levels, alpha = .8)
            f.colorbar(cf, ax=ax, label ='[m]')
            #ax.plot(Lon, Lat,'.')
            for sta in station_objects:
                ax.plot(sta.lon_dec,sta.lat_dec,'.k')
                coord_aux = [sta.lon_dec, sta.lat_dec]
            ax.plot(coord_aux,'.c', label = 'MT sta')
            #
            f.tight_layout()
            ax.set_xlabel('latitud [??]', size = textsize)
            ax.set_ylabel('longitud [??]', size = textsize)
            ax.set_title(name, size = textsize)            # save figure          
            if masl:
                ax.set_title(name+' m.a.s.l', size = textsize)
            else:
                ax.set_title(name, size = textsize)
            ax.legend(loc=1, prop={'size': textsize})
            # save figure
            file_name = name+'_MT_inv_tricontourf.png' 
            plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
            if masl:
                shutil.move(file_name, path_output+os.sep+file_name[:-4]+'_masl.png')
            else:
                shutil.move(file_name, path_output+os.sep+file_name)
            plt.clf()
        
        # plot countours with level
        # z1
        levels = np.arange(100,401,25) # for mean z1
        plot_2Darray_tricontourf(Z_z1_mean, name = 'z1 mean',levels = levels,  xlim = xlim, masl = False)
        # masl
        levels = np.arange(-50,351,50) # for mean z1
        plot_2Darray_tricontourf(Z_z1_mean, name = 'z1 mean',levels = levels, xlim = xlim, masl = True)
        # z1 + z2
        levels = np.arange(450,901,25)
        plot_2Darray_tricontourf(Z_z1_plus_z2_mean, name = 'z1+z2 mean',levels = None, xlim = xlim, masl = False)
        # masl
        levels = np.arange(-600,50,80)
        plot_2Darray_tricontourf(Z_z1_plus_z2_mean, name = 'z1+z2 mean', levels = levels, xlim = xlim, masl = True)


def triangulation_meb_results(station_objects, well_objects, path_base_image = None, ext_img = None, xlim = None, ylim = None, \
     file_name = None, format = None, value = None, vmin=None, vmax=None, filter_wells_Q = None):  
    """
    value: value to interpolate z1_mean, z2_mean, z1_std, z2_std.  vmin and vmax: range for the value.
    filter_wells_Q: filter wells based on their quality (See file). input: path to quality file 
    """
    if value:
        value = value
        if vmin:
            vmin = vmin
        if vmax:
            vmax = vmax
    else:
        value = False
        vmin = None
        vmax = None
    if filter_wells_Q: 
        filter_wells_Q = filter_wells_Q
    else: 
        filter_wells_Q = False
    well_objects_copy = well_objects

    # filter wells with 0 quality
    if filter_wells_Q:
        q_wells = [x for x in open(filter_wells_Q).readlines() if x[0]!='#']
        q_wells = [x.split() for x in q_wells] 
        q_wells = [x for x in q_wells if x[:][1]=='0'] 
        q_wells = [x[0] for x in q_wells]
        # create list of wells of quality diferent thab 0 
        well_objects_copy = [wl for wl in well_objects if wl.name not in q_wells] 

    lon_stas = []
    lat_stas = []
    #for sta in station_objects:
    #    lon_stas.append(sta.lon_dec)
    #    lat_stas.append(sta.lat_dec)
    for wl in well_objects_copy:
        if wl.meb:
            lon_stas.append(wl.lon_dec)
            lat_stas.append(wl.lat_dec)
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
    for wl in well_objects_copy:
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
            if value is 'z1_mean':
                values.append(wl.meb_z1_pars[0])
            if value is 'z2_mean':
                values.append(wl.meb_z2_pars[0])
            if value is 'z1_std':
                values.append(wl.meb_z1_pars[2])
            if value is 'z2_std':
                values.append(wl.meb_z2_pars[2])
            else:
                pass
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

    # meshgrid

    # Delaunay triangulazation 
    points = np.asarray(points)
    tri = Delaunay(points)
    
    if value: # interpolate values (z): LinearNDInterpolator
        # meshgrid
        X = np.linspace(min(lon_stas), max(lon_stas), num=1000)
        Y = np.linspace(min(lat_stas), max(lat_stas), num=1000)
        X, Y = np.meshgrid(X, Y)
        cmap = 'gist_rainbow'
        if True:
            #plt.figure()
            interp = LinearNDInterpolator(tri, values)
            Z0 = interp(X, Y)
            plt.pcolormesh(X, Y, Z0, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar() # Color Bar
            #plt.show()
        else:
            #plt.figure()
            func = interp2d(lon_stas, lat_stas, values)
            Z = func(X[0, :], Y[:, 0])
            plt.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar() # Color Bar
            #plt.show()

    #print(tri.simplices)
    #print(points[tri.simplices])
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy(), linewidth=.8, color = 'b')
    plt.plot(points[:,0], points[:,1], 'bo', label = 'MeB well', ms = 2)
    plt.plot(lon_stas, lat_stas, '*r', label = 'MT sta.', ms = 1)

    ax.legend(loc=1, prop={'size': 6})	
    ax.set_xlabel('latitud [??]', size = textsize)
    ax.set_ylabel('longitud [??]', size = textsize)
    if value:
        ax.set_title('Triangulation of MeB wells: '+value, size = textsize)
    else: 
        ax.set_title('Triangulation of MeB wells', size = textsize)


    # interpolation 
    
    # save figure
    if file_name is None:
        file_name = 'Trig_meb_wells_WRKNW5'
    if format is None: 
        format = 'png'

    # save figure
    plt.savefig(file_name+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
    shutil.move(file_name+'.png', '.'+os.sep+'plain_view_plots'+os.sep+file_name+'.png')
    plt.close(f)
    plt.clf()

def grid_meb_prior(wells_objects, coords, n_points = None,  slp = None, file_name = None, plot = None, 
        path_output = None, path_base_image = None, ext_img = None, xlim = None, ylim = None, 
        cont_plot = True, scat_plot = True):
    """
    fn for griding and calculate meb prior in grid (define by coords) points 
    output: file_name.txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
    """  
    # sort by latitud 
    wells_objects.sort(key=lambda x: x.lat_dec, reverse=False)

    if file_name is None:
        file_name = 'grid_meb_prior'
    if slp is None: 
        slp = 4*10.
    if n_points is None:
        n_points = 100
    if path_output is None: 
        path_output = '.'
    
    if True:#cont_plot:
        # vectors for coordinates    
        x = np.linspace(coords[0], coords[1], n_points) # long
        y = np.linspace(coords[2], coords[3], n_points) # lat
        X, Y = np.meshgrid(x, y)
        Z_z1_mean = X*0.
        Z_z1_std = X*0
        Z_z2_mean = X*0
        Z_z2_std = X*0
        # calculate MeB prior at each position 
        f = open(file_name+'.txt', "w")
        f.write("# lat\tlon\tmean_z1\tstd_z1\tmean_z2\tstd_z2\n")
        for j,lat in enumerate(y):
            for i,lon in enumerate(x):

                if False: # use 4 closest wells (search by quadrant approach)
                    dist_pre_q1 = []
                    dist_pre_q2 = []
                    dist_pre_q3 = []
                    dist_pre_q4 = []
                    #
                    name_aux_q1 = [] 
                    name_aux_q2 = []
                    name_aux_q3 = []
                    name_aux_q4 = []
                    wl_q1 = []
                    wl_q2 = []
                    wl_q3 = []
                    wl_q4 = []
                    for wl in wells_objects:
                        if wl.meb:
                            if True: # search by quadrant approach
                                # search for nearest well to MT station in quadrant 1 (Q1)
                                if (wl.lat_dec > lat and wl.lon_dec > lon): 
                                    # distance between station and well
                                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [lon, lat], type_coord = 'decimal')
                                    if not dist_pre_q1:
                                        dist_pre_q1 = dist
                                    # check if distance is longer than the previous wel 
                                    if dist <= dist_pre_q1: 
                                        name_aux_q1 = wl.name
                                        wl_q1 = wl
                                        dist_pre_q1 = dist
                                # search for nearest well to MT station in quadrant 2 (Q2)
                                if (wl.lat_dec < lat and wl.lon_dec > lon): 
                                    # distance between station and well
                                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [lon, lat], type_coord = 'decimal')
                                    if not dist_pre_q2:
                                        dist_pre_q2 = dist
                                    # check if distance is longer than the previous wel 
                                    if dist <= dist_pre_q2: 
                                        name_aux_q2 = wl.name
                                        wl_q2 = wl
                                        dist_pre_q2 = dist
                                # search for nearest well to MT station in quadrant 3 (Q3)
                                if (wl.lat_dec < lat and wl.lon_dec < lon): 
                                    # distance between station and well
                                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [lon, lat], type_coord = 'decimal')
                                    if not dist_pre_q3:
                                        dist_pre_q3 = dist
                                    # check if distance is longer than the previous wel 
                                    if dist <= dist_pre_q3: 
                                        name_aux_q3 = wl.name
                                        wl_q3 = wl
                                        dist_pre_q3 = dist
                                # search for nearest well to MT station in quadrant 4 (Q4)
                                if (wl.lat_dec > lat and wl.lon_dec < lon): 
                                    # distance between station and well
                                    dist = dist_two_points([wl.lon_dec, wl.lat_dec], [lon, lat], type_coord = 'decimal')
                                    if not dist_pre_q4:
                                        dist_pre_q4 = dist
                                    # check if distance is longer than the previous wel 
                                    if dist <= dist_pre_q4: 
                                        name_aux_q4 = wl.name
                                        wl_q4 = wl
                                        dist_pre_q4 = dist

                    # save names of nearest wells to be used for prior
                    near_wls = [wl_q1,wl_q2,wl_q3,wl_q4] #list of objects (wells)
                    near_wls = list(filter(None, near_wls))
                    dist_wels = [dist_pre_q1,dist_pre_q2,dist_pre_q3,dist_pre_q4]
                    dist_wels = list(filter(None, dist_wels))
                    # Calculate prior values for boundaries of the cc in station
                    # prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
                    # z1
                    z1_mean_prior = np.zeros(len(near_wls))
                    z1_std_prior = np.zeros(len(near_wls))
                    z2_mean_prior = np.zeros(len(near_wls))
                    z2_std_prior = np.zeros(len(near_wls))
                    #
                    z1_std_prior_incre = np.zeros(len(near_wls))
                    z2_std_prior_incre = np.zeros(len(near_wls))
                    count = 0
                    # extract meb mcmc results from nearest wells 
                    for wl in near_wls:
                        # extract meb mcmc results from file 
                        meb_mcmc_results = np.genfromtxt(wl.path_mcmc_meb+os.sep+"est_par.dat")
                        # values for mean a std for normal distribution representing the prior
                        z1_mean_prior[count] = meb_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
                        z1_std_prior[count] =  meb_mcmc_results[0,2] # std z1
                        z2_mean_prior[count] = meb_mcmc_results[1,1] # mean [1] z2 # median [3] z1
                        z2_std_prior[count] =  meb_mcmc_results[1,2] # std z2
                        # calc. increment in std. in the position of the station
                        # std. dev. increases as get farder from the well. It double its values per 2 km.
                        z1_std_prior_incre[count] = z1_std_prior[count]  + (dist_wels[count] *slp)
                        z2_std_prior_incre[count] = z2_std_prior[count]  + (dist_wels[count] *slp)
                        # load pars in well 
                        count+=1
                    # calculete z1 normal prior parameters
                    dist_weigth = [1./d for d in dist_wels]
                    z1_mean = np.dot(z1_mean_prior,dist_weigth)/np.sum(dist_weigth)
                    # std. dev. increases as get farder from the well. It double its values per km.  
                    z1_std = np.dot(z1_std_prior_incre,dist_weigth)/np.sum(dist_weigth)
                    # calculete z2 normal prior parameters
                    # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
                    #z2_mean_prior = z2_mean_prior - z1_mean_prior
                    #print(z2_mean_prior)
                    z2_mean = np.dot(z2_mean_prior,dist_weigth)/np.sum(dist_weigth)
                    #z2_mean = z2_mean 
                    if z2_mean < 0.:
                        raise ValueError
                    z2_std = np.dot(z2_std_prior_incre,dist_weigth)/np.sum(dist_weigth)

                if True: # use every well available
                    # save names of nearest wells to be used for prior
                    near_wls = [wl for wl in wells_objects if wl.meb] #list of objects (wells)
                    near_wls = list(filter(None, near_wls))
                    dist_wels = [dist_two_points([wl.lon_dec, wl.lat_dec], [lon, lat], type_coord = 'decimal')\
                        for wl in wells_objects if wl.meb]
                    dist_wels = list(filter(None, dist_wels))
                    # Calculate prior values for boundaries of the cc in station
                    # prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
                    # z1
                    z1_mean_prior = np.zeros(len(near_wls))
                    z1_std_prior = np.zeros(len(near_wls))
                    z2_mean_prior = np.zeros(len(near_wls))
                    z2_std_prior = np.zeros(len(near_wls))
                    #
                    z1_std_prior_incre = np.zeros(len(near_wls))
                    z2_std_prior_incre = np.zeros(len(near_wls))
                    count = 0
                    # extract meb mcmc results from nearest wells 
                    for wl in near_wls:
                        # extract meb mcmc results from file 
                        meb_mcmc_results = np.genfromtxt(wl.path_mcmc_meb+os.sep+"est_par.dat")
                        # values for mean a std for normal distribution representing the prior
                        z1_mean_prior[count] = meb_mcmc_results[0,1] # mean [1] z1 # median [3] z1 
                        z1_std_prior[count] =  meb_mcmc_results[0,2] # std z1
                        z2_mean_prior[count] = meb_mcmc_results[1,1] # mean [1] z2 # median [3] z1
                        z2_std_prior[count] =  meb_mcmc_results[1,2] # std z2
                        # calc. increment in std. in the position of the station
                        # std. dev. increases as get farder from the well. It double its values per 2 km.
                        z1_std_prior_incre[count] = z1_std_prior[count]  + (dist_wels[count] *slp)
                        z2_std_prior_incre[count] = z2_std_prior[count]  + (dist_wels[count] *slp)
                        # load pars in well 
                        count+=1
                    # calculete z1 normal prior parameters
                    dist_weigth = [1./d for d in dist_wels]
                    z1_mean = np.dot(z1_mean_prior,dist_weigth)/np.sum(dist_weigth)
                    # std. dev. increases as get farder from the well. It double its values per km.  
                    z1_std = np.dot(z1_std_prior_incre,dist_weigth)/np.sum(dist_weigth)
                    # calculete z2 normal prior parameters
                    # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
                    #z2_mean_prior = z2_mean_prior - z1_mean_prior
                    #print(z2_mean_prior)
                    z2_mean = np.dot(z2_mean_prior,dist_weigth)/np.sum(dist_weigth)
                    #z2_mean = z2_mean 
                    if z2_mean < 0.:
                        raise ValueError
                    z2_std = np.dot(z2_std_prior_incre,dist_weigth)/np.sum(dist_weigth)        
                
                # write values in .txt
                f.write("{:4.4f}\t{:4.4f}\t{:4.2f}\t{:4.2f}\t{:4.2f}\t{:4.2f}\n".\
                    format(lon,lat,z1_mean,z1_std,z2_mean-z1_mean,z2_std))
                #
                Z_z1_mean[j][i] = z1_mean
                Z_z1_std[j][i] = z1_std
                Z_z2_mean[j][i] = z2_mean+z1_mean
                Z_z2_std[j][i] = z2_std

        f.close()
        shutil.move('.'+os.sep+file_name+'.txt', path_output+os.sep+file_name+'.txt')

    if scat_plot:
        z1_mean = []
        z1_std = []
        z2_mean = []
        z2_std = []
        lon_stas = []
        lat_stas = []
        count = 0
        # extract conductor T1 and T2
        for i, wl in enumerate(wells_objects):
            if wl.meb:
                # values for mean a std for normal distribution representing the prior
                z1_mean.append(wl.meb_z1_pars[0]) # mean [1] z1 # median [3] z1 
                z1_std.append(wl.meb_z1_pars[1]) # std z1
                z2_mean.append(wl.meb_z2_pars[0]) # mean [1] z2 # median [3] z1
                z2_std.append(wl.meb_z2_pars[1]) # std z2
                lon_stas.append(wl.lon_dec)
                lat_stas.append(wl.lat_dec)

    if plot:
        ## 
        def plot_2Darray_contourf(array, name, levels = None):
            if levels is None:
                levels = np.arange(0,501,25)

            f = plt.figure(figsize=[12.5,10.5])
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
            
            cmap = plt.get_cmap('winter')
            ax.set_aspect('equal')

            # contourf plot
            if cont_plot:
                cf = ax.contourf(X,Y,array,levels = levels,cmap=cmap, alpha=.8, antialiased=True)
                f.colorbar(cf, ax=ax, label ='[m]')

            if not scat_plot:
                for wl in wells_objects:
                    if wl.meb:
                        ax.plot(wl.lon_dec,wl.lat_dec,'.k')
                        coord_aux = [wl.lon_dec, wl.lat_dec]
                ax.plot(coord_aux,'.k', label = 'MeB well')
            
            # scatter plot
            if scat_plot:
                if name == 'z1 mean':
                    data = z1_mean
                if name == 'z2 mean':
                    data = z2_mean
                if name == 'z1 std':
                    data = z1_std
                if name == 'z2 std':
                    data = z2_std
                # plot one circle for the legend
                ax.plot(lon_stas[0],lat_stas[0], c = 'lightgray', label = 'MeB well', zorder=0, \
                    marker='o', markersize = 8, markeredgecolor = 'k')
                
                #plt.scatter(lon_stas[0],lat_stas[0],s=2, c = 1,facecolors='none', label = 'MeB well', zorder=0)
                # scatter plot
                size = 50*np.ones(len(array))
                vmin = min(levels)
                vmax = max(levels)
                normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                scatter = ax.scatter(lon_stas,lat_stas, s = size, c = data, edgecolors = 'k', cmap = 'winter', \
                    norm = normalize)#alpha = 0.5)
                if not cont_plot:
                    f.colorbar(scatter, ax=ax, label ='[m]')
            
            f.tight_layout()
            ax.set_xlabel('latitud [??]', size = textsize)
            ax.set_ylabel('longitud [??]', size = textsize)
            if name == 'z1 mean':
                ax.set_title('Depth to the clay cap TOP boundary', size = textsize)
            if name == 'z2 mean':
                ax.set_title('Depth to the clay cap BOTTOM boundary', size = textsize)
            if name == 'z1 std':
                ax.set_title('Uncertainty of the TOP clay cap TOP boundary', size = textsize)
            if name == 'z2 std':
                ax.set_title('Uncertainty of the BOTTOM clay cap TOP boundary', size = textsize)
            
            ax.legend(loc=1, prop={'size': textsize})
            # save figure
            if cont_plot:
                file_name = name+'_meb_prior_contourf.png'
            if scat_plot:
                file_name = name+'_meb_prior_scatter.png'
            if cont_plot and scat_plot:
                file_name = name+'_meb_prior_cont_scat.png'
            plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
            shutil.move(file_name, path_output+os.sep+file_name)
            plt.clf()
       
        # plot countours with level
        levels = np.arange(125,501,25) # for std z1
        plot_2Darray_contourf(Z_z1_std, name = 'z1 std', levels = levels)
        plot_2Darray_contourf(Z_z2_std, name = 'z2 std', levels = levels)
        levels = np.arange(200,576,25) # for mean z1
        plot_2Darray_contourf(Z_z1_mean, name = 'z1 mean', levels = levels)
        levels = np.arange(700,1501,25) # for mean z2
        plot_2Darray_contourf(Z_z2_mean, name = 'z2 mean', levels = levels)

def grid_temp_conductor_bound(wells_objects, coords, n_points = None,  slp = None, file_name = None, plot = None, 
        path_output = None, path_base_image = None, ext_img = None, xlim = None, ylim = None, just_plot = None, masl = None):
    """
    fn for griding and calculate temperature at the top and bottom of the conductor in grid (define by coords) points 
    output: file_name.txt with [lon, lat, mean_z1, std_z1, mean_z2, std_z2]
    """   
    if file_name is None:
        file_name = 'grid_temp_bc'
    if slp is None: 
        slp = 1*10.
    if n_points is None:
        n_points = 20
    if path_output is None: 
        path_output = '.'
    
    #########################################################
    # vectors for coordinates    
    x = np.linspace(coords[0], coords[1], n_points) # long
    y = np.linspace(coords[2], coords[3], n_points) # lat
    X, Y = np.meshgrid(x, y)
    T_z1_mean = X*0.
    T_z1_std = X*0
    T_z2_mean = X*0
    T_z2_std = X*0
    # calculate MeB prior at each position 
    f = open(file_name+'.txt', "w")
    f.write("# lat\tlon\tmean_temp_z1\tstd_temp_z1\tmean_temp_z2\tstd_temp_z2\n")
    for j,lat in enumerate(y):
        for i,lon in enumerate(x):
            if True: # use every stations available
                # save names of nearest wells to be used for prior
                near_stas = [wl for wl in wells_objects] #list of objects (wells)
                near_stas = list(filter(None, near_stas))
                dist_stas = [dist_two_points([wl.lon_dec, wl.lat_dec], [lon, lat], type_coord = 'decimal')\
                    for wl in wells_objects]
                dist_stas = list(filter(None, dist_stas))
                # Calculate prior values for boundaries of the cc in station
                # prior consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
                # z1
                z1_mean_T = np.zeros(len(near_stas))
                z1_std_T = np.zeros(len(near_stas))
                z2_mean_T = np.zeros(len(near_stas))
                z2_std_T = np.zeros(len(near_stas))
                #
                z1_std_T_incre = np.zeros(len(near_stas))
                z2_std_T_incre = np.zeros(len(near_stas))
                count = 0
                # extract meb mcmc results from nearest wells 
                for wl in near_stas:
                    # extract meb mcmc results from file 
                    wl_temp_cond_bound = np.genfromtxt('.'+os.sep+'corr_temp_bc'+os.sep+wl.name+os.sep+"conductor_T1_T2.txt")
                    # values for mean a std for normal distribution representing the prior
                    z1_mean_T[count] = wl_temp_cond_bound[0] # mean [1] z1 # median [3] z1 
                    z1_std_T[count] =  wl_temp_cond_bound[1] # std z1
                    z2_mean_T[count] = wl_temp_cond_bound[2] # mean [1] z2 # median [3] z1
                    z2_std_T[count] =  wl_temp_cond_bound[3] # std z2
                    # calc. increment in std. in the position of the station
                    # std. dev. increases as get farder from the well. It double its values per 2 km.
                    z1_std_T_incre[count] = z1_std_T[count]  + (dist_stas[count] *slp)
                    z2_std_T_incre[count] = z2_std_T[count]  + (dist_stas[count] *slp)
                    # load pars in well 
                    count+=1

                # calculete z1 normal prior parameters
                dist_weigth = [1./d for d in dist_stas]
                z1_mean = np.dot(z1_mean_T,dist_weigth)/np.sum(dist_weigth)
                # std. dev. increases as get farder from the well. It double its values per km.  
                z1_std = np.dot(z1_std_T_incre,dist_weigth)/np.sum(dist_weigth)
                # calculete z2 normal prior parameters
                # change z2 from depth (meb mcmc) to tickness of second layer (mcmc MT)
                #z2_mean_prior = z2_mean_prior - z1_mean_prior
                #print(z2_mean_prior)
                z2_mean = np.dot(z2_mean_T,dist_weigth)/np.sum(dist_weigth)
                #z2_mean = z2_mean 
                if z2_mean < 0.:
                    raise ValueError
                z2_std = np.dot(z2_std_T_incre,dist_weigth)/np.sum(dist_weigth)        

                if masl:
                    z1_mean = wl.elev - z1_mean   # need to import topography (elevation in every point of the grid)
            
            # write values in .txt
            f.write("{:4.4f}\t{:4.4f}\t{:4.2f}\t{:4.2f}\t{:4.2f}\t{:4.2f}\n".\
                format(lon,lat,z1_mean,z1_std,z2_mean,z2_std))
            #
            T_z1_mean[j][i] = z1_mean
            T_z1_std[j][i] = z1_std
            T_z2_mean[j][i] = z2_mean
            T_z2_std[j][i] = z2_std

    f.close()
    if masl:
        shutil.move('.'+os.sep+file_name+'.txt', path_output+os.sep+file_name+'_masl.txt')
    else:
        shutil.move('.'+os.sep+file_name+'.txt', path_output+os.sep+file_name+'.txt')

    if plot:
        ## 
        def plot_2Darray_contourf_T(array, name, levels = None, xlim = None, masl = None):
            if levels is None:
                levels = np.arange(0,501,25)

            f = plt.figure(figsize=[12.5,10.5])
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
            
            cmap = plt.get_cmap('YlOrRd')
            ax.set_aspect('equal')

            cf = ax.contourf(X,Y,array,levels = levels,cmap=cmap, alpha=.8, antialiased=True)
            f.colorbar(cf, ax=ax, label ='Temperature ??C')

            for wl in wells_objects:
                ax.plot(wl.lon_dec,wl.lat_dec,'.k')
                coord_aux = [wl.lon_dec, wl.lat_dec]
            ax.plot(coord_aux,'.k', label = 'Well')
            
            f.tight_layout()
            ax.set_xlabel('latitud [??]', size = textsize)
            ax.set_ylabel('longitud [??]', size = textsize)
            if masl:
                ax.set_title(name+' m.a.s.l', size = textsize)
            else:
                ax.set_title(name, size = textsize)
            ax.legend(loc=1, prop={'size': textsize})
            # save figure
            file_name = name+'_contourf.png'
            plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
            if masl:
                shutil.move(file_name, path_output+os.sep+name+'_z1_contourf_masl.png')
            else:
                shutil.move(file_name, path_output+os.sep+file_name)
            plt.clf()
        
        # plot countours with level
        if masl:
            levels = np.arange(0,301,25) # for mean z1
        else:
            levels = np.arange(60,150,5) # for mean z1
        plot_2Darray_contourf_T(T_z1_mean, name = 'T1 mean', levels = levels, xlim = xlim, masl = masl)
        levels = np.arange(0,90,10) # for std z1
        plot_2Darray_contourf_T(T_z1_std, name = 'T1 std', levels = levels, xlim = xlim)
        ###
        if masl:
            levels = np.arange(-400,551,25) # for mean z2
        else:
            levels = np.arange(110,230,5) # for mean z2
        plot_2Darray_contourf_T(T_z2_mean, name = 'T2 mean', levels = levels, xlim = xlim)
        levels = np.arange(0,90,10) # for std z2
        plot_2Darray_contourf_T(T_z2_std, name = 'T2 std', levels = levels, xlim = xlim)
        ###

def scatter_temp_conductor_bound(wells_objects,  path_output = None, \
    path_base_image = None, ext_img = None, xlim = None, ylim = None, \
        alpha_img = None):
    """
    scatter plot of temperature at the top and bottom of the conductor
    """   
    if path_output is None: 
        path_output = '.'
    if ext_img is None: 
        pass
    else:
        ext = ext_img
        if alpha_img is None:
            alpha_img = 1.0
        else:
            alpha_img = alpha_img
    #########################################################
        # fill lists with temps at boundaries for each well 
    T1_mean = np.zeros(len(wells_objects))
    T1_std = np.zeros(len(wells_objects))
    T2_mean = np.zeros(len(wells_objects))
    T2_std = np.zeros(len(wells_objects))
    lon_wells = []
    lat_wells = []
    count = 0
    # extract conductor T1 and T2
    for i, wl in enumerate(wells_objects):
        # values for mean a std for normal distribution representing the prior
        T1_mean[i] =  wl.T1_pars[0] # mean [1] z1 # median [3] z1 
        T1_std[i] =  wl.T1_pars[1] # std z1
        T2_mean[i] = wl.T2_pars[0] # mean [1] z2 # median [3] z1
        T2_std[i] =  wl.T2_pars[1] # std z2
        lon_wells.append(wl.lon_dec)
        lat_wells.append(wl.lat_dec)

     # fn for scatter plot
    def plot_2Darray_scatter_T(lon_wells, lat_wells, data, data_std, name_data, \
        path_base_image = path_base_image, ext_img = None, xlim = None, ylim = None):
        # figure
        fig, ax = plt.subplots(figsize=(15,12))
        # plot base image (if given)
        if ext_img:
            img=mpimg.imread(path_base_image)
            ax.imshow(img, extent = ext, alpha = alpha_img) 
        if xlim is None:
            ax.set_xlim(ext[:2])
        else: 
            ax.set_xlim(xlim)

        if ylim is None:
            ax.set_ylim(ext[-2:])
        else: 
            ax.set_ylim(ylim)
        size = 200*np.ones(len(data))
        scatter = ax.scatter(lon_wells,lat_wells, s = size, c = data, cmap = 'YlOrRd')#alpha = 0.5)
        fig.colorbar(scatter, ax=ax, label ='Temperature ??C')
        # not sure if clay cap is there 
        ax.set_xlabel('Latitude [??]', size = textsize)
        ax.set_ylabel('Longitude [??]', size = textsize)

        if name_data == 'T2 mean':
            ax.set_title('T2: Temperature at the BOTTOM of the conductor', size = textsize)
        if name_data == 'T1 mean':
            ax.set_title('T1: Temperature at the TOP of the conductor', size = textsize)
        # save figure
        plt.savefig(name_data+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
        shutil.move(name_data+'.png', '.'+os.sep+path_output+os.sep+name_data+'_scatter.png')
        plt.tight_layout()
        plt.clf()
        # plot
    # call fn for figures
    plot_2Darray_scatter_T(lon_wells, lat_wells, T1_mean, T1_std, name_data = 'T1 mean', path_base_image = path_base_image, \
    ext_img = ext_img, xlim = xlim, ylim = ylim)
    plot_2Darray_scatter_T(lon_wells, lat_wells, T2_mean, T2_std, name_data = 'T2 mean', path_base_image = path_base_image, \
    ext_img = ext_img, xlim = xlim, ylim = ylim)


def scatter_MT_conductor_bound(station_objects,  path_output = None, \
    path_base_image = None, ext_img = None, xlim = None, ylim = None, alpha_img = None, \
            WK_resbound_line = None, taupo_lake_shoreline = None):
    """
    scatter plot of MT result depths at the top and bottom of the conductor
    """   
    if path_output is None: 
        path_output = '.'
    if ext_img is None: 
        pass
    else:
        ext = ext_img
        if alpha_img is None:
            alpha_img = 1.0
        else:
            alpha_img = alpha_img
    
    #########################################################
    # fill lists with temps at boundaries for each well 

    z1_mean = np.zeros(len(station_objects))
    z1_std = np.zeros(len(station_objects))
    z2_mean = np.zeros(len(station_objects))
    z2_std = np.zeros(len(station_objects))
    lon_stas = []
    lat_stas = []

    count = 0
    # extract conductor T1 and T2
    for i, sta in enumerate(station_objects):
        # values for mean a std for normal distribution representing the prior
        z1_mean[i] =  sta.z1_pars[0] # mean [1] z1 # median [3] z1 
        z1_std[i] =  sta.z1_pars[1] # std z1
        z2_mean[i] = sta.z2_pars[0] + sta.z1_pars[1] # mean [1] z2 # median [3] z1
        z2_std[i] =  sta.z2_pars[1] # std z2
        lon_stas.append(sta.lon_dec)
        lat_stas.append(sta.lat_dec)

    # fn for scatter plot
    def plot_2Darray_scatter_Z(lon_stas, lat_stas, data, data_std, name_data, 
            path_base_image = path_base_image, ext_img = None, xlim = None, ylim = None, unct_trans = True, 
                WK_resbound_line = None, taupo_lake_shoreline = None):
        '''
        unct_trans: transparency on scatter plot fn of std of z1 and z2
        '''
        # figure
        fig, ax = plt.subplots(figsize=(13,10))
        # plot base image (if given)
        if ext_img:
            img=mpimg.imread(path_base_image)
            ax.imshow(img, extent = ext, alpha = alpha_img) 
        if xlim is None:
            ax.set_xlim(ext[:2])
        else: 
            ax.set_xlim(xlim)
        if ylim is None:
            ax.set_ylim(ext[-2:])
        else: 
            ax.set_ylim(ylim)

        size = 200*np.ones(len(data))

        scatter = ax.scatter(lon_stas,lat_stas, s = size, c = data, edgecolors = 'k', cmap = 'winter')#alpha = 0.5)
        fig.colorbar(scatter, ax=ax, label ='Depth [m]')

        # plot one circle for the legend
        ax.plot(lon_stas[0],lat_stas[0], c = 'lightgray', label = 'MT station', zorder=0, \
            marker='o', markersize = 10, markeredgecolor = 'k')

        # not sure if clay cap is there 

        ax.set_xlabel('Latitude [??]', size = textsize)
        ax.set_ylabel('Longitude [??]', size = textsize)
        if name_data == 'z2 mean':
            ax.set_title('z2: depth at the BOTTOM of the conductor', size = textsize)

        if name_data == 'z1 mean':
            ax.set_title('z1: depth at the TOP of the conductor', size = textsize)

        # absence of CC (no_cc)
        for sta in station_objects:
            if (sta.z2_pars[0] < 50.):
                plt.plot(sta.lon_dec, sta.lat_dec,'w.', markersize=28) 
                plt.plot(sta.lon_dec, sta.lat_dec,'bx', markersize=12)

        if WK_resbound_line: 
            lats, lons = np.genfromtxt(WK_resbound_line, skip_header=1, delimiter=',').T
            plt.plot(lons, lats, color = 'r' ,linewidth = 4, label = 'DC resistivity boundary', zorder = 1)

        if taupo_lake_shoreline: 
            lats, lons = np.genfromtxt(taupo_lake_shoreline, skip_header=1, delimiter=',').T
            plt.plot(lons, lats, color = 'b' ,linewidth= 4, label = 'Taupo lake shoreline', zorder = 1)
            
        ax.legend(loc=1, prop={'size': textsize})
        # save figure
        plt.savefig(name_data+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	
        shutil.move(name_data+'.png', '.'+os.sep+path_output+os.sep+name_data+'_scatter.png')
        plt.tight_layout()
        plt.clf()
    
    # plot
    plot_2Darray_scatter_Z(lon_stas, lat_stas, z1_mean, z1_std, name_data = 'z1 mean', path_base_image = path_base_image, \
        ext_img = ext_img, xlim = xlim, ylim = ylim, WK_resbound_line = WK_resbound_line, taupo_lake_shoreline = taupo_lake_shoreline)
    plot_2Darray_scatter_Z(lon_stas, lat_stas, z2_mean, z2_std, name_data = 'z2 mean', path_base_image = path_base_image, \
        ext_img = ext_img, xlim = xlim, ylim = ylim, WK_resbound_line = WK_resbound_line, taupo_lake_shoreline = taupo_lake_shoreline)

    ###

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
    ax.set_xlabel('latitude [??]', size = textsize)
    ax.set_ylabel('longitude [??]', size = textsize)
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
            alphas[i] = 1. - data_std[i] / max(data_std) # -. increase the level of transparency

        rgba_colors = np.zeros((len(lon_stas),4))
        # for red the first column needs to be one
        rgba_colors[:,0] = .5
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas
        # size 
        size = abs(max(data) - data)/1
        if bound2plot == 'top': 
            scatter = ax.scatter(lon_stas,lat_stas, s = size/2, color = rgba_colors)#alpha = 0.5)
        if bound2plot == 'bottom': 
            scatter = ax.scatter(lon_stas,lat_stas, s = size/4, color = rgba_colors)#alpha = 0.5)
        # not sure if clay cap is there 
        for sta in station_objects:
            if (sta.z2_pars[0] < 50.):
                plt.plot(sta.lon_dec, sta.lat_dec,'w.', markersize=28) 
                plt.plot(sta.lon_dec, sta.lat_dec,'bx', markersize=12) 
        # absence of CC (no_cc)
        for sta in station_objects:
            if (sta.z2_pars[0] < 50.):
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
        ax.set_xlabel('Latitude [??]', size = textsize)
        ax.set_ylabel('Longitude [??]', size = textsize)
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

###########################################################################################
# Functions for base maps 

def base_map_region(path_topo = None , xlim = None, ylim = None,
    path_rest_bound = None, path_lake_shoreline = None, 
	path_faults = None, path_powerlines = None, two_cbar = None): 
    '''
    Create base map with topography and return figure. 
    '''
    # figure
    fig, ax = plt.subplots(figsize=(13,10))
    fig, ax = plt.subplots(figsize=(10,7))
    if two_cbar:
        fig, ax = plt.subplots(figsize=(16,10))
    ax.set_xlabel('Latitude [??]', size = textsize)
    ax.set_ylabel('Longitude [??]', size = textsize)
    if xlim is None:
        pass
    else: 
        ax.set_xlim(xlim)
    if ylim is None:
        pass
    else: 
        ax.set_ylim(ylim)

    # plot topography
    if path_topo: 
        lats, lons, elev = np.genfromtxt(path_topo, skip_header=1, delimiter=',').T
        #plt.plot(lons, lats, '.')
        ax.tricontour(lons, lats, elev, levels=10, linewidths=0.5, colors='g', alpha = 0.7)
        topo_cb = ax.tricontourf(lons, lats, elev, levels=10, cmap="BuGn", alpha = 0.7)
        #fig.colorbar(topo_cb, ax=ax, label ='elevation [m] (m.a.s.l.)')

    if path_rest_bound: 
        for path in path_rest_bound:
            lats, lons = np.genfromtxt(path, skip_header=1, delimiter=',').T
            plt.plot(lons, lats, color = 'orange' ,linewidth = 2, zorder = 1)
        plt.plot([],[], color = 'orange' ,linewidth = 2, label = 'DC resistivity boundary', zorder = 1)

    if path_lake_shoreline: 
        lats, lons = np.genfromtxt(path_lake_shoreline, skip_header=1, delimiter=',').T
        plt.plot(lons, lats, color = 'c' ,linewidth= 2, label = 'Taupo'+u'\u0304'+' lake shoreline', zorder = 1)

    if path_faults:
        def read_nzafd():
            reload = False
            if reload:
                with open('NZAFD_Dec_2018.kml') as data:
                    kml_soup = Soup(data, 'lxml-xml') # Parse as XML

                    faults = kml_soup.find_all('Placemark')
                    coords = {}
                    for fault in faults:
                        coord = fault.find_all('coordinates')[0].contents[0].strip().split()
                        coord = ([(np.float(ci.split(',')[0]), np.float(ci.split(',')[1])) for ci in coord])
                        try:
                            name = fault.find_all('name')[0].contents[0].strip()
                        except:
                            name = '{:d}'.format(np.random.randint(0,99999))
                            #continue
                            
                        try:
                            coords[name].append(coord)
                        except KeyError:
                            #print(name)
                            coords.update({name:[coord,]})
                coords = dict(coords)
                with open('nzafd.json', 'w') as fp:
                    json.dump(coords, fp)

            with open(path_faults, 'r') as fp:
                coords = json.load(fp)
                for k in coords.keys():
                    for i in range(len(coords[k])):
                        coords[k][i] = np.array(coords[k][i])
            return coords
        def add_faults(ax, c='c', labels=False, no_tfb=False, lw=0.45):
            #import cartopy.crs as ccrs

            coords = read_nzafd()

            zo = 3
            x_extent = np.array([175.95, 176.25])
            y_extent = np.array([-38.9, -38.55])

            for name,coord in coords.items():
                for coordi in coord:
                    if any([x_extent[0] < ci[0] < x_extent[1] and y_extent[0] < ci[1] < y_extent[1] for ci in coordi]):
                        #ax.plot(coordi[:,0],coordi[:,1],c+'-',lw=lw,transform=ccrs.PlateCarree(), zorder=zo) 
                        ax.plot(coordi[:,0],coordi[:,1],c+'-',lw=lw, zorder=zo) 
            ax.plot([],[],c+'-',lw=lw, zorder=zo, label = 'Faults') 
        add_faults(ax,'r',labels=True)

    if path_powerlines: # not working: need to be separated by nodes
        for path in path_powerlines:
            lats, lons = np.genfromtxt(path, skip_header=1, delimiter=',').T
            plt.plot(lons, lats, '-' , color = 'y', linewidth= 1, zorder = 3)
        plt.plot([], [], '-',color = 'y',linewidth= 1, label = 'Powerlines', zorder = 3)

    return fig, ax, topo_cb
    


