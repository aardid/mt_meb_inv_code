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
        locations4gearth.write(obj.name+'\t'+lon_dec+'\t'+lat_dec+'\t'+elev+'\n')
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

def uncert_bound_cc_plot_2D(sta_objects, pref_orient = 'EW'): 
    ## sta_objects: list of station objects
    ## sort list by longitud (min to max - East to West)
    sta_objects.sort(key=lambda x: x.lon_dec, reverse=True)
    ## calculate distances from first station to the others, save in array
    x_axis = np.zeros(len(sta_objects))
    i = 0
    for sta in sta_objects:
        coord1 = [sta_objects[0].lon_dec, sta_objects[0].lat_dec]
        coord2 = [sta.lon_dec, sta.lat_dec]
        x_axis[i] = dist_two_points(coord1, coord2, type_coord = 'decimal')
        i+=1
    ## plot envelopes 5% and 95% for cc boundaries
    