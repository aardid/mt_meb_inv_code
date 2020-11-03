"""
- Module Phase Tensor: functions to manipulate and plot the MT phase tensor
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2018
"""
# ==============================================================================
#  Imports
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.image as image

from matplotlib import gridspec
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math, cmath
import glob
from matplotlib.backends.backend_pdf import PdfPages
#from mpl_toolkits.basemap import pyproj
#from mpl_toolkits.basemap import Basemap
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from scipy.linalg import solve
from numpy import linalg as LA

textsize = 15
pale_orange_col = u'#ff7f0e' 
pale_blue_col = u'#1f77b4' 
pale_red_col = u'#EE6666'

# ==============================================================================
# Ellipses
# ==============================================================================

def ellipses_txt(Z,H): 

    name = Z[0]
    periods = Z[1]
    zxxr = Z[2]
    zxxi = Z[3]
    zxx = Z[4]
    zxx_var = Z[5]
    zxyr = Z[6]
    zxyi = Z[7]
    zxy = Z[8]
    zxy_var = Z[9]
    zyxr = Z[10]
    zyxi = Z[11]
    zyx = Z[12]
    zyx_var = Z[13]
    zyyr = Z[14]
    zyyi = Z[15]
    zyy = Z[16]
    zyy_var = Z[17]
    
    lat_dec, lon_dec = coord_dms2dec(H)
    
    #################################################
    
        #################################################
    # Calculate: Tensor Ellipses
    
    #Variables
    NUM = len(periods)
    det_R     = np.zeros(len(periods))
    T_fase_xx = np.zeros(len(periods))
    T_fase_xy = np.zeros(len(periods))
    T_fase_yx = np.zeros(len(periods))
    T_fase_yy = np.zeros(len(periods))

    phi_max = np.zeros(len(periods))
    phi_min = np.zeros(len(periods))
    beta    = np.zeros(len(periods))
    alpha   = np.zeros(len(periods))
    phi_1   = np.zeros(len(periods))
    phi_2   = np.zeros(len(periods))
    phi_3   = np.zeros(len(periods))

    diam_x = np.zeros(len(periods))
    diam_y = np.zeros(len(periods))
    angle = np.zeros(len(periods))
    e     = np.zeros(len(periods))

    eje_y = np.zeros(len(periods))
    a=range(NUM)
    eje_x = list(a) 
    aux_col = np.zeros(len(periods))
    
    file = name[:len(name)-4]
    ellip_txt= open(file+'_ellips.txt','w')
    
    numb = 0
    for p in periods:
        
        # Phase tensor 
        det_R[numb] = zxxr[numb]*zyyr[numb] - zxyr[numb]*zyxr[numb] # Determinant
    
        T_fase_xx[numb] = (1/det_R[numb])*(zyyr[numb]*zxxi[numb] - zxyr[numb]*zyxi[numb])
        T_fase_xy[numb] = (1/det_R[numb])*(zyyr[numb]*zxyi[numb] - zxyr[numb]*zyyi[numb])
        T_fase_yx[numb] = (1/det_R[numb])*(zxxr[numb]*zyxi[numb] - zyxr[numb]*zxxi[numb])
        T_fase_yy[numb] = (1/det_R[numb])*(zxxr[numb]*zyyi[numb] - zyxr[numb]*zxyi[numb])
        
        # ellipses components
        # beta[numb] = (1/2)*(360/(2*np.pi))*np.arctan((T_fase_xy[numb]-T_fase_yx[numb]) / (T_fase_xx[numb] + T_fase_yy[numb]))
        alpha[numb] = (1/2)*(360/(2*np.pi))*np.arctan((T_fase_xy[numb] + T_fase_yx[numb])/(T_fase_xx[numb] - T_fase_yy[numb]))

        phi_1[numb] = (T_fase_xx[numb] + T_fase_yy[numb])/2
        phi_3[numb] = (T_fase_xy[numb] - T_fase_yx[numb])/2
        
        beta[numb] = (1/2)*(360/(2*np.pi))*np.arctan(phi_3[numb]/phi_1[numb])
                               
        # phi2 depends on the value of determinant of TP 
        det_phi[numb] = T_fase_xx[numb]*T_fase_yy[numb] - T_fase_xy[numb]*T_fase_yx[numb]

        if det_phi[numb] < 0:

            phi_2[numb] = math.sqrt(abs(det_phi[numb]))
            
            if phi_2[numb]**2 > phi_1[numb]**2 + phi_3[numb]**2: # To verify
                phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(abs(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)))
                phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(abs(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
            else:
                phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
                phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
        
        else: # det_phi[numb] >= 0:
            phi_2[numb] = math.sqrt(det_phi[numb])
            phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
            phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
            
        # for ploting ellipses
        #diam_x[numb] = 2*phi_max[numb]
        #diam_y[numb] = 2*phi_min[numb]
        angle[numb] = alpha[numb]-beta[numb] 
        aux_col[numb] = (np.arctan(phi_min[numb]) + np.pi/2)/np.pi # normalice for a 0 - 1 scale
        e[numb] = phi_min[numb] / phi_max[numb]
        
        ellip_txt.write(str(p)+'\t'+str(lat_dec)+'\t'+str(lon_dec)+'\t'+str(e[numb])+'\t'+str(angle[numb])+'\t'+str(aux_col[numb])+'\n')
    
        numb = numb + 1
    #print(np.mean(alpha))
    #print(np.mean(beta))
    #print(np.mean(phi_max))
    #print(np.mean(phi_min))
    #print(np.mean(aux_col))
    #print(' ')
    ellip_txt.close()

def ellipses_var(Z,H): 

    name = Z[0]
    periods = Z[1]
    zxxr = Z[2]
    zxxi = Z[3]
    zxx = Z[4]
    zxx_var = Z[5]
    zxyr = Z[6]
    zxyi = Z[7]
    zxy = Z[8]
    zxy_var = Z[9]
    zyxr = Z[10]
    zyxi = Z[11]
    zyx = Z[12]
    zyx_var = Z[13]
    zyyr = Z[14]
    zyyi = Z[15]
    zyy = Z[16]
    zyy_var = Z[17]
    
    lat_dec, lon_dec = coord_dms2dec(H)
    
    #################################################
    
    #################################################
    # Calculate: Tensor Ellipses
    
    #Variables
    NUM = len(periods)
    det_R     = np.zeros(len(periods))
    T_fase_xx = np.zeros(len(periods))
    T_fase_xy = np.zeros(len(periods))
    T_fase_yx = np.zeros(len(periods))
    T_fase_yy = np.zeros(len(periods))
    
    det_phi = np.zeros(len(periods))
    phi_max = np.zeros(len(periods))
    phi_min = np.zeros(len(periods))
    beta    = np.zeros(len(periods))
    alpha   = np.zeros(len(periods))
    phi_1   = np.zeros(len(periods))
    phi_2   = np.zeros(len(periods))
    phi_3   = np.zeros(len(periods))

    diam_x = np.zeros(len(periods))
    diam_y = np.zeros(len(periods))
    angle = np.zeros(len(periods))
    e     = np.zeros(len(periods))
    aux_col = np.zeros(len(periods))
    file = name[:len(name)-4]
    
    # Matrix to fill 
    
    Z_ellips = []
    
    numb = 0
    for p in periods:
        
        # Phase tensor 
        det_R[numb] = zxxr[numb]*zyyr[numb] - zxyr[numb]*zyxr[numb] # Determinant
    
        T_fase_xx[numb] = (1/det_R[numb])*(zyyr[numb]*zxxi[numb] - zxyr[numb]*zyxi[numb])
        T_fase_xy[numb] = (1/det_R[numb])*(zyyr[numb]*zxyi[numb] - zxyr[numb]*zyyi[numb])
        T_fase_yx[numb] = (1/det_R[numb])*(zxxr[numb]*zyxi[numb] - zyxr[numb]*zxxi[numb])
        T_fase_yy[numb] = (1/det_R[numb])*(zxxr[numb]*zyyi[numb] - zyxr[numb]*zxyi[numb])
        
        # ellipses components
        # beta[numb] = (1/2)*(360/(2*np.pi))*np.arctan((T_fase_xy[numb]-T_fase_yx[numb]) / (T_fase_xx[numb] + T_fase_yy[numb]))
        alpha[numb] = (1/2)*(360/(2*np.pi))*np.arctan((T_fase_xy[numb] + T_fase_yx[numb])/(T_fase_xx[numb] - T_fase_yy[numb]))

        phi_1[numb] = (T_fase_xx[numb] + T_fase_yy[numb])/2
        phi_3[numb] = (T_fase_xy[numb] - T_fase_yx[numb])/2
        
        beta[numb] = (1/2)*(360/(2*np.pi))*np.arctan(phi_3[numb]/phi_1[numb])
                               
        # phi2 depends on the value of determinant of TP 
        det_phi[numb] = T_fase_xx[numb]*T_fase_yy[numb] - T_fase_xy[numb]*T_fase_yx[numb]

        if det_phi[numb] < 0:

            phi_2[numb] = math.sqrt(abs(det_phi[numb]))
            
            if phi_2[numb]**2 > phi_1[numb]**2 + phi_3[numb]**2: # To verify
                phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(abs(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)))
                phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(abs(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
            else:
                phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
                phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
        
        else: # det_phi[numb] >= 0:
            phi_2[numb] = math.sqrt(det_phi[numb])
            phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
            phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
            
        # for ploting ellipses
        #diam_x[numb] = 2*phi_max[numb]
        #diam_y[numb] = 2*phi_min[numb]
        angle[numb] = alpha[numb]-beta[numb] 
        aux_col[numb] = (np.arctan(phi_min[numb]) + np.pi/2)/np.pi # normalice for a 0 - 1 scale
        e[numb] = phi_min[numb] / phi_max[numb]
        
        Z_ellips_aux = [p, lat_dec, lon_dec, e[numb], angle[numb], aux_col[numb]]
        Z_ellips.append(Z_ellips_aux)
        numb = numb + 1
    return [Z_ellips]

# ==============================================================================
# Plots
# ==============================================================================

def region_map(): 
    
    fig = plt.figure(figsize=(8, 8))

    minLat = -55
    maxLat = -28
    minLon = 150
    maxLon = 199

    m = Basemap(projection='merc', resolution='c',
                width=8E6, height=8E6, 
                lat_0=-38,lon_0=176,
                llcrnrlon=minLon, llcrnrlat=minLat, urcrnrlon=maxLon, urcrnrlat=maxLat)

    m.etopo(scale=0.5, alpha=0.5)
    m.etopo()

    # Map (long, lat) to (x, y) for plotting
    x, y = m(176, -38)
    plt.plot(x, y, 'ok', markersize=5)
    plt.text(x, y, ' Taupo', fontsize=12);

    # draw lat-lon grids
    m.drawparallels(np.linspace(minLat, maxLat, 5), labels=[1,1,0,0], linewidth=0.1)
    m.drawmeridians(np.linspace(minLon, maxLon, 5), labels=[0,0,1,1], linewidth=0.1)
    #m.drawcoastlines() #dessiner les lignes
    
    plt.show()

def plot_ellips_planview(p, ellips_sta_array, frame, path_rest_bound = None):
    
    # ellips_sta[m][n][q], m: station (250),  n: period (52), q: variable (6)
    
    # find the postion of p in the vector of periods
    a = np.where(ellips_sta_array[0,:,0] == p)
    pos_p = a[0]
    
    # Figure (base)
    minLat = frame[0]
    maxLat = frame[1]
    minLon = frame[2]
    maxLon = frame[3]

    lat_0 = ellips_sta_array[0,0,1]
    lon_0 = ellips_sta_array[0,0,2]
    
    f,ax = plt.subplots()
    f.set_size_inches(6,4)

    # m = Basemap(projection='tmerc', resolution='c',
    #             width=8E6, height=8E6, 
    #             lat_0=lat_0,lon_0=lon_0,
    #             llcrnrlon=minLon, llcrnrlat=minLat, urcrnrlon=maxLon, urcrnrlat=maxLat)
    
    #m.etopo(scale=0.5, alpha=0.5)
    #m.etopo()
    #m.drawparallels(np.linspace(minLat, maxLat, 5), labels=[1,1,0,0], linewidth=0.1)
    #m.drawmeridians(np.linspace(minLon, maxLon, 5), labels=[0,0,1,1], linewidth=0.1)

    if path_rest_bound: 
        for path in path_rest_bound:
            lats, lons = np.genfromtxt(path, skip_header=1, delimiter=',').T
            plt.plot(lons, lats, color = 'orange' ,linewidth = 1.0, zorder = 0)
        plt.plot([],[], color = 'orange' ,linewidth = 2, label = 'DC resistivity boundary (Risk, 1984)', zorder = 1)

    #scale = .5e3         ############################ SCALE
    scale = .8e-2         ############################ SCALE

    for n in range(len(ellips_sta_array[:,0,0])): 
        # Variables ellipses
        lat_0=ellips_sta_array[n,pos_p,1]
        lon_0=ellips_sta_array[n,pos_p,2]
        e = ellips_sta_array[n,pos_p,3]
        angle = 90 - ellips_sta_array[n,pos_p,4] # Move to MT reference coord. system  
        color = ellips_sta_array[n,pos_p,5]#[0]
        #x,y=m(lon_0,lat_0)
        x,y=lon_0,lat_0
        ax.plot(x,y,'.', color = pale_blue_col, alpha =0.01)
        # Generated ellipses
        ells = [Ellipse((x, y), width= 1*scale, height=e*scale, angle=angle)]
        color_elip_aux = [1-abs(color/(np.pi/2)), abs(color/(np.pi/2)), 1];
        color_elip = [color_elip_aux[0][0], color_elip_aux[1][0], 1];
        # Plot ellipses
        for a in ells:
            ax.add_patch(a)
            a.set_facecolor(color_elip)
        

    #im = ax.imshow(np.arange(0).reshape((0, 0)),cmap=plt.cm.cool)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=1.5)
    #cbar = plt.colorbar(im)#, cax=cax)
    #cbar.ax.set_yticklabels(['0','','','','','45','','','','','90'])
    #cbar.set_label(r'arctg ($\phi_{min}$)', rotation=90)
    ax.set_title('Period: '+str(round(p,4))+' [s]', fontsize=textsize) 
    ax.grid(True, which='both', linewidth=0.4)
    plt.tight_layout()
    if False:
        plt.legend()
    return f

def plot_ellips_planview_mapBG(p, ellips_sta_array, frame, pic):
    
    # ellips_sta[m][n][q], m: station (250),  n: period (52), q: variable (6)
    
    # find the postion of p in the vector of periods
    a = np.where(ellips_sta_array[0,:,0] == p)
    pos_p = a[0]
    
    # Figure (base)
    minLat = frame[0]
    maxLat = frame[1]
    minLon = frame[2]
    maxLon = frame[3]

    lat_0 = ellips_sta_array[0,0,1]
    lon_0 = ellips_sta_array[0,0,2]
    
    f,ax = plt.subplots()
    f.set_size_inches(18,12)
    f.suptitle('Period: '+str(round(p,4))+' [s]', fontsize=textsize) 
    #implot = ax.imshow(pic, extent=[minLon, maxLon, minLat, maxLat])
    m = Basemap(projection='tmerc', resolution='c',
                width=8E6, height=8E6, 
                lat_0=lat_0,lon_0=lon_0,
                llcrnrlon=minLon, llcrnrlat=minLat, urcrnrlon=maxLon, urcrnrlat=maxLat)
    #m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= False)
    #m.arcgisimage(service=arcgisService, xpixels = 1500, verbose= False)
	#m.etopo(scale=0.5, alpha=0.5)
    #m.etopo()
    m.drawparallels(np.linspace(minLat, maxLat, 5), labels=[1,1,0,0], linewidth=0.1)
    m.drawmeridians(np.linspace(minLon, maxLon, 5), labels=[0,0,1,1], linewidth=0.1)

    scale = .5e3         ############################ SCALE
	
    for n in range(len(ellips_sta_array[:,0,0])): 
        # Variables ellipses
        lat_0=ellips_sta_array[n,pos_p,1]
        lon_0=ellips_sta_array[n,pos_p,2]
        e = ellips_sta_array[n,pos_p,3]
        angle = 90 - ellips_sta_array[n,pos_p,4] # Move to MT reference coord. system  
        color = ellips_sta_array[n,pos_p,5]#[0]
        x,y=m(lon_0,lat_0)
        # Generated ellipses
        ells = [Ellipse((x, y), width= 1*scale, height=e*scale, angle=angle)]
        color_elip_aux = [1-abs(color/(np.pi/2)), abs(color/(np.pi/2)), 1];
        color_elip = [color_elip_aux[0][0], color_elip_aux[1][0], 1];
        # Plot ellipses
        for a in ells:
            ax.add_patch(a)
            a.set_facecolor(color_elip)
            
    im = ax.imshow(np.arange(0).reshape((0, 0)),cmap=plt.cm.cool)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=1.5)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_yticklabels(['0','','','','','45','','','','','90'])
    cbar.set_label('arctg (phi_min)', rotation=90)
    ax.grid(True, which='both', linewidth=0.4)
    
    return f
	
def PT_plot_angle(Z):
    
    name = Z[0]
    periods = Z[1]
    zxxr = Z[2]
    zxxi = Z[3]
    zxx = Z[4]
    zxx_var = Z[5]
    zxyr = Z[6]
    zxyi = Z[7]
    zxy = Z[8]
    zxy_var = Z[9]
    zyxr = Z[10]
    zyxi = Z[11]
    zyx = Z[12]
    zyx_var = Z[13]
    zyyr = Z[14]
    zyyi = Z[15]
    zyy = Z[16]
    zyy_var = Z[17]
    
    mu=4*np.pi/(10^7)       # electrical permeability [Vs/Am]
    omega = 2*np.pi/periods
    cte = 2* (mu/(2*np.pi))*(10^6)
    
    zxx_app_res =mu/omega * np.square(abs(zxx))#mu/omega *(abs(zxy)^2)
    zxx_phase = (360/(2*np.pi)) * np.arctan(zxxi/ zxxr)
    zxx_app_res_error = np.sqrt(cte*periods*np.abs(zxx)*zxx_var)
    zxx_phase_error = np.abs((180/np.pi)*(np.sqrt(zxx_var/2)/np.abs(zxx)))
    
    zxy_app_res =mu/omega * np.square(abs(zxy))#mu/omega *(abs(zxy)^2)
    zxy_phase = (360/(2*np.pi)) * np.arctan(zxyi/ zxyr)
    zxy_app_res_error = np.sqrt(cte*periods*np.abs(zxy)*zxy_var)
    zxy_phase_error = np.abs((180/np.pi)*(np.sqrt(zxy_var/2)/np.abs(zxy)))
    
    zyx_app_res =mu/omega * np.square(abs(zyx))#mu/omega *(abs(zxy)^2)
    zyx_phase = (360/(2*np.pi)) * np.arctan(zyxi/ zyxr)
    zyx_app_res_error = np.sqrt(cte*periods*np.abs(zyx)*zyx_var)
    zyx_phase_error = np.abs((180/np.pi)*(np.sqrt(zyx_var/2)/np.abs(zyx)))
    
    zyy_app_res =mu/omega * np.square(abs(zyy))#mu/omega *(abs(zxy)^2)
    zyy_phase = (360/(2*np.pi)) * np.arctan(zyyi/ zyyr)
    zyy_app_res_error = np.sqrt(cte*periods*np.abs(zyy)*zyy_var)
    zyy_phase_error = np.abs((180/np.pi)*(np.sqrt(zyy_var/2)/np.abs(zyy)))
    
    # Calculate: Tensor Ellipses
    #Variables
    NUM = len(periods)
    det_R     = np.zeros(len(periods))
    T_fase_xx = np.zeros(len(periods))
    T_fase_xy = np.zeros(len(periods))
    T_fase_yx = np.zeros(len(periods))
    T_fase_yy = np.zeros(len(periods))

    phi_max = np.zeros(len(periods))
    phi_min = np.zeros(len(periods))
    beta    = np.zeros(len(periods))
    alpha   = np.zeros(len(periods))
    phi_1   = np.zeros(len(periods))
    phi_2   = np.zeros(len(periods))
    phi_3   = np.zeros(len(periods))
    arctg_phimin = np.zeros(len(periods))
    arctg_phimax = np.zeros(len(periods))

    diam_x = np.zeros(len(periods))
    diam_y = np.zeros(len(periods))
    angle = np.zeros(len(periods))
    e     = np.zeros(len(periods))

    eje_y = np.zeros(len(periods))
    a=range(NUM)
    eje_x = list(a) 
    aux_col = np.zeros(len(periods))
    
    numb = 0
    for p in periods:
        
        # Phase tensor 
        det_R[numb] = zxxr[numb]*zyyr[numb] - zxyr[numb]*zyxr[numb] # Determinant
    
        T_fase_xx[numb] = (1/det_R[numb])*(zyyr[numb]*zxxi[numb] - zxyr[numb]*zyxi[numb])
        T_fase_xy[numb] = (1/det_R[numb])*(zyyr[numb]*zxyi[numb] - zxyr[numb]*zyyi[numb])
        T_fase_yx[numb] = (1/det_R[numb])*(zxxr[numb]*zyxi[numb] - zyxr[numb]*zxxi[numb])
        T_fase_yy[numb] = (1/det_R[numb])*(zxxr[numb]*zyyi[numb] - zyxr[numb]*zxyi[numb])
        
        # ellipses components
        beta[numb] = (1/2)*(360/(2*np.pi))*np.arctan((T_fase_xy[numb]-T_fase_yx[numb]) / (T_fase_xx[numb] + T_fase_yy[numb]))
        alpha[numb] = (1/2)*(360/(2*np.pi))*np.arctan((T_fase_xy[numb] + T_fase_yx[numb])/(T_fase_xx[numb] - T_fase_yy[numb]))
    
        phi_1[numb] = (T_fase_xx[numb] + T_fase_yy[numb])/2
        #if (T_fase_xx[numb]*T_fase_yy[numb] - T_fase_xy[numb]*T_fase_yx[numb]) < 0:
        #        print('Raiz negativa phi2')
        phi_2[numb] = np.real(cmath.sqrt(T_fase_xx[numb]*T_fase_yy[numb] - T_fase_xy[numb]*T_fase_yx[numb]))
        phi_3[numb] = (T_fase_xy[numb] - T_fase_yx[numb])/2
    
        phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
        phi_min[numb] = phi_1[numb]**2 + phi_3[numb]**2 - math.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)

        # for ploting ellipses
        e[numb] = phi_min[numb] / phi_max[numb]
        angle[numb] = alpha[numb]-beta[numb] 
        aux_col[numb] = np.arctan(phi_min[numb]);
        e[numb] = phi_min[numb] / phi_max[numb]
        
        arctg_phimin[numb] = (360/(2*np.pi)) * np.arctan(phi_min[numb])
        arctg_phimax[numb] = (360/(2*np.pi)) * np.arctan(phi_max[numb])
    
        numb = numb + 1
    
    #################################################
    # Plot figure 
    
    f = plt.figure()
    f.set_size_inches(18,12)
    f.suptitle(name[:len(name)-4], fontsize=22)

    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    
    #################################################
    # Subplot 1:  Apparent Resistivity TE and TM   
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    #ax1.loglog(periods,app_res,'r*')
    ax1.errorbar(periods,zxy_app_res,zxy_app_res_error, fmt='ro')
    ax1.errorbar(periods,zyx_app_res,zyx_app_res_error, fmt='bo')
    #ax1.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    #ax1.set_xlim([1e-4, 1e3 + 100])
    ax1.set_ylim([1e-2,1.5e3])
    #ax1.set_xlabel('Period [s]', fontsize=14)
    ax1.set_ylabel('AppRes. [Ohm m]', fontsize=14)
    ax1.legend(['rho TE','rho TM'], loc="upper right")
    ax1.grid(True, which='both', linewidth=0.4)

    #################################################
    # Subplot 2:  Phase TE and TM   

    ax2.set_xscale("log")
    ax2.errorbar(periods,zxy_phase,zxy_phase_error, fmt='ro')
    ax2.errorbar(periods,zyx_phase,zyx_phase_error, fmt='bo')
    ax2.errorbar(periods,arctg_phimin,zxy_phase_error, fmt='go')
    ax2.errorbar(periods,arctg_phimax,zyx_phase_error, fmt='ko')
    #ax2.set_xlim([1e-4, 1e3 + 100])
    #ax2.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    ax2.set_ylim([0,90])
    #ax2.set_xlabel('Period [s]', fontsize=14)
    ax2.set_ylabel('Phase [°]', fontsize=14)
    ax2.legend(['phase TE','phase TM','arctg(phi_min)','arctg(phi_max)'], loc="upper right")
    ax2.grid(True, which='both', linewidth=0.4)
    
    #################################################
    # Subplot 3: Alpha - Beta    
    
    ax3.set_xscale("log")
    ax3.errorbar(periods,angle,zxy_phase_error, fmt='bo')
    #ax3.errorbar(periods,zyx_phase-180,zyx_phase_error, fmt='bo')
    #ax3.set_xlim([1e-4, 1e3 + 100])
    #ax3.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    ax3.set_ylim([-90,90])
    #ax3.set_xlabel('Period [s]', fontsize=14)
    ax3.set_ylabel('alpha - beta [°]', fontsize=14)
    ax3.legend(['alpha - beta'], loc="upper right")
    ax3.grid(True, which='both', linewidth=0.4)
    
    #################################################
    # Subplot 4: Beta    
    
    ax4.set_xscale("log")
    ax4.errorbar(periods,beta,zxy_phase_error, fmt='ro')
    #ax3.errorbar(periods,zyx_phase-180,zyx_phase_error, fmt='bo')
    #ax4.set_xlim([1e-4, 1e3 + 100])
    #ax3.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    ax4.set_ylim([-20,20])
    ax4.set_xlabel('Period [s]', fontsize=14)
    ax4.set_ylabel('beta [°]', fontsize=14)
    ax4.legend(['beta'], loc="upper right")
    ax4.grid(True, which='both', linewidth=0.4)

    #plt.show()
    return f

# ==============================================================================
# Coordinates
# ==============================================================================

def coord_dms2dec(H):
    lat = H[2]
    lon = H[3]
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
