# ==============================================================================
#  Imports
# ==============================================================================
import numpy as np
textsize = 15.
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
    split = False
    merge = False
    crop = True

    if split:
        # load file
        nzgm_coords = np.genfromtxt('Topography_25_m_vertices_NZMG.csv', delimiter = ',', skip_header = 1)
        
        nc = len(nzgm_coords)
        nf = 5 # number of batchs to be splited
        nb = 50000 # max nomber per batch 
        # loop to create new files 
        kloop = True
        line_count = 0

        for f in range(nf):
            file1 = open(str(f+1)+"_batch_nzgm_coords.txt","w")
            #print(f+1)
            for b in range(nb):
                    file1.write("{:.3f}".format(nzgm_coords[line_count][0])+','+"{:.3f}".format(nzgm_coords[line_count][1])+'\n')
                    line_count += 1
                    if line_count == nc:
                        break
            file1.close()   

    if merge: 
        # merge batch latlondec files into one and add elec from original (Topo...)
        file1 = open("Topography_25_m_vertices_LatLonDec.csv","w") # file to be written 
        file1.write('lat_dec, lon_dec, elev [m] \n')
        # open original 
        nzgm_coords = np.genfromtxt('Topography_25_m_vertices_NZMG.csv', delimiter = ',', skip_header = 1)
        # variables 
        nc = len(nzgm_coords)
        nf = 5 # number of batchs to be splited
        nb = 50000 # max nomber per batch 
        # loop to create new files 
        kloop = True
        line_count_gen = 0
        for f in range(nf):
            latlon_coords = np.genfromtxt(str(f+1)+"_batch_latlondec_coords.csv", delimiter = ',', skip_header = 1)
            line_count_par = 0
            for b in range(nb):
                file1.write("{:.3f}".format(latlon_coords[line_count_par][0])+','+"{:.3f}".format(latlon_coords[line_count_par][1])+','+"{:.2f}".format(nzgm_coords[line_count_gen][2])+'\n')
                line_count_par += 1
                line_count_gen += 1
                if line_count_gen == nc:
                    break
        file1.close() 
            
    if crop: 
        # open original 
        latlon_coords = np.genfromtxt('Topography_25_m_vertices_LatLonDec.csv', delimiter = ',', skip_header = 1)
        # filter per coords
        x_lim = [175.935, 176.255]
        y_lim = [-38.77,-38.545]
        rsample = np.arange(0,len(latlon_coords),100)
        file1 = open("Topography_zoom_WT_re_sample_vertices_LatLonDec.csv","w") # file to be written 
        file1.write('lat_dec, lon_dec, elev [m] \n')
        count = 0
        for i, coord in enumerate(latlon_coords): 
            if i in rsample:
                if (coord[1]>x_lim[0] and coord[1]<x_lim[1] and coord[0]>y_lim[0] and coord[0]<y_lim[1]):
                    if coord[2] != 357.00: # surface of lake Taupo
                        file1.write("{:.3f}".format(latlon_coords[i][0])+','+"{:.3f}".format(latlon_coords[i][1])+','+"{:.2f}".format(latlon_coords[i][2])+'\n')
        file1.close()   
