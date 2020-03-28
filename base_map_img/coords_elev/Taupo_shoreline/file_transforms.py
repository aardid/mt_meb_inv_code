# ==============================================================================
#  Imports
# ==============================================================================
import numpy as np
textsize = 15.
# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
    tav2coma = True

    import re
    if tav2coma:
        # load file
        #taupo_sl_coords = np.genfromtxt('taupo_ol_NZMG.dat', delimiter = '\t')
        file1 = open("taupo_ol_NZMG.txt","w")
        with open('taupo_ol_NZMG.dat') as f:
            for line in f:
                lines = line.rstrip('\n')
                file1.write("{:.3f}".format(lines[:)+','+"{:.3f}".format(taupo_sl_coords[f][1])+'\n')
            asdf
        
        sdfg
        nc = len(taupo_sl_coords)
        
        file1 = open("taupo_ol_NZMG.txt","w")
        for f in range(nc):
            if taupo_sl_coords[f][0] is None:
                pass
            else:
                file1.write("{:.3f}".format(taupo_sl_coords[f][0])+','+"{:.3f}".format(taupo_sl_coords[f][1])+'\n')

        file1.close()   
