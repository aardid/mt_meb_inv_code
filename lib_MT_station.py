"""
- Module EDI: Library for MT station. 
    - MT stations class
    - Functions to deal with edi files
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2018
.. conventions:: 
	:order in impedanze matrix [xx,xy,yx,yy]
"""
# ==============================================================================
#  Imports
# ==============================================================================

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
#from math import sqrt
#from cmath import sqrt as csqrt
import glob
from matplotlib.backends.backend_pdf import PdfPages
#from mpl_toolkits.basemap import pyproj
#from mpl_toolkits.basemap import Basemap
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from scipy.linalg import solve
from numpy import linalg as LA
from Maping_functions import*

# ==============================================================================
# MT stations class
# ==============================================================================

class Station(object): 
    """
    This class is for MT stations to be inverted.
	====================    ========================================== ==========
    Attributes              Description                                default
    =====================   ========================================== ==========
	name				    extracted from the name of the edi file
    ref		 			    reference number in the code
	path				    path to the edi file
	
	lat					    latitud	in dd:mm:ss
    lon					    longitud in dd:mm:ss
	lat_dec				    latitud	in decimal
    lon_dec				    longitud in decimal	
	elev					topography (elevation of the station)
	nearest_wl			    nearest wells (ref of wells in the code) 
	
	T						recorded periods
    num_T                   number of periods
	Z						impedanse tensor
	Z_xx			 		impedanse xx [real, img, mag, var]
	Z_xy					impedanse xy [real, img, mag, var]
	Z_yx					impedanse yx [real, img, mag, var]
	Z_yy					impedanse yy [real, img, mag, var]
	rho_app				    apparent resistivity (for 4 components of Z) 
	phase_deg				phase in degrees(for four components of Z)
	*rho_app_er		        error bar for apparent resistivity (for 4 components of Z) 
	*phase_deg_er			error bar for phase in degrees(for four components of Z)
	det_Z				    determinant of Z (magnitude) (rotational invariant)
	max_Z					maximum value of Z (magnitud) (rotational invariant)
    ssq_Z                   sum of squared elements of Z (rotational invariant)
	Z_strike				strike of  Z 
	Z_skew				    skew of Z 
	Zellip				    Z ellip (?)
	tip 					Tipper [txr, txi, txvar, tyr, tyi, tyvar, tmag]
	tip_mag				    tipper magnitud
	tip_phase				tipper phase
	tip_skew				tipper skew
	tip_strike			    tipper strike

    prior_meb_wl_names      names of nearest MeB wells consired in MeB prior
                            ['well 1', ..., 'well 4']
    prior_meb               mean and std for normal prior in station (for z1 and z2)
                            [[mean_z1, std_z1], [mean_z2, std_z2]
    prior_meb_wl_dist       distance to each wells to be consired in MeB prior
                            [dist well 1, ..., dist well 4']

	layer_mod				1D resistivity 3 layer model [z1,z2,r1,r2,r3]
    z1_pars                 distribution parameters for layer 1 
                            thickness (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d,e]
                            a: mean
                            b: standard deviation 
                            c: median
                            d: percentiles # [5%, 10%, ..., 95%] (19 elements)
    z2_pars                 distribution parameters for layer 2 
                            thickness (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    r1_pars                 distribution parameters for layer 1 
                            resistivity (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    r2_pars                 distribution parameters for layer 2 
                            resistivity (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    r3_pars                 distribution parameters for layer 3 
                            resistivity (model parameter) calculated 
                            from mcmc chain results: [a,b,c,d]
                            * See z1_pars vector description 
    af_mcmcinv              acceptance fraction of mcmc inversion 
                            [mean std thick1 thick2 res1 res2 res3] 
    act_mcmcinv             autocorrelation time of mcmc inversion 
                            [mean std thick1 thick2 res1 res2 res3] 

	temp_prof				temperaure profile

    betas_3l                beta normal distribution values for 3 layer model  
                            [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
    Tmin_3l                 Bottom temperature normal distribution values for 3 layer model 
                            [[mean_Tmin1, std_Tmin1],[mean_Tmin2, std_Tmin2],[mean_Tmin3, std_Tmin3]]
    Tmax_3l                 Top temperature normal distribution values for 3 layer model 
                            [[mean_Tmax1, std_Tmax1],[mean_Tmax2, std_Tmax2],[mean_Tmax3, std_Tmax3]]
    max_depth_temp          maximum depth for temp. profile based on weigthed average of nearest wells
                            Defined in calc_beta_sta_quadrant function 

    =====================   =====================================================
    Methods                 Description
    =====================   =====================================================
	read_edi_file	
	rotate_Z
	app_res_phase
    """
	
    def __init__(self, name, ref, path):
	# ==================== 
    # Attributes            
    # ===================== 
        self.name = name # name: extracted from the name of the edi file
        self.ref = ref	 # ref: reference number in the code
        self.path = path # path to the edi file
        ## Properties to be fill 
        # Position 
        self.lat = None 		# latitud in dd:mm:ss	
        self.lon = None 		# longitud in dd:mm:ss
        self.lat_dec = None 	# latitud in decimal	
        self.lon_dec = None 	# longitud  in decimal
        self.elev = None 		# topography (elevation of the station)
        self.nearest_wl = None  # nearest wells (ref of wells in the code, 2 wells) 
        # MT data
        self.T = None 			# recorded periods
        self.num_T = None       # number of periods 
        self.Z = None			# impedanse tensor 
        self.Z_xx = None		# impedanse xx [real, img, mag, var]
        self.Z_xy = None		# impedanse xy [real, img, mag, var]
        self.Z_yx = None		# impedanse yx [real, img, mag, var]
        self.Z_yy = None		# impedanse yy [real, img, mag, var]
        self.rho_app = None		# apparent resistivity (for four components of Z) 
        self.phase_deg = None	# phase (for four components of Z)
        self.rho_app_er = None	# apparent resistivity (for four components of Z) 
        self.phase_deg_er = None# phase (for four components of Z)
        self.max_Z = None		# maximum value of Z (rotational invariant)
        self.det_Z = None		# determinant of Z (rotational invariant)
        self.ssq_Z = None		# sum of squared elements of Z (rotational invariant)
        self.Z_strike = None	# strike of  Z 
        self.Z_skew = None		# skew of Z 
        self.Z_ellip = None		# Z ellip
        self.tip = None 		# Tipper [txr, txi, txvar, tyr, tyi, tyvar, tmag]	
        self.tip_mag = None		# tipper magnitud
        self.tip_phase = None	# tipper phase
        self.tip_skew = None	# tipper skew
        self.tip_strike = None	# tipper strike
		# MT estimates
        self.layer_mod = None		# 1D resistivity 3 layer model [z0,z1,z2,r1,r2,r3]
        self.z1_pars = None
        self.z2_pars = None
        self.r1_pars = None
        self.r2_pars = None
        self.r3_pars = None
		# Temperature estimates
        self.temp_prof = None		# temperaure profile
        self.slopes = None			# slopes values for each layer
        self.betas_3l = None        # beta normal distribution values for 3 layer model
        self.Tmin_3l = None         # bottom temperature normal distribution values for 3 layer model 
        self.Tmax_3l = None         # top temperature normal distribution values for 3 layer model 
        # MeB prior
        self.prior_meb_wl_names = None  # name of wells to be consired in MeB prior ()
        self.prior_meb = None           # distance to each wells to be consired in MeB prior
        self.prior_meb_wl_dist = None   # mean and std for normal prior in station (for z1 and z2) [ unit]

    # ===================== 
    # Methods               
    # =====================
    def read_edi_file(self):
		# Read edi file and set MT data atributes to object  
        pos_ast = self.path.find('*')
        [H, Z, T, Z_rot, Z_dim] = read_edi(self.path[:pos_ast]+self.name) 

		## Z = [file,periods,zxxr,zxxi,zxx,zxx_var,zxyr,zxyi,zxy,zxy_var,zyxr,zyxi,zyx,zyx_var,zyyr,zyyi,zyy,zyy_var]		
		## T = [txr, txi, txvar, tyr, tyi, tyvar, tmag]
		## Z_rot = [zrot]
		## Z_dim = [zskew, tstrike]
        self.lat = H[2]
        self.lon = H[3]
        self.lat_dec, self.lon_dec = coord_dms2dec(H)
        self.elev = H[4]
        self.Z = Z
        self.T = Z[1]
        self.num_T = len(Z[1])
        self.Z_xx = np.array([Z[2],Z[3],Z[4],Z[5]])
        self.Z_xy = np.array([Z[6],Z[7],Z[8],Z[9]])
        self.Z_yx = np.array([Z[10],Z[11],Z[12],Z[13]])
        self.Z_yy = np.array([Z[14],Z[15],Z[16],Z[17]])
        self.Z_skew = Z_dim[0]
        self.Z_strike = Z_rot[0]
        self.max_Z = self.calc_max_Z()
        self.det_Z = self.calc_det_Z()
        self.ssq_Z = self.calc_ssq_Z()
        self.tip = T[:-1]
        self.tip_mag = T[-1]
        self.tip_strike = Z_dim[1]
    
    def read_PT_Z(self, pc = None):
        # Read derotated data file of Z 
        # format: per Z11 Z21 Z12 Z22 err11 err21 err12 err22 
        if pc == "personalSuse":
            path = "/home/aardid/Documentos/data/Wairakei_Tauhara/MT_Survey/Unrotated_MTdata_PT/Z_Files"+os.sep+self.name[:-4]+".Z"
        if pc == "office": 
            path = "D:\workflow_data\MTdata_PT_unrotated\Z_Files"+os.sep+self.name[:-4]+".Z"

        Z_dr = np.genfromtxt(path).T
        self.T = Z_dr[0]
        # Zxx
        self.Z_xx[0] = Z_dr[1]
        self.Z_xx[1] = Z_dr[2]
        self.Z_xx[2] = [complex(Z_dr[1][i], Z_dr[2][i]) for i in range(len(Z_dr[0]))]
        self.Z_xx[3] = Z_dr[9]
        # Zyx
        self.Z_yx[0] = Z_dr[3]
        self.Z_yx[1] = Z_dr[4]
        self.Z_yx[2] = [complex(Z_dr[3][i], Z_dr[4][i]) for i in range(len(Z_dr[0]))]
        self.Z_yx[3] = Z_dr[10]
        # Zxy
        self.Z_xy[0] = Z_dr[5]
        self.Z_xy[1] = Z_dr[6]
        self.Z_xy[2] = [complex(Z_dr[5][i], Z_dr[6][i]) for i in range(len(Z_dr[0]))]
        self.Z_xy[3] = Z_dr[11]
        # Zxx
        self.Z_yy[0] = Z_dr[7]
        self.Z_yy[1] = Z_dr[8]
        self.Z_yy[2] = [complex(Z_dr[7][i], Z_dr[8][i]) for i in range(len(Z_dr[0]))]
        self.Z_yy[3] = Z_dr[12]

    def rotate_Z(self):
		## 2. Rotate Z to North (0°)
        #alpha = -1* self.Z_strike # Z_rot[0]  # rotate to cero (north)
        alpha = 360. - self.Z_strike # Z_rot[0]  # rotate to cero (north)
        #alpha = 0.*self.Z_strike 
        [Z_prime] = rotate_Z(self.Z, alpha)
        Z = Z_prime # for consider rotation 
        self.Z = Z
        self.Z_xx = [Z[2],Z[3],Z[4],Z[5]]
        self.Z_xy = [Z[6],Z[7],Z[8],Z[9]]
        self.Z_yx = [Z[10],Z[11],Z[12],Z[13]]
        self.Z_yy = [Z[14],Z[15],Z[16],Z[17]]

    def app_res_phase(self): 
		# Calculate apparent resistivity and phase for Z 
		## rho_app = [app_res_xx, app_res_xy, app_res_yx, app_res_yy]
		## phase_deg = [phase_de_xx, phase_de_xy, phase_de_yx, phase_de_yy]
        [self.rho_app, self.phase_deg, self.rho_app_er, self.phase_deg_er] = calc_app_res_phase(self.Z)

    def plot_app_res_phase(self): 
        # plot apparent resistivity and phase (4 components of the tensor)
        f = plot_Z_appres_phase(self.Z, title = self.name)
        return f

    def calc_max_Z(self): 
        max_Z = np.zeros(self.num_T) 
        for i in range(self.num_T):
            a = [abs(self.Z_xx[2,i]),abs(self.Z_xy[2,i]),abs(self.Z_yx[2,i]),abs(self.Z_yy[2,i])]
            max_Z[i] = np.max(a)
        return max_Z

    def calc_det_Z(self): 
        det_Z = np.zeros(self.num_T) 
        for i in range(self.num_T):
            a = (self.Z_xx[2,i]*self.Z_yy[2,i] - self.Z_yx[2,i]*self.Z_xy[2,i])
            #det_Z[i] = csqrt(a)
            det_Z[i] = np.sqrt(a)
        return det_Z

    def calc_ssq_Z(self): 
        ssq_Z = np.zeros(self.num_T) 
        for i in range(self.num_T):
            a = ((pow(self.Z_xx[2,i],2)+pow(self.Z_xy[2,i],2)+pow(self.Z_yx[2,i],2)+pow(self.Z_yy[2,i],2))/2)
            #ssq_Z[i] = csqrt(a)
            ssq_Z[i] = np.sqrt(a)
        return ssq_Z

    def temp_prof_est(self, plot_samples = None, ret_fig = None, Ns = None):
        """
        Estimate temperature profile in station position, based on 
        1D heat transfer model (See Bredehoeft, J. D., and I. S. Papadopulos (1965).
        Boundary conditions (betas, Tmins and Tmaxs) are calculated in calc_beta_sta_quadrant(). 
        Zmins and Zmaxs are extracted from layer model in station. 

        Inputs
        -----
        Ns: number of samples to generate
        plot
        plot_samples: create a figure of temperature profile samples
            save in path_temp_est
        ret_fig : return figure
        
        Attributes generated
        --------------------
        self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'+os.sep+self.name[:-4]

        """
        ## number of samples 
        if Ns is None:
            Ns = 100
        if os.path.isdir('.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'):
            pass
        else:
            os.mkdir('.'+os.sep+'temp_prof_samples'+os.sep+'MTstation')

        # directory to save results
        if os.path.isdir( '.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'+os.sep+self.name[:-4]):
            self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'+os.sep+self.name[:-4]
        else:
            os.mkdir('.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'+os.sep+self.name[:-4])
            self.path_temp_est = '.'+os.sep+'temp_prof_samples'+os.sep+'MTstation'+os.sep+self.name[:-4]
	
        # path for folder of the station 
        path = self.path_temp_est
        # extract mcmc inv results from file (z1 and z2 dist.) 
        mcmc_inv_results = np.genfromtxt('.'+os.sep+'mcmc_inversions'+os.sep+self.name[:-4]+os.sep+"est_par.dat")
        # values for mean a std for normal distribution representing the prior
        z1_mean, z1_std = mcmc_inv_results[0,1:3] # mean [1] z1 #  std [1] z1 # median [3] z1 
        z2_mean, z2_std = mcmc_inv_results[1,1:3] # mean [1] z2 #  std [1] z2 # median [3] z2 
        
        # maximum depth for profile -> thicknesses of two first layers plus 500 meters (arbitrary) 
        #max_depth = abs(z1_mean + z2_mean + 300.) # need to be checked (too arbitrary)
        max_depth = abs(self.max_depth_temp)
        # depths vector where temps are going to be estimated
        # zj = np.linspace(0.,max_depth,20) # depth profile sample. Set to the depth of the nearest well 
        delta_z = 2. 
        zj = np.arange(0.,max_depth , delta_z)

        ## text file of Test samples 
        t = open(path+os.sep+'temp_est_samples.txt', 'w')
        t.write('# Temperature profile samples: each line is a profile (starting from surface) \n')
        t.write('# First line correspond to depth values for temp. sample profiles \n')
        for z in zj:
            t.write('{}\t'.format(z))
        t.write('\n')
        if plot_samples:
            f,(ax1) = plt.subplots(1,1)
            f.set_size_inches(6,8)
            ax1.set_xscale("linear")
            ax1.set_yscale("linear")    
            ax1.set_xlabel('Temperature [deg C]', fontsize=14)
            ax1.set_ylabel('Depth [m]', fontsize=14)
            ax1.grid(True, which='both', linewidth=0.4)
            ax1.set_xlim([0.,300.])
            ax1.invert_yaxis()
            plt.title(self.name[:-4], fontsize=18,)
            # plot resistivity model  
            self.elev = float(self.elev)
            # upper bound
            ax1.plot([-5.,300.], [z1_mean, z1_mean],'y-', alpha = .5)
            ax1.plot([-5.,300.], [z1_mean + z1_std, z1_mean + z1_std],'y--', alpha = .3)
            ax1.plot([-5.,300.], [z1_mean - z1_std, z1_mean - z1_std],'y--', alpha = .3)
            # lower bound
            ax1.plot([-5.,300.], [z1_mean + z2_mean, z1_mean + z2_mean],'g-', alpha = .5)
            ax1.plot([-5.,300.], [z1_mean + z2_mean + z2_std, z1_mean + z2_mean + z2_std],'g--', alpha = .3)
            ax1.plot([-5.,300.], [z1_mean + z2_mean - z2_std, z1_mean + z2_mean - z2_std],'g--', alpha = .3)

        z1_z1_not_valid = True
        for i in range(Ns): 
            if True: # sample boundary conditions and betas
                ## Spatial boundary conditions (z0,z1,z2,z3)
                # z0 (surface) and z3 (bottom of third layer)
                z0, z3 = zj[0], zj[-1]
                # samples of z1 and z2 (top and bottom boundary of CC)
                z1 = np.abs(np.random.normal(z1_mean, z1_std, 1))+1. # 
                z2 = np.abs(np.random.normal(z2_mean, z2_std, 1))+1. # 
                while z1_z1_not_valid:
                    # condition for samples: sum of z1 (thick1) and z2(thick2) can't be larger than max depth of resample prof. 
                    if z1 + z2 < max_depth:
                        z1_z1_not_valid = False
                    else: 
                        z1 = np.abs(np.random.normal(z1_mean, z1_std, 1))+1.  
                        z2 = np.abs(np.random.normal(z2_mean, z2_std, 1))+1.
                z1_z1_not_valid = True 
                # 
                z1_aux = z1[0]
                z2_aux = z1_aux + z2[0]
                ## Temperature boundary conditions (T0,T1,T2,T3)
                T0_mean, T0_std = self.Tmin_3l[0]
                T1_mean, T1_std = self.Tmin_3l[1]
                T2_mean, T2_std = self.Tmin_3l[2]
                T3_mean, T3_std = self.Tmax_3l[2]
                # sample values
                T0 = np.abs(np.random.normal(T0_mean, T0_std + .1, 1)) # 
                T1 = np.abs(np.random.normal(T1_mean, T1_std, 1)) # 
                T2 = np.abs(np.random.normal(T2_mean, T2_std, 1)) # 
                T3 = np.abs(np.random.normal(T3_mean, T3_std, 1)) # 
                ## Beta for each layer
                b1_mean, b1_std = self.betas_3l[0]
                b2_mean, b2_std = self.betas_3l[1]
                b3_mean, b3_std = self.betas_3l[2]
                # sample values
                b1 = -1.*np.random.normal(b1_mean, b1_std, 1) # 
                b2 = -1.*np.random.normal(b2_mean, b2_std, 1) # 
                b3 = -1.*np.random.normal(b3_mean, b3_std, 1) # 
            ## construct profile
            # layer 1
            val, idx1 = find_nearest(zj, z1_aux)
            z_vec1 = zj[0:idx1+1]
            Test_l1 = Texp2(z_vec1,z0,z1_aux,T0,T1,b1)
            # layer 2
            val, idx2 = find_nearest(zj, z2_aux)
            z_vec2 = zj[idx1+1:idx2+1]
            Test_l2 = Texp2(z_vec2,z1_aux,z2_aux,T1,T2,b2)
            # layer 3
            z_vec3 = zj[idx2+1:len(zj)]
            Test_l3 = Texp2(z_vec3,z2_aux,z3,T2,T3,b3)

            # concatenate the estimated curves
            Test = np.concatenate((Test_l1, Test_l2,Test_l3), axis=0)

            # if z1_aux and z2_aux are similar:  construct the tempo profile using one layer
            # with boundary conditions for layer one 
            if z2_mean < 50.:
                Test = Texp2(zj,z0,z3,T0,T1,b1)

            for item in Test:
                t.write('{}\t'.format(item))
            t.write('\n')
           
            if plot_samples: # plot sample temperature profile
                ax1.plot(Test[:len(zj)] ,zj,'r-', linewidth =.5,  alpha=0.1)
                # if len(Test) == len(zj):
                #     ax1.plot(Test ,zj,'r-', linewidth =.5,alpha=0.1)
                # else:
                #     pass

        if plot_samples:
            ax1.plot(Test[:len(zj)] ,zj,'r-', linewidth =.5, alpha=0.1 ,label = 'sample')
            ax1.legend()
            plt.tight_layout()
            f.savefig("Test_samples.png", bbox_inches='tight')
            shutil.move('Test_samples.png',path+os.sep+'Test_samples.png')
            if ret_fig:
                return f
            else: 
                plt.close("all")
        t.close()

    def uncert_isotherms_percentils(self, isotherms = None, percentiels = None):
        """
        Calculate isotherm percetils (depths) from temperature profile samples of the station
        (assinged in temp_prof_est(self)). Save them in text file.

        Inputs
        -----------------
        isotherms: array of isotherms to calculate the depth percentils 
            isotherms = [120,150]
        percentiels: array of wanted percentiels
            percentiels = np.arange(5.,100.,45)

        File to be create
        -----------------
        'isotherms_percectils.txt'
        location: self.path_temp_est (assinged in temp_prof_est(self)
        Example of file format: 
            # Isotherms:	5.0%	50.0%		95.0%	
            120:	409.8	410.2	410.7
            150:	509.8	510.3	510.9	
        """
        if isotherms is None: 
            isotherms = [120,150,180,210,240]
        if percentiels is None: 
            percentiels = np.arange(5.,100.,5.)
        #####
        # path of folder samples  
        path = self.path_temp_est
        # create text file to save results 
        iso_perc = open(path+os.sep+'isotherms_percectils.txt', 'w')
        iso_perc.write('# Isotherms:\t')
        for per in percentiels: 
            iso_perc.write('{}%\t'.format(per))
        iso_perc.write('\n')
        # read the samples
        t_samples = np.genfromtxt(path+os.sep+'temp_est_samples.txt')
        d_samples = t_samples[0,:] # depths for samples 
        # loop over the isotherms 
        for iso in isotherms: # isotherms
            list_iso_z = []	
            # loop over samples
            for l in range(len(t_samples[:,0])-1): # -1 because first line are depths
                # extract isotherm value, add to list
                sample = t_samples[l+1,:]
                val, idx = find_nearest(sample, iso)
                # z value for idx
                z_iso = d_samples[idx]
                # add to list of depths with same isotherm
                list_iso_z.append(z_iso)
            list_iso_z = np.asarray(list_iso_z)
            # calc percentils of list
            per_iso = [np.percentile(list_iso_z,per) for per in percentiels] 
            # print in file 
            iso_perc.write('{}\t'.format(iso))
            for per in per_iso:
                iso_perc.write('{:3.1f}\t'.format(per))
            iso_perc.write('\n')
        iso_perc.close()   

# ==============================================================================
# Read EDI
# ==============================================================================
def read_edi(file):
    "Funcion que lee .edi y devuelve parametros de MT."
    infile = open(file, 'r')
    infile2 = open(file, 'r')

    n_linea = 1
    freq = []
    periods= []
    zrot = []
    
    zxxr = []
    zxxi = []
    zxx_var = []
    
    zxyr = []
    zxyi = []
    zxy_var = []

    zyxr = []
    zyxi = []
    zyx_var = []
    
    zyyr = []
    zyyi = []
    zyy_var = []
    
    txr = []
    txi = []
    txvar = []
    tyr = []
    tyi = []
    tyvar = []
    tmag = []
	
    zskew = []
    tstrike = []
    
    Z = [[periods],[zxyr],[zxyi],[zxy_var],[zyxr],[zyxi],[zyx_var]]

    # Search for the first and last line containing each element of the impedance tensor   
    for line in infile:
        #################################################
        #Buscar segmentos del header y tensor
        #################################################
        
        ################ Position 
        # LOC
        if "REFLOC" in line:
            n_linea_scan_loc = n_linea
        # LAT
        if "REFLAT" in line:
            n_linea_scan_lat = n_linea
        # LON
        if "REFLON" in line:
            n_linea_scan_lon = n_linea
        # ELEV
        if "REFELEV" in line:
            n_linea_scan_elev = n_linea
        
        ################ Frequency, rotation        
        #FREQ
        if ">FREQ" in line:
            n_linea_scanfrom_freq = n_linea + 1
        else: 
            if ">ZROT" in line: #">ZXXR" in line:
                n_linea_scanto_freq = n_linea - 2  
        #ZROT
        if ">ZROT" in line:
            n_linea_scanfrom_zrot = n_linea + 1
        else: 
            if ">ZXXR" in line: #">ZXXR" in line:
                n_linea_scanto_zrot  = n_linea - 2
					
        ################ ZXX
        #ZXXR
        if ">ZXXR" in line:
            n_linea_scanfrom_zxxr = n_linea + 1
        else: 
            if ">ZXXI" in line:
                n_linea_scanto_zxxr = n_linea - 1      
        #ZXXI
        if ">ZXXI" in line:
            n_linea_scanfrom_zxxi = n_linea + 1
        else: 
            if ">ZXX.VAR" in line:
                n_linea_scanto_zxxi = n_linea - 1
        #ZXX VAR
        if ">ZXX.VAR" in line:
            n_linea_scanfrom_zxxvar = n_linea + 1
        else: 
            if ">ZXYR" in line:
                n_linea_scanto_zxxvar = n_linea - 1 

        ################ ZXY
        #ZXYR
        if ">ZXYR" in line:
            n_linea_scanfrom_zxyr = n_linea + 1
        else: 
            if ">ZXYI" in line:
                n_linea_scanto_zxyr = n_linea - 1      
        #ZXYI
        if ">ZXYI" in line:
            n_linea_scanfrom_zxyi = n_linea + 1
        else: 
            if ">ZXY.VAR" in line:
                n_linea_scanto_zxyi = n_linea - 1
        #ZXY VAR
        if ">ZXY.VAR" in line:
            n_linea_scanfrom_zxyvar = n_linea + 1
        else: 
            if ">ZYXR" in line:
                n_linea_scanto_zxyvar = n_linea - 1
            
        ################ ZYX 
        if ">ZYXR" in line:
            n_linea_scanfrom_zyxr = n_linea + 1
        else: 
            if ">ZYXI" in line:
                n_linea_scanto_zyxr = n_linea - 1      
        #ZXYI
        if ">ZYXI" in line:
            n_linea_scanfrom_zyxi = n_linea + 1
        else: 
            if ">ZYX.VAR" in line:
                n_linea_scanto_zyxi = n_linea - 1
        #ZYX VAR
        if ">ZYX.VAR" in line:
            n_linea_scanfrom_zyxvar = n_linea + 1
        else: 
            if ">ZYYR" in line:
                n_linea_scanto_zyxvar = n_linea - 1
                
        ################ ZYY
        if ">ZYYR" in line:
            n_linea_scanfrom_zyyr = n_linea + 1
        else: 
            if ">ZYYI" in line:
                n_linea_scanto_zyyr = n_linea - 1      
        #ZYYI
        if ">ZYYI" in line:
            n_linea_scanfrom_zyyi = n_linea + 1
        else: 
            if ">ZYY.VAR" in line:
                n_linea_scanto_zyyi = n_linea - 1
        #ZYY VAR
        if ">ZYY.VAR" in line:
            n_linea_scanfrom_zyyvar = n_linea + 1
        else: 
            if ">RHOROT" in line:
                n_linea_scanto_zyyvar = n_linea - 1
            #else: 
            #    if ">!****TIPPER PARAMETERS****!" in line:
            #        n_linea_scanto_zyyvar = n_linea - 1
        
        #############################################
        # Tipper
        
        #TXR
        if ">TXR" in line:
            n_linea_scanfrom_txr = n_linea + 1
        else: 
            if ">TXI" in line:
                n_linea_scanto_txr = n_linea - 1
        #TXI
        if ">TXI" in line:
            n_linea_scanfrom_txi = n_linea + 1
        else: 
            if ">TXVAR" in line:
                n_linea_scanto_txi = n_linea - 1
        #TXVAR
        if ">TXVAR" in line:
            n_linea_scanfrom_txvar = n_linea + 1
        else: 
            if ">TYR" in line:
                n_linea_scanto_txvar = n_linea - 1
        #TYR
        if ">TYR" in line:
            n_linea_scanfrom_tyr = n_linea + 1
        else: 
            if ">TYI" in line:
                n_linea_scanto_tyr = n_linea - 1
        #TYI
        if ">TYI" in line:
            n_linea_scanfrom_tyi = n_linea + 1
        else: 
            if ">TYVAR" in line:
                n_linea_scanto_tyi = n_linea - 1
        #TYVAR
        if ">TYVAR" in line:
            n_linea_scanfrom_tyvar = n_linea + 1
        else: 
            if (">TIPMAG" in line and ">TIPMAG.VAR" not in line):
                n_linea_scanto_tyvar = n_linea - 1
            #else: 
            #    if ">END" in line:
            #        n_linea_scanto_tyvar = n_linea - 2
			
        #TIPMAG
        if ">TIPMAG " in line:
            n_linea_scanfrom_TIPMAG = n_linea + 1
        else: 
            if ">TIPMAG.VAR" in line:
                n_linea_scanto_TIPMAG  = n_linea - 1
				
        ################ Dimensional parameters     
        #ZSKEW
        if ">ZSKEW" in line:
            n_linea_scanfrom_ZSKEW = n_linea + 1
        else: 
            if ">ZELLIP" in line: #">ZXXR" in line:
                n_linea_scanto_ZSKEW = n_linea - 1 
				
        #TSTRIKE
        if ">TSTRIKE" in line:
            n_linea_scanfrom_TSTRIKE = n_linea + 1
        else: 
            if ">TSKEW" in line: #">ZXXR" in line:
                n_linea_scanto_TSTRIKE = n_linea - 1 
                
        #################################################
        n_linea=n_linea + 1 

    #print(n_linea_scanfrom_zyyvar)
    #print(n_linea_scanto_zyyvar)

    n_linea = 1
    for line in infile2:
        #################################################
        # Crear elementos y arreglos con variables del tensor 
        ################################################# 
        # Position
        #loc
        if n_linea == n_linea_scan_loc:
            pos_equal = line.find('=')
            loc = line[pos_equal+2:len(line)-2]

        #lat
        if n_linea == n_linea_scan_lat:
            pos_equal = line.find('=')
            lat = line[pos_equal+1:len(line)-1]

        #lon
        if n_linea == n_linea_scan_lon:
            pos_equal = line.find('=')
            lon = line[pos_equal+1:len(line)-1]

        #elev
        if n_linea == n_linea_scan_elev:
            pos_equal = line.find('=')
            elev = line[pos_equal+1:len(line)-1]
            
        #####################################################
        #freq
        if n_linea >= n_linea_scanfrom_freq:
            if n_linea <= n_linea_scanto_freq:
                b = line.split()
                c = map(float, b)
                freq.extend(c)
        #zrot
        if n_linea >= n_linea_scanfrom_zrot:
            if n_linea <= n_linea_scanto_zrot:
                b = line.split()
                c = map(float, b)
                zrot.extend(c)
        #zxxr
        if n_linea >= n_linea_scanfrom_zxxr:
            if n_linea <= n_linea_scanto_zxxr:
                b = line.split()
                c = map(float, b)
                zxxr.extend(c)
        #zxxi
        if n_linea >= n_linea_scanfrom_zxxi:
            if n_linea <= n_linea_scanto_zxxi:
                b = line.split()
                c = map(float, b)
                zxxi.extend(c)
        #zxxvar
        if n_linea >= n_linea_scanfrom_zxxvar:
            if n_linea <= n_linea_scanto_zxxvar:
                b = line.split()
                c = map(float, b)
                zxx_var.extend(c)
                
        #zxyr
        if n_linea >= n_linea_scanfrom_zxyr:
            if n_linea <= n_linea_scanto_zxyr:
                b = line.split()
                c = map(float, b)
                zxyr.extend(c)
        #zxyi
        if n_linea >= n_linea_scanfrom_zxyi:
            if n_linea <= n_linea_scanto_zxyi:
                b = line.split()
                c = map(float, b)
                zxyi.extend(c)
        #zxyvar
        if n_linea >= n_linea_scanfrom_zxyvar:
            if n_linea <= n_linea_scanto_zxyvar:
                b = line.split()
                c = map(float, b)
                zxy_var.extend(c)
        #zyxr
        if n_linea >= n_linea_scanfrom_zyxr:
            if n_linea <= n_linea_scanto_zyxr:
                b = line.split()
                c = map(float, b)
                zyxr.extend(c)
        #zyxi
        if n_linea >= n_linea_scanfrom_zyxi:
            if n_linea <= n_linea_scanto_zyxi:
                b = line.split()
                c = map(float, b)
                zyxi.extend(c)
        #zyxvar
        if n_linea >= n_linea_scanfrom_zyxvar:
            if n_linea <= n_linea_scanto_zyxvar:
                b = line.split()
                c = map(float, b)
                zyx_var.extend(c)
        #zyyr
        if n_linea >= n_linea_scanfrom_zyyr:
            if n_linea <= n_linea_scanto_zyyr:
                b = line.split()
                c = map(float, b)
                zyyr.extend(c)
        #zyyi
        if n_linea >= n_linea_scanfrom_zyyi:
            if n_linea <= n_linea_scanto_zyyi:
                b = line.split()
                c = map(float, b)
                zyyi.extend(c)
        #zyyvar
        if n_linea >= n_linea_scanfrom_zyyvar:
            if n_linea <= n_linea_scanto_zyyvar:
                b = line.split()
                c = map(float, b)
                zyy_var.extend(c)
                
        ######
        #Tipper
        
        #txr
        if n_linea >= n_linea_scanfrom_txr:
            if n_linea <= n_linea_scanto_txr:
                b = line.split()
                c = map(float, b)
                txr.extend(c)
        #txi
        if n_linea >= n_linea_scanfrom_txi:
            if n_linea <= n_linea_scanto_txi:
                b = line.split()
                c = map(float, b)
                txi.extend(c)
        #txvar
        if n_linea >= n_linea_scanfrom_txvar:
            if n_linea <= n_linea_scanto_txvar:
                b = line.split()
                c = map(float, b)
                txvar.extend(c)
        #tyr
        if n_linea >= n_linea_scanfrom_tyr:
            if n_linea <= n_linea_scanto_tyr:
                b = line.split()
                c = map(float, b)
                tyr.extend(c)
        #tyi
        if n_linea >= n_linea_scanfrom_tyi:
            if n_linea <= n_linea_scanto_tyi:
                b = line.split()
                c = map(float, b)
                tyi.extend(c)
        #tyvar
        if n_linea >= n_linea_scanfrom_tyvar:
            if n_linea <= n_linea_scanto_tyvar:
                b = line.split()
                c = map(float, b)
                tyvar.extend(c)
				
        #TIPMAG
        if n_linea >= n_linea_scanfrom_TIPMAG:
            if n_linea <= n_linea_scanto_TIPMAG:
                b = line.split()
                c = map(float, b)
                tmag.extend(c)
				
        #ZSKEW
        if n_linea >= n_linea_scanfrom_ZSKEW:
            if n_linea <= n_linea_scanto_ZSKEW:
                b = line.split()
                c = map(float, b)
                zskew.extend(c)
				
        #TSTRIKE
        if n_linea >= n_linea_scanfrom_TSTRIKE:
            if n_linea <= n_linea_scanto_TSTRIKE:
                b = line.split()
                c = map(float, b)
                tstrike.extend(c)
	
        n_linea=n_linea + 1
    
        ######
        #################################################

    # Header: location 

    # MT response variables

    freq = np.asarray(freq)
    zrot = np.asarray(zrot)
    periods = 1/freq
    omega = 2*np.pi/periods

    # zxx
    zxxr = np.asarray(zxxr)
    zxxi = np.asarray(zxxi)
    zxx = zxxr + zxxi*1j
    zxx_var = np.asarray(zxx_var)

    # zxy 
    zxyr = np.asarray(zxyr)
    zxyi = np.asarray(zxyi)
    zxy = zxyr + zxyi*1j
    zxy_var = np.asarray(zxy_var)

    # zyx
    zyxr = np.asarray(zyxr)
    zyxi = np.asarray(zyxi)
    zyx = zyxr + zyxi*1j
    zyx_var = np.asarray(zyx_var)
    
    # zyy
    zyyr = np.asarray(zyyr)
    zyyi = np.asarray(zyyi)
    zyy = zyyr + zyyi*1j
    zyy_var = np.asarray(zyy_var)
    
    # tipper
    txr = np.asarray(txr)
    txi = np.asarray(txi)
    txvar = np.asarray(txvar)
    tyr = np.asarray(tyr)
    tyi = np.asarray(tyi)
    tyvar = np.asarray(tyvar)
    tmag = np.asarray(tmag)
	
    # zskew
    zskew = np.asarray(zskew)
	
    # tstrike
    tstrike = np.asarray(tstrike)
	
    # Cerramos el fichero.
    infile.close()
    infile2.close()

    H = [file, loc, lat, lon, elev]    
    Z = [file,periods,zxxr,zxxi,zxx,zxx_var,zxyr,zxyi,zxy,zxy_var,zyxr,zyxi,zyx,zyx_var,zyyr,zyyi,zyy,zyy_var]
    T = [txr, txi, txvar, tyr, tyi, tyvar, tmag]
    Z_rot = [zrot]
    Z_dim = [zskew, tstrike]
	
    return [H, Z, T, Z_rot, Z_dim]

# ==============================================================================
# Manipulate Z
# ==============================================================================
def rotate_Z(Z, alpha):

    #alpha = rot[0]
    # name = Z[0]
    # periods = Z[1]
    # zxxr = Z[2]
    # zxxi = Z[3]
    # zxx = Z[4]
    # zxx_var = Z[5]
    # zxyr = Z[6]
    # zxyi = Z[7]
    # zxy = Z[8]
    # zxy_var = Z[9]
    # zyxr = Z[10]
    # zyxi = Z[11]
    # zyx = Z[12]
    # zyx_var = Z[13]
    # zyyr = Z[14]
    # zyyi = Z[15]
    # zyy = Z[16]
    # zyy_var = Z[17]
	
    for i in range(len(Z[1])):
        r_matrix = np.matrix([[np.cos(alpha[i]), np.sin(alpha[i])], [-1*np.sin(alpha[i]), np.cos(alpha[i])]]) 
        Z_teta = np.matrix([[Z[4][i], Z[8][i]], [Z[12][i], Z[16][i]]])
        Z_drot = np.matmul(r_matrix,np.matmul(Z_teta,r_matrix.T))
        
        # zxx, zxy, zyx, zyy
        Z[4][i] = Z_drot[0,0]
        Z[8][i] = Z_drot[0,1]
        Z[12][i] = Z_drot[1,0]
        Z[16][i] = Z_drot[1,1]

        # zxx real and imag, zxy real and imag, zyx real and imag, zyy real and imag
        Z[2][i] = np.real(Z[4][i])
        Z[3][i] = np.imag(Z[4][i])
        Z[6][i] = np.real(Z[8][i])
        Z[7][i] = np.imag(Z[8][i])
        Z[10][i] = np.real(Z[12][i])
        Z[11][i] = np.imag(Z[12][i])
        Z[14][i] = np.real(Z[16][i])
        Z[15][i] = np.imag(Z[16][i])

        # variance 
        # var(Z'ij)=a**2 var(Z11)+ b**2 var(Z12) .+c**2 var(Z21) + d**2 va(Z22)

        # rotate variance 
        # varXX
        Z[5][i] = \
            + (np.cos(alpha[i])*np.cos(alpha[i]))**2 * Z[5][i] \
            + (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[9][i] \
            + (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[13][i] \
            + (np.sin(alpha[i])*np.sin(alpha[i]))**2 * Z[17][i]
        # varXY
        Z[9][i] = \
            + (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[5][i] \
            + (np.cos(alpha[i])*np.cos(alpha[i]))**2 * Z[9][i] \
            - (np.sin(alpha[i])*np.sin(alpha[i]))**2 * Z[13][i] \
            + (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[17][i]
        # varYX
        Z[13][i] = \
            - (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[5][i] \
            - (np.sin(alpha[i])*np.sin(alpha[i]))**2 * Z[9][i] \
            + (np.cos(alpha[i])*np.cos(alpha[i]))**2 * Z[13][i] \
            + (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[17][i]
        # varYY
        Z[17][i] = \
            - (np.sin(alpha[i])*np.sin(alpha[i]))**2 * Z[5][i] \
            - (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[9][i] \
            + (np.sin(alpha[i])*np.cos(alpha[i]))**2 * Z[13][i] \
            + (np.cos(alpha[i])*np.cos(alpha[i]))**2 * Z[17][i]

    return [Z]

def calc_app_res_phase(Z): 
    p = Z[1]
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
    ## Zxx
    app_res_xx = p/5 * np.square(abs(zxx))
    phase_ra_xx = np.arctan(zxxi/zxxr)
    phase_de_xx = (360/(2*np.pi)) * phase_ra_xx
    ## std. Error based on Egbert, 98
    app_res_error_xx =  np.sqrt(2.*p*app_res_xx*(zxx_var**2.)/5.)
    phase_error_xx = (360/(2*np.pi*abs(zxx))) * np.sqrt(zxx_var**2. /2.)
    ## Zxy
    app_res_xy = p/5 * np.square(abs(zxy))
    phase_ra_xy = np.arctan(zxyi/zxyr)      	# radians
    phase_de_xy = (360/(2*np.pi)) * phase_ra_xy # degrees
        ## std. Error based on Egbert, 98
    app_res_error_xy =  np.sqrt(2.*p*app_res_xy*(zxy_var**2.)/5.)
    phase_error_xy = (360/(2*np.pi*abs(zxy))) * np.sqrt(zxy_var**2. /2.)
    ## Zyx
    app_res_yx = p/5 * np.square(abs(zyx))
    phase_ra_yx = np.arctan(zyxi/zyxr)
    phase_de_yx = (360/(2*np.pi)) * phase_ra_yx
        ## std. Error based on Egbert, 98
    app_res_error_yx =  np.sqrt(2.*p*app_res_yx*(zyx_var**2.)/5.)
    phase_error_yx = (360/(2*np.pi*abs(zyx))) * np.sqrt(zyx_var**2. /2.)
    ## Zyy
    app_res_yy = p/5 * np.square(abs(zyy))
    phase_ra_yy = np.arctan(zyyi/zyyr)
    phase_de_yy = (360/(2*np.pi)) * phase_ra_yy	
        ## std. Error based on Egbert, 98
    app_res_error_yy =  np.sqrt(2.*p*app_res_yy*(zyy_var**2.)/5.)
    phase_error_yy = (360/(2*np.pi*abs(zyy))) * np.sqrt(zyy_var**2. /2.)

    rho_app = [app_res_xx, app_res_xy, app_res_yx, app_res_yy]
    phase_deg = [phase_de_xx, phase_de_xy, phase_de_yx, phase_de_yy]
    rho_app_er = [app_res_error_xx, app_res_error_xy, app_res_error_yx, app_res_error_yy]
    phase_deg_er = [phase_error_xx,phase_error_xy,phase_error_yx,phase_error_yy]

    return [rho_app, phase_deg, rho_app_er, phase_deg_er]
# ==============================================================================
# Plots
# ==============================================================================
def plot_Z_appres_phase(Z, title = None):
    
    if title is None:
        title = name[len(name)-10:len(name)-4]
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
    
    zxx_app_res = periods/5 * np.square(abs(zxx))
    zxx_phase = (360/(2*np.pi)) * np.arctan(zxxi/ zxxr)
    ## std. Error based on Egbert, 98
    zxx_app_res_error =  np.sqrt(2.*periods*zxx_app_res*(zxx_var**2.)/5.)
    zxx_phase_error = (360/(2*np.pi*abs(zxx))) * np.sqrt(zxx_var**2. /2.)

    
    zxy_app_res = periods/5 * np.square(abs(zxy))
    zxy_phase = (360/(2*np.pi)) * np.arctan(zxyi/ zxyr)
    ## std. Error based on Egbert, 98
    zxy_app_res_error =  np.sqrt(2.*periods*zxy_app_res*(zxy_var**2.)/5.)
    zxy_phase_error = (360/(2*np.pi*abs(zxy))) * np.sqrt(zxy_var**2. /2.)

    
    zyx_app_res = periods/5 * np.square(abs(zyx))
    zyx_phase = (360/(2*np.pi)) * np.arctan(zyxi/ zyxr)
    ## std. Error based on Egbert, 98
    zyx_app_res_error =  np.sqrt(2.*periods*zyx_app_res*(zyx_var**2.)/5.)
    zyx_phase_error = (360/(2*np.pi*abs(zyx))) * np.sqrt(zyx_var**2. /2.)
    
    zyy_app_res = periods/5 * np.square(abs(zyy))
    zyy_phase = (360/(2*np.pi)) * np.arctan(zyyi/ zyyr)
    ## std. Error based on Egbert, 98
    zyy_app_res_error =  np.sqrt(2.*periods*zyy_app_res*(zyy_var**2.)/5.)
    zyy_phase_error = (360/(2*np.pi*abs(zyy))) * np.sqrt(zyy_var**2. /2.)
  
    #################################################
    # Plot figure with subplots of different sizes

    f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    #plt.title(file)
    #f = plt.figure(figsize=(12, 18)) 
    f.set_size_inches(12,18)
    f.suptitle(title, fontsize=22)
    #gs = gridspec.GridSpec(2, 1, width_ratios=[3, 1]) 
    
    #ax1 = plt.subplot(gs[0])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    #ax1.loglog(periods,app_res,'r*')
    ax1.errorbar(periods,zxy_app_res,zxy_app_res_error, fmt='ro')
    ax1.errorbar(periods,zyx_app_res,zyx_app_res_error, fmt='bo')
    #ax1.set_xlim([np.min(periods), np.max(periods)])
    ax1.set_xlim(1e-3, 1e3)
    ax1.set_ylim([1e0,1.5e3])
    ax1.set_xlabel('Period [s]', fontsize=18)
    ax1.set_ylabel('Ap. Resistiviy [Ohm m]', fontsize=18)
    ax1.legend(['RhoXY','RhoYX'])
    ax1.grid(True, which='both', linewidth=0.4)

    #ax2.semilogx(periods,phase,'b*')
    #ax2 = plt.subplot(gs[1])
    ax2.set_xscale("log")
    ax2.errorbar(periods,zxy_phase,zxy_phase_error, fmt='ro')
    ax2.errorbar(periods,zyx_phase-180,zyx_phase_error, fmt='bo')
    #ax2.set_xlim([np.min(periods), np.max(periods)])
    ax2.set_xlim(1e-3, 1e3)
    ax2.set_ylim([-190,190])
    ax2.set_xlabel('Period [s]', fontsize=18)
    ax2.set_ylabel('Phase [deg]', fontsize=18)
    ax2.legend(['PhaseXY','PhaseYX'])
    ax2.grid(True, which='both', linewidth=0.4)
    
    #ax1 = plt.subplot(gs[0])
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    #ax1.loglog(periods,app_res,'r*')
    ax3.errorbar(periods,zxx_app_res,zxx_app_res_error, fmt='ro')
    ax3.errorbar(periods,zyy_app_res,zyy_app_res_error, fmt='bo')
    #ax1.set_xlim([np.min(periods), np.max(periods)])
    ax3.set_xlim(1e-3, 1e3)
    ax3.set_ylim([1e0,1e3])
    ax3.set_xlabel('Period [s]', fontsize=18)
    ax3.set_ylabel('Ap. Resistiviy [Ohm m]', fontsize=18)
    ax3.legend(['RhoXX','RhoYY'])
    ax3.grid(True, which='both', linewidth=0.4)

    #ax2.semilogx(periods,phase,'b*')
    #ax2 = plt.subplot(gs[1])
    ax4.set_xscale("log")
    ax4.errorbar(periods,zxx_phase,zxx_phase_error, fmt='ro')
    ax4.errorbar(periods,zyy_phase-180,zyy_phase_error, fmt='bo')
    #ax2.set_xlim([np.min(periods), np.max(periods)])
    ax4.set_xlim(1e-3, 1e3)
    ax4.set_ylim([-190,190])
    ax4.set_xlabel('Period [s]', fontsize=18)
    ax4.set_ylabel('Phase [deg]', fontsize=18)
    ax4.legend(['PhaseXX','PhaseYY'])
    ax4.grid(True, which='both', linewidth=0.4)
    
    return f

def Z_plot_appres_phase_indvec_ellip(Z, T):  #Z_full_response(Z, T):
    
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
    
    txr = T[0]
    txi = T[1] 
    txvar = T[2]
    tyr = T[3]
    tyi = T[4]
    tyvar = T[5]
    
    #################################################
    # Plot figure with subplots of different sizes

    #f,(ax1,ax2,ax3,ax4) = plt.subplots()
    
    f = plt.figure()
    f.set_size_inches(12,18)
    f.suptitle(name[:len(name)-4], fontsize=22)

    gs = gridspec.GridSpec(4, 1, height_ratios=[4, 3, 2, 2])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    #################################################
    # Calculate: Apparent Resistivity and Phase    
    
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
    
    # Plot: Apparent Resistivity and Phase (XY and YX components)
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    #ax1.loglog(periods,app_res,'r*')
    ax1.errorbar(periods,zxy_app_res,zxy_app_res_error, fmt='ro')
    ax1.errorbar(periods,zyx_app_res,zyx_app_res_error, fmt='bo')
    ax1.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    ax1.set_xlim([1e-4, 1e3 + 100])
    #ax1.set_ylim([1e-1,1.5e3])
    #ax1.set_xlabel('Period [s]', fontsize=18)
    ax1.set_ylabel('Ap. Resistiviy [Ohm m]', fontsize=18)
    ax1.legend(['RhoXY','RhoYX'])
    ax1.grid(True, which='both', linewidth=0.4)

    ax2.set_xscale("log")
    ax2.errorbar(periods,zxy_phase,zxy_phase_error, fmt='ro')
    ax2.errorbar(periods,zyx_phase-180,zyx_phase_error, fmt='bo')
    ax2.set_xlim([1e-4, 1e3 + 100])
    #ax2.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    ax2.set_ylim([-190,190])
    #ax2.set_xlabel('Period [s]', fontsize=18)
    ax2.set_ylabel('Phase [deg]', fontsize=18)
    ax2.legend(['PhaseXY','PhaseYX'])
    ax2.grid(True, which='both', linewidth=0.4)
    
    #################################################
    # Calculate: Induction Vector  
    
    # Plot: Induction Vector 
    
    eje = np.zeros(len(periods))
    #f.set_size_inches(12,4)
    #ax3 = plt.axes()
    ax3.set_xscale("log")

    vec_r = ax3.quiver(periods,0,txr,tyr,
                       color='b',
                       units='y',
                       scale_units='y',
                       scale = 1,
                       headlength=1.0,
                       headaxislength=1,
                       width=0.01)

    vec_i = ax3.quiver(periods,0,txi,tyi,color='r',
                       units='y',scale_units='y',
                       scale = 1,headlength=1.0,
                       headaxislength=1,
                       width=0.01,alpha=0.5)

    #ax3.set_xlim([np.min(periods)/2, np.max(periods) + np.max(periods)])
    ax3.set_xlim([1e-4, 1e3 + 100])
    ax3.set_ylim([-.5,.5])
    ax3.set_xlabel('Period [s]', fontsize=18)
    ax3.set_ylabel(' ', fontsize=18)
    ax3.legend(['Ind. Vec.: Real','Ind. Vec.: Imaginary'])
    ax3.grid(True, which='both', linewidth=0.4)
    
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

            phi_2[numb] = np.sqrt(abs(det_phi[numb]))
            
            if phi_2[numb]**2 > phi_1[numb]**2 + phi_3[numb]**2: # To verify
                phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - np.sqrt(abs(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)))
                phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + np.sqrt(abs(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
            else:
                phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - np.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
                phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + np.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
        
        else: # det_phi[numb] >= 0:
            phi_2[numb] = np.sqrt(det_phi[numb])
            phi_min[numb] = 1*(phi_1[numb]**2 + phi_3[numb]**2 - np.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2))
            phi_max[numb] = phi_1[numb]**2 + phi_3[numb]**2 + np.sqrt(phi_1[numb]**2 + phi_3[numb]**2 - phi_2[numb]**2)
            
        # for ploting ellipses
        #diam_x[numb] = 2*phi_max[numb]
        #diam_y[numb] = 2*phi_min[numb]
        angle[numb] = alpha[numb]-beta[numb] 
        aux_col[numb] = (np.arctan(phi_min[numb]) + np.pi/2)/np.pi # normalice for a 0 - 1 scale
        e[numb] = phi_min[numb] / phi_max[numb]
    
        numb = numb + 1
    
    # Plot: Tensor Ellipses
    
    #f, ax4 = plt.subplots(subplot_kw={'aspect': 'equal'})
    #fig, ax = plt.subplots(1,1)
    #fig.set_size_inches(14,8)
    #fig.set_size_inches(14,14)

    for i in range(NUM):
        color_elip = [1-abs(aux_col[i]/(np.pi/2)), abs(aux_col[i]/(np.pi/2)), 1];  
        # Color scale 'cool': varies between light blue and pink 
        ells = [Ellipse(xy=[eje_x[i],eje_y[i]], width= 1, height=e[i], angle=angle[i])]
        for a in ells:
            ax4.add_patch(a)
            a.set_facecolor(color_elip)
            #plt.close("all")
    
    ax4.grid(True, which='both', linewidth=0.4)
    #ax4.set_xlim([-2,54])
    ax4.set_xlim([-2,len(periods)+1])
    ax4.set_ylim([-4,4])
    im = ax4.imshow(np.arange(0).reshape((0, 0)),cmap=plt.cm.cool)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_yticklabels(['0','','','','','90'])
    cbar.set_label('arctg (phi_min)', rotation=90)
    #plt.show()
    return f
	
# ==============================================================================
#  Functions
# ==============================================================================

def calc_beta_sta_quadrant(station_objects, wells_objects): 
    """
    Function that calculates betas, Tmins and Tmaxs in MT station positions based those values in
    nearest wells. First, for each quadrant around the station, the nearest wells are found. 
    Second, using the beta estimatimation results, the parameters are calculated as a weigthed 
    average of those in the nearest wells. Third, the results are assigned as attributes to the MT objects. 
    
    Attributes generated:
    sta_obj.betas_3l = [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
    sta_obj.Tmin_3l = [[mean_Tmin1, std_Tmin1],[mean_Tmin2, std_Tmin2],[mean_Tmin3, std_Tmin3]]
    sta_obj.Tmax_3l = [[mean_Tmax1, std_Tmax1],[mean_Tmax2, std_Tmax2],[mean_Tmax3, std_Tmax3]]
    self.max_depth_temp : maximum depth for temp. profile based on weigthed average 

    Note:
    wl.read_temp_prof_est_wells() needs to be run first for every well to load wl attributes of betas and others. 
    .. conventions::
	: z1 and z2 in MT object refer to thickness of two first layers
    : distances in meters
    : temperature in celcius
    """
    for sta_obj in station_objects:
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
            # search for nearest well to MT station in quadrant 1 (Q1)
            if (wl.lat_dec > sta_obj.lat_dec and wl.lon_dec > sta_obj.lon_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q1:
                    dist_pre_q1 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q1: 
                    name_aux_q1 = wl.name
                    wl_q1 = wl
                    dist_pre_q1 = dist
            # search for nearest well to MT station in quadrant 2 (Q2)
            if (wl.lat_dec < sta_obj.lat_dec and wl.lon_dec > sta_obj.lon_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q2:
                    dist_pre_q2 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q2: 
                    name_aux_q2 = wl.name
                    wl_q2 = wl
                    dist_pre_q2 = dist
            # search for nearest well to MT station in quadrant 3 (Q3)
            if (wl.lat_dec < sta_obj.lat_dec and wl.lon_dec < sta_obj.lon_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q3:
                    dist_pre_q3 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q3: 
                    name_aux_q3 = wl.name
                    wl_q3 = wl
                    dist_pre_q3 = dist
            # search for nearest well to MT station in quadrant 4 (Q4)
            if (wl.lat_dec > sta_obj.lat_dec and wl.lon_dec < sta_obj.lon_dec): 
                # distance between station and well
                dist = dist_two_points([wl.lon_dec, wl.lat_dec], [sta_obj.lon_dec, sta_obj.lat_dec], type_coord = 'decimal')
                if not dist_pre_q4:
                    dist_pre_q4 = dist
                # check if distance is longer than the previous wel 
                if dist <= dist_pre_q4: 
                    name_aux_q4 = wl.name
                    wl_q4 = wl
                    dist_pre_q4 = dist

        # save names of nearest wells to be used for prior
        sta_obj.beta_wl_names = [name_aux_q1, name_aux_q2, name_aux_q3, name_aux_q4]
        sta_obj.beta_wl_names = list(filter(None, sta_obj.prior_meb_wl_names))
        near_wls = [wl_q1,wl_q2,wl_q3,wl_q4] #list of objects (wells)
        near_wls = list(filter(None, near_wls))
        dist_wels = [dist_pre_q1,dist_pre_q2,dist_pre_q3,dist_pre_q4]
        dist_wels = list(filter(None, dist_wels))
        sta_obj.beta_wl_dist = dist_wels

        # maximum depth for temp. profile based on weigthed average 
        depth_near_wls = [wl.red_depth[-1] for wl in near_wls]
        sta_obj.max_depth_temp = np.dot(depth_near_wls,dist_wels)/np.sum(dist_wels)


        #### betas
        # betas consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
        b1_mean = np.zeros(len(near_wls))
        b1_std = np.zeros(len(near_wls))
        b2_mean = np.zeros(len(near_wls))
        b2_std = np.zeros(len(near_wls))
        b3_mean = np.zeros(len(near_wls))
        b3_std = np.zeros(len(near_wls))
        count = 0
        # extract betas from nearest wells 
        #print(sta_obj.beta_wl_names)
        #print(sta_obj.beta_wl_dist)
        
        for wl in near_wls:
            # extract beta values from wl
            #wl.betas_3l = [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
            b1_mean[count], b1_std[count] = wl.betas_3l[0]
            b2_mean[count], b2_std[count] = wl.betas_3l[1]
            b3_mean[count], b3_std[count] = wl.betas_3l[2]
            count+=1
        # calculete betas normal dist. for MT stations
        b1_mean = np.dot(b1_mean,dist_wels)/np.sum(dist_wels)
        b1_std = np.dot(b1_std,dist_wels)/np.sum(dist_wels)
        b2_mean = np.dot(b2_mean,dist_wels)/np.sum(dist_wels)
        b2_std = np.dot(b2_std,dist_wels)/np.sum(dist_wels)
        b3_mean = np.dot(b3_mean,dist_wels)/np.sum(dist_wels)
        b3_std = np.dot(b3_std,dist_wels)/np.sum(dist_wels)

        # assign result to attribute
        sta_obj.betas_3l = [[b1_mean, b1_std],[b2_mean, b2_std],[b3_mean, b3_std]]
        #### Tmin
        # consist of mean and std for parameter, calculate as weighted(distance) average from nearest wells
        t1_mean = np.zeros(len(near_wls))
        t1_std = np.zeros(len(near_wls))
        t2_mean = np.zeros(len(near_wls))
        t2_std = np.zeros(len(near_wls))
        t3_mean = np.zeros(len(near_wls))
        t3_std = np.zeros(len(near_wls))
        count = 0
        # extract meb mcmc results from nearest wells 
        for wl in near_wls:
            # extract beta values from wl
            #wl.betas_3l = [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
            t1_mean[count], t1_std[count] = wl.Tmin_3l[0]
            t2_mean[count], t2_std[count] = wl.Tmin_3l[1]
            t3_mean[count], t3_std[count] = wl.Tmin_3l[2]
            count+=1
        # calculete betas normal dist. for MT stations
        t1_mean = np.dot(t1_mean,dist_wels)/np.sum(dist_wels)
        t1_std = np.dot(t1_std,dist_wels)/np.sum(dist_wels)
        t2_mean = np.dot(t2_mean,dist_wels)/np.sum(dist_wels)
        t2_std = np.dot(t2_std,dist_wels)/np.sum(dist_wels)
        t3_mean = np.dot(t3_mean,dist_wels)/np.sum(dist_wels)
        t3_std = np.dot(t3_std,dist_wels)/np.sum(dist_wels)

        # assign result to attribute
        sta_obj.Tmin_3l = [[t1_mean, t1_std],[t2_mean, t2_std],[t3_mean, t3_std]]

        #### Tmax
        t1_mean = np.zeros(len(near_wls))
        t1_std = np.zeros(len(near_wls))
        t2_mean = np.zeros(len(near_wls))
        t2_std = np.zeros(len(near_wls))
        t3_mean = np.zeros(len(near_wls))
        t3_std = np.zeros(len(near_wls))
        count = 0
        # extract T bound. cond. from nearest wells 
        for wl in near_wls:
            # extract beta values from wl
            #wl.betas_3l = [[mean_beta1, std_beta1],[mean_beta2, std_beta2],[mean_beta3, std_beta3]]
            t1_mean[count], t1_std[count] = wl.Tmax_3l[0]
            t2_mean[count], t2_std[count] = wl.Tmax_3l[1]
            # assign std for bottom Temp bound condition as the std of previus boundary (To be checked)
            t3_mean[count], t3_std[count] = wl.Tmax_3l[2][0], wl.Tmax_3l[1][1]
            count+=1
        # calculete betas normal dist. for MT stations
        t1_mean = np.dot(t1_mean,dist_wels)/np.sum(dist_wels)
        t1_std = np.dot(t1_std,dist_wels)/np.sum(dist_wels)
        t2_mean = np.dot(t2_mean,dist_wels)/np.sum(dist_wels)
        t2_std = np.dot(t2_std,dist_wels)/np.sum(dist_wels)
        t3_mean = np.dot(t3_mean,dist_wels)/np.sum(dist_wels)
        t3_std = np.dot(t3_std,dist_wels)/np.sum(dist_wels)

        # assign result to attribute
        sta_obj.Tmax_3l = [[t1_mean, t1_std],[t2_mean, t2_std],[t3_mean, t3_std]]

def Texp2(z,Zmin,Zmax,Tmin,Tmax,beta): 
    """
    Calculate temp. profile based on model 1D heat transfer model (See Bredehoeft, J. D., and I. S. Papadopulos (1965))

    Input:
    - z: depths to calculate
    - Zmin,Zmax,Tmin,Tmax: boundary conditions
    - beta: beta coeficient for the model 

    Output:
    - Test: estimated temperature profile (array)

    """
    Test = (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta)-1) + Tmin
    return Test

def find_nearest(array, value):
    """
    Find nearest to value in an array
    
    Input:
    - array: numpy array to look in
    - value: value to search
    
    Output:
    - array[idx]: closest element in array to value

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx