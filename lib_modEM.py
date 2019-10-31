"""
- Module ModEM: Class to deal with ModEM results 
   :synopsis: Forward and inversion ot MT using ModEM software.
# Author: Alberto Ardid
# Institution: University of Auckland
# Date: 2019
.. conventions::
CC	: Clay Cap 
"""

# ==============================================================================
#  Imports
# ==============================================================================

from glob import glob
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
import cmath, traceback, os, sys, shutil
from mulgrids import *
from scipy.interpolate import griddata
from doe_lhs import lhs
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.stats import cauchy
from scipy.signal import savgol_filter
import scipy.sparse as sp
import math
import cmath
import corner, emcee
import time
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

#from scipy.interpolate import griddata
from lib_sample_data import *
from misc_functios import *
#from SimPEG import Utils, Solver, Mesh, DataMisfit, Regularization, Optimization, InvProblem, Directives, Inversion
J = cmath.sqrt(-1)

exe = r'D:\ModEM\ModEM\f90\Mod2DMT.exe' # office
#exe = r'C:\ModEM\f90\Mod2DMT.exe' # house
textsize = 15.

# ==============================================================================
#  ModEM class
# ==============================================================================

class Modem(object):
    """
    This class is for dealing with ModEM software
	====================    ========================================== ==========
    Attributes              Description                                default
    =====================   ========================================== ==========
    nx                       
    ny
    nz
    x0
    y0
    z0
    work_dir
    rho
    f
    T
    =====================   =====================================================
    Methods                 Description
    =====================   =====================================================
    make_grid
    plot_grid
    rho_box
    rho_blocks
    stations
    assemble_grid
    read_input
    write_input
    read_data
    write_data
    run
    invert
    plot_rho2D
    plot_impedance
    """
    def __init__(self, work_dir = None):
        self.nx = 1
        self.ny = 1
        self.nz = 1
        self.x0 = 0.
        self.y0 = 0.
        self.z0 = 0.
        self.rho = None
        if work_dir is None: 
            self.work_dir = '.'
        else:
            self.work_dir = work_dir
            if not os.path.isdir(self.work_dir): 
                os.makedirs(self.work_dir)
        self.verbose = True
    def make_grid(self,xe,ye,ze):
        # x coords, depend on dimensionality
        if len(xe) == 0:
            self.xe = []
            self.x = []
            self.dim = 2
        else:
            self.xe = np.array(xe)
            self.x = 0.5*(self.xe[1:]+self.xe[:-1])
            xmin = 0.5*(np.min(self.x)+np.max(self.x))
            self.x -= xmin
            self.xe -= xmin
            self.dim = 3
            self.nx = len(self.x)
        # y coords
        self.ye = np.array(ye)
        self.y = 0.5*(self.ye[1:]+self.ye[:-1])
        ymin = 0.5*(np.min(self.y)+np.max(self.y))
        self.y -= ymin
        self.ye -= ymin
        self.ny = len(self.y)
        # z coords
        self.ze = np.array(ze)
        self.z = 0.5*(self.ze[1:]+self.ze[:-1])
        self.nz = len(self.z)
        # create empty rho structure of appropriate dimension
        if self.dim == 2: self.rho = np.zeros((self.ny, self.nz))
        else: self.rho = np.zeros((self.nx, self.ny, self.nz))
    def plot_grid(self, filename, ylim = None, zlim = None):
        # plot simple contours 
        plt.clf()
        fig = plt.figure(figsize=[3.5,3.5])
        ax = plt.axes([0.18,0.25,0.70,0.50])
        
        y0,y1 = self.y[0]/1.e3, self.y[-1]/1.e3
        z0,z1 = self.z[0]/1.e3, self.z[-1]/1.e3
        for y in [y0,y1]: ax.plot([y,y], [z0,z1], 'k-')
        for z in [z0,z1]: ax.plot([y0,y1], [z,z], 'k-')
        
        for x in self.ye[1:-1]: ax.plot([x/1.e3, x/1.e3],[z0,z1], '-', color = [0.5,0.5,0.5], zorder = 10, lw = 0.5) 
        for y in self.ze[1:-1]: ax.plot([y0,y1],[y/1.e3, y/1.e3], '-', color = [0.5,0.5,0.5], zorder = 10, lw = 0.5) 
        
        if ylim is not None: ax.set_xlim(ylim)		
        if zlim is not None: ax.set_ylim(zlim)
        
        ax.set_xlabel("y / km", size = textsize)
        ax.set_ylabel("z / km", size = textsize)
        
        for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(textsize)
        
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        plt.close(fig)
    def rho_box(self, rho, box=None):
        if box is None: 
            box = [self.ye[0]-0.1, self.ze[-1]-0.1, self.ye[-1]-self.ye[0]+0.2, self.ze[0]-self.ze[-1]+0.2]
        y0,y1 = box[0],box[0]+box[2]
        z0,z1 = box[1],box[1]+box[3]
        
        for i,yi in enumerate(self.y):
            for j,zi in enumerate(self.z):
                if (yi>y0)&(yi<=y1)&(zi>z0)&(zi<=z1): self.rho[i,j] = rho
    def rho_blocks(self, rho, y, z):
        for pi,yi,zi in zip(rho, y, z):
            if pi <= 0.: continue 		# zero or negative resitivity indicates no change
            
            # find nearest point
            i = np.argmin(abs(self.y - yi))
            j = np.argmin(abs(self.z - zi))
                        
            # check coordinate within bounding box (otherwise ignore as out of grid)
            if not ((yi>self.ye[i])&(yi<self.ye[i+1])): continue 
            if not ((zi<self.ze[j])&(zi>self.ze[j+1])): continue
            
            # change value
            self.rho[i,j] = pi
    def stations(self, ys, zs, f):
        try: iter(zs) 
        except TypeError: zs = zs*np.ones((1,len(ys)))[0]
        try: iter(ys) 
        except TypeError: ys = ys*np.ones((1,len(ys)))[0]
        self.ys = np.array(ys)
        self.xs = 0.*self.ys
        self.zs = zs
        self.f = np.array([f for i in range(len(ys))]).T
        
        dat = np.zeros((len(f),len(self.xs)), dtype = complex)
        for dtype in ['TE_Impedance', 'TM_Impedance']:
            self.__setattr__(dtype, dat)
    def assemble_grid(self):
        # list to array
        self.xe = np.array(self.x) # list to array
        self.ye = np.array(self.y) # list to array
        self.ze = np.array(self.z) # list to array

        # change from cell size to postition on grid
        # horizontal coords
        y = 1.*self.ye
        self.ye = np.zeros((1,len(y)+1))[0]
        self.ye[0] = 0
        # center at the left of the section 
        self.ye[1:] = np.cumsum(y)
        self.y = 0.5*(self.ye[1:]+self.ye[:-1])
        ymin = 0.5*(np.min(self.y)+np.max(self.y))
        # center at the middle at the section  
        self.y -= ymin
        self.ye -= ymin
        if self.dim == 2:
            # vertical coords
            z = 1.*self.ze
            self.ze = np.zeros((1,len(z)+1))[0]
            self.ze[0] = 0
            self.ze[1:] = -np.cumsum(z)
            self.z = 0.5*(self.ze[1:]+self.ze[:-1])
        else:
            # horizontal coords
            x = 1.*self.xe
            self.xe = np.zeros((1,len(x)+1))[0]
            self.xe[0] = 0
            # center at the left of the section
            self.xe[1:] = np.cumsum(x)
            self.x = 0.5*(self.xe[1:]+self.xe[:-1])
            xmin = 0.5*(np.min(self.x)+np.max(self.x))
            self.x -= xmin
            self.xe -= xmin

            # vertical coords
            z = 1.*self.ze
            self.ze = np.zeros((1,len(z)+1))[0]
            self.ze[0] = 0
            self.ze[1:] = -np.cumsum(z)
            #self.ze[1:] = np.cumsum(z)
            self.z = 0.5*(self.ze[1:]+self.ze[:-1])
            ## from here vector self.x,y,z contains positions in the grid 
            ## center at the middle (ex. self.x = [-3,-1,1,3])

    def read_input(self, filename):
        self.x = []
        self.y = []
        self.z = []
        fp = open(filename,'r')
        # read header 
        ln = fp.readline().rstrip()
        # if fisrt line is a comment, go to next line
        if '#' in ln:
            ln = fp.readline().rstrip()
        nums = ln.split()
        # log format?
        if nums[-1] == "LOGE": logR = True; nums = nums[:-1]
        else: logR = False
        
        if len(nums) == 2:
            self.dim = 2
            self.ny = int(float(nums[0]))
            self.nz = int(float(nums[1]))
        else:
            self.dim = 3
            self.nx = int(float(nums[0]))
            self.ny = int(float(nums[1]))
            self.nz = int(float(nums[2]))
            
        if self.dim == 2: self.rho = np.zeros((self.ny, self.nz))
        else: self.rho = np.zeros((self.nx, self.ny, self.nz))
        
        # read grid size
        keepLooping = True
        ivec = 0
        cnt = 0
        if self.dim == 2: 
            CNT = (self.ny+self.nz)
            vec = self.y
            n = self.ny
        else: 
            CNT = (self.nx+self.ny+self.nz)
            vec = self.x
            n = self.nx
        while keepLooping:
            # read the next line
            ln = fp.readline().rstrip()
            nums = ln.split()
            # for each number
            for num in nums:
                # check if need to move to next vector
                if len(vec) == n:
                    if ivec == 0:
                        if self.dim == 2:
                            vec = self.z
                            n = self.nz
                        else:
                            vec = self.y
                            n = self.ny
                    elif ivec == 1:
                        vec = self.z
                        n = self.nz
                    else:
                        adsf					
                    ivec += 1
                # append value to vector
                vec.append(float(num))
                cnt += 1
                if cnt == CNT: 
                    keepLooping = False
                    break
            # condition to break from loop
        #print(self.x)
        #print(self.y)
        #print(self.z)
        
        # assemble grid
        self.assemble_grid()
        ## from here vector self.x,y,z contains positions in the grid 
        ## center at the middle (ex. self.x = [-3,-1,1,3])
        #print(self.x)
        #print(self.y)
        #print(self.z)
        
        # read resistivities
        ln = fp.readline() 		# skip this line		
        i,j,k,cnt = [0,0,0,0]
        while True:
            # read the next line
            ln = fp.readline().rstrip()
            nums = ln.split()
            # for each number
            for num in nums:
                if self.dim == 2: self.rho[j,k] = float(num)
                else: self.rho[i,j,k] = float(num) 
                
                i += 1
                if i == self.nx: i = 0; j += 1
                if j == self.ny: j = 0; k += 1
                cnt += 1
                
            # condition to break from loop
            if all(np.array([i,j,k]) == np.array([self.nx, self.ny, self.nz])): break 
                
            if cnt == (self.nx*self.ny*self.nz): break
                    
        if logR: self.rho = np.exp(self.rho)
        fp.close()

    def write_input(self, filename):
        with open(filename, 'w') as fp:
            # write the mesh
            fp.write("%i %i LOGE\n"%(len(self.ye)-1, len(self.ze)-1))
            cnt = 0
            for y0,y1 in zip(self.ye[0:], self.ye[1:]):
                fp.write("%8.7f "%(y1-y0))
                cnt += 1
            fp.write("\n")			
    #				if cnt == 10: fp.write("\n"); cnt = 0
    #			if cnt !=0: fp.write("\n")			
            for z0,z1 in zip(self.ze[0:], self.ze[1:]):
                fp.write("%8.7f "%(z0-z1))
                cnt += 1
            fp.write("\n")			
    #				if cnt == 10: fp.write("\n"); cnt = 0
    #			if cnt !=0: fp.write("\n")			
            fp.write("1\n")
            
            # write the conductivities
            cnt = 0
            for k in range(len(self.z)):
                for j in range(len(self.y)):
                    fp.write("%8.7e "%np.log(self.rho[j,k]))
                    cnt += 1
                    if cnt == 19: fp.write("\n"); cnt = 0
                if cnt !=0: fp.write("\n")
            if cnt !=0: fp.write("\n")

    def read_data(self, filename, add_noise = None, polar = None, plot_noise = None):
        fp = open(filename,'r')
        
        keepReading = True
        ln = fp.readline().rstrip() # discard
        cnt = 0
        while keepReading: 	# outer loop, more data records
            # read header 
            ln = fp.readline().rstrip() # discard
            
            ln = fp.readline().rstrip()[1:] # data type	
            data_type = ln.strip()
            print("reading %s"%data_type)
            ln = fp.readline().rstrip() # sign convention
            ln = fp.readline().rstrip() # units
            
            ln = fp.readline().rstrip() # rotation
            ln = fp.readline().rstrip()[1:] # origin
            nums = ln.split()
            if len(nums) == 2:
                self.x0 = 0.; self.y0 = float(nums[0]); self.z0 = float(nums[1])
            else:
                self.x0 = float(nums[0]); self.y0 = float(nums[1]); self.z0 = float(nums[2])
            ln = fp.readline().rstrip()[1:] # periods and stations
            nums = ln.split(); nT = int(float(nums[0])); nS = int(float(nums[1]))
            
            # allocate space
            if cnt == 0:
                self.xs = np.zeros((1,nS))[0]	# station locations
                self.ys = np.zeros((1,nS))[0]	# station locations
                self.zs = np.zeros((1,nS))[0]	# station locations
                
                self.f = np.zeros((nT,nS))
                self.T = np.zeros((nT,nS))
            
            dat = np.zeros((nT,nS), dtype = complex)
                    
            for i in range(nS):					
                for j in range(nT):
                    ln = fp.readline().rstrip()
                    nums = ln.split()
                    
                    # save station location information
                    if (cnt == 0) and (j == 0):
                        self.xs[i] = float(nums[4])+self.x0
                        self.ys[i] = float(nums[5])+self.y0
                        self.zs[i] = float(nums[6])+self.z0
                        
                    if cnt == 0:
                        self.f[j,i] = 1./float(nums[0])
                        self.T[j,i] = float(nums[0])
                        
                    # save station data
                    dat[j,i] = np.complex(float(nums[8]),float(nums[9]))
        
            self.__setattr__(data_type, dat) 	# for data
            # condition to break from outer loop
            ln = fp.readline().rstrip() # discard
            if ln == '':  	# another data record present
                keepReading = False
            cnt += 1
        
        self.ys = self.ys - (self.ye[-1]-self.ye[0])/2.				

        if add_noise: 
            if polar: 
                # add noise in polar coordinates. ratio noise amplitud to noise phase = 10/3 
                # calc. amplitud and phase 
                # from xy to polar
                TE_amp = abs(self.TE_Impedance)
                TE_phase = np.arctan(self.TE_Impedance.imag/self.TE_Impedance.real)
                TM_amp = abs(self.TM_Impedance)
                TM_phase = np.arctan(self.TM_Impedance.imag/self.TM_Impedance.real)

                # add noise to phase 
                # abs of perc_noise % magnitud of each impedance
                perc = add_noise/100.
                mu = 0.
                sigma = 1.
                error_TE_amp = 0*TE_amp
                error_TM_amp = 0*TE_amp
                error_TE_phase = 0*TE_amp
                error_TM_phase = 0*TE_amp

                if plot_noise:
                    pp = PdfPages('noise_per_station.pdf')

                # for each element, calculate its add_noise% 
                for k in range(np.shape(self.TE_Impedance)[1]):# stations
                    for i in range(np.shape(self.TE_Impedance)[0]):# periods 
                        # add error to amplitudes
                        error_TE_amp[i][k] = perc * np.max(TE_amp[i][k])*(sigma*np.random.randn()+mu)
                        error_TM_amp[i][k] = perc * np.max(TM_amp[i][k])*(sigma*np.random.randn()+mu)
                        TE_amp[i][k] =  TE_amp[i][k] + error_TE_amp[i][k]
                        TM_amp[i][k] =  TM_amp[i][k] + error_TM_amp[i][k]
                        # add error to phases
                        error_TE_phase[i][k] = .5*perc * np.max(TE_phase[i][k])*(sigma*np.random.randn()+mu)
                        error_TM_phase[i][k] = .5*perc * np.max(TM_phase[i][k])*(sigma*np.random.randn()+mu)
                        TE_phase[i][k] =  TE_phase[i][k] + error_TE_phase[i][k]
                        TM_phase[i][k] =  TM_phase[i][k] + error_TM_phase[i][k]

                    if plot_noise:
                        # plot histograms of errors per stations
                        f = plt.figure()
                        f.set_size_inches(8,8)
                         
                        #f.suptitle(self.name, size = textsize)
                        errors = [error_TE_amp[:,k], error_TM_amp[:,k], error_TE_phase[:,k], error_TM_phase[:,k]]
                        e_x_label = ['noise TE amp.','noise TM amp.', 'noise TE phase', 'noise TE phase']
                        for i in range(len(errors)): 
                            ax = plt.subplot(2, 2, i+1)
                            b1 = errors[i]
                            bins = np.linspace(np.min(b1), np.max(b1[:-1]), 3*int(np.sqrt(len(b1))))
                            h,e = np.histogram(b1, bins, density = True)
                            m = 0.5*(e[:-1]+e[1:])
                            plt.bar(e[:-1], h, e[1]-e[0])
                            ax.set_xlabel(e_x_label[i], fontsize=10)
                            ax.set_ylabel('freq.', fontsize=10)
                            plt.grid(True, which='both', linewidth=0.1)
                        
                        f.suptitle('station: {:}'.format(k), fontsize=14)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        pp.savefig(f)
                        plt.close(f)

                if plot_noise:
                    pp.close()
                    shutil.move('noise_per_station.pdf','.'+os.sep+self.work_dir +os.sep+'noise_per_station.pdf')

                # from polar to xy 
                for i in range(np.shape(self.TE_Impedance)[0]):# periods 
                    for k in range(np.shape(self.TE_Impedance)[1]):# stations
                        # add half of the error to each component (real and complex) 
                        self.TE_Impedance[i][k] = complex(TE_amp[i][k]*np.cos(TE_phase[i][k]) ,TE_amp[i][k]*np.sin(TE_phase[i][k])) 
                        self.TM_Impedance[i][k] = complex(TM_amp[i][k]*np.cos(TM_phase[i][k]) ,TM_amp[i][k]*np.sin(TM_phase[i][k]))                                   

            else: 
                # abs of perc_noise % magnitud of each impedance
                perc = add_noise/100.
                TE_noise_values = abs(self.TE_Impedance*perc)
                TM_noise_values = abs(self.TM_Impedance*perc)

                # for each element, calculate its add_noise% 
                for i in range(np.shape(self.TE_Impedance)[0]):# periods 
                    for k in range(np.shape(self.TE_Impedance)[1]):# stations
                        # add half of the error to each component (real and complex) 
                        self.TE_Impedance[i][k] = complex((TE_noise_values[i][k] *np.random.randn()) ,(TE_noise_values[i][k] *np.random.randn())) \
                                                + self.TE_Impedance[i][k]
                        self.TM_Impedance[i][k] = complex((TM_noise_values[i][k] *np.random.randn()) ,(TM_noise_values[i][k] *np.random.randn())) \
                                                + self.TM_Impedance[i][k]
		
        # compute apparent resistivity
        om = 2.*np.pi*self.f
        self.TE_Resistivity = np.real(np.conj(self.TE_Impedance)*self.TE_Impedance)/(om*4.*np.pi*1.e-7)
        self.TM_Resistivity = np.real(np.conj(self.TM_Impedance)*self.TM_Impedance)/(om*4.*np.pi*1.e-7)
    
    def write_data(self, filename):
        with open(filename, 'w') as fp:
            for dtype in ['TE_Impedance','TM_Impedance']:				
                fp.write("# data file written from Python\n")
                fp.write("# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error \n")
                fp.write("> %s\n"%dtype)
                fp.write(">  exp(-i\omega t)\n")
                #fp.write("> [V/m]/[T]\n")
                fp.write("> [V/m]/[A/m]\n")
                fp.write("> 0.0\n")
                fp.write("> %8.7e %8.7e\n"%(self.y0, self.z0))
                
                nS = len(self.ys)
                nT = len(self.f[:,0])
                fp.write("> %i %i \n"%(nT, nS))
                dat = self.__getattribute__(dtype)
                for i in range(nS):					
                    for j in range(nT):
                        T = 1./self.f[j,i]
                        xs,ys,zs = [self.xs[i]-self.x0, self.ys[i]-self.y0,self.zs[i]-self.z0]
                        ys = ys + (self.ye[-1]-self.ye[0])/2.
                        fp.write(" %8.7e %03i 0.00 0.00 %8.7e %8.7e %8.7e %s %8.7e %8.7e %8.7e\n"%(T,i+1,xs,ys,zs, dtype[:2], dat[j,i].real, dat[j,i].imag, 1.e-3))
            fp.write("\n")

    def run(self, input, output, exe, verbose = True):
                
        self.verbose = verbose		
        assert os.path.isfile(exe), "no exe at location %s"%exe
        
        # change to working directory
        odir = os.getcwd()		# save old directory
        os.chdir(self.work_dir)
        # write input
        try:
            self.write_input(input)
        except:
            os.chdir(odir)
            traceback.print_exc()
            raise
            
        # write output template
        try:
            template = output.split('.')
            template = template[0]+'_template.'+template[1]			
            self.write_data(template)
        except:
            os.chdir(odir)
            traceback.print_exc()
            raise
        
        # run model
        try:
            command = [exe,]
            # input file
            command.append('-F')	 # forward model
            command.append(input)	
            command.append(template)		
            command.append(output)	
            
            # run simulation
            p = Popen(command,stdout=PIPE)
            if self.verbose:
                # pipe output to screen
    #				for c in iter(lambda: p.stdout.read(1), ''):
    #					sys.stdout.write(c)
                for line in p.stdout:
                    sys.stdout.write(line.decode("utf-8"))
            else:			
                # wait for simulation to finish
                p.communicate()
        except:
            os.chdir(odir)
            traceback.print_exc()
            raise
            
        os.chdir(odir)
    def invert(self, input, output, exe, parameters = None, verbose = True):
                
        self.verbose = verbose		
        assert os.path.isfile(exe), "no exe at location %s"%exe
        
        # change to working directory
        odir = os.getcwd()		# save old directory
        os.chdir(self.work_dir)
        # write input file as template (using output name)
        try:
            self.write_input(output)
        except:
            os.chdir(odir)
            traceback.print_exc()
            raise
            
        # write data file (using input name)
        try:
            self.write_data(input)
        except:
            os.chdir(odir)
            traceback.print_exc()
            raise
        
        # run model
        try:
            command = [exe,]
            # input file
            command.append('-I')	 # forward model
            command.append('NLCG')	 # forward model
            command.append(output)	
            command.append(input)	
            if parameters is not None:
                command.append(parameters)		
            
            print(command)
            # run simulation
            p = Popen(command,stdout=PIPE)
            if self.verbose:
                # pipe output to screen
                #for c in iter(lambda: p.stdout.read(1), ''):
                    #sys.stdout.write(c)
                for line in p.stdout:
                    sys.stdout.write(line.decode("utf-8"))
            else:			
                # wait for simulation to finish
                p.communicate()
        except:
            os.chdir(odir)
            traceback.print_exc()
            raise
            
        os.chdir(odir)
    
    def plot_rho2D(self, filename, xlim = None, ylim = None, gridlines = False, clim = None, overlay = None, overlay_name = None, format = None):
        # check data available for plotting
        try: self.rho
        except:	raise ValueError("no resistivity data to plot")
                
        # plot simple contours 
        plt.clf()
        fig = plt.figure(figsize=[7.5,5.5])
        ax = plt.axes([0.18,0.25,0.70,0.50])
        
        yy,zz = np.meshgrid(self.y,self.z)
        
        rho = self.rho*1.
        #rho = self.rho[25][:][:]

        if clim is None:
            lmin = np.floor(np.min(np.log10(rho)))
            lmax = np.ceil(np.max(np.log10(rho)))
        else:
            lmin = np.floor(np.min(np.log10(clim[0])))
            lmax = np.ceil(np.max(np.log10(clim[1])))
        levels = 10**np.arange(lmin, lmax+0.25, 0.25)

        CS = ax.contourf(yy.T/1.e3,zz.T/1.e3, rho, levels = levels, zorder = 1,norm = LogNorm(), cmap='jet_r')
        cax = plt.colorbar(CS, ax = ax)
        cax.set_label(r'resistivity [$\Omega$ m]', size = textsize)
        
        ax.set_xlabel('y [km]', size = textsize)
        ax.set_ylabel('z [km]', size = textsize)
            
        ax.set_title('Resistivity Distribution', size = textsize)
        
        if overlay is not None:			
            #dat = overlay[0]
            #level = overlay[1]
            #rho = dat.rho*1.
            levels = [overlay[i] for i in range(len(overlay))]			
            CS = ax.contour(yy.T/1.e3,zz.T/1.e3, rho, levels = levels, colors = [[0.4,0.4,0.4],], linewidths = 0.5)#, cmap='jet')
        
            for i in range(len(overlay)):
                x = np.asarray([])
                y = np.asarray([])
                for k in range(len(CS.collections[0].get_paths())): 
                    p = CS.collections[0].get_paths()[k]
                    #p = CS.collections[0].get_paths()[k]
                    v = p.vertices
                    x_aux = v[:,0]
                    y_aux = v[:,1]
                    if len(x_aux) > 10.:
                        x = np.append(x,x_aux)
                        y = np.append(y,y_aux)
                if len(x)<90.: 
                    print(len(x))
                    raise ValueError("overlay: few point for valid contour")
                if overlay_name is None:
                    f = open(self.work_dir+os.sep+"contour_res_{}.txt".format(str(levels[i])), "w")
                else:
                    f = open(overlay_name+'_res_{}.txt'.format(str(levels[i])), "w") 
                f.write('# x\ty\n')
                for j in range(len(x)): 
                    f.write(str(x[j])+'\t'+str(y[j])+'\n')
                f.close()
        
        for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(textsize)
        for t in cax.ax.get_yticklabels(): t.set_fontsize(textsize)
        
        if xlim is not None: ax.set_xlim(xlim)		
        if ylim is not None: ax.set_ylim(ylim)
        
        if gridlines:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for x in self.ye[1:-1]: ax.plot([x/1.e3, x/1.e3],ylim, '-', color = [0.5,0.5,0.5], zorder = 10, lw = 0.5) 
            for y in self.ze[1:-1]: ax.plot(xlim,[y/1.e3, y/1.e3], '-', color = [0.5,0.5,0.5], zorder = 10, lw = 0.5) 
        
        if format is 'png':
            plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        else:
            plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
            
        #plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
        #	orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

        plt.close(fig)
    
    def plot_impedance(self, filename, xlim = None, frequency = True, TE = True, clim = None):
        # plot simple contours 
        plt.clf()
        fig = plt.figure(figsize=[7.5,5.5])
        ax = plt.axes([0.18,0.25,0.70,0.50])
        
        # plot resistivity vs. frequency for the station immediately above the centre
        i = np.argmin(abs(self.ys))
        if frequency: 
            ff = self.f
        else: 
            ff = 1./self.f
        xx,f = np.meshgrid(self.ys/1.e3, self.f[:,i])
        
        if TE:
            rho = self.TE_Resistivity
        else:
            rho = self.TM_Resistivity
                                
        if clim is None:
            lmin = np.floor(np.min(np.log10(rho)))
            lmax = np.ceil(np.max(np.log10(rho)))
        else:
            lmin = np.floor(np.min(np.log10(clim[0])))
            lmax = np.ceil(np.max(np.log10(clim[1])))
        levels = 10**np.arange(lmin, lmax+0.25, 0.25)

        CS = ax.contourf(xx,ff, rho, levels = levels, norm = LogNorm(), cmap='jet')
        cax = plt.colorbar(CS, ax = ax)
        cax.set_label(r'resistivity [$\Omega$ m]', size = textsize)
        
        ax.set_yscale('log')
        
        if frequency:
            ax.set_ylabel('frequency [Hz]', size = textsize)
        else:
            ax.set_ylabel('period [s]', size = textsize)
            ax.invert_yaxis()
        ax.set_xlabel('y [km]', size = textsize)
        
        for t in ax.get_xticklabels()+ax.get_yticklabels()+cax.ax.get_yticklabels(): t.set_fontsize(textsize)
        
        #if xlim is not None: ax.set_xlim(xlim)		
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
        plt.close(fig)

    def intp_1D_prof(self, x_intp, y_intp):
        """
        Inteporlate 1D profile at surface position lat,lon
        (from grid defined by self.x,y,z) 
        Depths of interpolation are grid ones (self.z) 

        Inputs: 
            x_intp: surface position in x axis on grid.
            x_intp: surface position in y axis on grid.

        Return: 
            Vector containing interpolate values at depths given in self.z
        """ 

        ## construct vectors to be used in griddata: points, values and xi
        # (i) construct 'points' and 'values' vectors
        points = np.zeros([self.nx*self.ny*self.nz, 3])
        values = np.zeros([self.nx*self.ny*self.nz])
        # loop over self.nx, ny, nz to fill points and values
        n = 0
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz): 
                    #print(n)
                    points[n] = [self.x[i], self.y[j], self.z[k]]
                    values[n] = self.rho[i,j,k]
                    n+=1
        ## save points in txt file
        #t = open('points.txt', 'w')
        #for i in range(n):
        #    t.write('{}\t{}\t{}\n'.format(points[i][0],points[i][1],points[i][2]))
        #t.close()
        
        # (ii) construct 'xi' vector
        xi = np.zeros([self.nz, 3])
        for k in range(self.nz): 
            xi[k,:] = [x_intp,y_intp,self.z[k]]

        ## (iii) griddata: create interpolate profile 
        grid_z = griddata(points, values, xi, method='nearest')

        return grid_z
    
    def plot_comp_mcmc(self, z_intp, z0, z1_mcmc, z2_mcmc, r2_mcmc):
        """
        Plot profile from 'intp_1D_prof' method with mcmc CC boundaries. 

        Return:
            figure
        """
        # create figure
        f = plt.figure(figsize=[5.5,5.5])
        ## plot profile 3d inv
        ax = plt.axes()
        ax.plot(np.log10(z_intp), self.z -  z0,'m-', label='profile from 3D inv.')
        ## plot mcmc inv 
        #ax.plot(r2_mcmc[0],-1*z1_mcmc[0],'r*')
        #ax.errorbar(r2_mcmc[0],-1*z1_mcmc[0],r2_mcmc[1],'r*')
        ax.errorbar(np.log10(r2_mcmc[0]),-1*z1_mcmc[0],z1_mcmc[1],np.log10(r2_mcmc[1]),'r*', label = 'top bound. CC mcmc')
        #ax.plot(r2_mcmc[0],-1*(z1_mcmc[0] + z2_mcmc[0]),'b*')
        #ax.errorbar(r2_mcmc[0],-1*(z1_mcmc[0] + z2_mcmc[0]),r2_mcmc[1],'b*')
        ax.errorbar(np.log10(r2_mcmc[0]),-1*(z1_mcmc[0] + z2_mcmc[0]),z2_mcmc[1],np.log10(r2_mcmc[1]),'b*', label = 'bottom bound. CC mcmc')
        plt.ylim([-600,0])
        plt.xlim([-1,3])
        ax.legend(loc = 2)
        ax.set_xlabel('Resistivity [Ohm m]')
        ax.set_ylabel('Depth [m]')
        ax.set_title('3Dinv profile and MCMC CC boundaries')
        ax.grid(alpha = 0.3)
        plt.tight_layout()
        #plt.show()

        return f

    def plot_uncert_comp(self, res_intp, z0, z1_mcmc, z2_mcmc, r2_mcmc):
        """
        a. Plot interpolated profile on new sample positions every 1 m 
        b. Plot CC equivalent boundaries in 3Dinv profile compare to MCMC CC boundaries 

        Return: 
            Figures a. and b. 
        """

        ## calculate positions of r2_mcmc in z_inerp (dephts in self.z)
        # (0) resample  res_intp and self.z in dz ~ 1m. 
        # vector to resample in 
        val, idx = find_nearest(self.z, -2000.)
        z_rs = np.arange(self.z[0],self.z[idx]-1.,-1.)
        if False: # cubic spline interpolation 
            log_r_rs = cubic_spline_interpolation(self.z,np.log10(res_intp),z_rs,rev = True)
        if True: # piecewise linear interpolation 
            log_r_rs = piecewise_interpolation(self.z[::-1],np.log10(res_intp[::-1]),z_rs[::-1])
            log_r_rs = log_r_rs[::-1]

        # create figure
        f = plt.figure(figsize=[5.5,5.5])
        ax = plt.axes()
        ax.plot(np.log10(res_intp), self.z -  z0,'b*-', label='profile from 3D inv.')
        ax.plot(log_r_rs, z_rs[:-1] -  z0,'m-', label='rs profile')
        plt.ylim([-1500,0])
        plt.xlim([-1,4])
        ax.legend(loc = 3)
        ax.set_xlabel('log10 Resistivity [Ohm m]')
        ax.set_ylabel('Depth [m]')
        ax.set_title('3Dinv profile interpolation')
        ax.grid(alpha = 0.3)
        plt.tight_layout()
        #plt.show()

        # (1) divide profile in 2 sections to look for upper and lower boundary
        val_mid, idx_mid = find_nearest(abs(log_r_rs - np.min(log_r_rs)), 0.)
        # (2) extract depth (index) of r2_mcmc in res_intp
        
        ## first ocurrence (upper bound)
        # mean
        val, idx = find_nearest(log_r_rs[0:idx_mid], np.log10(r2_mcmc[0]))
        z1_3dinv_mean = z_rs[idx] - z0
        # mean + std
        val, idx = find_nearest(log_r_rs[0:idx_mid], np.log10(r2_mcmc[0]+r2_mcmc[1]))
        z1_3dinv_mean_plus_std = z_rs[idx] - z0
        # mean - std
        val, idx = find_nearest(log_r_rs[0:idx_mid], np.log10(r2_mcmc[0]-r2_mcmc[1]))
        z1_3dinv_mean_minus_std = z_rs[idx] - z0

        ## second ocurrence 
        # mean
        val, idx = find_nearest(log_r_rs[idx_mid:], np.log10(r2_mcmc[0]))
        z2_3dinv_mean = z_rs[idx+idx_mid] - z0
        # mean + std
        val, idx = find_nearest(log_r_rs[idx_mid:], np.log10(r2_mcmc[0]+r2_mcmc[1]))
        z2_3dinv_mean_plus_std = z_rs[idx+idx_mid] - z0
        # mean - std
        val, idx = find_nearest(log_r_rs[idx_mid:], np.log10(r2_mcmc[0]-r2_mcmc[1]))
        z2_3dinv_mean_minus_std = z_rs[idx+idx_mid] - z0

        # create figure
        g = plt.figure(figsize=[5.0,5.0])
        ax = plt.axes()
        # plot 3dinv bounds
        ax.plot(0.,z1_3dinv_mean,'ro')
        ax.plot(0.,z2_3dinv_mean,'bo')
        ax.plot([0.,0.],[z1_3dinv_mean_plus_std,z1_3dinv_mean_minus_std],'r-', label = 'mcmc inv, top bound.')
        ax.plot([0.,0.],[z2_3dinv_mean_plus_std,z2_3dinv_mean_minus_std],'b-', label = 'mcmc inv, bottom bound.')

        # plot mcmc bounds
        ax.plot(1.,-1*z1_mcmc[0],'r*')
        ax.plot(1.,-1*(z1_mcmc[0] + z2_mcmc[0]),'b*')
        ax.plot([1.,1.],[-1*(z1_mcmc[0] - z1_mcmc[1]),-1*(z1_mcmc[0] + z1_mcmc[1])],'r--', label = '3D inv, top bound.')
        ax.plot([1.,1.],[-1*(z1_mcmc[0] + z2_mcmc[0] - z2_mcmc[1]),-1*(z1_mcmc[0] + z2_mcmc[0] + z2_mcmc[1])],'b--', label = '3D inv, bottom bound.')

        #ax.plot(np.log10(res_intp), self.z -  z0,'b*-', label='profile from 3D inv.')
        #ax.plot(log_r_rs, z_rs -  z0,'m-', label='rs profile')
        #plt.ylim([-500,0.])
        plt.xlim([-1,5])
        ax.legend(loc = 0)
        #ax.set_xlabel('log10 Resistivity [Ohm m]')
        ax.set_ylabel('Depth [m]')
        ax.set_title('Comparition CC bound: 3Dinv and MCMCinv')
        ax.grid(alpha = 0.3)
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        plt.tight_layout()
        #plt.show()

        return f, g

    def plot_true_CC_bound(self,filename, xlim = None, ylim = None, gridlines = False):
        # check data available for plotting
        try: self.rho
        except:	raise ValueError("no resistivity data to plot")
                
        # plot simple contours 
        plt.clf()
        fig = plt.figure(figsize=[7.5,5.5])
        ax = plt.axes([0.18,0.25,0.70,0.50])
        
        yy,zz = np.meshgrid(self.y,self.z)

        level = [1,10]
  
        CS = ax.contourf(yy.T/1.e3,zz.T/1.e3, rho, levels = level)
        cax = plt.colorbar(CS, ax = ax)
        cax.set_label(r'resistivity [$\Omega$ m]', size = textsize)
        
        ax.set_xlabel('y [km]', size = textsize)
        ax.set_ylabel('z [km]', size = textsize)
 
        plt.show()
        asdf
            
        ax.set_title('Resistivity Distribution', size = textsize)
        for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(textsize)
        for t in cax.ax.get_yticklabels(): t.set_fontsize(textsize)
        
        if xlim is not None: ax.set_xlim(xlim)		
        if ylim is not None: ax.set_ylim(ylim)

        if gridlines:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for x in self.ye[1:-1]: ax.plot([x/1.e3, x/1.e3],ylim, '-', color = [0.5,0.5,0.5], zorder = 10, lw = 0.5) 
            for y in self.ze[1:-1]: ax.plot(xlim,[y/1.e3, y/1.e3], '-', color = [0.5,0.5,0.5], zorder = 10, lw = 0.5) 
        
        plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
            
        #plt.savefig(filename, dpi=300, facecolor='w', edgecolor='w',
        #	orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

        plt.close(fig)

# Functions for constructing resistivity model
def f(x): 
	return (1-6*x**2+4*x**3)/np.sqrt(1-12*x**2+24*x**3-12*x**4)
def df(x): 
	return (12*x**2-12*x)/np.sqrt(1-12*x**2+24*x**3-12*x**4) - (-48*x**3+72*x**2-24*x)*(4*x**3-6*x**2+1)/np.sqrt(1-12*x**2+24*x**3-12*x**4)**3/2.
def rho(x, par):
	xm, xw, xel, xer, rho0, rhoa = par
	# check iterable
	try: iter(x) 
	except TypeError: x = [x]
	x = np.array(x)
	# apply transfer
	rho = []
	drho = np.log10(rhoa) - np.log10(rho0)
	
	for xi in x:
		if (xi<(xm-xw/2-xel/2)) or (xi>(xm+xw/2+xer/2)): 
			rho.append(rho0)
		elif (xi>(xm-xw/2+xel/2)) and (xi<(xm+xw/2-xer/2)): 
			rho.append(rhoa)
		elif (xi<=(xm-xw/2+xel/2)) and (xi>=(xm-xw/2-xel/2)):
			rho.append(10**(np.log10(rho0) + (1-f((xi - (xm-xw/2-xel/2))/xel))/2.*drho))
		else:
			rho.append(10**(np.log10(rho0) + (1+f((xi - (xm+xw/2-xer/2))/xer))/2.*drho))
	return np.array(rho)
def rho_noise(x, par, var_rho0, var_rhoa):
	xm, xw, xel, xer, rho0, rhoa = par
	
# check iterable
	try: iter(x) 
	except TypeError: x = [x]
	x = np.array(x)
	# apply transfer
	rho = []
	drho = np.log10(rhoa) - np.log10(rho0)
	
	for xi in x:
		if (xi<(xm-xw/2-xel/2)) or (xi>(xm+xw/2+xer/2)): 								# range 1 and 5
			mu = rho0
			sigma = np.sqrt(var_rho0)
			noise = sigma * np.random.randn()
			rho0_noise = rho0 + noise
			rho.append(rho0_noise)
		elif (xi>(xm-xw/2+xel/2)) and (xi<(xm+xw/2-xer/2)):  							# range 3
			mu = rhoa
			sigma = np.sqrt(var_rhoa)
			noise = sigma * np.random.randn()
			rhoa_noise = rhoa + noise
			rho.append(rhoa_noise)
		elif (xi<=(xm-xw/2+xel/2)) and (xi>=(xm-xw/2-xel/2)): 							# range 2
			mu = rho0
			sigma = np.sqrt(var_rho0)
			noise = sigma * np.random.randn()
			rho0_noise = rho0 + noise
			rho.append(10**(np.log10(rho0_noise) + (1-f((xi - (xm-xw/2-xel/2))/xel))/2.*drho)) 
		else:																			 # range 4
			mu = rho0
			sigma = np.sqrt(var_rho0)
			noise = sigma * np.random.randn()
			rho0_noise = rho0 + noise
			rho.append(10**(np.log10(rho0_noise) + (1+f((xi - (xm+xw/2-xer/2))/xer))/2.*drho)) 
	
	return np.array(rho)
def rho2(xi, xm, xw, xel, xer, rho0, rhoa):
	if xw<0.: return 1.e32
	drho = np.log10(rhoa) - np.log10(rho0)
	if (xi<(xm-xw/2-xel)) or (xi>(xm+xw/2+xer)): 
		return rho0
	elif (xi>(xm-xw/2)) and (xi<(xm+xw/2)): 
		return rhoa
	elif (xi<=(xm-xw/2)) and (xi>=(xm-xw/2-xel)):
		return 10**(np.log10(rho0) + (1-f((xi - (xm-xw/2-xel))/xel))/2.*drho)
	else:
		return 10**(np.log10(rho0) + (1+f((xi - (xm+xw/2))/xer))/2.*drho)
def rho3(x, *par): return np.array([rho2(xi,*par) for xi in x])
def setup_simulation(work_dir = None):
	# use the same grid for all simulations
	AUTOUGHGeo = r'D:\ModEM\modem_python\00_Temp_model_grid\g31kBlock_vEWA_BH.dat' # Office
	#AUTOUGHGeo = r'C:\Users\ajara\Desktop\Temp_file\g31kBlock_vEWA_BH.dat' # House

	geo = mulgrid(AUTOUGHGeo)		# get tough 2 geometry

	# layer edges
	ze = np.array([l.bottom for l in geo.layerlist])
	ze -= np.max(ze)
	# column edges
	xe = np.unique([c.bounding_box[0][0] for c in geo.columnlist]+[c.bounding_box[1][0] for c in geo.columnlist])
	ye = np.unique([c.bounding_box[0][1] for c in geo.columnlist]+[c.bounding_box[1][1] for c in geo.columnlist])
	
	# read temperature data and assign conductivity model
	x,y,z,T = np.genfromtxt(r'D:\ModEM\modem_python\00_Temp_model_grid\31kBlockEWA_BH_final_xyz_T.dat', delimiter = ',').T # office
	#x,y,z,T = np.genfromtxt(r'C:\Users\ajara\Desktop\Temp_file\31kBlockEWA_BH_final_xyz_T.dat', delimiter = ',').T  # house
	
	# get min x slice of data
	xmin = np.min(x); tol = 1.e-6
	inds = np.where(abs(x-xmin)<tol)
	y = y[inds]
	z = z[inds]
	z = z - np.max(z)
	T = T[inds]
	
	# interpolate temperatures to finer grid in z direction
	ynew = np.unique(y)
	
	ymax = 5.e3
	dy = 100.
	yu = np.unique(y)
	i = np.argmin(abs(yu - ymax))		
	ynew = list(np.linspace(dy/2.,ymax+dy/2., abs(ymax)/dy+1))+ list(yu[i+1:])
	ye = list(np.linspace(0,ye[i+1], abs(ye[i+1])/dy+1))+ list(ye[i+2:])
	
	zmin = -1.4e3
	dz = 50.
	zu = np.flipud(np.unique(z))
	i = np.argmin(abs(zu - zmin))		
	znew = list(np.linspace(-dz/2.,zmin-dz/2., abs(zmin)/dz+1))+ list(zu[i+1:])
	ze = list(np.linspace(0,ze[i+1], abs(ze[i+1])/dz+1))+ list(ze[i+2:])
			
	[ynew,znew] = np.meshgrid(ynew, znew)
	ynew = ynew.flatten()
	znew = znew.flatten()
	Tnew = griddata(np.array([y,z]).T,T,np.array([ynew,znew]).T, method='linear', fill_value = np.min(T))
	yi,zi,Ti = np.array([ynew,znew,Tnew])
	
	# create a grid
	dat = Modem(work_dir = work_dir)
	x = []
	base = 10
	
	ny = 10
	y = np.array(list(ye) + list(np.logspace(np.log10(ye[-1]),np.log10(80.e3),ny+1,base))[1:])
	y = list(reversed(-y))[:-1] + list(y)
	
	nz = 10
	z = list(ze) + list(-np.logspace(np.log10(-ze[-1]),np.log10(80.e3),nz+1,base))[1:]
	
	dat.make_grid(x,y,z)
	
	nT = 41
	nS = 41
	base = 10
	T = np.logspace(np.log(0.01)/np.log(base), np.log(400)/np.log(base), nT, base = base) 		
	dat.stations(np.linspace(-10.e3,10.e3,nS),0.,1./T)
	
	return dat, [yi,zi,Ti]
def set_rho(dat, yi, zi, rhoi, rhob):
    rho_crust, rho_greywacke, rho_fill = rhob
    dat.rho_box(rho_crust) 		# lower crust
    dat.rho_box(rho_greywacke, box = [-80.e3,-10.e3,160.e3,8.3e3]) 	# greywacke
    dat.rho_box(rho_fill, box = [-80.e3,-0.e3,160.e3,2.e3]) 	# fill
    # rigth side
    dat.rho_blocks(rhoi, yi, zi[:np.floor(len(zi)/3)])
    dat.rho_blocks(rhoi, -yi,zi[:np.floor(len(zi)/2)])

    # left side
    #dat.rho_blocks(rhoi, -yi,zi[:np.floor(len(zi)/1)])
    #if mod == 2:
    #    dat.rho_blocks(rhoi, -yi,zi[:np.floor(len(zi)/1)])
    #else: 
    #    dat.rho_blocks(rhoi, -yi,zi[:np.floor(len(zi)/3)])
    #dat.plot_rho2D(dat.work_dir+os.sep+'rho.png', xlim = [-12., 12.], ylim = [-5.,0], gridlines = True, clim = [1,1000])	
    return dat

def run_simulation(dat, dir0 = None):
	# run model
	dat.run(input = 'in.dat', output = 'out.dat', exe = r'D:\ModEM\ModEM\f90\Mod2DMT.exe')		
	# visualize input and output		
	dat.read_data(dat.work_dir+os.sep+'out.dat')	
	
	if dir0 is None:
		with open(dat.work_dir+os.sep+'TE_Resistivity0.txt','w') as fp:
			for tei in dat.TE_Resistivity.flatten():
				fp.write("%8.7e\n"%tei)
		with open(dat.work_dir+os.sep+'TM_Resistivity0.txt','w') as fp:
			for tei in dat.TM_Resistivity.flatten():
				fp.write("%8.7e\n"%tei)
	else:
		TE_Resistivity0 = np.genfromtxt(dir0+os.sep+'TE_Resistivity0.txt')
		TM_Resistivity0 = np.genfromtxt(dir0+os.sep+'TM_Resistivity0.txt')
		
		rms = np.sum((dat.TE_Resistivity.flatten() - TE_Resistivity0)**2)
		rms += np.sum((dat.TM_Resistivity.flatten() - TM_Resistivity0)**2)
		rms = np.sqrt(rms/(2.*dat.TE_Resistivity.shape[0]*dat.TM_Resistivity.shape[1]))
	
		with open(dat.work_dir+os.sep+'rms.txt','w') as fp:
			fp.write("%8.7e\n"%rms)
def resistivity_model(par, wd):
	# run a proposed case
	plt.clf()
	fig = plt.figure(figsize=[7.5,5.5])
	ax = plt.axes([0.18,0.25,0.70,0.50])
			
	T = np.linspace(50,300,1001)
	ax.plot(T,rho(T, par),'b*')
	ax.set_yscale('log')

	ax.set_ylim(0.1,1000)
	ax.set_ylabel(r'resistivity / $\Omega$ m', size = textsize)
	ax.set_xlabel(r'temperature / $^\circ$C', size = textsize)
	
	plt.savefig(wd+os.sep+'resistivity_model.pdf', dpi=300, facecolor='w', edgecolor='w',
		orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
	plt.close(fig)
	
	# save parameter
	with open(wd+os.sep+'parameters.txt','w') as fp:
		fp.write("xm = %8.7f\n"%par[0])
		fp.write("xw = %8.7f\n"%par[1])
		fp.write("xel = %8.7f\n"%par[2])
		fp.write("xer = %8.7f\n"%par[3])
		fp.write("rhoa = %8.7f\n"%par[5])
def resistivity_model_noise(par, wd, var_rho0, var_rhoa):
	# run a proposed case
	plt.clf()
	fig = plt.figure(figsize=[7.5,5.5])
	ax = plt.axes([0.18,0.25,0.70,0.50])
			
	T = np.linspace(50,300,1001)
	ax.plot(T,rho_noise(T, par, var_rho0, var_rhoa),'r*')
	ax.plot(T,rho(T, par),'b-')
	ax.set_yscale('log')
	
	ax.set_ylim(1,3000)
	ax.set_ylabel(r'resistivity [$\Omega$ m]', size = textsize)
	ax.set_xlabel(r'temperature [$^\circ$C]', size = textsize)
	ax.set_title(r'$\rho (T)$ model', size = textsize)
	plt.savefig(wd+os.sep+'resistivity_model_noise.pdf', dpi=300, facecolor='w', edgecolor='w',
		orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
	plt.savefig(wd+os.sep+'resistivity_model_noise.png', dpi=300, facecolor='w', edgecolor='w',
		orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)

	plt.close(fig)
	
	# save parameter
	with open(wd+os.sep+'parameters.txt','w') as fp:
		fp.write("xm = %8.7f\n"%par[0])
		fp.write("xw = %8.7f\n"%par[1])
		fp.write("xel = %8.7f\n"%par[2])
		fp.write("xer = %8.7f\n"%par[3])
		fp.write("rhoa = %8.7f\n"%par[5])
def	run_one_simulation(sample):
	i, xm, xw, xel, xer, rhoa, rho_crust, rho_greywacke, rho_fill = sample
	print("\n\nSAMPLE %i\n\n"%i)
	par_run = [xm,xw,xel,xer,rho_fill,rhoa]
	dat, t2t  = setup_simulation(work_dir = 'run%04i'%i)
	yi,zi,Ti = t2t
	rhoi = rho(Ti, par_run)
	dat = set_rho(dat, yi, zi, rhoi, sample[-3:])
	resistivity_model(par_run, dat.work_dir)				
	run_simulation(dat, 'true')
	os.remove(dat.work_dir+os.sep+'out_template.dat')
def get_T(y,z,T,ystations,zinterp):
	yi,zi = np.meshgrid(ystations, zinterp)	
	Tinterp = griddata(np.array([y,z]).T,T,np.array([yi.flatten(),zi.flatten()]).T,method='linear')		
	return zi,Tinterp.reshape([len(zinterp), len(ystations)]).T
def get_rho(dat, ystations, zinterp):	
	yi,zi = np.meshgrid(ystations, zinterp)	
	yy,zz = np.meshgrid(dat.y, dat.z)
	Tinterp = griddata(np.array([yy.flatten(),zz.flatten()]).T,dat.rho.T.flatten(),np.array([yi.flatten(),zi.flatten()]).T,method='linear')		
	return zi,Tinterp.reshape([len(zinterp), len(ystations)]).T
def cp(x, m, su, sl, a, b): 
	return np.array([b-a*su*cauchy.pdf(xi, m, su) if xi>m else b-a*sl*cauchy.pdf(xi, m, sl) for xi in x])
def Texp(z,a,b,c): return 275.*(np.exp(-a*(z-c)/b)-1)/(np.exp(a) - 1)+25.
def normal(z, m, s, a, b): return b-a*np.exp(-(z-m)**2/(2*s**2))
def Texp2(z,Zmax,Zmin,Tmin,Tmax,beta): return (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta)-1) + Tmin
def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]
def T_est_demo(beta, z, Zmin, Zmax, Tmin, Tmax):
	z = np.asarray(z)
	# look where the normal estimation is negative: built the Test from there  
	#if any(TT[:,sta_obj.pos] <= 0.):
	# if sta_obj.pos >= 6:
	if False:
		inds_z_inicial_aux = np.where(TT[:,sta_obj.pos] <= 0. )
		inds_z_inicial = np.min(inds_z_inicial_aux) + sta_obj.pos##+ (12 - abs(sta_obj.pos-8)) # + abs(j-11)*1
		#if j == 4:
		#	inds_z_inicial = np.min(inds_z_inicial_aux) + (j-3)*4
						
		# find index of boundaries (Z)				
		#layer 1
		inds_z = np.where(z == find_nearest(z, Zmax[0]))
		inds_z_l1 = int(inds_z[0][0])
		# layer 2
		inds_z = np.where(z == find_nearest(z, Zmax[1]))
		inds_z_l2 = int(inds_z[0][0])
		# layer 3
		inds_z = np.where(z == find_nearest(z, Zmax[2]))
		inds_z_l3 = int(inds_z[0][0])
		# construct profile by layer						
		# def Texp2(z,Zmax,Zmin,Tmin,Tmax,beta): 
		# return (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta)-1) + Tmin
		Test_l1 = Texp2(z[inds_z_l1:len(z)],Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta[0])
		Test_l2 = Texp2(z[inds_z_l2:inds_z_l1],Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta[1])
		Test_l3 = Texp2(z[inds_z_l3:inds_z_l2],Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta[2])

		Test = np.concatenate((Test_l3, Test_l2, Test_l1),axis=0)
					
		if inds_z_inicial < len(z):
			Test[0:inds_z_inicial-1] = Test[len(z)-inds_z_inicial+1:len(z)]
			Test[inds_z_inicial-1:len(z)] = Test[inds_z_inicial-2]
	else: 
		inds_z_inicial = len(z)
		#print('hola')
		# find index of boundaries (Z)				
		#  layer 1
		inds_z = np.where(z == find_nearest(z, Zmax[0]))
		inds_z_l1 = int(inds_z[0][0])
		# layer 2
		inds_z = np.where(z == find_nearest(z, Zmax[1]))
		inds_z_l2 = int(inds_z[0][0])
		# layer 3
		inds_z = np.where(z == find_nearest(z, Zmax[2]))
		inds_z_l3 = int(inds_z[0][0])
		# construct profile by layer
		Test_l1 = Texp2(z[inds_z_l1:len(z)],Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta[0])
		Test_l2 = Texp2(z[inds_z_l2:inds_z_l1],Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta[1])
		Test_l3 = Texp2(z[inds_z_l3:inds_z_l2],Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta[2])

		Test = np.concatenate((Test_l3, Test_l2, Test_l1),axis=0)
						
	#print(np.mean(Test - sta_obj.temp_profile[1]))
	return Test
def T_BC_trans(Zmin, Zmax, slopes, obj):
					
	sigma = 5. # std from the isotherm value 
					
	Tmin_l1 = obj.temp_profile[1][-1]
	Tmax_l1 = np.random.normal(levels[0], sigma, 1)[0] #levels[0] 
					
	Tmin_l2 = Tmax_l1
	Tmax_l2 = np.random.normal(levels[1], sigma, 1)[0] #levels[1] 
	Tmin_l3 = Tmax_l2
	Tmax_l3 = Tmin_l3 + slopes[2]*(Zmin[2]-Zmax[2])
					
	Tmin = [Tmin_l1, Tmin_l2, Tmin_l3] 
	Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
					
	#print(Tmax)
	#print(sta_obj.temp_profile[1][0])

	return Tmin, Tmax	
	# Functions for MCMMC inversion  	
def s_depth(period,rho):
    return 500*np.sqrt(rho*period)
def anaMT1D_f_nlayers(h,rho,T):
    # Base parameters:
    mu=4*math.pi*(10**-7)             # Electrical permitivity [Vs/Am]
    phi = T.copy()
    rho_ap = T.copy()

    # Parameters:Thicknesses and resistivities (funtion of number of layers)
    n_layers= len(h) # number of layers
    #layers=np.linspace(1,n_layers,n_layers)
    #h = np.linspace(1,n_layers,n_layers) # Thickness
    #h[0] = h_1
    #h[1] = h_2
    #h[2] = h_3
    #rho = np.linspace(1,n_layers+1,n_layers+1)
    #rho[0] = rho_1
    #rho[1] = rho_2
    #rho[2] = rho_3
    #rho_hs = rho_hs # half space

    # Recursion
    for k in range(0,len(T)):
        pe=T[k]
        omega=(2*math.pi)/pe
        # Half space parameters
        gamM = cmath.sqrt(1j*omega*mu*(1/rho[-1]))
        C = 1/gamM
        # Interaction: inferior layer -> superior layer
        for l in range(0,n_layers):
            gam = cmath.sqrt(1j*omega*mu*(1/rho[(n_layers-1)-l]))
            r = (1-(gam*C))/(1+(gam*C))
            C=(1-r*cmath.exp(-2*gam*h[n_layers-(l+1)]))/(gam*(1+r*cmath.exp(-2*gam*h[n_layers-(l+1)])))
        Z=1j*omega*C                                              # Impedance
        phi[k]= (math.atan(Z.imag/Z.real))*360/(2*math.pi)        # Phase in degrees
        rho_ap[k]=(mu/omega)*(abs(Z)**2)                          # Apparent resistivity 
    
    return rho_ap, phi
def anaMT1D_f_3layers(h_1,h_2,h_3,rho_1,rho_2,rho_3,rho_hs,T):
    # Base parameters:
    mu=4*math.pi*(10**-7)             # Electrical permitivity [Vs/Am]
    phi = T.copy()
    rho_ap = T.copy()

    # Parameters:Thicknesses and resistivities (funtion of number of layers)
    n_layers= 3 # number of layers
    layers=np.linspace(1,n_layers,n_layers)
    h = np.linspace(1,n_layers,n_layers) # Thickness
    h[0] = h_1
    h[1] = h_2
    h[2] = h_3
    rho = np.linspace(1,n_layers+1,n_layers+1)
    rho[0] = rho_1
    rho[1] = rho_2
    rho[2] = rho_3
    rho_hs = rho_hs # half space

    # Recursion
    for k in range(0,len(T)):
        pe=T[k]
        omega=(2*math.pi)/pe
        # Half space parameters
        gamM = cmath.sqrt(1j*omega*mu*(1/rho_hs))
        C = 1/gamM
        # Interaction: inferior layer -> superior layer
        for l in range(0,n_layers):
            gam = cmath.sqrt(1j*omega*mu*(1/rho[(n_layers-1)-l]))
            r = (1-(gam*C))/(1+(gam*C))
            C=(1-r*cmath.exp(-2*gam*h[n_layers-(l+1)]))/(gam*(1+r*cmath.exp(-2*gam*h[n_layers-(l+1)])))
        Z=1j*omega*C                                              # Impedance
        phi[k]= (math.atan(Z.imag/Z.real))*360/(2*math.pi)        # Phase in degrees
        rho_ap[k]=(mu/omega)*(abs(Z)**2)                          # Apparent resistivity 
    
    return rho_ap, phi
def anaMT1D_f_2layers(h_1,h_2,rho_1,rho_2,rho_hs,T):
    # Base parameters:
    mu=4*math.pi*(10**-7)             # Electrical permitivity [Vs/Am]
    phi = T.copy()
    rho_ap = T.copy()

    # Parameters:Thicknesses and resistivities (funtion of number of layers)
    n_layers= 2 # number of layers
    layers=np.linspace(1,n_layers,n_layers)
    h = np.linspace(1,n_layers,n_layers) # Thickness
    h[0] = h_1
    h[1] = h_2
    rho = np.linspace(1,n_layers+1,n_layers+1)
    rho[0] = rho_1
    rho[1] = rho_2
    rho_hs = rho_hs # half space

    # Recursion
    for k in range(0,len(T)):
        pe=T[k]
        omega=(2*math.pi)/pe
        # Half space parameters
        gamM = cmath.sqrt(1j*omega*mu*(1/rho_hs))
        C = 1/gamM
        # Interaction: inferior layer -> superior layer
        for l in range(0,n_layers):
            gam = cmath.sqrt(1j*omega*mu*(1/rho[(n_layers-1)-l]))
            r = (1-(gam*C))/(1+(gam*C))
            C=(1-r*cmath.exp(-2*gam*h[n_layers-(l+1)]))/(gam*(1+r*cmath.exp(-2*gam*h[n_layers-(l+1)])))
        Z=1j*omega*C                                              # Impedance
        phi[k]= (math.atan(Z.imag/Z.real))*360/(2*math.pi)        # Phase in degrees
        rho_ap[k]=(mu/omega)*(abs(Z)**2)                          # Apparent resistivity 
    
    return rho_ap, phi
def num_model_mesh(periods,target_depth,depth_layer1,depth_layer2,depth_layer3,rho_layer1,rho_layer2,rho_layer3,
                   rho_hs,thickness_layer1,thickness_layer2,thickness_layer3): 
    
    p_min = np.min(periods)   # s
    p_max = np.max(periods)   # s
    
    # skin depth 
    s_depth_min = s_depth(p_min,rho_hs)
    s_depth_max = s_depth(p_max,rho_hs)
    cell_size =  s_depth_min/4   # m
    depth_extantion = 2*s_depth_max # m 
    
    n_cells = int(np.floor(target_depth/cell_size))        # number of cells
    p_cells = int(np.floor(n_cells/4))                     # padding cells
    
    v4TensorMesh =  [(cell_size, p_cells, -1.3), (cell_size, n_cells)]  # (?) -1.3
    mesh = Mesh.TensorMesh([v4TensorMesh],'N') 
    
    # Model (layers)
    sigma_model = 1./rho_hs * np.ones(mesh.nC)
    
    # find the indices of the layer 1
    layer_inds_layer1 = (
        (mesh.vectorCCx<=-depth_layer1) & 
        (mesh.vectorCCx>-(depth_layer1+thickness_layer1))
    )
    
    # find the indices of the layer 2
    layer_inds_layer2 = (
        (mesh.vectorCCx<=-depth_layer2) & 
        (mesh.vectorCCx>-(depth_layer2+thickness_layer2))
    )
    # find the indices of the layer 3
    layer_inds_layer3 = (
        (mesh.vectorCCx<=-depth_layer3) & 
        (mesh.vectorCCx>-(depth_layer3+thickness_layer3))
    )
    # Build de model
    sigma_model[layer_inds_layer1] = 1./rho_layer1  # add layer 1
    sigma_model[layer_inds_layer2] = 1./rho_layer2  # add layer 2
    sigma_model[layer_inds_layer3] = 1./rho_layer3  # add layer 3
    
    return sigma_model, mesh	
def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return int(idx), array[idx]
def closest_wells(well_pos, pos):
	# search left of the station
	well_pos_left = []
	for i in well_pos: 
		if i < pos:
			well_pos_left.append(i)
	# search rigth of the station
	well_pos_rigth = []
	for i in well_pos: 
		if i > pos:
			well_pos_rigth.append(i)
	closest_wells_2 = [well_pos_left[-1],well_pos_rigth[0]]
	return closest_wells_2
	

def est_rest_cc_prior(pos, distances):
	#(0) well to consider : 2 nearest
	n = 2
	n_wells =  [i for i in range(n)]
	#(1) Search for two neighbor wells  # well_pos =  [0,3,9], sta_pos = [...rest...]
	#dist_pos_well = [i-pos for i in well_pos]
	#dist_pos_well = [abs(i) for i in  dist_pos_well]
	#indx_sort = np.argsort(dist_pos_well)
	#closest_wells = [well_pos[indx_sort[i]] for i in n_wells] # location of closest 2 wells (in ystation vector) 
	
	#search left of the station
	well_pos_left = []
	for i in well_pos: 
		if i < pos:
			well_pos_left.append(i)
	#search rigth of the station
	well_pos_rigth = []
	for i in well_pos: 
		if i > pos:
			well_pos_rigth.append(i)
	closest_wells = [well_pos_left[-1],well_pos_rigth[0]]
	
	#closest_wells = closest_wells(well_pos, pos)
	
	#(2) Calculate distance between station and wells (2)
	dist_sta_wells = [distances[i] for i in closest_wells]	
	#(3) Calculate resistivity for prior 
	idx_aux = [find_nearest(well_pos,closest_wells[i]) for i in range(n)] # index of closest wells in well_pos
	idx_aux = [idx_aux[i][0] for i in range(len(idx_aux))]
	rest_cc_nwells = [well_objects[idx_aux[i]].clay_cap_resistivity for i in range(n)] # cc resistivities in closest wells
	# (4) Calculate prior res for estation as ponderate sum of cc rest of closest wells  
	rest_cc_wells_vec = [rest_cc_nwells[i][0]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
	rest_cc_wells = sum(rest_cc_wells_vec)
	return rest_cc_wells
def est_rest_l1_l3_prior(pos, distances):
	#(0) well to consider : 2 nearest
	n = 2
	n_wells =  [i for i in range(n)]
	#(1) Search for two neighbor wells  # well_pos =  [0,3,9], sta_pos = [...rest...]
	#dist_pos_well = [i-pos for i in well_pos]
	#dist_pos_well = [abs(i) for i in  dist_pos_well]
	#indx_sort = np.argsort(dist_pos_well)
	#closest_wells = [well_pos[indx_sort[i]] for i in n_wells] # location of closest 2 wells (in ystation vector) 
	# search left of the station
	well_pos_left = []
	for i in well_pos: 
		if i < pos:
			well_pos_left.append(i)
	#search rigth of the station
	well_pos_rigth = []
	for i in well_pos: 
		if i > pos:
			well_pos_rigth.append(i)
	closest_wells = [well_pos_left[-1],well_pos_rigth[0]]
	
	#closest_wells = closest_wells(well_pos, pos)
	#(2) Calculate distance between station and wells (2)
	dist_sta_wells = [distances[i] for i in closest_wells]	
	#(3) Calculate resistivity for prior 
	idx_aux = [find_nearest(well_pos,closest_wells[i]) for i in range(n)] # index of closest wells in well_pos
	idx_aux = [idx_aux[i][0] for i in range(len(idx_aux))]
	rest_l1_nwells = [well_objects[idx_aux[i]].l1_resistivity for i in range(n)] # layer 1 resistivities in closest wells
	rest_hs_nwells = [well_objects[idx_aux[i]].hs_resistivity for i in range(n)] # layer 1 resistivities in closest wells
	# (4) Calculate prior res for estation as ponderate sum of cc rest of closest wells  
	# layer 1
	rest_l1_wells_vec = [rest_l1_nwells[i]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
	rest_l1_wells = sum(rest_l1_wells_vec)
	# layer hs
	rest_hs_wells_vec = [rest_hs_nwells[i]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
	rest_hs_wells = sum(rest_hs_wells_vec)
	return rest_l1_wells, rest_hs_wells
def est_cc_bound_prior(pos, distances):
	##(0) well to consider : 2 nearest
	n = 2
	n_wells =  [i for i in range(n)]
	##(1) Search for two neighbor wells  # well_pos =  [0,3,9], sta_pos = [...rest...]
	##dist_pos_well = [i-pos for i in well_pos]
	##dist_pos_well = [abs(i) for i in  dist_pos_well]
	##indx_sort = np.argsort(dist_pos_well)
	##closest_wells = [well_pos[indx_sort[i]] for i in n_wells] # location of closest 2 wells (in ystation vector) 
	# search left of the station
	well_pos_left = []
	for i in well_pos: 
		if i < pos:
			well_pos_left.append(i)
	#search rigth of the station
	well_pos_rigth = []
	for i in well_pos: 
		if i > pos:
			well_pos_rigth.append(i)
	closest_wells = [well_pos_left[-1],well_pos_rigth[0]]
	
	#closest_wells = closest_wells(well_pos, pos)
	##(2) Calculate distance between station and wells (2)
	dist_sta_wells = [distances[i] for i in closest_wells]	
	##(3) Calculate boundaries for prior 
	idx_aux = [find_nearest(well_pos,closest_wells[i]) for i in range(n)] # index of closest wells in well_pos
	idx_aux = [idx_aux[i][0] for i in range(len(idx_aux))]
	bounds_cc_nwells = [well_objects[idx_aux[i]].clay_cap_boundaries for i in range(n)] # cc resistivities in closest wells
	##(4) Calculate prior res for estation as ponderate sum of cc rest of closest wells 
	upper_bound_cc_wells_vec = [bounds_cc_nwells[i][0]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
	upper_bound_cc_wells = sum(upper_bound_cc_wells_vec)
	lower_bound_cc_wells_vec = [bounds_cc_nwells[i][1]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
	lower_bound_cc_wells = sum(lower_bound_cc_wells_vec)
	return upper_bound_cc_wells, lower_bound_cc_wells	
def est_sigma_thickness_prior(pos, distances):
	##(0) well to consider : 2 nearest
	n = 2
	n_wells =  [i for i in range(n)]
	#search left of the station
	well_pos_left = []
	for i in well_pos: 
		if i < pos:
			well_pos_left.append(i)
	#search rigth of the station
	well_pos_rigth = []
	for i in well_pos: 
		if i > pos:
			well_pos_rigth.append(i)
	closest_wells = [well_pos_left[-1],well_pos_rigth[0]]
	
	#closest_wells = closest_wells(well_pos, pos)
	
	##(2) Search for values of paramter in wells and maximum variablity between them
	idx_aux = [find_nearest(well_pos,closest_wells[i]) for i in range(n)] # index of closest wells in well_pos
	idx_aux = [idx_aux[i][0] for i in range(len(idx_aux))]
	bounds_cc_nwells = [well_objects[idx_aux[i]].clay_cap_boundaries for i in range(n)] # cc boundaries in closest wells
	max_var_thick_l1 = [bounds_cc_nwells[i][0] for i in range(n)] # verctos of thickness of layer 1 in closest wells 
	max_var_thick_l1 = abs(max_var_thick_l1[1]-max_var_thick_l1[0]) #np.std(max_var_thick_l1) # std dev for thickness of layer 1 in closest wells

	max_var_thick_l2 = [bounds_cc_nwells[i][1] for i in range(n)] # verctos of thickness of layer 1 in closest wells 
	max_var_thick_l2 = abs(max_var_thick_l2[1]-max_var_thick_l2[0])#np.std(max_var_thick_l2) # std dev for thickness of layer 1 in closest wells
	##(3) Calculate distance between station and wells (2)
	dist_sta_wells = [distances[i] for i in closest_wells]
	half_dist_between_wells = sum(np.abs(dist_sta_wells))/len(dist_sta_wells)
	## (4) Calculate sigma of thickness for each layer as a linear function of maximum variability from closest wells 
	sigma_thick_l1 = .5*(-1.*(max_var_thick_l1/half_dist_between_wells)* abs(half_dist_between_wells - np.min(dist_sta_wells))+max_var_thick_l1) # two wells, 2D
	#sigma_thick_l2 = .5*(-1.*(max_var_thick_l2/half_dist_between_wells)* abs(dist_sta_wells[1]-dist_sta_wells[0])+max_var_thick_l2) # two wells, 2D
	sigma_thick_l2 = .5*(-1.*(max_var_thick_l2/half_dist_between_wells)* abs(half_dist_between_wells - np.min(dist_sta_wells))+max_var_thick_l2) # two wells, 2D
	return sigma_thick_l1, sigma_thick_l2

	
def read_modem_model_column(file):
    """
    Read resistivity model from modEM inversion results in column format. 
    Format: file should be # Long Lat Z(m) Rho 
    """
    model = np.genfromtxt(file).T
 
    model_lat = model[1,:]
    model_lon = model[0,:]
    model_z  = model[2,:]
    model_rho = model[3,:]

    return model, model_lat, model_lon, model_z, model_rho

def intp_1D_prof_from_model(model, x_surf, y_surf, method = None, dz = None ,fig = None, name = None):
    """
    Interpolate a 1D profile from model in the surface position 
    given by x_surf and y_surf 
 
    fig: if True, return figure 
    """
    if method is None: 
        method = 'linear'
    if dz is None: 
        dz = 25.

    model_lat = model[1,:]
    model_lon = model[0,:]
    model_z  = model[2,:]
    model_rho = model[3,:]

    # (i) construct 'points' and 'values' vectors
    points = np.zeros([len(model_lat), 3])
    values = np.zeros([len(model_lat)])

    for i in range(len(model_lat)):
        points[i,:] = model[0:-1,i]
    values = model[3,:]

    # (ii) construct 'xi' vector
    z_vec = np.arange(min(model_z),max(model_z),dz)
    xi = np.zeros([len(z_vec), 3])
    for k in range(len(z_vec)): 
        xi[k,:] = [x_surf,y_surf,z_vec[k]]

    ## (iii) griddata: create interpolate profile 
    grid_z = griddata(points, values, xi, method=method)

    if fig: 
        ## plot profile 3d inv
        # create figure
        f = plt.figure(figsize=[5.5,5.5])
        ax = plt.axes()
        ax.plot(np.log10(grid_z),-z_vec,'m-')
        plt.ylim([-2000,1000])
        plt.xlim([-1,4])
        #ax.legend(loc = 2)
        ax.set_xlabel('Resistivity [Ohm m]', size = textsize)
        ax.set_ylabel('Depth [m]', size = textsize)
        if name is None:
            ax.set_title('1D prof. from 3D model', size = textsize)
        else: 
            ax.set_title(name+': 1D prof. from 3D model', size = textsize)
        ax.grid(alpha = 0.3)
        plt.tight_layout()
        #plt.show()
        
        return grid_z, z_vec, f
    
    else: 
        return grid_z


 











