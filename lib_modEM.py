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
from scipy.interpolate import griddata
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

class modEM(object):
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
    def read_data(self, filename, extract_origin = None):
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
                #self.x0 = 0.; self.y0 = float(nums[0]); self.z0 = float(nums[1])
                self.x0 = float(nums[0]); self.y0 = float(nums[1]); self.z0 = 0.
            else:
                self.x0 = float(nums[0]); self.y0 = float(nums[1]); self.z0 = float(nums[2])
            ln = fp.readline().rstrip()[1:] # periods and stations
            nums = ln.split(); nT = int(float(nums[0])); nS = int(float(nums[1]))
            
            if extract_origin: 
                return 
            
            # allocate space
            if cnt == 0:
                self.xs = np.zeros((1,nS))[0]	# station locations
                self.ys = np.zeros((1,nS))[0]	# station locations
                self.zs = np.zeros((1,nS))[0]	# station locations
                
                self.f = np.zeros((nT,nS))
            
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
                        
                    # save station data
                    dat[j,i] = np.complex(float(nums[8]),float(nums[9]))
        
            self.__setattr__(data_type, dat) 	# for data
            # condition to break from outer loop
            ln = fp.readline().rstrip() # discard
            if ln == '':  	# another data record present
                keepReading = False
            cnt += 1
        
        self.ys = self.ys - (self.ye[-1]-self.ye[0])/2.		
        
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
                        
                        fp.write(" %8.7e %03i 0.00 0.00 %8.7e %8.7e %8.7e %s %8.7e %8.7e %8.7e\n"%(T,i+1,xs,ys,zs, dtype[:2], np.real(dat[j,i]), np.imag(dat[j,i]), 1.e-3))
                
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
    def plot_rho2D(self, filename, xlim = None, ylim = None, gridlines = False, clim = None, overlay = None):
        # check data available for plotting
        try: self.rho
        except:	raise ValueError("no resistivity data to plot")
                
        # plot simple contours 
        plt.clf()
        fig = plt.figure(figsize=[7.5,5.5])
        ax = plt.axes([0.18,0.25,0.70,0.50])
        
        yy,zz = np.meshgrid(self.y,self.z)
        
        rho = self.rho*1.
        rho = self.rho[25][:][:]

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
            dat = overlay[0]
            level = overlay[1]
            rho = dat.rho*1.
            levels = [level,1.01*level]			
            CS = ax.contour(yy.T/1.e3,zz.T/1.e3, rho, levels = levels, colors = [[0.4,0.4,0.4],], linewidths = 0.5, cmap='jet')

        
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






 











