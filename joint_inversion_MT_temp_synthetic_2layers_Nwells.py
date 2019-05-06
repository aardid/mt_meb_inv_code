"""
.. module:: ModEM
   :synopsis: Forward and inversion ot MT using ModEM software.
				Joint inversion using well temperature.
				Estimate Temperature distribution. 

.. moduleauthor:: David Dempsey, Alberto Ardid 
				  University of Auckland 
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
#from SimPEG import Utils, Solver, Mesh, DataMisfit, Regularization, Optimization, InvProblem, Directives, Inversion
J = cmath.sqrt(-1)

exe = r'D:\ModEM\ModEM\f90\Mod2DMT.exe' # office
#exe = r'C:\ModEM\f90\Mod2DMT.exe' # house
textsize = 15.

# ==============================================================================
#  Objects
# ==============================================================================

class Modem(object):
	def __init__(self, work_dir = None):
		self.nx = 1
		self.ny = 1
		self.nz = 1
		self.x0 = 0.
		self.y0 = 0.
		self.z0 = 0.
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
		self.xe = np.array(self.x)
		self.ye = np.array(self.y)
		self.ze = np.array(self.z)
		
		# horizontal coords
		y = 1.*self.ye
		self.ye = np.zeros((1,len(y)+1))[0]
		self.ye[0] = 0
		self.ye[1:] = np.cumsum(y)
		self.y = 0.5*(self.ye[1:]+self.ye[:-1])
		ymin = 0.5*(np.min(self.y)+np.max(self.y))
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
			# vertical coords
			x = 1.*self.ye
			self.xe = np.zeros((1,len(x)+1))[0]
			self.xe[0] = 0
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
			self.z = 0.5*(self.ze[1:]+self.ze[:-1])
	def read_input(self, filename):
		self.x = []
		self.y = []
		self.z = []
		fp = open(filename,'r')
		
		# read header 
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
		
		# assemble grid
		self.assemble_grid()
		
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
	def read_data(self, filename):
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

class Well(object):
    # Methods 
    def __init__(self, name, pos, rho_profile, temp_profile, temp_profile_est, beta, slopes, clay_cap_boundaries, clay_cap_resistivity, distances, l1_resistivity, hs_resistivity):

        self.name = name
        self.pos = pos
		#self.lat = lat
        #self.lon = lon
        self.rho_profile = rho_profile
        self.temp_profile = temp_profile
        self.temp_profile_est = temp_profile_est
        self.beta = beta
        self.slopes = slopes
        self.clay_cap_boundaries = clay_cap_boundaries
        self.clay_cap_resistivity = clay_cap_resistivity
        self.distances = distances
        self.l1_resistivity = l1_resistivity
        self.hs_resistivity = hs_resistivity

class Station(object):
    # Methods 
    def __init__(self, name, pos, rho_profile, temp_profile, temp_profile_est, beta, slopes, clay_cap_boundaries_est, resistivity_layers_est, distances, clay_cap_resistivity_prior, resistivity_prior_l1_l3, clay_cap_thick_prior, l1_thick_prior, sigma_thick_l1, sigma_thick_l2):

        self.name = name
        self.pos = pos
		#self.lat = lat
        #self.lon = lon
        self.rho_profile = rho_profile
        self.temp_profile = temp_profile
        self.temp_profile_est = temp_profile_est
        self.beta = beta
        self.slopes = slopes
        self.clay_cap_boundaries_est = clay_cap_boundaries_est
        self.resistivity_layers_est = resistivity_layers_est
        self.distances = distances
        self.clay_cap_resistivity_prior = clay_cap_resistivity_prior
        self.resistivity_prior_l1_l3 = resistivity_prior_l1_l3
        self.clay_cap_thick_prior = clay_cap_thick_prior
        self.l1_thick_prior = l1_thick_prior
        self.sigma_thick_l1 = sigma_thick_l1
        self.sigma_thick_l2 = sigma_thick_l2

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
	dat.rho_box(rho_greywacke, box = [-80.e3,-10.e3,160.e3,8.e3]) 	# greywacke
	dat.rho_box(rho_fill, box = [-80.e3,-2.e3,160.e3,2.e3]) 	# greywacke
	
	dat.rho_blocks(rhoi, yi, zi)
	dat.rho_blocks(rhoi, -yi, zi)
	
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

	
if __name__ == "__main__":
##############################################################################################	
##############################################################################################
	
	# Improving DDEMPSEY2014
		
	# Steps (1), (2) , (4) and (5) (flowchart)
	# Temp -> ResTrue -> ResEst (Folder: 01_Temp2Rest)
	# Calculate (inv) resistivitity distribution from true resistivity distribution
	
	if False:
		
		# 1. Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 
		dat, t2t = setup_simulation(work_dir = '01_Temp2Rest')
		yi,zi,Ti = t2t
		
		# plot temperature distribution 
		if True: 
			
			yi = np.asarray(yi)
			zi = np.asarray(zi)
			Ti = np.asarray(Ti)

			# Through the unstructured data get the structured data by interpolation
			y = np.linspace(np.min(yi), np.max(yi), 1000)
			z = np.linspace(np.min(zi), np.max(zi), 1000)
			
			from matplotlib.mlab import griddata
			T_grid = griddata(yi, zi, Ti, y, z, interp='linear')
			
			plt.clf()
			fig = plt.figure(figsize=[7.5,5.5])
			ax = plt.axes([0.18,0.25,0.70,0.50])
			CS = ax.contour(y/1e3, z/1e3, T_grid, 15, linewidths=0.5, colors='k')
			CS = ax.contourf(y/1e3, z/1e3, T_grid, 15, cmap='jet') #vmax=abs(np.max(z)), vmin=abs(np.min(z)))
			cax = plt.colorbar(CS, ax = ax)
			ax.set_xlabel('y [km]', size = textsize)
			ax.set_ylabel('z [km]', size = textsize)
			cax.set_label('Temperature [°C]', size = textsize)
			
			ax.set_title('Temperature Distribution', size = textsize)
			# plt.show()
			
			#if xlim is not None: ax.set_xlim(xlim)		
			plt.savefig(dat.work_dir+os.sep+'True_Temperature_Distribution.png', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
			#plt.close(fig)
			
			plt.savefig(dat.work_dir+os.sep+'True_Temperature_Distribution.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			plt.close(fig)
			
		if False: 
		
			## 2. choose true transfer function and get true resistivity
			rhob = [700, 1000, 400]		
			#par_true = [200,40,10,10,rhob[-1],10]
			par_true = [200,40,40,40,rhob[-1],5]

			var_rho0 = 5000
			var_rhoa = 500
			rhoi = rho_noise(Ti, par_true, var_rho0, var_rhoa)
			#rhoi = rho(Ti, par_true)
			dat = set_rho(dat, yi, zi, rhoi, rhob)
			
			## 3. Plot: grid, transfer function T -> Rest, and rho distribution (true)	
			dat.plot_grid(dat.work_dir+os.sep+'grid.png', ylim = [-12., 12.], zlim = [-5.,0])
			#resistivity_model(par_true, dat.work_dir)		
			resistivity_model_noise(par_true, dat.work_dir, var_rho0, var_rhoa)	
			dat.plot_rho2D(dat.work_dir+os.sep+'rho_true.pdf', xlim = [-7., 7.], ylim = [-3.,0], gridlines = False, clim = [1e0,1.e4])

			
			#4. Set stations 		
			nT = 41
			nS = 41
			base = 10
			T = np.logspace(np.log(0.01)/np.log(base), np.log(400)/np.log(base), nT, base = base) 		
			dat.stations(np.linspace(-10.e3,10.e3,nS),0.,1./T)
			
			#5. run forward model 
			dat.run(input = 'in.dat', output = 'out.dat', exe = r'D:\ModEM\ModEM\f90\Mod2DMT.exe') # office
			#dat.run(input = 'in.dat', output = 'out.dat', exe = r'C:\ModEM\f90\Mod2DMT.exe') # house
			
			#6. visualize output of forward
			dat.read_data(dat.work_dir+os.sep+'out.dat')	
			dat.plot_impedance(dat.work_dir+os.sep+'TEimpedances.pdf', frequency = False)
			dat.plot_impedance(dat.work_dir+os.sep+'TMimpedances.pdf', frequency = False, TE = False)
			
			#7. run inversion 
			parfile = dat.work_dir+os.sep+'par.inv' # load parameters of inversion 
			#parfile = r'C:\Users\ajara\Desktop\MTcodes\par.inv' # load parameters of inversion 
			dat.rho_box(1000.) 		# background (basement)
			dat.rho_box(500., box = [-80.e3,-2.e3,160.e3,2.e3]) 	# surface infill
			dat.plot_rho2D(dat.work_dir+os.sep+'rho_initial.pdf', xlim = [-12., 12.], ylim = [-5.,0], gridlines = True, clim = [1e0,1.e4])
			dat.invert(input = 'out.dat', output = 'inv.dat', exe = r'D:\ModEM\ModEM\f90\Mod2DMT.exe', parameters = 'par.inv') # office
			#dat.invert(input = 'out.dat', output = 'inv.dat', exe = r'C:\ModEM\f90\Mod2DMT.exe', parameters = 'par.inv') # house

			#8. read and plot the last iteration 
			dat.read_input(dat.work_dir+os.sep+'inv.dat')	
			fls = glob(dat.work_dir + os.sep+'*.rho')
			newest = max(fls, key = lambda x: os.path.getctime(x))
			dat.read_input(newest)	
			dat.plot_rho2D(dat.work_dir+os.sep+'rho_inv.pdf', xlim = [-7., 7.], ylim = [-3.,0], gridlines = False, clim = [1e0,1.e4])
			
	# Steps (3), (6) and (7) (flowchart)
	# joint inversion: fit values of temp and resistivity and extrapolate away from well  
	# (Folder: 02_Rest2Temp)

	if False: # joint inversion through fit to normal distribution 
	
		# 1. Create modem object and import temperature distribution
		#    Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 
		
		dat0, t2t = setup_simulation(work_dir = '02_Rest2Temp')
		dat, t2t = setup_simulation(work_dir = '02_Rest2Temp')
		yi,zi,Ti = t2t
		
		# 2. Recover inverted model 
		dat_aux, t2t = setup_simulation(work_dir = '01_Temp2Rest')		
		fls = glob(dat_aux.work_dir + os.sep+'*.rho')
		newest = max(fls, key = lambda x: os.path.getctime(x))

		#dat0.read_input(dat.work_dir+os.sep+'results_noise2'+os.sep+'Modular_NLCG_090.rho')
		dat0.read_input(newest)	
		
		#dat0.plot_rho2D(dat.work_dir+os.sep+'rho_inv.png', xlim = [-10,10], ylim = [-5,0], gridlines = True, clim = [1,1000])
		dat0.plot_rho2D(dat.work_dir+os.sep+'rho_inv.pdf', xlim = [-10,10], ylim = [-5,0], gridlines = True, clim = [1,1000])
		
		# 1. Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 

		#dat0.read_input(dat.work_dir+os.sep+'BLOCK2_NLCG_030.rho')
		#dat_aux.plot_rho2D(dat0.work_dir+os.sep+'rho_est.png', xlim = [-12., 12.], ylim = [-5.,0], gridlines = True, clim = [1,1000])
		
		# 3. get T profile in position of the wells (concident with stations) 
		zinterp = np.linspace(-1.e3,-25.,101)
		ystation = np.arange(50., 5.0e3, 500.) # position station dx = 200 m
		
		ywell = 0.0e3  # position well 1 (1000 m)		
		ydemo = 2.0e3  # position well 2 (1500 m)
		
		i = np.argmin(abs(ystation - ywell)) # position of well with respect to the stations 
		j = np.argmin(abs(ystation - ydemo)) # position of well with respect to the stations
		ywell = ystation[i]
		ydemo = ystation[j]
		z,Td = get_T(yi, zi, Ti, [ydemo], zinterp)  # temperature profiles (at single station)
		z,Tw = get_T(yi, zi, Ti, [ywell], zinterp) 	# well temperature profile
		
		# 4. get true rho in ystation positions
		z,rho0 = get_rho(dat0, ystation, zinterp)	
		
		# 5. Fit resistivity and temp profiles with normal dist. curve
		#	 in position of well => obtain fit parameters (to use in extrapolation of temp)
		#Fit resistivity profile
		x = zinterp
		y = np.log10(rho0[i,:])
		normalplot = True # True: fit to normal dist   False: fit to Cauchy dist
		if normalplot:
			p0 = [x[np.argmin(y)],200.,(np.max(y) - np.min(y)), np.max(y)]
			pcal,pcov = curve_fit(normal, x, y, p0) 
		else:
			p0 = [x[np.argmin(y)],200.,200.,(np.max(y) - np.min(y)), np.max(y)]
			pcal,pcov = curve_fit(cp, x, y, p0)     
		#print(pcal)
		perr = np.sqrt(np.diag(pcov)) # standard deviation errors
		#print(perr)
		#Fit temperature profile
		x = zinterp
		y = Tw[0]
		#p0 = [-20.]
		p0 = [-20., 0.5e3, 200.]
		Tcal,pcov = curve_fit(Texp, x, y, p0)
		#print(Tcal)
		perr = np.sqrt(np.diag(pcov)) # standard deviation errors
		#print(perr)
		
		# 6. Extrapolate temp away from the well using fit paramters estimated for well
		normalplot = True # True: fit to normal dist   False: fit to Cauchy dist
		
		x = zinterp
		yy = np.array(ystation)
		yy,zz = np.meshgrid(yy, zinterp)
		TT = yy*0.
		ny = len(ystation)
		for k,ystationi in enumerate(ystation):			
			y = np.log10(rho0[k,:])
			if normalplot:
				p0 = [x[np.argmin(y)],200.,(np.max(y) - np.min(y)), np.max(y)]
			else:
				p0 = [x[np.argmin(y)],200.,200.,(np.max(y) - np.min(y)), np.max(y)]			
			try:
				if normalplot:
					pcali,pcov = curve_fit(normal, x, y, p0)
				else:
					pcali,pcov = curve_fit(cp, x, y, p0)
			except RuntimeError:
				continue
						
			zmax0 = pcal[0]
			zmaxi = pcali[0]
			zj = zinterp*1.
			Tj = Texp(zj, Tcal[0], Tcal[1]+(zmaxi-zmax0), Tcal[2]+(zmaxi-zmax0))
			#Tj = Texp(zj, Tcal[0], Tcal[1]*(zmaxi/zmax0), Tcal[2]*(zmaxi/zmax0))
			TT[:,k] = Tj
			
			#if i == 10: break
			#if k == i: asdf
		#print(TT[-1,j])
		
		f = open(dat.work_dir+os.sep+"T_est_normal_dist.dat", "w")
		for k in range(0,len(TT[0,:])):
			for i in range(0,len(TT[:,0])):
				f.write("{:3.2f}\t".format(TT[i,k]))
			f.write("\n".format())
		f.close()

		# 7. Plot: (1) temperature real and fit, and res in well position (true values) (ax1a)
		#          (2) temperature real and extrapolate, and res in 'j' position (away form well)
		#		   (3) Countour Temp map of extrapolate temp (ax)
		
		plt.clf()
		
		textsize = 12.
		fig = plt.figure(figsize=[14.0,7.0])
		ax = plt.axes([0.58,0.25,0.35,0.50])
		ax1= plt.axes([0.10,0.25,0.17,0.50])
		ax2= plt.axes([0.33,0.25,0.17,0.50])
		ax1a=ax1.twiny()
		ax2a=ax2.twiny()
				
		levels = range(50,301,50)
		
		CS = ax.contourf(yy/1.e3, zz, TT, levels = levels, cmap='jet')
		cax = plt.colorbar(CS, ax = ax)
		
		xlim = ax.get_xlim()
		xlim = [0,4]
		ax.set_xlim(xlim)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim)
		
		ny = len(np.unique(yi))
		nz = len(np.unique(zi))
		yi = yi.reshape([nz,ny])
		zi = zi.reshape([nz,ny])
		Ti = Ti.reshape([nz,ny])
		CS = ax.contour(yi/1.e3,zi, Ti, levels = levels, colors = 'k', linewidths = 0.5)
		plt.clabel(CS, fmt = "%i")
		
		ax.set_ylabel('depth [m]', size = textsize)
		ax.set_xlabel('y [km]', size = textsize)
		
		# plot well temperature
		i = np.argmin(abs(ystation - ywell))
		j = np.argmin(abs(ystation - ydemo))
		ax1a.plot(Tw[0],zinterp, 'r-')
		ax1a.plot(Texp(zinterp, *Tcal), zinterp, 'r--', lw = 0.7)
		ax2a.plot(Td[0],zinterp, 'r-')
		ax2a.plot(TT[:,j],zinterp, 'r--', lw = 0.7)
		ax1.plot(rho0[i,:],zinterp,'b-')		
		ax2.plot(rho0[j,:],zinterp,'b-')
		
		x = zinterp
		y = np.log10(rho0[i,:])
		if normalplot:
			p0 = [x[np.argmin(y)],200.,(np.max(y) - np.min(y)), np.max(y)]
			prho0,pcov = curve_fit(normal, x, y, p0)
			ax1.plot(10**normal(zinterp, *prho0), zinterp,'b--', lw = 0.5)
		else:
			p0 = [x[np.argmin(y)],200.,200.,(np.max(y) - np.min(y)), np.max(y)]
			prho0,pcov = curve_fit(cp, x, y, p0)
			ax1.plot(10**cp(zinterp, *prho0), zinterp,'b--', lw = 0.5)
		
		x = zinterp
		y = np.log10(rho0[j,:])
		if normalplot:
			p0 = [x[np.argmin(y)],200.,(np.max(y) - np.min(y)), np.max(y)]
			prho0,pcov = curve_fit(normal, x, y, p0)
			ax2.plot(10**normal(zinterp, *prho0), zinterp,'b--', lw = 0.5)
		else:
			p0 = [x[np.argmin(y)],200.,200.,(np.max(y) - np.min(y)), np.max(y)]
			prho0,pcov = curve_fit(cp, x, y, p0)
			ax2.plot(10**cp(zinterp, *prho0), zinterp,'b--', lw = 0.5)
			
		for axi in [ax1,ax2]:
			#axi.set_ylabel('depth [km]', size = textsize)
			axi.set_xlabel('resistivity [$\Omega$m]', size = textsize)
			axi.set_xlim([5, 800])
			axi.set_xscale('log')
		ax1.set_ylabel('depth [m]', size = textsize)	
		
		for axi in [ax1a,ax2a]:
			axi.set_xlabel('temperature [$^\circ$C]', size = textsize)
			axi.set_xlim([50, 300])
		
		ax.set_title('(c) Temperature distribution', size = textsize+2, y=1.15)
		ax1.set_title('(a) Well profile ', size = textsize+2, y=1.15)
		ax2.set_title('(b) Station profile', size = textsize+2, y=1.15)
		
		for axi in [ax, ax1, ax2, ax1a, ax2a]:
			for t in axi.get_xticklabels()+axi.get_yticklabels(): t.set_fontsize(textsize)
		for t in cax.ax.get_yticklabels(): t.set_fontsize(textsize)
		
		plt.savefig(dat.work_dir+os.sep+'inferred_temperature.png', dpi=300, facecolor='w', edgecolor='w',
			orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
		plt.savefig(dat.work_dir+os.sep+'inferred_temperature.pdf', dpi=300, facecolor='w', edgecolor='w',
			orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
		plt.close(fig)
	
	if False:  # joint inversion through fit to layer model :  one station 
	
		# 1. Create modem object and import temperature distribution
		#    Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 
		dat0, t2t = setup_simulation(work_dir = '02_Rest2Temp')
		dat, t2t = setup_simulation(work_dir = '02_Rest2Temp')
		yi,zi,Ti = t2t
		
		# 2. Recover inverted model 
		dat_aux, t2t = setup_simulation(work_dir = '01_Temp2Rest')		
		fls = glob(dat_aux.work_dir + os.sep+'*.rho')
		newest = max(fls, key = lambda x: os.path.getctime(x))

		#dat0.read_input(dat.work_dir+os.sep+'results_noise2'+os.sep+'Modular_NLCG_090.rho')
		dat0.read_input(newest)	
		
		#dat0.plot_rho2D(dat.work_dir+os.sep+'rho_inv.png', xlim = [-10,10], ylim = [-5,0], gridlines = True, clim = [1,1000])
		dat0.plot_rho2D(dat.work_dir+os.sep+'rho_inv.pdf', xlim = [-10,10], ylim = [-5,0], gridlines = True, clim = [1,1000])
		
		# 1. Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 

		#dat0.read_input(dat.work_dir+os.sep+'BLOCK2_NLCG_030.rho')
		#dat_aux.plot_rho2D(dat0.work_dir+os.sep+'rho_est.png', xlim = [-12., 12.], ylim = [-5.,0], gridlines = True, clim = [1,1000])
		
		# 3. get T profile in position of the wells (concident with stations) 
		zinterp = np.linspace(-1.e3,-25.,101)
		ystation = np.arange(50., 5.0e3, 500.) # position station dx = 200 m
		
		ywell = 0.0e3  # position well 1 (1000 m)		
		ydemo = 0.5e3  # position well 2 (1000 m)
		
		i = np.argmin(abs(ystation - ywell)) # position of well with respect to the stations 
		j = np.argmin(abs(ystation - ydemo)) # position of demo with respect to the stations
		ywell = ystation[i]
		print(ywell)
		ydemo = ystation[j]
		print(ydemo)
		z,Td = get_T(yi, zi, Ti, [ydemo], zinterp)  # temperature profiles (at single station)
		z,Tw = get_T(yi, zi, Ti, [ywell], zinterp) 	# well temperature profile
		z_vec = np.concatenate(z,  axis=0 )
		Tw_vec = np.concatenate(Tw,  axis=0 )
		Td_vec = np.concatenate(Td,  axis=0 )
		
		# 4. get true rho in ystation positions
		z,rho0 = get_rho(dat0, ystation, zinterp)	
		
		# 5. Joint inversion in position of well 
		
		# 5.1 Calculate apparent resistivity associated with true (extracted from inversion result) resistivity profile in the well position 	
		z_aux = np.linspace(25.,1.e3,101) 
		dzi = z_aux[2] - z_aux[1]
		
		z_thickness = np.ones(len(zinterp)-1)*dzi
		periods = np.logspace(-3, 3,50) # Periods range 10**-3 to 10**3 [s]
		
		rho0_flip_well = np.flipud(rho0[i])
		rho0_flip_demo = np.flipud(rho0[j])
		app_res_vec_ana, phase_vec_ana = anaMT1D_f_nlayers(z_thickness,rho0_flip_well,periods)
		app_res_vec_ana_demo, phase_vec_ana_demo = anaMT1D_f_nlayers(z_thickness,rho0_flip_demo,periods)
		
		# plot apparent resistivity
		if False:
			plt.clf()
			fig = plt.figure(figsize=[7.0,7.0])
			ax = plt.axes([0.18,0.25,0.70,0.50])
			ax.loglog(periods,app_res_vec_ana,'r-',label='True')
			#ax.loglog(periods,app_res_vec_noise,'b*',label='Observed')
			ax.set_ylabel(r'app. resistivity  $[\Omega$m]', size = textsize)
			ax.set_xlabel(r'period  $[s]$', size = textsize)
			plt.savefig(dat.work_dir+os.sep+'app_res_well.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			plt.close(fig)
			
			plt.clf()
			fig = plt.figure(figsize=[7.0,7.0])
			ax = plt.axes([0.18,0.25,0.70,0.50])
			ax.loglog(periods,app_res_vec_ana_demo,'r-',label='True')
			#ax.loglog(periods,app_res_vec_noise,'b*',label='Observed')
			ax.set_ylabel(r'app. resistivity  $[\Omega$m]', size = textsize)
			ax.set_xlabel(r'period  $[s]$', size = textsize)
			plt.savefig(dat.work_dir+os.sep+'app_res_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			plt.close(fig)

		
		# 5.2 Invert apparent resistivity to obtain a 3 layer model  (using MCMC)	
		# Estimate the conductivity parameters (via LLK)
		
		# run MCMC: well and demo
		if False: 
		
			# log likelihood for the model, given the data
			def lnprob(pars, obs):
				v = 0.15
				if any(x<0 for x in pars):
					return -10e10
				else:
					return -np.sum((obs[:,1]-anaMT1D_f_3layers(*pars,obs[:,0]))**2)/v

			ndim = 7                # parameter space dimensionality
			nwalkers=20*140           # number of walkers
			
			#well
			
			# create the emcee object (set threads>1 for multiprocessing)
			data = np.array([periods,app_res_vec_ana]).T
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=1, args=[data,])

			# set the initial location of the walkers
			## initial model
			# Layer 1
			rho_layer1 = 120 # ohm m
			depth_layer1 = 0  # m
			thickness_layer1 = 200  # m
			# Layer 2
			rho_layer2 = 50  # ohm m
			depth_layer2 = thickness_layer1  # m
			thickness_layer2 = 100   # m
			# Layer 3
			rho_layer3 = 500  # ohm m
			depth_layer3 = thickness_layer1 + thickness_layer2
			thickness_layer3 = 1000 - (thickness_layer1 + thickness_layer2) # m
			# Layer hs
			rho_hs = 500  # ohm m
			sigma_hs = 1/rho_hs
			
			pars = [thickness_layer1,thickness_layer2,thickness_layer3,rho_layer1,rho_layer2,rho_layer3,rho_hs]  # initial guess
			p0 = np.array([pars + 1e2*np.random.randn(ndim) for i in range(nwalkers)])  # add some noise

			# set the emcee sampler to start at the initial guess and run 100 burn-in jumps
			pos,prob,state=sampler.run_mcmc(p0,200)
			# sampler.reset()
			#print(1)
			f = open("chain.dat", "w")
			nk,nit,ndim=sampler.chain.shape
			for k in range(nk):
				for i in range(nit):
					f.write("{:d} {:d} ".format(k, i))
					for j in range(ndim):
						f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
					f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
			f.close()
			
			## run mcmc for demo 
			
			#ndim = 7                # parameter space dimensionality
			#nwalkers=2*140           # number of walkers
			# create the emcee object (set threads>1 for multiprocessing)
			data_demo = np.array([periods,app_res_vec_ana_demo]).T
			sampler.reset()
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=1, args=[data_demo,])

			# set the initial location of the walkers
			## True Model
			# Layer 1
			rho_layer1 = 120 # ohm m
			depth_layer1 = 0  # m
			thickness_layer1 = 200  # m
			# Layer 2
			rho_layer2 = 50  # ohm m
			depth_layer2 = thickness_layer1  # m
			thickness_layer2 = 100   # m
			# Layer 3
			rho_layer3 = 500  # ohm m
			depth_layer3 = thickness_layer1 + thickness_layer2
			thickness_layer3 = 1000 - (thickness_layer1 + thickness_layer2) # m
			# Layer hs
			rho_hs = 500  # ohm m
			sigma_hs = 1/rho_hs
			
			pars = [thickness_layer1,thickness_layer2,thickness_layer3,rho_layer1,rho_layer2,rho_layer3,rho_hs]  # initial guess
			p0 = np.array([pars + 1e2*np.random.randn(ndim) for i in range(nwalkers)])  # add some noise

			# set the emcee sampler to start at the initial guess and run 100 burn-in jumps
			pos,prob,state=sampler.run_mcmc(p0,200)
			# sampler.reset()
			#print(1)
			d = open("chain_demo.dat", "w")
			nk,nit,ndim=sampler.chain.shape
			for k in range(nk):
				for i in range(nit):
					d.write("{:d} {:d} ".format(k, i))
					for j in range(ndim):
						d.write("{:15.7f} ".format(sampler.chain[k,i,j]))
					d.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
			d.close()
		# Plot results MCMC: well and demo
		if False:
			# show corner plot
			chain = np.genfromtxt('chain.dat')
			weights = chain[:,-1]
			weights -= np.max(weights)
			weights = np.exp(weights)
			labels = ['thick. 1','thick. 2','thick. 3','rest. 1','rest. 2','rest. 3','rest. hs']
			fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30)
			plt.savefig(dat.work_dir+os.sep+'mcmc_results_well.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			plt.close(fig)
		# Sample posterior and plot predictions vs true known: well and demo
		if False: 
					
			# Layer 1
			rho_layer1 = 120 # ohm m
			depth_layer1 = 25  # m
			thickness_layer1 = 200  # m
			# Layer 2
			rho_layer2 = 50  # ohm m
			depth_layer2 = thickness_layer1  # m
			thickness_layer2 = 100   # m
			# Layer 3
			rho_layer3 = 500  # ohm m
			depth_layer3 = thickness_layer1 + thickness_layer2
			thickness_layer3 = 1000 - (thickness_layer1 + thickness_layer2) # m
			# Layer hs
			rho_hs = 500  # ohm m
			sigma_hs = 1/rho_hs
			
			######################################################################
			# well
			# reproducability
			np.random.seed(1)
			# load in the posterior
			chain = np.genfromtxt('chain.dat')
			params = chain[:,2:-1]

			# define parameter sets for forward runs
			Nruns = 1000
			pars = []
				# generate Nruns random integers in parameter set range (as a way of sampling this dist)
			Nsamples = 0
			llnmax=np.max(chain[:,-1])
			ln = chain[:,-1]-llnmax

			while Nsamples != Nruns:
				id = np.random.randint(0,params.shape[0]-1)
			#    lni = ln[id]
				# probability test
			#    lni = np.exp(lni)
			#    if lni<np.random.random():
			#        continue
				# in sample test
				par_new = [params[id,0], params[id,1], params[id,2], params[id,3], params[id,4], params[id,5], params[id,6]]
			#    for par in pars:
			#        if all([(abs(p-pn)/abs(p))<1.e-4 for p,pn in zip(par,par_new)]):
			#            continue
				pars.append([Nsamples, params[id,0], params[id,1], params[id,2], params[id,3], params[id,4], params[id,5], params[id,6]])
				Nsamples += 1
			   

			f,ax = plt.subplots(1,1)
			f.set_size_inches(8,8) 
			ax.loglog(periods,app_res_vec_ana,'r-',label='True', lw = 1.0, alpha=0.7)
			ax.set_xlim([np.min(periods), np.max(periods)])
			ax.set_ylim([1e1,1e3])
			ax.set_xlabel('Period [s]')
			ax.set_ylabel('Ap. Resistiviy [Ohm m]')

			# plot eq data as vertical lines
			for par in pars:
				if all(x > 0. for x in par):
					app_res_vec_aux, phase_vec_aux = anaMT1D_f_3layers(*par[1:8],periods)
					ax.plot(periods, app_res_vec_aux,'b-', lw = 0.1, alpha=0.1, zorder=0)
			plt.savefig(dat.work_dir+os.sep+'mcmc_app_res_well.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			plt.close(f) 
			
			# plot conductivity models	
			if True: 
				target_depth = 500
				sigma_model, mesh = num_model_mesh(periods,target_depth,0,thickness_layer1,
												   thickness_layer1+thickness_layer2,rho_layer1,rho_layer2,rho_layer3,
												   rho_hs,thickness_layer1,thickness_layer2,thickness_layer3)
				rho_model = 1./sigma_model
				fig1, ax = plt.subplots(1,1)
				fig1.set_size_inches(8,4)
				z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
				z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
				rho_model_plt = np.repeat(rho_model, 2, axis=0)
				ax.semilogy(z, rho_model_plt,'r-',label='True', lw = 0.5, alpha=.7)
				ax.set_xlim([-500., 0.])
				ax.set_ylim([1, 1500])
				ax.invert_xaxis() # plot the surface on the left
				ax.set_xlabel("Elevation (m)")
				ax.set_ylabel("Resistivity (Ohm m)") 

				# plot eq data as vertical lines
				for par in pars:
					if all(x > 0. for x in par):
						sigma_model, mesh = num_model_mesh(periods,target_depth,0,par[1],par[1]+par[2],par[4],par[5],par[6],par[7],par[1],par[2],par[3])
						rho_model = 1./sigma_model
						z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
						z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
						rho_model_plt = np.repeat(rho_model, 2, axis=0)
						ax.semilogy(z, rho_model_plt,'b-', lw = 0.1, alpha=0.1, zorder=0)	
				
				plt.savefig(dat.work_dir+os.sep+'mcmc_rest_models_3layers_well.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			######################################################################			
			# demo
			
			# reproducability
			np.random.seed(1)
			# load in the posterior
			chain = np.genfromtxt('chain_demo.dat')
			params = chain[:,2:-1]

			# define parameter sets for forward runs
			Nruns = 1000
			pars = []
				# generate Nruns random integers in parameter set range (as a way of sampling this dist)
			Nsamples = 0
			llnmax=np.max(chain[:,-1])
			ln = chain[:,-1]-llnmax

			while Nsamples != Nruns:
				id = np.random.randint(0,params.shape[0]-1)
			#    lni = ln[id]
				# probability test
			#    lni = np.exp(lni)
			#    if lni<np.random.random():
			#        continue
				# in sample test
				par_new = [params[id,0], params[id,1], params[id,2], params[id,3], params[id,4], params[id,5], params[id,6]]
			#    for par in pars:
			#        if all([(abs(p-pn)/abs(p))<1.e-4 for p,pn in zip(par,par_new)]):
			#            continue
				pars.append([Nsamples, params[id,0], params[id,1], params[id,2], params[id,3], params[id,4], params[id,5], params[id,6]])
				Nsamples += 1
			   

			f,ax = plt.subplots(1,1)
			f.set_size_inches(8,8) 
			ax.loglog(periods,app_res_vec_ana_demo,'r-',label='True', lw = 1.0, alpha=0.7)
			ax.set_xlim([np.min(periods), np.max(periods)])
			ax.set_ylim([1e1,1e3])
			ax.set_xlabel('Period [s]')
			ax.set_ylabel('Ap. Resistiviy [Ohm m]')

			# plot eq data as vertical lines
			for par in pars:
				if all(x > 0. for x in par):			
					app_res_vec_aux, phase_vec_aux = anaMT1D_f_3layers(*par[1:8],periods)
					ax.plot(periods, app_res_vec_aux,'b-', lw = 0.1, alpha=0.1, zorder=0)
				
			plt.savefig(dat.work_dir+os.sep+'mcmc_app_res_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
			plt.close(f) 
			
			# plot conductivity models for demo 
			if True: 
				target_depth = 500
				sigma_model, mesh = num_model_mesh(periods,target_depth,0,thickness_layer1,
												   thickness_layer1+thickness_layer2,rho_layer1,rho_layer2,rho_layer3,
												   rho_hs,thickness_layer1,thickness_layer2,thickness_layer3)
				rho_model = 1./sigma_model
				fig1, ax = plt.subplots(1,1)
				fig1.set_size_inches(8,4)
				z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
				z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
				rho_model_plt = np.repeat(rho_model, 2, axis=0)
				ax.semilogy(z, rho_model_plt,'r-',label='True', lw = 0.5, alpha=.7)
				ax.set_xlim([-500., 0.])
				ax.set_ylim([1, 1500])
				ax.invert_xaxis() # plot the surface on the left
				ax.set_xlabel("Elevation (m)")
				ax.set_ylabel("Resistivity (Ohm m)") 

				# plot eq data as vertical lines
				for par in pars:
					if all(x > 0. for x in par):
						sigma_model, mesh = num_model_mesh(periods,target_depth,0,par[1],par[1]+par[2],par[4],par[5],par[6],par[7],par[1],par[2],par[3])
						rho_model = 1./sigma_model
						z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
						z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
						rho_model_plt = np.repeat(rho_model, 2, axis=0)
						ax.semilogy(z, rho_model_plt,'b-', lw = 0.1, alpha=0.1, zorder=0)	
				
				plt.savefig(dat.work_dir+os.sep+'mcmc_rest_models_3layers_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)			
		# look for the best fit solution of MCMC: well and demo
		if False:
			# Well 		
			data = np.genfromtxt('chain.dat')
			misfit = data[:,9]
			inds = np.where(misfit == np.max(misfit)) 
	
			## BestFit Model
			# Layer 1
			bf_rho_layer1 = data[inds[0][0],:][5]
			bf_depth_layer1 = 25  # m
			bf_thickness_layer1 = data[inds[0][0],:][2]
			# Layer 2
			bf_rho_layer2 = data[inds[0][0],:][6]
			bf_depth_layer2 = bf_thickness_layer1 + bf_depth_layer1  
			bf_thickness_layer2 = data[inds[0][0],:][3]
			# Layer 3
			bf_rho_layer3 = data[inds[0][0],:][7]
			bf_depth_layer3 = bf_depth_layer2 + bf_thickness_layer2
			bf_thickness_layer3 = data[inds[0][0],:][4]
			# Layer hs
			bf_rho_hs = data[inds[0][0],:][8]
			bf_sigma_hs = 1/bf_rho_hs
			
			#print(inds)
			#print(data[inds[0][0],:])
			bf_pars = [data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4],
																	data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],
																	data[inds[0][0],:][8]]
			bf_pars_well = bf_pars
			
			if True:# plot best fit: app. res and res. model
				# pars = [thickness_layer1,thickness_layer2,thickness_layer3,rho_layer1,rho_layer2,rho_layer3,rho_hs]  
				# anaMT1D_f_3layers(h_1,h_2,h_3,rho_1,rho_2,rho_3,rho_hs,T)
				app_res_best_fit, phase_best_fit = anaMT1D_f_3layers(data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4],
																	data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],
																	data[inds[0][0],:][8],periods)
				# fit: app. resistivity
				plt.clf()
				fig = plt.figure(figsize=[7.0,7.0])
				ax = plt.axes([0.18,0.25,0.70,0.50])
				ax.loglog(periods,app_res_vec_ana,'r-',label='True')
				ax.loglog(periods,app_res_best_fit,'b-',label='Best Fit')
				ax.set_ylabel(r'app. resistivity  $[\Omega$m]', size = textsize)
				ax.set_xlabel(r'period  $[s]$', size = textsize)
				plt.savefig(dat.work_dir+os.sep+'mcmc_bfit_app_res_well.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
				plt.close(fig)
				plt.clf()

				# layer model
				target_depth = 1000
				sigma_model, mesh = num_model_mesh(periods,target_depth,25,data[inds[0][0],:][2],data[inds[0][0],:][2]+data[inds[0][0],:][3],
													data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],data[inds[0][0],:][8],
													data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4])
				rho_model = 1./sigma_model
				fig1, ax = plt.subplots(1,1)
				fig1.set_size_inches(8,4)
				z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
				z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
				rho_model_plt = np.repeat(rho_model, 2, axis=0)
				ax.semilogy(z, rho_model_plt,'r-',label='True', lw = 1., alpha=.7)
				ax.set_xlim([-1000., 0.])
				ax.set_ylim([1, 1500])
				ax.invert_xaxis() # plot the surface on the left
				ax.set_xlabel("Elevation (m)")
				ax.set_ylabel("Resistivity (Ohm m)") 
				
				plt.savefig(dat.work_dir+os.sep+'mcmc_bfit_model_res_well.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
																	
			# Demo	
			data = []		
			data = np.genfromtxt('chain_demo.dat')
			
			misfit = data[:,9]
			inds = np.where(misfit == np.max(misfit)) 
			
			## BestFit Model
			# Layer 1
			bf_rho_layer1_demo = data[inds[0][0],:][5]
			bf_depth_layer1_demo = 25  # m
			bf_thickness_layer1_demo = data[inds[0][0],:][2]
			# Layer 2
			bf_rho_layer2_demo = data[inds[0][0],:][6]
			bf_depth_layer2_demo = bf_thickness_layer1_demo  + bf_depth_layer1_demo
			bf_thickness_layer2_demo = data[inds[0][0],:][3]
			# Layer 3
			bf_rho_layer3_demo = data[inds[0][0],:][7]
			bf_depth_layer3_demo = bf_depth_layer2_demo + bf_thickness_layer2_demo
			bf_thickness_layer3_demo = data[inds[0][0],:][4]
			# Layer hs
			bf_rho_hs_demo = data[inds[0][0],:][8]
			bf_sigma_hs_demo = 1/bf_rho_hs_demo
			
			#print(data[inds[0][0],:])
			bf_pars = [data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4],
																	data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],
																	data[inds[0][0],:][8]]
			bf_pars_demo = bf_pars
			if True:# plot best fit: app. res and res. model
				# pars = [thickness_layer1,thickness_layer2,thickness_layer3,rho_layer1,rho_layer2,rho_layer3,rho_hs]  
				# anaMT1D_f_3layers(h_1,h_2,h_3,rho_1,rho_2,rho_3,rho_hs,T)
				app_res_best_fit, phase_best_fit = anaMT1D_f_3layers(data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4],
																	data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],
																	data[inds[0][0],:][8],periods)
				# fit: app. resistivity
				plt.clf()
				fig = plt.figure(figsize=[7.0,7.0])
				ax = plt.axes([0.18,0.25,0.70,0.50])
				ax.loglog(periods,app_res_vec_ana_demo,'r-',label='True')
				ax.loglog(periods,app_res_best_fit,'b-',label='Best Fit')
				ax.set_ylabel(r'app. resistivity  $[\Omega$m]', size = textsize)
				ax.set_xlabel(r'period  $[s]$', size = textsize)
				plt.savefig(dat.work_dir+os.sep+'mcmc_bfit_app_res_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
				plt.close(fig)
				plt.clf()

				# layer model
				target_depth = 1000
				sigma_model_demo, mesh = num_model_mesh(periods,target_depth,25,data[inds[0][0],:][2],data[inds[0][0],:][2]+data[inds[0][0],:][3],
													data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],data[inds[0][0],:][8],
													data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4])
				rho_model = 1./sigma_model_demo
				fig1, ax = plt.subplots(1,1)
				fig1.set_size_inches(8,4)
				z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
				z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
				rho_model_plt = np.repeat(rho_model, 2, axis=0)
				ax.semilogy(z, rho_model_plt,'r-',label='True', lw = 1., alpha=.7)
				ax.set_xlim([-1000., 0.])
				ax.set_ylim([1, 1500])
				ax.invert_xaxis() # plot the surface on the left
				ax.set_xlabel("Elevation (m)")
				ax.set_ylabel("Resistivity (Ohm m)") 
				
				plt.savefig(dat.work_dir+os.sep+'mcmc_bfit_model_res_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
		
		# Fit: observed temperature profile for layer model and construct estimated temperature profile in well position
		# Calculate beta parameter for the layers model (well) to use in demo
		if False: 

			# Define boundaries (Z) for Temp. function by layer
			Zmin = [bf_depth_layer1, bf_depth_layer2, bf_depth_layer3]
			Zmax = [bf_depth_layer1 + bf_thickness_layer1, bf_depth_layer2 + bf_thickness_layer2, -z_vec[0]]# bf_depth_layer3 + bf_thickness_layer3]
			# find temperatures in boundaries layers 
			def find_nearest(array, value):
				array = np.asarray(array)
				idx = (np.abs(array - value)).argmin()
				return array[idx]
			def T_est(Tw, z, Zmin, Zmax):
				z = np.asarray(z)			
				# Tmin y T max for layer 1
				inds_z = np.where(z == find_nearest(z, Zmin[0]))
				inds_z_l1 = int(inds_z[0][0])
				Tmin_l1 = Tw[inds_z_l1]
				inds_z = np.where(z == find_nearest(z, Zmax[0]))
				inds_z_l1 = int(inds_z[0][0])
				Tmax_l1 = Tw[inds_z_l1]
				# Tmin y T max for layer 2
				inds_z = np.where(z == find_nearest(z, Zmin[1]))
				inds_z_l2 = int(inds_z[0][0])
				Tmin_l2 = Tw[inds_z_l2]
				inds_z = np.where(z == find_nearest(z, Zmax[1]))
				inds_z_l2 = int(inds_z[0][0])
				Tmax_l2 = Tw[inds_z_l2]
				# Tmin y T max for layer 3
				inds_z = np.where(z == find_nearest(z, Zmin[2]))
				inds_z_l3 = int(inds_z[0][0])
				Tmin_l3 = Tw[inds_z_l3]
				inds_z = np.where(z == find_nearest(z, Zmax[2]))
				inds_z_l3 = int(inds_z[0][0])
				Tmax_l3 = Tw[inds_z_l3]
				# T boundary condition 
				Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
				Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]

				# Fit Twell with Texp
				beta_range = np.arange(-3.05, -0.4, 0.05)
				
				# layer 1
				misfit_aux = 10.e3
				beta_opt_l1 = 100
				
				for beta in beta_range: 
					Test_aux = Texp2(z[inds_z_l1:len(z)],Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta)
					aux = np.sum((Test_aux-Tw[inds_z_l1:len(z)])**2)
					if aux < misfit_aux:
						misfit_aux = np.sum((Test_aux-Tw[inds_z_l1:len(z)])**2)
						beta_opt_l1 = beta
				
				Test_l1 = Texp2(z[inds_z_l1:len(z)],Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta_opt_l1)
	
				# layer 2
				misfit_aux = 10.e3
				beta_opt_l2 = 100
				
				for beta in beta_range: 
					Test_aux = Texp2(z[inds_z_l2:inds_z_l1],Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta)
					aux = np.sum((Test_aux-Tw[inds_z_l2:inds_z_l1])**2)
					if aux < misfit_aux:
						misfit_aux = np.sum((Test_aux-Tw[inds_z_l2:inds_z_l1])**2)
						beta_opt_l2 = beta
				Test_l2 = Texp2(z[inds_z_l2:inds_z_l1],Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta_opt_l2)	
				#print(Test_l2)
				#print(Tw[inds_z_l2:inds_z_l1])
				#print(beta_opt_l2)
			
			
				# layer 3
				misfit_aux = 10.e3
				beta_opt_l3 = 100
				
				for beta in beta_range: 
					Test_aux = Texp2(z[inds_z_l3:inds_z_l2],Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta)
					aux = np.sum((Test_aux-Tw[inds_z_l3:inds_z_l2])**2)
					if aux < misfit_aux:
						misfit_aux = np.sum((Test_aux-Tw[inds_z_l3:inds_z_l2])**2)
						beta_opt_l3 = beta
				Test_l3 = Texp2(z[inds_z_l3:inds_z_l2],Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta_opt_l3)	

				Test = np.concatenate((Test_l3, Test_l2, Test_l1),axis=0) 
				beta = [beta_opt_l1, beta_opt_l2, beta_opt_l3]
				slopes = [(Tmax_l1-Tmin_l1)/(Zmax[0]-Zmin[0]),(Tmax_l2-Tmin_l2)/(Zmax[1]-Zmin[1]),(Tmax_l3-Tmin_l3)/(Zmax[2]-Zmin[2])]		

				return Test, beta, Tmin, Tmax, slopes
				
			Test, beta, Tmin, Tmax, slopes = T_est(Tw_vec, -z_vec, Zmin, Zmax) # beta: convection coefficients for each layer  
			#print(Test[-1])
			#print(Tw_vec[-1])
			if True: # plot Temp estimated and observed in well
				plt.clf()
				fig = plt.figure(figsize=[4.5,7.5])
				ax = plt.axes([0.18,0.25,0.70,0.50])
				#ax = plt.axes([0.58,0.25,0.35,0.50])
				ax.plot(Test, z_vec, 'r--')
				ax.plot(Tw_vec, z_vec, 'b-')
				ax.plot(TT[:,i],zinterp, 'g--')
				ax.set_ylabel('depth [km]', size = textsize)
				ax.set_xlabel('Temp [C]', size = textsize)
				ax.set_xlim([50, 300])
				plt.savefig(dat.work_dir+os.sep+'temp_fit_well.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
		# demo profile: estimated layer model and calculated temp. profile using beta calculate for the well
		# Approach 1
		if False:
			# Define boundaries (Z) for Temp. function by layer
			Zmin = [bf_depth_layer1_demo, bf_depth_layer2_demo, bf_depth_layer3_demo]
			Zmax = [bf_depth_layer1_demo + bf_thickness_layer1_demo, bf_depth_layer2_demo + bf_thickness_layer2_demo, -z_vec[0]]# bf_depth_layer3 + bf_thickness_layer3]
			
			def Tdemo_BC(T, z, Zmin, Zmax):
				z = np.asarray(z)			
				# Tmin y T max for layer 1
				inds_z = np.where(z == find_nearest(z, Zmin[0]))
				inds_z_l1 = int(inds_z[0][0])
				Tmin_l1 = T[inds_z_l1]
				inds_z = np.where(z == find_nearest(z, Zmax[0]))
				inds_z_l1 = int(inds_z[0][0])
				Tmax_l1 = T[inds_z_l1]
				# Tmin y T max for layer 2
				inds_z = np.where(z == find_nearest(z, Zmin[1]))
				inds_z_l2 = int(inds_z[0][0])
				Tmin_l2 = T[inds_z_l2]
				inds_z = np.where(z == find_nearest(z, Zmax[1]))
				inds_z_l2 = int(inds_z[0][0])
				Tmax_l2 = T[inds_z_l2]
				# Tmin y T max for layer 3
				inds_z = np.where(z == find_nearest(z, Zmin[2]))
				inds_z_l3 = int(inds_z[0][0])
				Tmin_l3 = T[inds_z_l3]
				inds_z = np.where(z == find_nearest(z, Zmax[2]))
				inds_z_l3 = int(inds_z[0][0])
				Tmax_l3 = T[inds_z_l3]
				# T boundary condition 
				Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
				Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
				
				return Tmin, Tmax

			def T_BC_projection(z, Zini, beta, Tini):
				# (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta) - 1) + Tmin
				return (300 - Tini)*(np.exp(beta*(z - Zini)/(1000. - Zini)) - 1.)/(np.exp(beta) - 1.) + Tini		
			
			def Tdemo_BC2(T, Zmin, Zmax, beta, TT):

				# T boundary condition
				# Tmin y T max for layer 1
				Tmin_l1 = TT[-1,j]
				Tmax_l1 = T_BC_projection(Zmax[0], Zmin[0], beta[0], Tmin_l1)
				# Tmin y T max for layer 2
				Tmin_l2 = Tmax_l1
				Tmax_l2 = T_BC_projection(Zmax[1], Zmin[1], beta[1], Tmin_l2)
				# Tmin y T max for layer 3
				Tmin_l3 = Tmax_l2
				Tmax_l3 = T_BC_projection(Zmax[2], Zmin[2], beta[2], Tmin_l3)
				
				Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
				Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
				
				return Tmin, Tmax
			
			# find index of boundaries (Z)
			def T_est_demo(beta, z, Zmin, Zmax, Tmin, Tmax):
				z = np.asarray(z)				
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
				return Test
			
			#Tmin_demo, Tmax_demo = Tdemo_BC(Test, -z_vec, Zmin, Zmax) 
			Tmin_demo, Tmax_demo = Tdemo_BC2(Test, Zmin, Zmax, beta, TT) # TT: array of estimate temperatures by normal fit
			
			#print(Zmin)
			#print(beta)
			#print(Zmax)
			#print(Tmin_demo)
			#print(Tmax_demo)
			
			Test_demo = T_est_demo(beta, -z_vec, Zmin, Zmax, Tmin_demo, Tmax_demo)

			if True: # plot Temp estimated and observed in well
				plt.clf()
				fig = plt.figure(figsize=[4.5,7.5])
				ax = plt.axes([0.18,0.25,0.70,0.50])
				#ax = plt.axes([0.58,0.25,0.35,0.50])
				ax.plot(Test_demo, z_vec, 'r--')
				ax.plot(Td_vec, z_vec, 'b-')
				ax.set_ylabel('depth [km]', size = textsize)
				ax.set_xlabel('Temp [C]', size = textsize)
				ax.set_xlim([1, 300])
				plt.savefig(dat.work_dir+os.sep+'temp_fit_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
		
		# Approach 2
		if True:
		
			# Define boundaries (Z) for Temp. function by layer
			Zmin = [bf_depth_layer1_demo, bf_depth_layer2_demo, bf_depth_layer3_demo]
			Zmax = [bf_depth_layer1_demo + bf_thickness_layer1_demo, bf_depth_layer2_demo + bf_thickness_layer2_demo, -z_vec[0]]# bf_depth_layer3 + bf_thickness_layer3]
			#beta[1] = beta[0]
			def Tdemo_BC(T, z, Zmin, Zmax):
				z = np.asarray(z)			
				# Tmin y T max for layer 1
				inds_z = np.where(z == find_nearest(z, Zmin[0]))
				inds_z_l1 = int(inds_z[0][0])
				Tmin_l1 = T[inds_z_l1]
				inds_z = np.where(z == find_nearest(z, Zmax[0]))
				inds_z_l1 = int(inds_z[0][0])
				Tmax_l1 = T[inds_z_l1]
				# Tmin y T max for layer 2
				inds_z = np.where(z == find_nearest(z, Zmin[1]))
				inds_z_l2 = int(inds_z[0][0])
				Tmin_l2 = T[inds_z_l2]
				inds_z = np.where(z == find_nearest(z, Zmax[1]))
				inds_z_l2 = int(inds_z[0][0])
				Tmax_l2 = T[inds_z_l2]
				# Tmin y T max for layer 3
				inds_z = np.where(z == find_nearest(z, Zmin[2]))
				inds_z_l3 = int(inds_z[0][0])
				Tmin_l3 = T[inds_z_l3]
				inds_z = np.where(z == find_nearest(z, Zmax[2]))
				inds_z_l3 = int(inds_z[0][0])
				Tmax_l3 = T[inds_z_l3]
				# T boundary condition 
				Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
				Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
				
				return Tmin, Tmax

			def T_BC_projection(z, Zini, beta, Tini, Tfin):
				# (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta) - 1) + Tmin
				return (Tfin - Tini)*(np.exp(beta*(z - Zini)/(1000. - Zini)) - 1.)/(np.exp(beta) - 1.) + Tini		
			
			def Tdemo_BC_slope(T, Zmin, Zmax, beta, TT):

				# T boundary condition
				# Tmin y T max for layer 1
				Tmin_l1 = TT[-1,j]

				# look where the normal estimation is negative: built the Test from there  
				inds_z_inicial_aux = np.where(TT[:,j] <= 0. )
				inds_z_inicial = np.min(inds_z_inicial_aux)
				
				if Tmin_l1 <= 0:
					Tmin_l1 = TT[inds_z_inicial-1,j]
				T_max_BC_l1 = Tmin_l1 + slopes[0]*(Zmax[2]-Zmin[0])
				Tmax_l1 = T_BC_projection(Zmax[0], Zmin[0], beta[0], Tmin_l1, T_max_BC_l1)
				
				# Tmin y T max for layer 2
				Tmin_l2 = Tmax_l1
				#T_max_BC_l2 = Tmin_l1 + slopes[1]*(Zmax[2]-Zmin[0])
				T_max_BC_l2 = Tmin_l2 + slopes[1]*(Zmax[2]-Zmin[1])
				Tmax_l2 = T_BC_projection(Zmax[1], Zmin[1], beta[1], Tmin_l2, T_max_BC_l2)
				
				# Tmin y T max for layer 3
				Tmin_l3 = Tmax_l2
				T_max_BC_l3 = TT[0,j]
				#T_max_BC_l3 = Tmin_l3 + slopes[2]*(Zmax[2]-Zmin[2])
				Tmax_l3 = T_BC_projection(Zmax[2], Zmin[2], beta[2], Tmin_l3, T_max_BC_l3)
				
				Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
				Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
				
				return Tmin, Tmax
			
			# find index of boundaries (Z)
			def T_est_demo(beta, z, Zmin, Zmax, Tmin, Tmax):
				z = np.asarray(z)
				
				# look where the normal estimation is negative: built the Test from there  
				inds_z_inicial_aux = np.where(TT[:,j] <= 0. )
				inds_z_inicial = np.min(inds_z_inicial_aux)
				
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
				
				if inds_z_inicial < len(z):
					Test[0:inds_z_inicial-1] = Test[len(z)-inds_z_inicial+1:len(z)]
					Test[inds_z_inicial-1:len(z)] = Test[inds_z_inicial-2]

				return Test
			
			#Tmin_demo, Tmax_demo = Tdemo_BC(Test, -z_vec, Zmin, Zmax) 
			Tmin_demo, Tmax_demo = Tdemo_BC_slope(Test, Zmin, Zmax, beta, TT) # TT: array of estimate temperatures by normal fit
			Test_demo = T_est_demo(beta, -z_vec, Zmin, Zmax, Tmin_demo, Tmax_demo)

			print(beta)
			#print(slopes)
			#print(Zmin)
			#print(Zmax)
			#print(Ti)
			#print(Tmin_demo)
			#print(Tmax_demo)
			
			if True: # plot Temp estimated and observed in demo
				plt.clf()
				fig = plt.figure(figsize=[4.5,7.5])
				ax = plt.axes([0.18,0.25,0.70,0.50])
				#ax = plt.axes([0.58,0.25,0.35,0.50])
				ax.plot(Test_demo, z_vec, 'r--')
				ax.plot(Td_vec, z_vec, 'b-')
				ax.plot(TT[:,j],zinterp, 'g--')
				ax.set_ylabel('depth [km]', size = textsize)
				ax.set_xlabel('Temp [C]', size = textsize)
				ax.set_xlim([-300, 300])
				plt.savefig(dat.work_dir+os.sep+'temp_fit_demo.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
		
#########################################################################################################################################
#########################################################################################################################################
		
	if True:  # joint inversion through fit to layer model :  all the stations
	
		# Location 
		
		location = 1  # 0: house / 1: office  
		
		# 1. Create modem object and import temperature distribution
		#    Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 
		dat0, t2t = setup_simulation(work_dir = '02_Rest2Temp')
		dat, t2t = setup_simulation(work_dir = '03_Rest2Temp_allStations')
		yi,zi,Ti = t2t
		
		# 2. Recover inverted model 
		dat_aux, t2t = setup_simulation(work_dir = '01_Temp2Rest')		
		fls = glob(dat_aux.work_dir + os.sep+'*.rho')
		newest = max(fls, key = lambda x: os.path.getctime(x))

		#dat0.read_input(dat0.work_dir+os.sep+'results_noise2'+os.sep+'Modular_NLCG_090.rho')
		#dat0.read_input(dat0.work_dir+os.sep+'results_noise2'+os.sep+'Modular_NLCG_090.rho')
		dat0.read_input(newest)	
		
		#dat0.plot_rho2D(dat.work_dir+os.sep+'rho_inv.png', xlim = [-10,10], ylim = [-5,0], gridlines = True, clim = [1,1000])
		dat0.plot_rho2D(dat.work_dir+os.sep+'rho_inv.pdf', xlim = [-10,10], ylim = [-5,0], gridlines = True, clim = [0.1,1000])
		
		# 1. Setud simultation -> return modem object (dat) and temp profile T(y,z) (t2t) 

		#dat0.read_input(dat.work_dir+os.sep+'BLOCK2_NLCG_030.rho')
		#dat_aux.plot_rho2D(dat0.work_dir+os.sep+'rho_est.png', xlim = [-12., 12.], ylim = [-5.,0], gridlines = True, clim = [1,1000])
		
		# 3. get T profile in position of the wells (concident with stations) 
		zinterp = np.linspace(-1.5e3,-25.,101)
		ystation = np.arange(50., 5.0e3, 500.) # position station dx = 200 m

		# 4. get true rho in ystation positions
		z,rho0 = get_rho(dat0, ystation, zinterp)	
		# 5. Joint inversion in position of well 
		
		# 5.1 Calculate apparent resistivity associated with true (extracted from inversion result) resistivity profile in the well position 	
		z_aux = np.linspace(25.,2.e3,101) 
		dzi = z_aux[2] - z_aux[1]
		
		z_thickness = np.ones(len(zinterp)-1)*dzi
		periods = np.logspace(-3, 3,50) # Periods range 10**-3 to 10**3 [s]
		
		levels = [160,240] # clay cap isotherms boundaries
		positions = [i for i in range(len(ystation))]
		##################################################################################################################################
		# Define wells (by position) 
		well_pos = [0,2,8,9] #[0,1] # position of well with respect to the stations
		well_objects = []

		for i in range(len(well_pos)):
			j = i + 1
			#Attibuttes: name, pos, rho_profile, temp_profile, clay_cap_boundaries, clay_cap_resistivity, distances (to other stations). 
			name = 'well_{}'.format(j)
			pos = well_pos[i] 
			rho_profile = rho0[pos]
			clay_cap_resistivity= [np.min(rho_profile)]
			z,Tw = get_T(yi, zi, Ti, [ystation[pos]], zinterp)
			Tw_vec = np.concatenate(Tw,  axis=0)
			z_vec = np.concatenate(z,  axis=0)
			temp_profile = [z_vec, Tw_vec]
			temp_profile_est = [z_vec, []]
			beta = []
			slopes = []
			# temp profile going up
			Tw_vec_rev = Tw_vec[::-1]
			z_rev = z[::-1] 
			# boundaries cc: 
			grad_temp = np.gradient(savgol_filter(Tw_vec_rev, 51, 3))
			grad_temp[0] = grad_temp[1] 
			grad_temp[-1] = grad_temp[-2] 

			#print(Tw_vec_rev)
			#print(savgol_filter(Tw_vec_rev, len(Tw_vec_rev), 2))
			
			
			
			#for i in range(len(grad_temp)-1):
			#	if Tw_vec_rev[i] == Tw_vec_rev[i+1]:
			#		grad_temp[i] = 0. 
			#	if Tw_vec_rev[i] = 
			
			if (all(t > 0. for t in grad_temp) and any(t > levels[1] for t in Tw_vec)): # case when both temp boundaries are in the temp profile, only once. example well in position 0. 
				#print('1')
				idx1 ,value1 = find_nearest(Tw_vec,levels[0])
				idx2 ,value2 = find_nearest(Tw_vec,levels[1])
				clay_cap_boundaries = [z_vec[idx1],z_vec[idx2]]
				#print(clay_cap_boundaries)
				
				
			if (any(t < 0. for t in grad_temp) and any(t > levels[1] for t in Tw_vec)): # case when both temp boundaries are in the temp profile, more than ones. example well in position 4.
				#print('2')
				idx_aux ,value1 = find_nearest(grad_temp,0)
				idx1_aux ,value1 = find_nearest(Tw_vec_rev[0:idx_aux],levels[0])
				idx2_aux ,value2 = find_nearest(Tw_vec_rev[0:idx_aux],levels[1])
				idx1 = len(Tw_vec) + 1 - idx1_aux 		    # index in the not-reverse array 
				idx2 = len(Tw_vec) + 1 - idx2_aux 
				clay_cap_boundaries = [z_vec[idx1],z_vec[idx2]]
				#print(clay_cap_boundaries)

			if (all(t < levels[0] for t in Tw_vec_rev)):# case when temp never richs 160°. example well in position 9.
				#print('3')
				idx1 = -2 # top
				idx2 = -3 # top
				clay_cap_boundaries = [z_vec[idx1],z_vec[idx2]]
				#print(clay_cap_boundaries)
				
			if (any(t > levels[0] for t in Tw_vec_rev) and all(t < levels[1] for t in Tw_vec_rev)): # case when temp richs 160, but never reachs 220°, and have another 160°. example well in position 7 and 8.
				# case 1: temp goes up and never reach 220°
				#print('4')
				if (all(t > 0. for t in grad_temp)):
					#print('4.1')
					idx1 ,value1 = find_nearest(Tw_vec,levels[0])
					idx2 =  [0]
					clay_cap_boundaries = [z_vec[idx1],z_vec[idx2]]
					#print(clay_cap_boundaries)
					
				# case 2: temp start to decrease at some point and reach a second 160°. Example well in position 7 
				if (any(t < 0. for t in grad_temp)):
					#print('4.2')
					idx_aux ,valu1 = find_nearest(Tw_vec_rev,100.)	
					grad_temp[0:idx_aux] = 10.
					idx_aux ,value1 = find_nearest(grad_temp,0)
					idx1_aux ,value1 = find_nearest(Tw_vec_rev[0:idx_aux ],levels[0])
					idx2_aux ,value2 = find_nearest(Tw_vec_rev[idx_aux:-1],levels[0])
					
					idx1 = len(Tw_vec) + 1 - idx1_aux 		    # index in the not-reverse array 
					idx2 = len(Tw_vec) + 1 - (idx2_aux + idx_aux)
					clay_cap_boundaries = [z_vec[idx1],z_vec[idx2]]
					#print(clay_cap_boundaries)
			
			print(clay_cap_boundaries)
			distances = abs(ystation - ystation[pos])
			# to be define
			l1_resistivity = 600.
			hs_resistivity = 600.
			
			#locals()[name] = Well(name, pos, rho_profile, temp_profile, clay_cap_boundaries, clay_cap_resistivity, distances, l1_resistivity, hs_resistivity)
			well_objects.append(Well(name, pos, rho_profile, temp_profile, temp_profile_est, beta, slopes, clay_cap_boundaries, clay_cap_resistivity, distances, l1_resistivity, hs_resistivity))

		##################################################################################################################################
		# Define stations (by position) 
		sta_pos = [0 for i in range(len(ystation)-len(well_pos))]

		station_objects = []
		j = 0
		for i in range(len(ystation)): 
			if i not in well_pos: 
				sta_pos[j] =  i 
				j = j + 1
	
		for i in range(len(sta_pos)):
			j = i + 1
			# name, pos, rho_profile, temp_profile, clay_cap_boundaries, clay_cap_resistivity
			name = 'sta_{}'.format(j)
			#print(name)
			pos = sta_pos[i]
			#lat = 
			#lon = 
			rho_profile = rho0[pos]
			# import true temp profile: only valid for synthetic
			z,Tw = get_T(yi, zi, Ti, [ystation[pos]], zinterp)
			Tw_vec = np.concatenate(Tw,  axis=0)
			z_vec = np.concatenate(z,  axis=0)
			temp_profile = [z_vec, Tw_vec]
			#
			temp_profile_est = [] # to be estimated 
			beta = []
			slopes = []
			clay_cap_boundaries_est = [] # to be estimated 
			resistivity_layers_est = [] # to be estimated 
			distances = abs(ystation - ystation[pos])
			clay_cap_resistivity_prior = est_rest_cc_prior(pos, distances)
			resistivity_prior_l1_l3 = est_rest_l1_l3_prior(pos, distances)
			cc_bounds = est_cc_bound_prior(pos, distances)
			clay_cap_thick_prior = cc_bounds[1]-cc_bounds[0]
			#print(clay_cap_thick_prior)
			l1_thick_prior = cc_bounds[0]

			sigma_thick_l1, sigma_thick_l2 = est_sigma_thickness_prior(pos, distances)	
		
			#locals()[name] = Station(name, pos, rho_profile, temp_profile, clay_cap_boundaries, clay_cap_resistivity, distances, clay_cap_resistivity_prior, resistivity_prior_l1_l3)
			station_objects.append(Station(name, pos, rho_profile, temp_profile, temp_profile_est, beta, slopes, clay_cap_boundaries_est, resistivity_layers_est, distances, clay_cap_resistivity_prior, resistivity_prior_l1_l3, clay_cap_thick_prior, l1_thick_prior,sigma_thick_l1, sigma_thick_l2))

		# 5.2 Invert apparent resistivity to obtain a 3 layer model  (using MCMC)	
		# Estimate the conductivity parameters (via LLK)

		# run MCMC: all the stations
		if False: 
			##########################################################################################################################
			def mcmc_station(app_res_vec, periods, obj):
				# Create chain.dat
				def lnprob(pars, obs):
					# log likelihood for the model, given the data
					v = 0.15
					# Parameter constrain
					if (any(x<0 for x in pars)):
						return -np.Inf
					else:
						if 'well' in obj.name:
							prob = -np.sum((np.log10(obs[:,1])-np.log10(anaMT1D_f_2layers(*pars,obs[:,0])))**2)/v + lnprior_well(*pars)
						if 'well' not in obj.name:
							prob = -np.sum((np.log10(obs[:,1])-np.log10(anaMT1D_f_2layers(*pars,obs[:,0])))**2)/v + lnprior_station(*pars)
							
						if prob!=prob:
							return -np.Inf
						else: 
							return prob
						
				def lnprior_well(*pars):
					# pars = [thickness_layer1[0],thickness_layer2[1],rho_layer1[2],rho_layer2[3],rho_hs[4]]
					
					# 1th: high resistivity, around 600 ohm m 
					lp_rho_layer1 = -(pars[2] - obj.l1_resistivity)**2/ (2*.005**2)
					lp_thick_layer1 = -(pars[0] - abs(obj.clay_cap_boundaries[0]))**2/ (2*.005**2)
	
					# 2th: low resistivity, around 10 ohm m 
					# lp_alpha_l2 = -(pars[1]*pars[4] - well_alpha_2)**2/ (2*10**2)
					lp_rho_layer2 = -(pars[3] - obj.clay_cap_resistivity)**2/ (2*.005**2)
					lp_thick_layer2 = -(pars[1] - (abs(obj.clay_cap_boundaries[1])-abs(obj.clay_cap_boundaries[0])))**2/ (2*.005**2)
					
					# 3th (hs): low resistivity, around 600 ohm m
					#lp_thick_layer3 = -(pars[2] - well_thick_3)**2/ (2*20**2)					
					lp_rho_hs = -(pars[4] - obj.hs_resistivity)**2/ (2*.005**2)
					
					return  lp_rho_layer2 + lp_thick_layer2 + lp_rho_layer1 + lp_thick_layer1 + lp_rho_hs
					
				def lnprior_station(*pars):
					# pars = [thickness_layer1[0],thickness_layer2[1],rho_layer1[2],rho_layer2[3],rho_hs[4]]

					# 1th: high resistivity, around 600 ohm m 
					lp_thick_layer1 = -(pars[0] - abs(obj.l1_thick_prior))**2/ (2*obj.sigma_thick_l1**2)
					lp_rho_layer1 = -(pars[2] - obj.resistivity_prior_l1_l3[0])**2/ (2*500**2)
					
					# 2th layer: product between thickness and resistivity
					#lp_alpha_l2 = -(pars[1]*pars[3] - well_alpha_2)**2/ (2*1000**2)
					lp_rho_layer2 = -(pars[3] - obj.clay_cap_resistivity_prior)**2/ (2*50**2)
					lp_thick_layer2 = -(pars[1] - abs(obj.clay_cap_thick_prior))**2/ (2*obj.sigma_thick_l2**2)
					
					# 3th: high resistivity, around 600 ohm m 
					#lp_rho_layer3 = -(pars[5] - well_res_3)**2/ (2*200**2)
					#lp_thick_layer3 = -(pars[2] - well_thick_3)**2/ (2*20**2)	
					lp_rho_hs = -(pars[4] - obj.resistivity_prior_l1_l3[1])**2/ (2*500**2)
					
					return  lp_thick_layer1 + lp_rho_layer1 + lp_rho_layer2 + lp_thick_layer2 + lp_rho_hs
					
						
				ndim = 5                  # parameter space dimensionality
				nwalkers= 50           # number of walkers
				
				# create the emcee object (set threads>1 for multiprocessing)
				data = np.array([periods,app_res_vec]).T
				sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=1, args=[data,])

				# set the initial location of the walkers
				## initial model
				# pars = [thickness_layer1[0],thickness_layer2[1],rho_layer1[2],rho_layer2[3],rho_hs[4]]
				pars = [200,100,150,50,500]  # initial guess
				p0 = np.array([pars + 1e2*np.random.randn(ndim) for i in range(nwalkers)])  # add some noise

				# set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
				pos,prob,state=sampler.run_mcmc(p0,2000)
				# sampler.reset()
				#print(1)
				f = open("chain.dat", "w")
				nk,nit,ndim=sampler.chain.shape
				for k in range(nk):
					for i in range(nit):
						f.write("{:d} {:d} ".format(k, i))
						for j in range(ndim):
							f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
						f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
				f.close()
				
			print('run MCMC: wells')
			for well_obj in well_objects:
				print(well_obj.name)
				rho0_flip_demo = np.flipud(well_obj.rho_profile) # get rho true model 
				app_res_vec_demo, phase_vec_demo = anaMT1D_f_nlayers(z_thickness,rho0_flip_demo,periods) # get app. resistivity
				mcmc_station(app_res_vec_demo, periods, well_obj) # run MCMC
				#shutil.rmtree(dat.work_dir+os.sep+str(well_obj.pos))
				os.mkdir(dat.work_dir+os.sep+str(well_obj.pos))
				shutil.move('chain.dat', dat.work_dir+os.sep+str(well_obj.pos)+os.sep+'chain.dat')# move chain.dat to station folder
			
			print('run MCMC: MT stations')
			for sta_obj in station_objects:
				print(sta_obj.name)
				rho0_flip_demo = np.flipud(sta_obj.rho_profile) # get rho true model
				app_res_vec_demo, phase_vec_demo = anaMT1D_f_nlayers(z_thickness,rho0_flip_demo,periods) # get app. resistivity
				mcmc_station(app_res_vec_demo, periods, sta_obj) # run MCMC
				#shutil.rmtree(dat.work_dir+os.sep+str(sta_obj.pos))
				os.mkdir(dat.work_dir+os.sep+str(sta_obj.pos))
				shutil.move('chain.dat', dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+'chain.dat')# move chain.dat to station folder
					
		# Plot results MCMC: all the stations
		if False:
			print('Plot results MCMC')
			for k,ystationi in enumerate(ystation):
				print(str(k))
				# show corner plot
				chain = np.genfromtxt(dat.work_dir+os.sep+str(k)+os.sep+'chain.dat')
				weights = chain[:,-1]
				weights -= np.max(weights)
				weights = np.exp(weights)
				labels = ['thick. 1','thick. 2','rest. 1','rest. 2','rest. hs']
				fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30)
				plt.savefig(dat.work_dir+os.sep+str(k)+os.sep+'mcmc_results.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
				plt.close(fig)
			
				# walker plot
				if True:
					#chain = np.genfromtxt(chain_file)
					npar = int(chain.shape[1] - 3)
					f,axs = plt.subplots(npar,1)
					f.set_size_inches([8,8])
					# if npar == 3:
						# labels = ['k','pc','LLK']
					# elif npar == 4:
						# labels = ['ka','kb','pc','LLK']
					# elif npar == 5:
						# labels = ['ka','kb','pc','LLK_max','LLK']
					# elif npar == 6:
						# labels = ['ka','kb','pc','ln','LLK_max','LLK']
					# elif npar == 7:
						# labels = ['t1','t2','t3','r1','r2','r3','r4']

					for i,ax,label in zip(range(npar),axs,labels):
						for j in np.unique(chain[:,0]):
							ind = np.where(chain[:,0] == j)
							it = chain[ind,1]
							par = chain[ind,2+i]
							ax.plot(it[0],par[0],'k-')
						ax.set_ylabel(label)
					
					plt.savefig(dat.work_dir+os.sep+str(k)+os.sep+'walkers.png', dpi=300)
			
		# Sample posterior and plot predictions vs true known: all the stations
		if False: 
			print('Samples posterior')
			#for k in [0]:
			for k,ystationi in enumerate(ystation):

				print(str(k))
				######################################################################
				# reproducability
				np.random.seed(1)
				# load in the posterior
				chain = np.genfromtxt(dat.work_dir+os.sep+str(k)+os.sep+'chain.dat')
				params = chain[:,2:-1]
				
				# define parameter sets for forward runs
				Nruns = 2000
				pars = []
				pars_order = []
					# generate Nruns random integers in parameter set range (as a way of sampling this dist)
				Nsamples = 0
				llnmax=np.max(chain[:,-1])
				ln = chain[:,-1]-llnmax

				while Nsamples != Nruns:
					id = np.random.randint(0,params.shape[0]-1)
				#    lni = ln[id]
					# probability test
				#    lni = np.exp(lni)
				#    if lni<np.random.random():
				#        continue
					# in sample test
					par_new = [params[id,0], params[id,1], params[id,2], params[id,3], params[id,4]]
				#    for par in pars:
				#        if all([(abs(p-pn)/abs(p))<1.e-4 for p,pn in zip(par,par_new)]):
				#            continue
					pars.append([Nsamples, params[id,0], params[id,1], params[id,2], params[id,3], params[id,4]])
					pars_order.append([chain[id,0], chain[id,1],chain[id,2], chain[id,3],chain[id,4],chain[id,5],chain[id,6],chain[id,7]])
					Nsamples += 1
				
				# Write in .dat paramateres sampled in order of fit (best fit a the top)
				pars_order= np.asarray(pars_order)
				pars_order = pars_order[pars_order[:,7].argsort()[::]]

				f = open(dat.work_dir+os.sep+str(k)+os.sep+"chain_sample_order.dat", "w")
				for l in reversed(range(len(pars_order[:,1]))): #(len(data_order[:,1])):
					j= l - len(pars_order[:,1])
					#if (pars_order[j,3]<300 and j!=4 and j!=6):
					#	f.write("{:4.0f}\t {:4.0f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\n".format(pars_order[j,0], pars_order[j,1], pars_order[j,2], pars_order[j,3], pars_order[j,4], pars_order[j,5], pars_order[j,6], pars_order[j,7], pars_order[j,8], pars_order[j,9]))
					f.write("{:4.0f}\t {:4.0f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\n".format(pars_order[j,0], pars_order[j,1], pars_order[j,2], pars_order[j,3], pars_order[j,4], pars_order[j,5], pars_order[j,6], pars_order[j,7]))
				f.close()

				f,ax = plt.subplots(1,1)
				f.set_size_inches(8,4) 
				#ax.loglog(periods,app_res_vec_ana,'r-',label='True', lw = 1.0, alpha=0.7)
				ax.set_xlim([np.min(periods), np.max(periods)])
				ax.set_ylim([1e1,1e3])
				ax.set_xlabel('period [s]', size = textsize)
				ax.set_ylabel('resistiviy [Ohm m]', size = textsize)
				ax.set_title('(a) Apparent Resistivity: MCMC posterior samples', size = textsize)

				# plot eq data as vertical lines
				for par in pars:
					if all(x > 0. for x in par):
						app_res_vec_aux, phase_vec_aux = anaMT1D_f_2layers(*par[1:6],periods)
						ax.loglog(periods, app_res_vec_aux,'b-', lw = 0.1, alpha=0.1, zorder=0)
				
				rho0_flip_demo = np.flipud(rho0[k]) # get rho true model in 'k' station
				app_res_vec_demo, phase_vec_demo = anaMT1D_f_nlayers(z_thickness,rho0_flip_demo,periods) # get app. resistivity
				
				ax.loglog(periods, app_res_vec_demo,'r-', lw = 1.5, alpha=0.7, zorder=0)
				
				plt.savefig(dat.work_dir+os.sep+str(k)+os.sep+'mcmc_app_res.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
				plt.close(f) 
				
			# plot conductivity models	
			
				##rho_model = 1./sigma_model
				# fig1, ax = plt.subplots(1,1)
				# fig1.set_size_inches(8,4)
				# ax.set_xlim([-2000., 0.])
				# ax.set_ylim([0.1e1, 1.5e3])
				# ax.invert_xaxis() # plot the surface on the left
				# ax.set_xlabel("depth [m]", size = textsize)
				# ax.set_ylabel("resistivity [Ohm m]", size = textsize)
				# ax.set_title('(b) 1D three-layer model: MCMC posterior samples', size = textsize)				

				##plot eq data as vertical lines
				
				# data = np.genfromtxt(dat.work_dir+os.sep+str(k)+os.sep+'chain_sample_order.dat')
				# misfit = data[:,9]
				# inds = np.where(misfit == np.max(misfit)) 
				
				# for par in pars:
					# if all(x > 0. for x in par):
						# sigma_model, mesh = num_model_mesh(periods,1500,0,par[1],par[1]+par[2],par[4],par[5],par[6],par[7],par[1],par[2],par[3])
						# rho_model = 1./sigma_model
						# z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
						# z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
						# rho_model_plt = np.repeat(rho_model, 2, axis=0)
						# if len(z) == len(rho_model_plt):
							# ax.semilogy(z, rho_model_plt,'b-', lw = 0.1, alpha=0.1, zorder=0)

				# sigma_model, mesh = num_model_mesh(periods,1500,0,data[inds[0][0],:][2],data[inds[0][0],:][2]+data[inds[0][0],:][3],
													# data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],data[inds[0][0],:][8],
													# data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4])
				# rho_model = 1./sigma_model
				# z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
				# z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
				# rho_model_plt = np.repeat(rho_model, 2, axis=0)
				# ax.semilogy(z, rho_model_plt,'r-', lw = 1.5, alpha=.7, zorder=0)						
				 
				# plt.savefig(dat.work_dir+os.sep+str(k)+os.sep+'mcmc_rest_models_3layers.pdf', dpi=300, facecolor='w', edgecolor='w',
					# orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)

		#################################################

			## (1) boundaries clay cap: uncertainty z1 and z2
			deface = -25.
			#deface = -0.
			num_samples = Nruns
			sta_numb = len(ystation)
			z1_s_array = np.zeros((sta_numb,2))
			z2_s_array = np.zeros((sta_numb,2))
			z1_s_array_per=np.zeros((sta_numb,3))
			z2_s_array_per=np.zeros((sta_numb,3))
			
			# for j,ystationi in enumerate(ystation):
			for j in range(sta_numb):
				rest_mod_samples = np.genfromtxt(dat.work_dir+os.sep+str(j)+os.sep+"chain_sample_order.dat")
				z1_s = rest_mod_samples[0:num_samples,2]
				z2_s = rest_mod_samples[0:num_samples,3]
				z1_s_array[j,:] = [np.mean(z1_s), np.std(z1_s)]
				z2_s_array[j,:] = [np.mean(z2_s), np.std(z2_s)]
				z1_s_array_per[j,:] = [np.median(z1_s), np.mean(z1_s) - (np.mean(z1_s) - np.percentile(z1_s,10)),np.mean(z1_s) + abs(np.mean(z1_s) -  np.percentile(z1_s,90))]
				z2_s_array_per[j,:] = [np.median(z1_s) + np.median(z2_s), np.median(z1_s) + np.mean(z2_s) - (np.mean(z2_s) - np.percentile(z2_s,10)),np.median(z1_s) + np.mean(z2_s) + abs(np.mean(z2_s) -  np.percentile(z2_s,90))]

			fig = plt.figure(figsize=[7.5,5.5])
			ax = plt.axes([0.18,0.25,0.70,0.50])
			
			# plot using mean a std
			#ax.errorbar(ystation/1.e3, -z2_s_array[:,0], yerr=z2_s_array[:,1], fmt='bo', label='bottom')
			#ax.plot(ystation/1.e3, -z2_s_array[:,0],'bo-', label='bottom')
			#ax.fill_between(ystation/1.e3, -z2_s_array[:,0]-z2_s_array[:,1], -z2_s_array[:,0]+z2_s_array[:,1],  alpha=.05, edgecolor='b', facecolor='b')
			#ax.errorbar(ystation/1.e3, -z1_s_array[:,0], yerr=z1_s_array[:,1], fmt='ro', label='top')
			#ax.plot(ystation/1.e3, -z1_s_array[:,0], 'ro-', label='top')
			#ax.fill_between(ystation/1.e3, -z1_s_array[:,0]-z1_s_array[:,1], -z1_s_array[:,0]+z1_s_array[:,1],  alpha=.4, edgecolor='r', facecolor='r')
			
			# plot using meadian a percentile
			ax.plot(ystation/1.e3, deface -z2_s_array_per[:,0],'bo-', label='bottom')
			ax.fill_between(ystation/1.e3,deface -z2_s_array_per[:,1], deface -z2_s_array_per[:,2],  alpha=.5, edgecolor='b', facecolor='b')
			ax.plot(ystation/1.e3,deface -z1_s_array_per[:,0],'ro-', label='top')
			ax.fill_between(ystation/1.e3,deface -z1_s_array_per[:,1], deface -z1_s_array_per[:,2],  alpha=.5, edgecolor='r', facecolor='r')
			
			# True boundaries 
			
			ny2 = len(np.unique(yi))
			nz2 = len(np.unique(zi))
			yi2 = yi.reshape([nz2,ny2])
			zi2 = zi.reshape([nz2,ny2])
			Ti2 = Ti.reshape([nz2,ny2])
			levels = [160,240]
			CS = ax.contour(yi2/1.e3,zi2, Ti2, levels = levels, colors = 'g', linewidths = 0.8)
			#X0, Y0 = CS.collections[0].get_paths()[0].vertices.T
			#X1, Y1 = CS.collections[1].get_paths()[0].vertices.T
			#print(len(X0))
			#print(len(X1))
			
			#ax.fill_between(X0,Y0,Y0,  alpha=.1, edgecolor='g', facecolor='g')
			#plt.clabel(CS, fmt = "%i")
			
			ax.set_xlim([ystation[0]/1.e3, ystation[-1]/1.e3])
			ax.set_ylim([-2.2e3,0e1])
			ax.set_xlabel('y [km]', size = textsize)
			ax.set_ylabel('depth [m]', size = textsize)
			ax.set_title('Clay cap boundaries depth  ', size = textsize)
			ax.legend(loc=3, prop={'size': 10})	
			plt.savefig(dat.work_dir+os.sep+'z1_z2_uncert.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
			plt.savefig(dat.work_dir+os.sep+'z1_z2_uncert.png', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)	

			#for sta_obj in station_objects:	
			#	f = open(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+"beta_sta.dat", "w")
				
			#	rest_mod_samples = np.genfromtxt(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+"chain_sample_order.dat")
			#	z1_s = rest_mod_samples[0:num_samples,2]
			#	z2_s = rest_mod_samples[0:num_samples,3]
			#	sta_obj.beta = beta
				
		
		# look for the best fit solution of MCMC: all the stations
		if False:
			print('Best fit MCMC')
			for k,ystationi in enumerate(ystation):
			#for k in [0]:
				print(str(k))
				data = np.genfromtxt(dat.work_dir+os.sep+str(k)+os.sep+'chain.dat')
				misfit = data[:,9]
				inds = np.where(misfit == np.max(misfit)) 
				f = open(dat.work_dir+os.sep+str(k)+os.sep+"best_model.dat", "w")
				f.write("{:3.3f}\t".format(data[inds[0][0],:][2]))
				f.write("{:3.3f}\t".format(data[inds[0][0],:][3]))
				f.write("{:3.3f}\t".format(data[inds[0][0],:][4]))
				f.write("{:3.3f}\t".format(data[inds[0][0],:][5]))
				f.write("{:3.3f}\t".format(data[inds[0][0],:][6]))
				f.write("{:3.3f}\t".format(data[inds[0][0],:][7]))
				f.write("{:3.3f}\n".format(data[inds[0][0],:][8]))
				f.close()
				
				# data_order = data[data[:,9].argsort()[::-1]]

				# for l in range(len(data_order[:,1]) - 1):
					# if data_order[l,9] == data_order[l+1,9]:
						# data_order = np.delete(data_order,(l), axis=0)

				# f = open(dat.work_dir+os.sep+str(k)+os.sep+"chain_order.dat", "w")
				
				# for j in range(100): #(len(data_order[:,1])):
					# f.write("{:4.0f}\t {:4.0f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\t {:3.3f}\n".format(data_order[j,0], data_order[j,1], data_order[j,2], data_order[j,3], data_order[j,4], data_order[j,5], data_order[j,6], data_order[j,7], data_order[j,8], data_order[j,9]))
				# f.close()
				
				# layer model
				target_depth = 500
				sigma_model, mesh = num_model_mesh(periods,target_depth,25,data[inds[0][0],:][2],data[inds[0][0],:][2]+data[inds[0][0],:][3],
													data[inds[0][0],:][5],data[inds[0][0],:][6],data[inds[0][0],:][7],data[inds[0][0],:][8],
													data[inds[0][0],:][2],data[inds[0][0],:][3],data[inds[0][0],:][4])
				rho_model = 1./sigma_model
				fig1, ax = plt.subplots(1,1)
				fig1.set_size_inches(8,4)
				z = np.repeat(mesh.vectorNx[1:-1], 2, axis=0)
				z = np.r_[mesh.vectorNx[0], z, mesh.vectorNx[-1]]
				rho_model_plt = np.repeat(rho_model, 2, axis=0)
				ax.semilogy(z, rho_model_plt,'r-',label='True', lw = 1., alpha=.7)
				ax.set_xlim([-1000., 0.])
				ax.set_ylim([1, 1500])
				ax.invert_xaxis() # plot the surface on the left
				ax.set_xlabel("Elevation (m)")
				ax.set_ylabel("Resistivity (Ohm m)")
				
				plt.savefig(dat.work_dir+os.sep+str(k)+os.sep+'mcmc_bfit_model_res.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)

		# Calculate beta parameter for the layers model in wells and stations (estimated)
		if False: 
			print('Calculating beta')
			# wells
			# well_pos =  [0] #[0,1] # position of well with respect to the stations
			
			for well_obj in well_objects:
				print(well_obj.pos)
				data = np.genfromtxt(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+'chain_sample_order.dat')
				misfit = data[:,-1] # extract the column of misfit (max. prob)
				inds = np.where(misfit == np.max(misfit))  # find index of value with maximum prob of ocurrence

				# Define boundaries (Z) for Temp. function by layer
				Zmin = [well_obj.temp_profile[0][-1], well_obj.clay_cap_boundaries[0], well_obj.clay_cap_boundaries[1]]
				Zmax = [well_obj.clay_cap_boundaries[0],well_obj.clay_cap_boundaries[1], well_obj.temp_profile[0][0]]# 	
				def find_nearest(array, value):
					array = np.asarray(array)
					idx = (np.abs(array - value)).argmin()
					return array[idx]
				def T_beta_est(Tw, z, Zmin, Zmax):
					#z = np.asarray(z)			
					# Search for index of ccboud in depth profile
					# Tmin y T max for layer 1
					inds_z = np.where(z == find_nearest(z, Zmin[0]))
					inds_z_l1_top = int(inds_z[0][0])
					Tmin_l1 = Tw[inds_z_l1_top]
					
					inds_z = np.where(z == find_nearest(z, Zmax[0]))
					inds_z_l1_bot = int(inds_z[0][0])
					Tmax_l1 = Tw[inds_z_l1_bot]
					
					# Tmin y T max for layer 2
					inds_z = np.where(z == find_nearest(z, Zmin[1]))
					inds_z_l2_top = int(inds_z[0][0])
					Tmin_l2 = Tw[inds_z_l2_top]
					
					inds_z = np.where(z == find_nearest(z, Zmax[1]))
					inds_z_l2_bot = int(inds_z[0][0])
					Tmax_l2 = Tw[inds_z_l2_bot]
					
					# Tmin y T max for hs
					inds_z = np.where(z == find_nearest(z, Zmin[2]))
					inds_z_l3_top = int(inds_z[0][0])
					Tmin_l3 = Tw[inds_z_l3_top]
					inds_z = np.where(z == find_nearest(z, Zmax[2]))
					inds_z_l3_bot = int(inds_z[0][0])
					Tmax_l3 = Tw[inds_z_l3_bot]
					
					# T boundary condition 
					Tmin = [Tmin_l1, Tmin_l2, Tmin_l3]
					Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]

					# Fit Twell with Texp
					beta_range = np.arange(-30.0, 30.0, 0.5)
					beta_def = -2.5
					#print(beta_range)
					
					### Layer 1
					# Calculate beta that best fot the true temp profile 
					popt, pcov = curve_fit(Texp2, z[inds_z_l1_bot:inds_z_l1_top+1], Tw[inds_z_l1_bot:inds_z_l1_top+1], p0=[Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta_def], bounds=([Zmax[0]-1.,Zmin[0]-1.,Tmin[0]-1,Tmax[0]-1., beta_range[0]], [Zmax[0]+1.,Zmin[0]+1.,Tmin[0]+1.,Tmax[0]+1,beta_range[-1]]))
					
					beta_opt_l1 = popt[-1]
					Test_l1 = Texp2(z[inds_z_l1_bot:inds_z_l1_top+1],Zmax[0],Zmin[0],Tmin[0],Tmax[0],beta_opt_l1)

					### Layer 2
					# Calculate beta that best fot the true temp profile 
					popt, pcov = curve_fit(Texp2, z[inds_z_l2_bot:inds_z_l2_top+1], Tw[inds_z_l2_bot:inds_z_l2_top+1], p0=[Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta_def], bounds=([Zmax[1]-1.,Zmin[1]-1.,Tmin[1]-1,Tmax[1]-1., beta_range[0]], [Zmax[1]+1.,Zmin[1]+1.,Tmin[1]+1.,Tmax[1]+1,beta_range[-1]]))
					
					beta_opt_l2 = popt[-1]
					Test_l2 = Texp2(z[inds_z_l2_bot:inds_z_l2_top+1],Zmax[1],Zmin[1],Tmin[1],Tmax[1],beta_opt_l2)

					# layer 3
					# Calculate beta that best fot the true temp profile 
					popt, pcov = curve_fit(Texp2, z[inds_z_l3_bot:inds_z_l3_top+1], Tw[inds_z_l3_bot:inds_z_l3_top+1], p0=[Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta_def], bounds=([Zmax[2]-1.,Zmin[2]-1.,Tmin[2]-1,Tmax[2]-1., beta_range[0]], [Zmax[2]+1.,Zmin[2]+1.,Tmin[2]+1.,Tmax[2]+1,beta_range[-1]]))
					
					beta_opt_l3 = popt[-1]
					Test_l3 = Texp2(z[inds_z_l3_bot:inds_z_l3_top+1],Zmax[2],Zmin[2],Tmin[2],Tmax[2],beta_opt_l3)
					
					# concatenate the estimated curves
					Test = np.concatenate((Test_l3[:,], Test_l2[1:], Test_l1[1:]),axis=0) 
					beta = [beta_opt_l1, beta_opt_l2, beta_opt_l3]
					slopes = [(Tmax_l1-Tmin_l1)/(Zmax[0]-Zmin[0]),(Tmax_l2-Tmin_l2)/(Zmax[1]-Zmin[1]),(Tmax_l3-Tmin_l3)/(Zmax[2]-Zmin[2])]		
					
					return Test, beta, Tmin, Tmax, slopes
				
				# Calculate beta and estimated temp profile
				Test, beta, Tmin, Tmax, slopes = T_beta_est(well_obj.temp_profile[1], well_obj.temp_profile[0], Zmin, Zmax) # 

				# fill object well atributes (beta and Test)
				well_obj.beta = beta
				
				if well_obj.pos == 9:
					beta_9 = well_obj.beta[-1]
					well_obj.beta = [beta_9, beta_9, beta_9]
					print('beta 9')
					print(well_obj.beta)
				
				well_obj.slopes = slopes
				well_obj.temp_profile_est[1] = Test
				f = open(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+"beta_well.dat", "w")
				for k in beta:
					f.write("{:2.2f}\n".format(k))
				f.close()
				
				f = open(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+"slopes_well.dat", "w")
				for k in slopes:
					f.write("{:2.3f}\n".format(k))
				f.close()
				
				if True: # plot Temp estimated and observed in well
					plt.clf()
					fig = plt.figure(figsize=[4.5,7.5])
					ax = plt.axes([0.18,0.25,0.70,0.50])
					#ax = plt.axes([0.58,0.25,0.35,0.50])
					ax.plot(well_obj.temp_profile_est[1], well_obj.temp_profile_est[0], 'r--', label='Estimated Temp.')
					ax.plot(well_obj.temp_profile[1], well_obj.temp_profile[0], 'b-', label='True Temp.')
					ax.set_ylabel('depth [km]', size = textsize)
					ax.set_xlabel('Temp [C]', size = textsize)
					ax.set_xlim([50, 300])
					ax.legend()
					plt.savefig(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+'temp_fit_well.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
				
				f = open(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+"temp_fit_well.dat", "w")
				for k in Test:
					f.write("{:3.3f}\t".format(k))
				f.close()	
			
				
				
				#################################################################################################
				# Calculate T estimate for every sample of the posterior
				
				TT = np.genfromtxt(dat0.work_dir+os.sep+os.sep+'T_est_normal_dist.dat')
				TT = TT.transpose()
				TT2 = TT
				
				TT2[:,0] = Test
				#TT_true[:,j] = Td_vec
				data = np.genfromtxt(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+'chain_sample_order.dat')
				# s = [len(data[:,1]),len(TT2[:,0])]
				s = [1000,len(TT2[:,0])]
				TT_samples = np.zeros(s)

				for l in range(s[0]):

					# Define boundaries (Z) for Temp. function by layer
					t1 = data[l,:][2] # thickness layer 1
					t2 = data[l,:][3] # thickness layer 2

					Zmin_sample = [-z_vec[-1], -z_vec[-1] + t1  , -z_vec[-1] + t1 + t2]
					Zmax_sample = [Zmin_sample[1],Zmin_sample[2], -z_vec[0]]

					Tmin_sample, Tmax_sample = T_BC_trans(Zmin_sample, Zmax_sample, well_obj.slopes, well_obj)
					Test_sample = T_est_demo(well_obj.beta, -z_vec, Zmin_sample, Zmax_sample, Tmin_sample, Tmax_sample)
					
					#TT_samples[l,:] = Test_sample
					TT_samples[l,:] = well_obj.temp_profile_est[1]
					
				f = open(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+"T_est_samples.dat", "w")
				for k in range(0,len(TT_samples[0,:])):
					for i in range(0,len(TT_samples[:,0])):
						f.write("{:3.2f}\t".format(TT_samples[i,k]))
					f.write("\n".format())
				f.close()
				
				if True: # plot Temp estimated and observed in demo
					plt.clf()
					fig = plt.figure(figsize=[4.5,7.5])
					ax = plt.axes([0.18,0.25,0.70,0.50])
					#ax = plt.axes([0.58,0.25,0.35,0.50])
					for l in range(len(TT_samples[:,1])):
						ax.plot(TT_samples[l,:], z_vec, 'r-', lw = 0.5, alpha=0.1, zorder=0)
						
					ax.plot(well_obj.temp_profile_est[1], well_obj.temp_profile[0], 'r-', label = 'Est. profiles')
					ax.plot(well_obj.temp_profile[1], well_obj.temp_profile[0], 'b-', label = 'True profiles')
					#ax.plot(TT[:,j],zinterp, 'g--', lw = 0.2)
					ax.set_ylabel('depth [km]', size = textsize)
					ax.set_xlabel('Temp [C]', size = textsize)
					ax.set_xlim([-100, 350])
					ax.legend()
					plt.savefig(dat.work_dir+os.sep+str(well_obj.pos)+os.sep+'temp_fit_multi_'+str(well_obj.pos)+'.pdf', dpi=300, facecolor='w', edgecolor='w',
						orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)				
				#####				
################################################################################################
################################################################################################
			
			for sta_obj in station_objects:
				print(sta_obj.pos)
				
				# (1) calculate beta 
				# Calculate beta parameter for the stations 
				def est_beta_sta(pos, distances):
					#(0) well to consider : 2 nearest
					n = 2
					n_wells =  [i for i in range(n)]
					#(1) Search for two neighbor wells  # well_pos =  [0,3,9], sta_pos = [...rest...]
					dist_pos_well = [i-pos for i in well_pos]
					dist_pos_well = [abs(i) for i in  dist_pos_well]
					indx_sort = np.argsort(dist_pos_well)
					################
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

					#(2) Calculate distance between station and wells (2)
					dist_sta_wells = [distances[i] for i in closest_wells]	
					#(3) Extract beta from nearest wells 
						# closest well are the position in the general array 
						# calculate closest well in wells array
					numb = 0
					closest_wells_ref = [-1,-1] 
					for well in well_objects:
						if well.pos == closest_wells[0]:
							closest_wells_ref[0] = numb
						if well.pos == closest_wells[1]:
							closest_wells_ref[1] = numb
						numb = numb + 1
					beta_nwells = [well_objects[i].beta for i in closest_wells_ref] # layer 1 beta in closest wells
					
					# (4) Calculate beta for estation as ponderate sum of beta of closest wells  
					# layer 1
					
					beta_sta_l1 = [beta_nwells[i][0]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
					beta_l1 = sum(beta_sta_l1)
					# layer 2
					beta_sta_l2 = [beta_nwells[i][1]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
					beta_l2 = sum(beta_sta_l2)
					# layer 3
					beta_sta_l3 = [beta_nwells[i][2]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
					beta_l3 = sum(beta_sta_l3)
					beta = [beta_l1, beta_l2, beta_l3]
					return beta
					
				beta = est_beta_sta(sta_obj.pos, sta_obj.distances)
				sta_obj.beta = beta
				f = open(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+"beta_sta.dat", "w")
				for k in beta:
					f.write("{:2.2f}\n".format(k))
				f.close()
				
				# (1) extract boundaries and res from MCMC inversion and put in object  			
				#clay_cap_boundaries_est = [] # to be estimated 
				#clay_cap_resistivity_est = [] # to be estimated 

				data = np.genfromtxt(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+'chain_sample_order.dat')
				t1 = np.mean(data[:,2]) # extract the column of misfit (max. prob)
				t2 = np.mean(data[:,3])
				r1 = np.mean(data[:,4])
				r2 = np.mean(data[:,5])
				r3 = np.mean(data[:,6])
				sta_obj.clay_cap_boundaries_est = [-(t1),-(t1+t2)]
				print(sta_obj.clay_cap_boundaries_est)
				sta_obj.resistivity_layers_est = [r1,r2,r3]
				
				# (1) Calculate slopes

				def est_slopes_sta(pos, distances):
					#(0) well to consider : 2 nearest
					n = 2
					n_wells =  [i for i in range(n)]
					#(1) Search for two neighbor wells  # well_pos =  [0,3,9], sta_pos = [...rest...]
					dist_pos_well = [i-pos for i in well_pos]
					dist_pos_well = [abs(i) for i in  dist_pos_well]
					indx_sort = np.argsort(dist_pos_well)
					################
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

					#(2) Calculate distance between station and wells (2)
					dist_sta_wells = [distances[i] for i in closest_wells]	
					#(3) Extract beta from nearest wells 
						# closest well are the position in the general array 
						# calculate closest well in wells array
					numb = 0
					closest_wells_ref = [-1,-1] 
					for well in well_objects:
						if well.pos == closest_wells[0]:
							closest_wells_ref[0] = numb
						if well.pos == closest_wells[1]:
							closest_wells_ref[1] = numb
						numb = numb + 1
					slopes_nwells = [well_objects[i].slopes for i in closest_wells_ref] # layer 1 beta in closest wells
					
					# (4) Calculate beta for estation as ponderate sum of beta of closest wells  
					# layer 1
					
					slope_sta_l1 = [slopes_nwells[i][0]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
					slope_l1 = sum(slope_sta_l1)
					# layer 2
					slope_sta_l2 = [slopes_nwells[i][1]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
					slope_l2 = sum(slope_sta_l2)
					# layer 3
					slope_sta_l3 = [slopes_nwells[i][2]*(sum(dist_sta_wells)-dist_sta_wells[i])/sum(dist_sta_wells) for i in range(n)]
					slope_l3 = sum(slope_sta_l3)
					slopes = [slope_l1, slope_l2, slope_l3]
					return slopes
				slopes = est_slopes_sta(sta_obj.pos, sta_obj.distances )				
				sta_obj.slopes = slopes
			
		# Calculated temp. profile: all the stations			
		if False:
		
			print('Calc. Temp. profile')	
			for sta_obj in station_objects:	
				print(sta_obj.pos)
				
				TT = np.genfromtxt(dat0.work_dir+os.sep+os.sep+'T_est_normal_dist.dat')
				TT = TT.transpose()
				TT2 = TT

				Zmin = [-z_vec[-1], -sta_obj.clay_cap_boundaries_est[0], -sta_obj.clay_cap_boundaries_est[1]]				
				Zmax = [-sta_obj.clay_cap_boundaries_est[0],-sta_obj.clay_cap_boundaries_est[1], -z_vec[0]]#
					
				# Calculate temperature boundary conditions 
				def T_BC_slopes(Zmin, Zmax, slopes):
					
					Tmin_l1 = sta_obj.temp_profile[1][-1]
					Tmax_l1	= Tmin_l1 + slopes[0]*(Zmin[0] - Zmax[0])
					
					Tmin_l2 = Tmax_l1
					Tmax_l2	= Tmin_l2 + slopes[1]*(Zmin[1]-Zmax[1])
					
					Tmin_l3 = Tmax_l2
					Tmax_l3 = Tmin_l3 + slopes[2]*(Zmin[2]-Zmax[2])
					
					Tmin = [Tmin_l1, Tmin_l2, Tmin_l3] 
					Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
					
					#Tmax_2 = Tmin_l3 + slopes[2]*(Zmin[0]-Zmax[2])
					#print(Tmax_l3)
					#print(Tmax)
					#print(sta_obj.temp_profile[1][0])
					
					return Tmin, Tmax

				# def T_BC_trans(Zmin, Zmax, slopes):
					
					# sigma = 5. # std from the isotherm value 
					
					# Tmin_l1 = sta_obj.temp_profile[1][-1]
					# Tmax_l1 = np.random.normal(levels[0], sigma, 1)[0] #levels[0] 
					
					# Tmin_l2 = Tmax_l1
					# Tmax_l2 = np.random.normal(levels[1], sigma, 1)[0] #levels[1] 
					
					# Tmin_l3 = Tmax_l2
					# Tmax_l3 = Tmin_l3 + slopes[2]*(Zmin[2]-Zmax[2])
					
					# Tmin = [Tmin_l1, Tmin_l2, Tmin_l3] 
					# Tmax = [Tmax_l1, Tmax_l2, Tmax_l3]
					
					#print(Tmax)
					#print(sta_obj.temp_profile[1][0])

					# return Tmin, Tmax
				
				Tmin_demo, Tmax_demo = T_BC_slopes(Zmin, Zmax, sta_obj.slopes) # Temperature boundaries by slope method
				Tmin_demo, Tmax_demo = T_BC_trans(Zmin, Zmax, sta_obj.slopes, sta_obj) # Temperature boundaries by transition temp. 
				#print(Tmin_demo)
				#print(Tmax_demo)
				#print(sta_obj.temp_profile)

				# Calculate temperature profile 
		
				Test_demo = T_est_demo(sta_obj.beta, -z_vec, Zmin, Zmax, Tmin_demo, Tmax_demo)
				sta_obj.temp_profile_est = Test_demo
				
				#################################################################################################
				# Calculate T estimate for every sample of the posterior
				
				# TT2[:,0] = Test_demo
				##TT_true[:,j] = Td_vec
				# data = np.genfromtxt(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+'chain_sample_order.dat')
				##s = [len(data[:,1]),len(TT2[:,0])]
				##s = [500,len(TT2[:,0])]
				# s = [500,106]
				# TT_samples = np.zeros(s)
				
				TT = np.genfromtxt(dat0.work_dir+os.sep+os.sep+'T_est_normal_dist.dat')
				TT = TT.transpose()
				TT2 = TT
				
				TT2[:,0] = Test
				#TT_true[:,j] = Td_vec
				data = np.genfromtxt(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+'chain_sample_order.dat')
				# s = [len(data[:,1]),len(TT2[:,0])]
				s = [1000,len(TT2[:,0])]
				TT_samples = np.zeros(s)

				for l in range(s[0]):
				
				
				#data = np.genfromtxt(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+'chain_sample_order.dat')
					t1 = data[l,2] # extract the column of misfit (max. prob)
					t2 = data[l,3]
					r1 = data[l,4]
					r2 = data[l,5]
					r3 = data[l,6]
					#sta_obj.clay_cap_boundaries_est = [-(t1),-(t1+t2)]
					#sta_obj.resistivity_layers_est = [r1,r2,r3]

					Zmin_sample = [-z_vec[-1], -z_vec[-1] + t1  , -z_vec[-1] + t1 + t2]
					Zmax_sample = [Zmin_sample[1],Zmin_sample[2], -z_vec[0]]

					Tmin_sample, Tmax_sample = T_BC_trans(Zmin_sample, Zmax_sample, sta_obj.slopes, sta_obj)
					Test_sample = T_est_demo(sta_obj.beta, -z_vec, Zmin_sample, Zmax_sample, Tmin_sample, Tmax_sample)
					
					TT_samples[l,:] = Test_sample
					
				f = open(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+"T_est_samples.dat", "w")
				for k in range(0,len(TT_samples[0,:])):
					for i in range(0,len(TT_samples[:,0])):
						f.write("{:3.2f}\t".format(TT_samples[i,k]))
					f.write("\n".format())
				f.close()
				
				if True: # plot Temp estimated and observed in demo
					plt.clf()
					fig = plt.figure(figsize=[4.5,7.5])
					ax = plt.axes([0.18,0.25,0.70,0.50])
					#ax = plt.axes([0.58,0.25,0.35,0.50])
					for l in range(len(TT_samples[:,0])):
						ax.plot(TT_samples[l,:], sta_obj.temp_profile[0], 'r-', lw = 0.5, alpha=0.1)
						#ax.plot(sta_obj.temp_profile_est, sta_obj.temp_profile[0], 'r-', lw = 0.5, alpha=0.1, zorder=0)
					
					ax.plot(sta_obj.temp_profile_est, sta_obj.temp_profile[0], 'r-', label = 'Est. profiles')
					ax.plot(sta_obj.temp_profile[1], sta_obj.temp_profile[0], 'b-', label = 'True profiles')
					#ax.plot(TT[:,j],zinterp, 'g--', lw = 0.2)
					ax.set_ylabel('depth [km]', size = textsize)
					ax.set_xlabel('Temp [C]', size = textsize)
					ax.set_xlim([-10, 350])
					ax.legend(loc='lower left')
					plt.savefig(dat.work_dir+os.sep+str(sta_obj.pos)+os.sep+'temp_fit_sta_'+str(sta_obj.pos)+'.pdf', dpi=300, facecolor='w', edgecolor='w',
						orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
					
					
			############################################################################################################
			############################################################################################################
			
			# f = open(dat.work_dir+os.sep+"T_est_layers.dat", "w")
			# for k in range(0,len(TT2[0,:])):
				# for i in range(0,len(TT2[:,0])):
					# f.write("{:3.2f}\t".format(TT2[i,k]))
				# f.write("\n".format())
			# f.close()
			
			# f = open(dat.work_dir+os.sep+"T_true.dat", "w")
			# for k in range(0,len(TT_true[0,:])):
				# for i in range(0,len(TT_true[:,0])):
					# f.write("{:3.2f}\t".format(TT_true[i,k]))
				# f.write("\n".format())
			# f.close()		

		
			if False: # plot temperature figures # old temp figure 
			
			
				# 7. Plot: (1) temperature real and fit, and res in well position (true values) (ax1a)
				#          (2) temperature real and extrapolate, and res in 'j' position (away form well)
				#		   (3) Countour Temp map of extrapolate temp (ax)
				
				#T_true = np.genfromtxt(dat.work_dir+os.sep+'T_true.dat')
				
				plt.clf()
				fig = plt.figure(figsize=[14.0,7.0])
				ax = plt.axes([0.58,0.25,0.35,0.50])
				ax1= plt.axes([0.10,0.25,0.17,0.50])
				ax2= plt.axes([0.33,0.25,0.17,0.50])
				ax1a=ax1.twiny()
				ax2a=ax2.twiny()

				# contour surface of estimate model (app2)
				x = zinterp
				yy = np.array(ystation)
				yy,zz = np.meshgrid(yy, zinterp)
						
				levels = range(50,301,50)
				
				# use values from true temp. in well position
				#pwell = 0 # position well
				#Tw_est = np.genfromtxt(dat.work_dir+os.sep+str(pwell)+os.sep+'temp_fit_well.dat')
				#Tw_est= Tw_est.transpose()
				#TT2[:,0] = Tw_est
				
				# use values from app. normal dist in poisition p_app1
				p_app1 = 1 # position well
				TT2[:,p_app1] = TT[:,p_app1]
				p_app1 = 0 # position well
				TT2[:,p_app1] = TT[:,p_app1]
				
				# contour surface of temperature 
				
				CS = ax.contourf(yy/1.e3, zz, TT2, levels = levels, cmap='jet')
				cax = plt.colorbar(CS, ax = ax)
				
				
				xlim = ax.get_xlim()
				xlim = [0,4]
				ax.set_xlim(xlim)
				ylim = ax.get_ylim()
				ax.set_ylim(ylim)
				
				# contour lines of true model 
				
				ny = len(np.unique(yi))
				nz = len(np.unique(zi))
				yi = yi.reshape([nz,ny])
				zi = zi.reshape([nz,ny])
				Ti = Ti.reshape([nz,ny])
				
				CS = ax.contour(yi/1.e3,zi, Ti, levels = levels, colors = 'k', linewidths = 0.5)
				plt.clabel(CS, fmt = "%i")
				
				ax.set_ylabel('depth [km]', size = textsize)
				ax.set_xlabel('y [km]', size = textsize)
				
				#plot well temperature
				pwell = 0 # position well
				Tw_est = np.genfromtxt(dat.work_dir+os.sep+str(pwell)+os.sep+'temp_fit_well.dat')
				Tw_est= Tw_est.transpose()
				ax1a.plot(Tw_est, z_vec, 'r--', lw = 0.7)
				#ax1a.plot(TT_true[:, pwell], z_vec, 'r-')
				ax1.plot(rho0[pwell,:],zinterp,'b-')
				
				pdemo = 3 # position demo
				#ax2a.plot(TT_true[:, pdemo],zinterp, 'r-')
				ax2a.plot(TT2[:,pdemo],zinterp, 'r--', lw = 0.7)
				ax2.plot(rho0[pdemo,:],zinterp,'b-')
				
				for axi in [ax1,ax2]:
					axi.set_ylabel('depth [m]', size = textsize)
					axi.set_xlabel('resistivity [$\Omega$m]', size = textsize)
					axi.set_xlim([5, 800])
					axi.set_xscale('log')
					
				for axi in [ax1a,ax2a]:
					axi.set_xlabel('temperature [$^\circ$C]', size = textsize)
					axi.set_xlim([50, 300])
				
				ax.set_title('joint temperature inversion from MT and well data', size = textsize)
				
				for axi in [ax, ax1, ax2, ax1a, ax2a]:
					for t in axi.get_xticklabels()+axi.get_yticklabels(): t.set_fontsize(textsize)
				for t in cax.ax.get_yticklabels(): t.set_fontsize(textsize)
				
				# plt.savefig(dat.work_dir+os.sep+'inferred_temperature.png', dpi=300, facecolor='w', edgecolor='w',
					# orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
				plt.savefig(dat.work_dir+os.sep+'inferred_temperature.pdf', dpi=300, facecolor='w', edgecolor='w',
					orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=0.1)
				plt.close(fig)

			# T_true = np.genfromtxt(dat.work_dir+os.sep+'T_true.dat')
			# beta = np.genfromtxt(dat.work_dir+os.sep+'0'+os.sep+'beta_well.dat')

			#Initial temperature condition 
			#(1) well
			# well_pos = 0
			# T_well = T_true[well_pos,:]
			# j = well_pos
			# num_samples = 100
			# rest_mod_samples = np.genfromtxt(dat.work_dir+os.sep+str(j)+os.sep+"chain_sample_order.dat")
			
			#Search z1 and z2 
			# z1_s = rest_mod_samples[0:num_samples,2]
			# z1_mean_std = [np.mean(z1_s), np.std(z1_s)]
			# z2_s = rest_mod_samples[0:num_samples,3]
			# z2_mean_std = [np.mean(z2_s), np.std(z2_s)]
			## Calculate beta for layer 1 and layer 2
			# indx_z1 = int((np.abs(-zinterp - z1)).argmin())
			# indx_z2 = int((np.abs(-zinterp - z2)).argmin())	

			# def Texp3(z,Zmin,Zmax,Tmin,Tmax,beta): return (Tmax - Tmin)*(np.exp(beta*(z-Zmin)/(Zmax-Zmin))-1)/(np.exp(beta)-1) + Tmin
			# popt, pcov = curve_fit(Texp3, -zinterp[indx_z1:], T_well[indx_z1:], 
									# p0=[-zinterp[indx_z1],-zinterp[-1],T_well[indx_z1],T_well[-1],-0.4], 
									# bounds=([-zinterp[indx_z1]-1,-zinterp[-1]-1,T_well[indx_z1]-1,T_well[-1]-1,-3.0], [-zinterp[indx_z1]+1, -zinterp[-1]+1,T_well[indx_z1]+1,T_well[-1]+1,3.0]))
			
			# print(popt[4])
			#z1 = z1_mean_std[0] # choose in a distribution
			#z2 = z2_mean_std[0] # choose in a distribution
			
			
		if True: # plot quality control: using TE and TM 
			print('Quality control plot')
			
			dat00, t2t = setup_simulation(work_dir = '01_Temp2Rest')
			yi,zi,Ti = t2t

			dat00.read_data(dat00.work_dir+os.sep+'out.dat')
			
			#print(dat.TE_Resistivity[0])
			#print(dat.TM_Resistivity[0])
			#print(dat.TM_Resistivity[0]-dat.TE_Resistivity[0])
			#print(np.linalg.norm(dat.TM_Resistivity[0]-dat.TE_Resistivity[0]))
			
			#print(dat.ys[dat.ys >= dat.y0-1])
			
			TM_TE_array = np.log(dat00.TE_Resistivity)-np.log(dat00.TM_Resistivity) # Array of diferences between TE and TM for every station 
			TM_TE_vec = np.linalg.norm(TM_TE_array, axis = 1) # The two-norm of each column (station)
			
			#TM_TE_norm_sta = TM_TE_vec[ystation[-1] >= dat.ys >= dat.y0-1]			
			periods = dat00.f[:,0]

			# plot of TM and TE for every station  
			f,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(1,10)
			f.set_size_inches(20,8) 
			# n = 19 # start at 0 km
			n = 18
			ax1.loglog(dat00.TE_Resistivity[n+1],periods)
			ax1.loglog(dat00.TM_Resistivity[n+1],periods)
			ax1.invert_yaxis()
			ax2.loglog(dat00.TE_Resistivity[n+2],periods)
			ax2.loglog(dat00.TM_Resistivity[n+2],periods)
			ax2.invert_yaxis()
			ax3.loglog(dat00.TE_Resistivity[n+3],periods)
			ax3.loglog(dat00.TM_Resistivity[n+3],periods)
			ax3.invert_yaxis()
			ax4.loglog(dat00.TE_Resistivity[n+4],periods)
			ax4.plot(dat00.TM_Resistivity[n+4],periods)
			ax4.invert_yaxis()
			ax5.loglog(dat00.TE_Resistivity[n+5],periods)
			ax5.loglog(dat00.TM_Resistivity[n+5],periods)
			ax5.invert_yaxis()
			ax6.loglog(dat00.TE_Resistivity[n+6],periods)
			ax6.loglog(dat00.TM_Resistivity[n+6],periods)
			ax6.invert_yaxis()
			ax7.loglog(dat00.TE_Resistivity[n+7],periods)
			ax7.loglog(dat00.TM_Resistivity[n+7],periods)
			ax7.invert_yaxis()
			ax8.loglog(dat00.TE_Resistivity[n+8],periods)
			ax8.loglog(dat00.TM_Resistivity[n+8],periods)
			ax8.invert_yaxis()
			ax9.loglog(dat00.TE_Resistivity[n+9],periods)
			ax9.loglog(dat00.TM_Resistivity[n+9],periods)
			ax9.invert_yaxis()
			ax10.loglog(dat00.TE_Resistivity[n+10],periods)
			ax10.loglog(dat00.TM_Resistivity[n+10],periods)
			ax10.invert_yaxis()
			
			plt.savefig(dat.work_dir+os.sep+'TM_TE.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
			
		
			fig = plt.figure(figsize=[7.5,5.5])
			ax = plt.axes([0.18,0.25,0.70,0.50])
			
			ax.plot(dat00.ys, TM_TE_vec/np.max(TM_TE_vec),'*-', color = 'b')
			
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			ax.set_xlim([0, 1.0e4])
			ax.set_ylim([0.6,1.1])

			ax.set_xlabel('y [km]', size = textsize)
			ax.set_ylabel('', size = textsize)
			ax.set_title('Quality factor ', size = textsize)
			#ax.legend(loc=3, prop={'size': 10})
			
			#print(dat.TE_Resistivity[dat.ys >= dat.y0-1][0])
			
			plt.savefig(dat.work_dir+os.sep+'quality_factor.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)

			#dat.plot_impedance(dat.work_dir+os.sep+'TEimpedances.pdf', frequency = True)
			#dat.plot_impedance(dat.work_dir+os.sep+'TMimpedances.pdf', frequency = True, TE = False)
			
			
		if True: # plot uncertainty: resistivity model (boundaries) and temperature
			print('Uncertain Isotherms')

			# (2) Uncertaity temperatures: isotherms errors
			deface = -25.
			num_samples = 1000
			sta_numb = len(ystation)
			
			#isotherms = [100, 150, 200, 250]
			#isotherms = [120, 160, 200, 240]
			#iso_col = ['b', 'g', 'r', 'c'] #, 'm']
			isotherms = [100, 220]#180] 
			iso_col = ['b', 'r', 'c','m','g']
			#isotherms = [100, 180]
			#iso_col = ['b', 'm', 'c','m','g']
			z_iso = np.zeros((sta_numb,num_samples))
			z_iso2 = np.zeros((sta_numb,2))
			z_iso2_per = np.zeros((sta_numb,3))
			
			fig = plt.figure(figsize=[7.5,5.5])
			ax = plt.axes([0.18,0.25,0.70,0.50])
			to_where = sta_numb
			aux_to_where = 0
			
			# Estimated isotherms 
			
			for k in range(len(isotherms)): # isotherms			
				for j in range(sta_numb): # stations
					T_esp_samples = np.genfromtxt(dat.work_dir+os.sep+str(j)+os.sep+"T_est_samples.dat")
					T_esp_samples = T_esp_samples.T	
					for l in range(num_samples): # samples

						indx = int((np.abs(T_esp_samples[l,:] - isotherms[k])).argmin())
							
						if 	j > 6 and np.min(np.abs(T_esp_samples[l,:] - isotherms[k])) > 20.: 
							z_iso[j,l] = 2000.
						else:
							z_iso[j,l] = -1*zinterp[-1] + (len(zinterp)-indx) * (zinterp[1]-zinterp[0]) 

						#print(z_iso)
					z_iso2[j,:] = [np.median(z_iso[j,:]), np.std(z_iso[j,:])]	

					# if every z_iso is assign to 2000., fill (plot) only until there
					if j > 6 and np.std(z_iso[j,:]) == 0.0 and np.mean(z_iso[j,:]) > 2000. - 1. and aux_to_where == 0: 
						print('hola')
						to_where = j 
						aux_to_where = 1
					
					z_iso2_per[j,:] = [np.median(z_iso[j,:])
									, np.median(z_iso[j,:]) - (np.median(z_iso[j,:]) -  np.percentile(z_iso[j,:], 5))
									, np.median(z_iso[j,:]) + abs(np.median(z_iso[j,:]) -  np.percentile(z_iso[j,:], 95))]
					
				# plot using mean and std
				#ax.errorbar(ystation/1.e3, -z_iso2[:,0], yerr=z_iso2[:,1], label=str(isotherms[k])+' C')
				#ax.plot(ystation/1.e3, -z_iso2[:,0], label=str(isotherms[k])+' C')
				#ax.fill_between(ystation/1.e3, -z_iso2[:,0]-z_iso2[:,1], -z_iso2[:,0]+z_iso2[:,1],  alpha=.1)
				
				# plot using meadian a percentile
				#ax.errorbar(ystation/1.e3, -z_iso2[:,0], yerr=z_iso2[:,1], label=str(isotherms[k])+' C')
				#ax.plot(ystation/1.e3, -z_iso2[:,0], label=str(isotherms[k])+' C', color = iso_col[k])
				print(to_where)

				ax.fill_between(ystation/1.e3, -z_iso2_per[:,1], -z_iso2_per[:,2],  alpha=.4, edgecolor=iso_col[k], facecolor=iso_col[k], where = ystation/1.e3 < to_where/2.)

				to_where = sta_numb
				aux_to_where = 0
				#print(z_iso2)
				
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			ax.set_xlim([ystation[0]/1.e3 - 0.1, ystation[-1]/1.e3 + 0.1])
			ax.set_ylim([-2.0e3,0e1])
			ax.set_xlabel('y [km]', size = textsize)
			ax.set_ylabel('depth [m]', size = textsize)
			#ax.set_title('(b) Uncertain isotherms ', size = textsize)
			#ax.legend(loc=3, prop={'size': 10})

			# True Isothrems
			# yi,zi,Ti = t2t # temperature grid
			ny2 = len(np.unique(yi))
			nz2 = len(np.unique(zi))
			yi2 = yi.reshape([nz2,ny2])
			zi2 = zi.reshape([nz2,ny2])
			Ti2 = Ti.reshape([nz2,ny2])
			#levels = range(120,241,40)
			#CS = ax.contour(yi2/1.e3,zi2, Ti2, levels = isotherms, colors = 'k', linewidths = 0.3)
			CS = ax.contour(yi2/1.e3,zi2, Ti2, levels = isotherms, colors = iso_col, linewidths = 0.5, linestyles='dashed')

			plt.clabel(CS, fmt = "%i")
			#plt.legend(CS[0], CS[1], CS[2])
			
			plt.savefig(dat.work_dir+os.sep+'isot_uncert_'+str(isotherms[1])+'.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
			plt.savefig(dat.work_dir+os.sep+'isot_uncert_'+str(isotherms[1])+'.png', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=.1)
				
				
				
			#################################################################################################################	
			### Figure with quality control 
			
			
			deface = -25.
			
			fig = plt.figure() 
			
			gs = gridspec.GridSpec(4, 3) 
			ax1=fig.add_subplot(gs[0:3,:]) # First row, first column
			ax2=fig.add_subplot(gs[3,:]) # First row, first column
			
			for k in range(len(isotherms)): # isotherms			
				for j in range(sta_numb): # stations
					T_esp_samples = np.genfromtxt(dat.work_dir+os.sep+str(j)+os.sep+"T_est_samples.dat")
					T_esp_samples = T_esp_samples.T	
					for l in range(num_samples): # samples
						indx = int((np.abs(T_esp_samples[l,:] - isotherms[k])).argmin())
						z_iso[j,l] = -1*zinterp[-1] + (len(zinterp)-indx) * (zinterp[1]-zinterp[0]) 
					z_iso2[j,:] = [np.mean(z_iso[j,:]), np.std(z_iso[j,:])]	
					z_iso2_per[j,:] = [np.median(z_iso[j,:])
									, np.median(z_iso[j,:]) - (np.median(z_iso[j,:]) -  np.percentile(z_iso[j,:], 5))
									, np.median(z_iso[j,:]) + abs(np.median(z_iso[j,:]) -  np.percentile(z_iso[j,:], 95))]
					
				# plot using mean and std
				#ax.errorbar(ystation/1.e3, -z_iso2[:,0], yerr=z_iso2[:,1], label=str(isotherms[k])+' C')
				#ax.plot(ystation/1.e3, -z_iso2[:,0], label=str(isotherms[k])+' C')
				#ax.fill_between(ystation/1.e3, -z_iso2[:,0]-z_iso2[:,1], -z_iso2[:,0]+z_iso2[:,1],  alpha=.1)
				
				# plot using meadian a percentile
				#ax.errorbar(ystation/1.e3, -z_iso2[:,0], yerr=z_iso2[:,1], label=str(isotherms[k])+' C')
				#ax.plot(ystation/1.e3, -z_iso2[:,0], label=str(isotherms[k])+' C', color = iso_col[k])
				#ax1.plot(ystation/1.e3, -z_iso2[:,0],'o', color = iso_col[k],  alpha=.2)
				ax1.fill_between(ystation/1.e3, -z_iso2_per[:,1], -z_iso2_per[:,2],  alpha=.4, edgecolor=iso_col[k], facecolor=iso_col[k])
			
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			ax1.set_xlim(xlim)
			ax1.set_ylim(ylim)
			ax1.set_xlim([ystation[0]/1.e3 - 0.1, ystation[-1]/1.e3 + 0.1])
			ax1.set_ylim([-2.0e3,0e1])
			#ax1.set_xlabel('y [km]', size = textsize-2)
			ax1.set_ylabel('depth [m]', size = textsize-4)
			ax1.set_title('(b) Estimated isotherms ', size = textsize)
			#ax.legend(loc=3, prop={'size': 10})

			# True Isothrems

			CS = ax1.contour(yi2/1.e3,zi2, Ti2, levels = isotherms, colors = iso_col, linewidths = 0.5, linestyles='dashed')
			plt.clabel(CS, fmt = "%i")
			#plt.legend(CS[0], CS[1], CS[2])

			ax2.plot(dat00.ys/1.e3, TM_TE_vec/np.max(TM_TE_vec),'*-', color = 'b')
			
			xlim = ax2.get_xlim()
			ylim = ax2.get_ylim()
			ax2.set_xlim(xlim)
			ax2.set_ylim(ylim)
			ax2.set_xlim([ystation[0]/1.e3 - 0.1, ystation[-1]/1.e3 + 0.1])
			ax2.set_ylim([0.67,0.9])
			ax2.set_xlabel('y [km]', size = textsize-4)
			ax2.set_ylabel('Quality factor', size = textsize-4)
			
			#f.tight_layout()
			
			plt.savefig(dat.work_dir+os.sep+'isot_uncert_quality_control.pdf', dpi=300, facecolor='w', edgecolor='w',
				orientation='portrait', format='pdf',transparent=True, bbox_inches=None, pad_inches=.1)
				
				
				
#########################################################################################################################################
#########################################################################################################################################



