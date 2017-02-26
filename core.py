#!/usr/bin/env python
# encoding: utf-8
"""
Foundation classes for the `ddmatch` library.

Documentation guidelines are available `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Created by Klas Modin on 2014-11-03.
"""

import numpy as np 
import numba

def generate_optimized_image_composition(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
	def image_compose_2d(I,xphi,yphi,Iout):
		for i in range(s):
			for j in range(s):
				xind = int(xphi[i,j])
				yind = int(yphi[i,j])
				xindp1 = xind+1
				yindp1 = yind+1
				deltax = xphi[i,j]-float(xind)
				deltay = yphi[i,j]-float(yind)
				
				# Id xdelta is negative it means that xphi is negative, so xind
				# is larger than xphi. We then reduce xind and xindp1 by 1 and
				# after that impose the periodic boundary conditions.
				if (deltax < 0 or xind < 0):
					deltax += 1.0
					xind -= 1
					xind %= s
					xindp1 -= 1
					xindp1 %= s
				elif (xind >= s):
					xind %= s
					xindp1 %= s
				elif (xindp1 >= s):
					xindp1 %= s

				if (deltay < 0 or xind < 0):
					deltay += 1.0
					yind -= 1
					yind %= s
					yindp1 -= 1
					yindp1 %= s
				elif (yind >= s):
					yind %= s
					yindp1 %= s
				elif (yindp1 >= s):
					yindp1 %= s
					
				onemdeltax = 1.-deltax
				onemdeltay = 1.-deltay
				Iout[i,j] = I[yind,xind]*onemdeltax*onemdeltay+\
					I[yind,xindp1]*deltax*onemdeltay+\
					I[yindp1,xind]*deltay*onemdeltax+\
					I[yindp1,xindp1]*deltay*deltax

	return image_compose_2d

def generate_optimized_diffeo_evaluation(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:],f8[:],f8[:],f8[:])')
	def eval_diffeo_2d(xpsi,ypsi,xvect,yvect,xout,yout):
	# Evaluate diffeo psi(x,y) for each pair in xvect, yvect. 
	# Assuming psi is periodic.

		d = xvect.shape[0]

		for i in range(d):
			xind = int(xvect[i])
			yind = int(yvect[i])
			xindp1 = xind+1
			yindp1 = yind+1
			deltax = xvect[i]-float(xind)
			deltay = yvect[i]-float(yind)
			xshift = 0.0
			xshiftp1 = 0.0
			yshift = 0.0
			yshiftp1 = 0.0
			
			# Id xdelta is negative it means that xphi is negative, so xind
			# is larger than xphi. We then reduce xind and xindp1 by 1 and
			# after that impose the periodic boundary conditions.
			if (deltax < 0 or xind < 0):
				deltax += 1.0
				xind -= 1
				xindp1 -= 1
				xind %= s
				xshift = -float(s) # Should use floor_divide here instead.
				if (xindp1 < 0):
					xindp1 %= s
					xshiftp1 = -float(s)
			elif (xind >= s):
				xind %= s
				xindp1 %= s
				xshift = float(s)
				xshiftp1 = float(s)
			elif (xindp1 >= s):
				xindp1 %= s
				xshiftp1 = float(s)

			if (deltay < 0 or yind < 0):
				deltay += 1.0
				yind -= 1
				yindp1 -= 1
				yind %= s
				yshift = -float(s) # Should use floor_divide here instead.
				if (yindp1 < 0):
					yindp1 %= s
					yshiftp1 = -float(s)
			elif (yind >= s):
				yind %= s
				yindp1 %= s
				yshift = float(s)
				yshiftp1 = float(s)
			elif (yindp1 >= s):
				yindp1 %= s
				yshiftp1 = float(s)
				
			xout[i] = (xpsi[yind,xind]+xshift)*(1.-deltax)*(1.-deltay)+\
				(xpsi[yind,xindp1]+xshiftp1)*deltax*(1.-deltay)+\
				(xpsi[yindp1,xind]+xshift)*deltay*(1.-deltax)+\
				(xpsi[yindp1,xindp1]+xshiftp1)*deltay*deltax
			
			yout[i] = (ypsi[yind,xind]+yshift)*(1.-deltax)*(1.-deltay)+\
				(ypsi[yind,xindp1]+yshift)*deltax*(1.-deltay)+\
				(ypsi[yindp1,xind]+yshiftp1)*deltay*(1.-deltax)+\
				(ypsi[yindp1,xindp1]+yshiftp1)*deltay*deltax

	return eval_diffeo_2d

def generate_optimized_diffeo_composition(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
	def diffeo_compose_2d(xpsi,ypsi,xphi,yphi,xout,yout):
	# Compute composition psi o phi. 
	# Assuming psi and phi are periodic.

		for i in range(s):
			for j in range(s):
				xind = int(xphi[i,j])
				yind = int(yphi[i,j])
				xindp1 = xind+1
				yindp1 = yind+1
				deltax = xphi[i,j]-float(xind)
				deltay = yphi[i,j]-float(yind)
				xshift = 0.0
				xshiftp1 = 0.0
				yshift = 0.0
				yshiftp1 = 0.0
				
				# Id xdelta is negative it means that xphi is negative, so xind
				# is larger than xphi. We then reduce xind and xindp1 by 1 and
				# after that impose the periodic boundary conditions.
				if (deltax < 0 or xind < 0):
					deltax += 1.0
					xind -= 1
					xindp1 -= 1
					xind %= s
					xshift = -float(s) # Should use floor_divide here instead.
					if (xindp1 < 0):
						xindp1 %= s
						xshiftp1 = -float(s)
				elif (xind >= s):
					xind %= s
					xindp1 %= s
					xshift = float(s)
					xshiftp1 = float(s)
				elif (xindp1 >= s):
					xindp1 %= s
					xshiftp1 = float(s)

				if (deltay < 0 or yind < 0):
					deltay += 1.0
					yind -= 1
					yindp1 -= 1
					yind %= s
					yshift = -float(s) # Should use floor_divide here instead.
					if (yindp1 < 0):
						yindp1 %= s
						yshiftp1 = -float(s)
				elif (yind >= s):
					yind %= s
					yindp1 %= s
					yshift = float(s)
					yshiftp1 = float(s)
				elif (yindp1 >= s):
					yindp1 %= s
					yshiftp1 = float(s)
					
				xout[i,j] = (xpsi[yind,xind]+xshift)*(1.-deltax)*(1.-deltay)+\
					(xpsi[yind,xindp1]+xshiftp1)*deltax*(1.-deltay)+\
					(xpsi[yindp1,xind]+xshift)*deltay*(1.-deltax)+\
					(xpsi[yindp1,xindp1]+xshiftp1)*deltay*deltax
				
				yout[i,j] = (ypsi[yind,xind]+yshift)*(1.-deltax)*(1.-deltay)+\
					(ypsi[yind,xindp1]+yshift)*deltax*(1.-deltay)+\
					(ypsi[yindp1,xind]+yshiftp1)*deltay*(1.-deltax)+\
					(ypsi[yindp1,xindp1]+yshiftp1)*deltay*deltax

	return diffeo_compose_2d

def generate_optimized_image_gradient(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def image_gradient_2d(I,dIdx,dIdy):
		im1 = s-1
		jm1 = s-1
		for i in range(s-1):
			ip1 = i+1
			for j in range(s-1):
				jp1 = j+1
				dIdy[i,j] = (I[ip1,j]-I[im1,j])/2.0
				dIdx[i,j] = (I[i,jp1]-I[i,jm1])/2.0
				jm1 = j
			dIdy[i,s-1] = (I[ip1,s-1]-I[im1,s-1])/2.0
			dIdx[i,s-1] = (I[i,0]-I[i,s-2])/2.0
			jm1 = s-1
			im1 = i
		for j in range(s-1):
			jp1 = j+1
			dIdy[s-1,j] = (I[0,j]-I[im1,j])/2.0
			dIdx[s-1,j] = (I[s-1,jp1]-I[s-1,jm1])/2.0
			jm1 = j
		dIdy[s-1,s-1] = (I[0,s-1]-I[s-2,s-1])/2.0
		dIdx[s-1,s-1] = (I[s-1,0]-I[s-1,s-2])/2.0

	return image_gradient_2d


def generate_optimized_divergence(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def divergence_2d(vx,vy,divv):
		im1 = s-1
		jm1 = s-1
		for i in range(s-1):
			ip1 = i+1
			for j in range(s-1):
				jp1 = j+1
				divv[i,j] = (vy[ip1,j]-vy[im1,j] + vx[i,jp1]-vx[i,jm1])/2.0
				jm1 = j
			divv[i,s-1] = (vy[ip1,s-1]-vy[im1,s-1] + vx[i,0]-vx[i,s-2])/2.0
			jm1 = s-1
			im1 = i
		for j in range(s-1):
			jp1 = j+1
			divv[s-1,j] = (vy[0,j]-vy[im1,j] + vx[s-1,jp1]-vx[s-1,jm1])/2.0
			jm1 = j
		divv[s-1,s-1] = (vy[0,s-1]-vy[s-2,s-1] + vx[s-1,0]-vx[s-1,s-2])/2.0

	return divergence_2d

def generate_optimized_jacobian(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('f8(f8,f8,f8,f8)')
	def det_2d(a11,a21,a12,a22):
		return a12*a21-a11*a22 

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def jacobian_2d(xphi,yphi,jac):
		for i in range(s-1):
			for j in range(s-1):
				jac[i,j] = det_2d(xphi[i+1,j]-xphi[i,j],yphi[i+1,j]-yphi[i,j],\
								xphi[i,j+1]-xphi[i,j],yphi[i,j+1]-yphi[i,j])
			jac[i,s-1] = det_2d(xphi[i+1,s-1]-xphi[i,s-1],yphi[i+1,s-1]-yphi[i,s-1],\
							xphi[i,0]+s-xphi[i,s-1],yphi[i,0]+s-yphi[i,s-1])
		for j in range(s-1):
			jac[s-1,j] = det_2d(xphi[0,j]+s-xphi[s-1,j],yphi[0,j]+s-yphi[s-1,j],\
							xphi[s-1,j+1]-xphi[s-1,j],yphi[s-1,j+1]-yphi[s-1,j])
		jac[s-1,s-1] = det_2d(xphi[0,s-1]-xphi[s-1,s-1],yphi[0,s-1]+s-yphi[s-1,s-1],\
						xphi[s-1,0]+s-xphi[s-1,s-1],yphi[s-1,0]-yphi[s-1,s-1])

	return jacobian_2d


def generate_optimized_density_match_L2_gradient(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8,f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
	def density_match_L2_gradient_2d(sigma, sqrtJ, dsqrtJdx, dsqrtJdy, W, dWdx, dWdy, W0, dW0dx, dW0dy, doutdx, doutdy):
		for i in range(s):
			for j in range(s):
				doutdx[i,j] = sigma*dsqrtJdx[i,j] + W0[i,j]*dWdx[i,j] - W[i,j]*dW0dx[i,j]
				doutdy[i,j] = sigma*dsqrtJdy[i,j] + W0[i,j]*dWdy[i,j] - W[i,j]*dW0dy[i,j]

	return density_match_L2_gradient_2d


class CompatibleConformalDensityMatching(object):
	"""
	Implementation of the comptible matching algorithm using the conformal metric.

	Objects from this class are efficient implementations of the diffeomorphic
	density matching algorithm detailed in the paper by Bauer, Joshi, and Modin (2015).
	The computations are accelerated using the `numba` library.
	"""

	def __init__(self, source, target, compute_phi=True):
		"""
		Initialize the matching process.

		Implements to algorithm in the paper by Bauer, Joshi, and Modin 2015.

		Parameters
		----------
		source : array_like
			Numpy array (float64) for the source image.
		target : array_like
			Numpy array (float64) for the target image.
			Must be of the same shape as `source`.
		alpha : float
			Parameter specifying the midpoint to be used for the conformal metric.
		compute_phi : bool
			Whether to compute the forward phi mapping or not.

		Returns
		-------
		None or Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""	
		
		self.source = source
		self.target = target
		self.compute_phi = compute_phi
		I1 = target
		I0 = source

		# Check input
		if (I0.shape != I1.shape):
			raise(TypeError('Source and target images must have the same shape.'))
		if (I0.dtype != I1.dtype):
			raise(TypeError('Source and target images must have the same dtype.'))
		for d in I1.shape:
			if (d != I1.shape[0]):
				raise(NotImplementedError('Only square images allowed so far.'))
		if (len(I1.shape) != 2):
			raise(NotImplementedError('Only 2d images allowed so far.'))

		# Create optimized algorithm functions
		self.image_compose = generate_optimized_image_composition(I1)
		self.diffeo_compose = generate_optimized_diffeo_composition(I1)
		self.image_gradient = generate_optimized_image_gradient(I1)
		self.divergence = generate_optimized_divergence(I1)
		# self.density_match_L2_gradient = generate_optimized_density_match_L2_gradient(I1)
		self.evaluate = generate_optimized_diffeo_evaluation(I1)

		# Allocate and initialize variables
		self.E = []
		self.s = I1.shape[0]
		self.E = []
		self.I0 = I0
		self.I1 = I1
		self.I = np.ones_like(I1)
		self.f = np.zeros_like(I1)
		self.mu = np.ones_like(I1)
		self.mudot = np.zeros_like(I1)
		# self.W0 = np.sqrt(I0)
		# self.dW0dx = np.zeros_like(I1)
		# self.dW0dy = np.zeros_like(I1)
		# self.W1 = np.sqrt(I1)
		# self.W1 *= np.linalg.norm(self.W0,'fro')/np.linalg.norm(self.W1,'fro')
		# self.W = self.W1.copy()
		# self.dWdx = np.zeros_like(I1)
		# self.dWdy = np.zeros_like(I1)
		self.vx = np.zeros_like(I1)
		self.vy = np.zeros_like(I1)
				
		# Allocate and initialize the diffeos
		x = np.linspace(0, self.s, self.s, endpoint=False)
		[self.idx, self.idy] = np.meshgrid(x, x)
		self.phiinvx = self.idx.copy()
		self.phiinvy = self.idy.copy()
		self.psiinvx = self.idx.copy()
		self.psiinvy = self.idy.copy()
		if self.compute_phi:
			self.phix = self.idx.copy()
			self.phiy = self.idy.copy()	
			self.psix = self.idx.copy()
			self.psiy = self.idy.copy()	
		self.tmpx = self.idx.copy()
		self.tmpy = self.idy.copy()
		self.conformal_met = np.ones_like(I1)
		self.conformal_met /= self.conformal_met.sum()


		# Compute the Laplace and inverse Laplace operator
		self.L = 4. - 2.*(np.cos(2.*np.pi*self.idx/self.s) + np.cos(2.*np.pi*self.idy/self.s))
		self.L[0,0] = 1.
		self.Linv = 1./self.L
		self.L[0,0] = 0.
		self.Linv[0,0] = 1.

	def run(self, epsilon=0.1, alpha=0.5, use_conformal=True):
		"""
		Carry out the matching process.

		Implements to algorithm in the paper by Bauer, Joshi, and Modin 2015.

		Parameters
		----------
		epsilon : float
			The stepsize in the interval [0,1].
		yielditer : bool
			If `True`, then a yield statement is executed at the start of
			each iterations. This is useful for example when animating 
			the warp in real-time.

		Returns
		-------
		None or Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""		

		# Initial computations
		if not np.allclose(self.I0.sum(),self.I1.sum()):
			raise(ValueError('I0 and I1 must have the same total volume.'))
		if (alpha < 0) or (alpha > 1):
			raise(ValueError('Parameter alpha must be between 0 and 1.'))


		# Compute volume and angle
		self.use_conformal = use_conformal
		self.W0 = np.sqrt(self.I0)
		self.W1 = np.sqrt(self.I1)
		self.totvol = self.I0.sum();
		self.theta = np.arccos((self.W0*self.W1).sum()/self.totvol)
		if use_conformal:
			self.conformal_met = (1.-alpha)*self.I0 + alpha*self.I1

		tvec = np.arange(0,1,epsilon) + alpha*epsilon
		for t in tvec:
			
			# OUTPUT
			# np.copyto(self.tmpx, self.sqrtJ)
			# self.tmpx -= 1.
			# self.tmpx *= self.sigma
			# self.tmpx **= 2
			# self.E.append(self.tmpx.sum())
			# np.copyto(self.tmpx, self.W)
			# self.tmpx -= self.W0
			# self.tmpx **= 2
			# self.E[-1] += self.tmpx.sum()

			# ALGORITHM

			# Calculate geodesic
			# self.W = np.sin((1.-t)*self.theta)/np.sin(self.theta)*self.W0 + (np.sin(t*self.theta)/np.sin(self.theta))*self.W1
			# self.mu = self.W**2

			# Specify point on path and its derivative
			self.mu = (1-t)*self.I0 + t*self.I1
			self.mudot = self.I1 - self.I0

			# Compute Ik in STEP 2 of algorithm
			np.copyto(self.tmpx, self.mudot)
			self.tmpx /= self.mu
			self.image_compose(self.tmpx, self.phiinvx, self.phiinvy, self.I)

			# Solve the Poisson equation in STEP 2
			if use_conformal:
				self.I *= self.conformal_met
			Ihat = np.fft.fftn(self.I)
			Ihat *= self.Linv
			self.f[:] = np.fft.ifftn(Ihat).real

			# Compute the gradient of STEP 3
			self.image_gradient(self.f, self.vx, self.vy)
			if use_conformal:
				self.vx /= self.conformal_met
				self.vy /= self.conformal_met

			# Compute the incremental diffeos according to STEP 4
			np.copyto(self.tmpx, self.vx)
			self.tmpx *= epsilon
			np.copyto(self.psiinvx, self.idx)
			self.psiinvx -= self.tmpx
			if self.compute_phi: # Compute forward psi also (only for output purposes)
				np.copyto(self.psix, self.idx)
				self.psix += self.tmpx

			np.copyto(self.tmpy, self.vy)
			self.tmpy *= epsilon
			np.copyto(self.psiinvy, self.idy)
			self.psiinvy -= self.tmpy
			if self.compute_phi: # Compute forward psi also (only for output purposes)
				np.copyto(self.psiy, self.idy)
				self.psiy += self.tmpy

			# Update the diffeos according to STEP 5
			self.diffeo_compose(self.phiinvx, self.phiinvy, self.psiinvx, self.psiinvy, \
								self.tmpx, self.tmpy)
			np.copyto(self.phiinvx, self.tmpx)
			np.copyto(self.phiinvy, self.tmpy)
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				self.diffeo_compose(self.psix, self.psiy, self.phix, self.phiy, \
									self.tmpx, self.tmpy)
				np.copyto(self.phix, self.tmpx)
				np.copyto(self.phiy, self.tmpy)


			



class TwoComponentDensityMatching(object):
	"""
	Implementation of the two component density matching algorithm.

	Objects from this class are efficient implementations of the diffeomorphic
	density matching algorithm detailed in the paper by Bauer, Joshi, and Modin (2015).
	The computations are accelerated using the `numba` library.
	"""

	def __init__(self, source, target, sigma=0.5, compute_phi=True):
		"""
		Initialize the matching process.

		Implements to algorithm in the paper by Bauer, Joshi, and Modin 2015.

		Parameters
		----------
		source : array_like
			Numpy array (float64) for the source image.
		target : array_like
			Numpy array (float64) for the target image.
			Must be of the same shape as `source`.
		sigma : float
			Parameter for penalizing change of volume (divergence).
		compute_phi : bool
			Whether to compute the forward phi mapping or not.

		Returns
		-------
		None or Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""	
		
		self.source = source
		self.target = target
		self.compute_phi = compute_phi
		I0 = target
		I1 = source

		# Check input
		if (I0.shape != I1.shape):
			raise(TypeError('Source and target images must have the same shape.'))
		if (I0.dtype != I1.dtype):
			raise(TypeError('Source and target images must have the same dtype.'))
		if (sigma < 0):
			raise(ValueError('Paramter sigma must be positive.'))
		for d in I1.shape:
			if (d != I1.shape[0]):
				raise(NotImplementedError('Only square images allowed so far.'))
		if (len(I1.shape) != 2):
			raise(NotImplementedError('Only 2d images allowed so far.'))

		# Create optimized algorithm functions
		self.image_compose = generate_optimized_image_composition(I1)
		self.diffeo_compose = generate_optimized_diffeo_composition(I1)
		self.image_gradient = generate_optimized_image_gradient(I1)
		self.divergence = generate_optimized_divergence(I1)
		self.density_match_L2_gradient = generate_optimized_density_match_L2_gradient(I1)
		self.evaluate = generate_optimized_diffeo_evaluation(I1)

		# Allocate and initialize variables
		self.sigma = sigma
		self.s = I1.shape[0]
		self.E = []
		self.I0 = I0
		self.I1 = I1
		self.J = np.ones_like(I1)
		self.sqrtJ = np.ones_like(I1)
		self.dsqrtJdx = np.zeros_like(I1)
		self.dsqrtJdy = np.zeros_like(I1)
		self.W0 = np.sqrt(I0)
		self.dW0dx = np.zeros_like(I1)
		self.dW0dy = np.zeros_like(I1)
		self.W1 = np.sqrt(I1)
		self.W1 *= np.linalg.norm(self.W0,'fro')/np.linalg.norm(self.W1,'fro')
		self.W = self.W1.copy()
		self.dWdx = np.zeros_like(I1)
		self.dWdy = np.zeros_like(I1)
		self.image_gradient(self.W0, self.dW0dx, self.dW0dy)
		self.vx = np.zeros_like(I1)
		self.vy = np.zeros_like(I1)
		self.divv = np.zeros_like(I1)
				
		# Allocate and initialize the diffeos
		x = np.linspace(0, self.s, self.s, endpoint=False)
		[self.idx, self.idy] = np.meshgrid(x, x)
		self.phiinvx = self.idx.copy()
		self.phiinvy = self.idy.copy()
		self.psiinvx = self.idx.copy()
		self.psiinvy = self.idy.copy()
		if self.compute_phi:
			self.phix = self.idx.copy()
			self.phiy = self.idy.copy()	
			self.psix = self.idx.copy()
			self.psiy = self.idy.copy()	
		self.tmpx = self.idx.copy()
		self.tmpy = self.idy.copy()

		# Compute the Laplace and inverse Laplace operator
		self.L = 4. - 2.*(np.cos(2.*np.pi*self.idx/self.s) + np.cos(2.*np.pi*self.idy/self.s))
		self.L[0,0] = 1.
		self.Linv = 1./self.L
		self.L[0,0] = 0.
		self.Linv[0,0] = 1.

		
	def run(self, niter=300, epsilon=0.1, yielditer=False):
		"""
		Carry out the matching process.

		Implements to algorithm in the paper by Bauer, Joshi, and Modin 2015.

		Parameters
		----------
		niter : int
			Number of iterations to take.
		epsilon : float
			The stepsize in the gradient descent method.
		yielditer : bool
			If `True`, then a yield statement is executed at the start of
			each iterations. This is useful for example when animating 
			the warp in real-time.

		Returns
		-------
		None or Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""		

		kE = len(self.E)
		self.E = np.hstack((self.E,np.zeros(niter)))
		
		for k in range(niter):
			
			# OUTPUT
			np.copyto(self.tmpx, self.sqrtJ)
			self.tmpx -= 1.
			self.tmpx *= self.sigma
			self.tmpx **= 2
			self.E[k+kE] = self.tmpx.sum()
			np.copyto(self.tmpx, self.W)
			self.tmpx -= self.W0
			self.tmpx **= 2
			self.E[k+kE] += self.tmpx.sum()

			
			# STEP 1
			self.image_compose(self.W1, self.phiinvx, self.phiinvy, self.W)
			self.W *= self.sqrtJ
			
			# STEP 2
			self.image_gradient(self.sqrtJ, self.dsqrtJdx, self.dsqrtJdy)
			self.image_gradient(self.W, self.dWdx, self.dWdy)
			self.density_match_L2_gradient(self.sigma, self.sqrtJ, self.dsqrtJdx, self.dsqrtJdy, \
										   self.W, self.dWdx, self.dWdy, \
										   self.W0, self.dW0dx, self.dW0dy, self.vx, self.vy)
			
			# STEP 3
#             print('Mean v before: [%f,%f]'%(np.mean(self.vx),np.mean(self.vy)))
			fftx = np.fft.fftn(self.vx)
			ffty = np.fft.fftn(self.vy)
			fftx *= self.Linv
			ffty *= self.Linv
			self.vx[:] = -np.fft.ifftn(fftx).real
			self.vy[:] = -np.fft.ifftn(ffty).real
#             print('Mean v after: [%f,%f]'%(-np.mean(self.vx),-np.mean(self.vy)))
			
			# STEP 4 (v = -grad E, so to compute the inverse we solve \psiinv' = -epsilon*v o \psiinv)
			np.copyto(self.tmpx, self.vx)
			self.tmpx *= epsilon
			np.copyto(self.psiinvx, self.idx)
			self.psiinvx -= self.tmpx
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				np.copyto(self.psix, self.idx)
				self.psix += self.tmpx

			np.copyto(self.tmpy, self.vy)
			self.tmpy *= epsilon
			np.copyto(self.psiinvy, self.idy)
			self.psiinvy -= self.tmpy
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				np.copyto(self.psiy, self.idy)
				self.psiy += self.tmpy

			
			# STEP 5
			self.diffeo_compose(self.phiinvx, self.phiinvy, self.psiinvx, self.psiinvy, \
								self.tmpx, self.tmpy)
			np.copyto(self.phiinvx, self.tmpx)
			np.copyto(self.phiinvy, self.tmpy)
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				self.diffeo_compose(self.phix, self.phiy, self.psix, self.psiy, \
									self.tmpx, self.tmpy)
				np.copyto(self.phix, self.tmpx)
				np.copyto(self.phiy, self.tmpy)

			
			# STEP 6
			self.image_compose(self.J, self.psiinvx, self.psiinvy, self.sqrtJ)
			np.copyto(self.J, self.sqrtJ)
			self.divergence(self.vx, self.vy, self.divv)
			self.divv *= -epsilon
			np.exp(self.divv, out=self.sqrtJ)
			self.J *= self.sqrtJ
			np.sqrt(self.J, out=self.sqrtJ)


if __name__ == '__main__':
	pass
