#!/usr/bin/env python
# encoding: utf-8
"""
Display classes for the `ddmatch` library.

Documentation guidelines are available `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Created by Klas Modin on 2014-11-03.
"""

import numpy as np 
import matplotlib.pyplot as plt

class MatchPlot(object):
	"""
	Presentation of a matching process computed using any of the `ddmatch`
	core classes.

	The visualization depends on the `matplotlib` library.
	"""
	def __init__(self, axes = None):
		super(MatchPlot, self).__init__()
		if not axes:
			axes = plt.figure().gca()
		self.axes = axes

	def plot_density(self, I, *arg, **kwarg):
		if 'cmap' not in kwarg:
			kwarg['cmap'] = 'bone'
		if 'vmin' not in kwarg:
			kwarg['vmin'] = I.min()
		if 'vmax' not in kwarg:
			kwarg['vmax'] = I.max()

		self.dens = self.axes.imshow(I, *arg, **kwarg)
		return self

	def update_density(self, I):
		self.dens.set_data(I)

	def plot_warp(self, phix, phiy, color='lightskyblue', downsample='auto', **kwarg):
		if (downsample == 'auto'):
			self.skip = int(np.max([phix.shape[0]/32,1]))
		elif (downsample == 'no'):
			self.skip = 1
		else:
			self.skip = downsample
		if 'linewidth' not in kwarg:
			kwarg['linewidth'] = 1

		skip = self.skip
		self.vert_lines=self.axes.plot(phix[:,skip::skip],phiy[:,skip::skip],color,**kwarg)
		self.hor_lines=self.axes.plot(phix[skip::skip,:].T,phiy[skip::skip,:].T,color,**kwarg)
		return self

	def update_warp(self, phix, phiy):
		skip = self.skip
		for (k,line) in zip(range(len(self.vert_lines)),self.vert_lines):
			line.set_xdata(phix[:,skip::skip][:,k])
			line.set_ydata(phiy[:,skip::skip][:,k])
		for (k,line) in zip(range(len(self.hor_lines)),self.hor_lines):
			line.set_xdata(phix[skip::skip,:].T[:,k])
			line.set_ydata(phiy[skip::skip,:].T[:,k])

