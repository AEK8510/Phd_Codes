


import numpy as np
import getfem as gf


import os
import sys
#

def load_volumic_data(mf, mim, md, value):
	f = mf.eval(value)
	md.add_initialized_fem_data('VolumicData', mf, f)
	md.add_source_term_brick(mim, 'u', 'VolumicData')
	return md

## Global Dirichlet boundary
def Global_dirichlet_boundary(m, mf, md, mim,  value):
	BOTTOM = 4
	# border = m.outer_faces()
	# fnor = m.normal_of_faces(border)
	# #print(fnor)
	# bbottom = abs(fnor[1, :]+1) < 1e-14
	# fdown = np.compress(bbottom, border, axis=1)
	# m.set_region(BOTTOM, fdown)
	g = mf.eval(value)
	md.add_initialized_fem_data('DirichletData', mf, g)
	md.add_Dirichlet_condition_with_multipliers(
		mim, 'u', mf, BOTTOM, 'DirichletData')
	return (m, md, mf, mim)

