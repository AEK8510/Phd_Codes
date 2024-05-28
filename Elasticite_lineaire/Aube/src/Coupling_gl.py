import numpy as np
import getfem as gf
import os
import sys

def Global_to_local_coupling(Fmf, Fmd, Fmim, Fnbd, Gmf, region_num):
	Interpo = gf.asm_interpolation_matrix(Gmf, Fmf)[Fmf.basic_dof_on_region(region_num)[
		:, np.newaxis], Gmf.basic_dof_on_region(region_num)]
	FM = gf.asm_mass_matrix(Fmim, Fmf, Fmf, region_num)[
		Fmf.basic_dof_on_region(region_num), Fmf.basic_dof_on_region(region_num)]
	uFd = np.zeros(Fnbd)
	Fmd.add_initialized_fem_data('InterDiri', Fmf, uFd)
	Fmd.add_multiplier('Fmult'+str(region_num), Fmf, 'u')
	Fmd.add_Dirichlet_condition_with_multipliers(Fmim, 'u', 'Fmult'+str(region_num), region_num, 'InterDiri')
	return(uFd, Interpo, FM)



