import numpy as np
import getfem as gf
import os
import sys


def global_auxiliary_renumbering(Gmf, region_interface, BORD, j):
	b = [[]]*(len(region_interface))
	for i in range(0,len(region_interface)):
		b[i] = Gmf.basic_dof_on_region(region_interface[i])[:]
	Gbound_bis = [];
	Gbound_bis = np.unique(np.concatenate((b), ));
	if BORD:
		id = 1000000 + j
		a = Gmf.basic_dof_on_region(id);
		bbord = np.intersect1d(Gbound_bis, a);
		inter = np.argwhere(np.in1d(Gbound_bis, a));
		Gbound_bis = np.delete(Gbound_bis, inter, 0);
	return (Gbound_bis)

def Global_to_local_coupling(Fmim, Fmf, Gmf, region_interface, BORD, bbord_fin, j, interface_id):
	# Auxiliary and Fine regions after deleting the border elemnents
	FM = gf.asm_mass_matrix(Fmim, Fmf, Fmf, interface_id)[
		Fmf.basic_dof_on_region(interface_id), Fmf.basic_dof_on_region(interface_id)];
	if BORD:
		new_region_100_fin = np.delete(Fmf.basic_dof_on_region(interface_id), bbord_fin, 0);
		FM = np.delete(FM, bbord_fin, 1)
		FM = np.delete(FM, bbord_fin, 0)
	else :
		new_region_100_fin = Fmf.basic_dof_on_region(interface_id);
	# Global to fine renumbering
	(Gbound) = global_auxiliary_renumbering(Gmf, region_interface, BORD, j);
	# Interpolation matrix between the global and the fine on the interfaces
	Interpo = gf.asm_interpolation_matrix(Gmf, Fmf)[new_region_100_fin[ :, np.newaxis], Gbound];
	return (Interpo, new_region_100_fin, Gbound, FM)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def Add_Lagrangian_multipliers(Fmd, Fmf, Fnbd, new_region_100_fin):
	uFd = np.zeros(Fnbd)
	Fmd.add_variable('Fmult', len(new_region_100_fin));
	C = gf.Spmat('empty', len(new_region_100_fin), Fmf.nbdof());
	C.add(np.arange(len(new_region_100_fin)), new_region_100_fin,
	      gf.Spmat('identity', len(new_region_100_fin)));
	F_index = Fmd.add_constraint_with_multipliers('u', 'Fmult', C, uFd);
	return(C,F_index,uFd)
