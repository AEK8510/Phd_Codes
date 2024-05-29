import numpy as np
import getfem as gf
import os
import sys
#os.chdir('./results/cube/')
Path = os.getcwd()

def Gradient_comput(mf, Dm, u, size, ind, DU):
	for i in range(0, size):
		DU[i] = gf.compute_gradient(mf[i], u[i], Dm[i])
		Dm[i].export_to_pos(Path + '/DU' + str(ind[i]) + '.pos',Dm[i] ,DU[i], 'DU_F' + str(ind[i]))
		Dm[i].export_to_vtk(Path + '/DU' + str(ind[i]) + '.vtk', 'ascii', DU[i], 'DUF')
	return 0



def Solution_export(mf, u, size, ind):
	for i in range(0, size):
		mf[i].export_to_pos(Path + '/u' + str(ind[i]) + '.pos', u[i], 'U_F'+str(ind[i]))
		mf[i].export_to_vtk(Path + '/u' + str(ind[i]) + '.vtk', 'ascii', u[i], 'U_GL')
	return 0

def VM_comput_P(m, md, ind, mim, size):
	GMV = {}
	VM = {}
	for i in range(0, size):
		GMV[i] = gf.MeshFem(m[i])
		GMV[i].set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))
		#md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu', region=m.regions()[0])
		VM[i] = md[i].compute_isotropic_linearized_Von_Mises_or_Tresca('u', 'lambda', 'mu', GMV[i]);
		VM[i] = md[i].small_strain_elastoplasticity_Von_Mises(mim[i], GMV[i], 'Prandtl Reuss linear hardening',
		                                    'displacement only', 'u', 'xi', 'Previous_Ep', 'Previous_alpha', 'lambda', 'mu', 'sigma_y','H_k', 'H_i', m[i].regions()[1])
		#VM[i] = md[i].compute_finite_strain_elastoplasticity_Von_Mises(mim[i], GMV[i], 'Prandtl Reuss linear hardening', 'displacement only', 'u'
		#                                                       , 'xi', 'Previous_Ep', 'lambda', 'mu', 'sigma_y','H_k', 'H_i')
		#VM[i] = md[i].compute_elastoplasticity_Von_Mises_or_Tresca('u', 'sigma_y', GMV[i], version=None)
		#if VM[i].max() > 3464:
		print(VM[i].max(), "----------------", ind[i])
		sl = gf.Slice(('none',), m[i], 1)
		sl.export_to_pos(Path + '/VM' +str(ind[i]) + '.pos', GMV[i], VM[i], 'Von Mises Stress')
		sl.export_to_vtk(Path + '/VM' +str(ind[i]) + '.vtk', 'ascii', GMV[i], VM[i], 'Von Mises Stress')
	return 0

def VM_comput_G(m, md, rank):
	GMV = gf.MeshFem(m, 1)
	GMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
	VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u', 'clambda', 'cmu', GMV);
	print(VM.max(), "-----------", "Von mises Glbal")
	sl = gf.Slice(('none',), m, 1)
	sl.export_to_pos(Path + '/VM_G.pos', GMV, VM, 'Von Mises Stress')
	sl.export_to_vtk(Path + '/VM_G.vtk', 'ascii', GMV, VM, 'Von Mises Stress')
	return 0
