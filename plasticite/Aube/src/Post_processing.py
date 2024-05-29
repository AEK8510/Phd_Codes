import numpy as np
import getfem as gf
import os
import sys


Path = os.getcwd()

def Gradient_comput(mf, Dm, u, rank):
	DU = gf.compute_gradient(mf, u, Dm)
	Dm.export_to_pos(Path + '/DU' + str(rank) + '.pos',Dm ,DU, 'DUF')
	Dm.export_to_vtk(Path + '/DU' + str(rank) + '.vtk', 'ascii', DU, 'DUF')
	return 0

def VM_comput_P(m, md, rank, Dim, mim):
	if Dim == 2:
		GMV = gf.MeshFem(m, 1)
		GMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(2,1)'))
	if Dim == 3:
		GMV = gf.MeshFem(m, 1)
		GMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))
	VM = md.small_strain_elastoplasticity_Von_Mises (mim, GMV, 'Prandtl Reuss linear hardening', 'displacement only', 'u', 'xi', 'Previous_Ep', 'previous_alpha',
										'lambda', 'mu', 'sigma_y', 'H_k', 'H_i')
	#VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u', 'clambda', 'cmu', GMV);
	print(VM.max(),"--------------------", rank)
	sl = gf.Slice(('none',), m, 1)
	sl.export_to_pos(Path + '/VM' +str(rank) + '.pos', GMV, VM, 'Von Mises Stress')
	sl.export_to_vtk(Path + '/VM' +str(rank) + '.vtk', 'ascii', GMV, VM, 'Von Mises Stress')
	return 0


def VM_comput_G(m, md, rank, Dim):
	if Dim == 2:
		GMV = gf.MeshFem(m, 1)
		GMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(2,1)'))
	if Dim == 3:
		GMV = gf.MeshFem(m, 1)
		GMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))
	VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u', 'clambda', 'cmu', GMV);
	print(VM.max(), "------------", rank)
	sl = gf.Slice(('none',), m, 1)
	sl.export_to_pos(Path + '/VM' +str(rank) + '.pos', GMV, VM, 'Von Mises Stress')
	sl.export_to_vtk(Path + '/VM' +str(rank) + '.vtk', 'ascii', GMV, VM, 'Von Mises Stress')
	return VM, GMV

def Complementary_solution(Mesh, uG, Gmf, Dim, VMG, GMV):
	Cm = gf.Mesh('import', 'gmsh', Mesh)
	if Dim == 2:
		Cmf = gf.MeshFem(Cm, 2)
		Cmf.set_fem(gf.Fem('FEM_PK(2,1)'))
		Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))
	if Dim == 3:
		Cmf = gf.MeshFem(Cm, 3)
		Cmf.set_fem(gf.Fem('FEM_PK(3,1)'))
		Cmim = gf.MeshIm(Cm, gf.Integ('IM_TETRAHEDRON(5)'))
	#Cmf.set_fem(gf.Fem('FEM_PK(2,2)'))
	#Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))
	Cmd = gf.Model('real')
	uC = gf.compute_interpolate_on(Gmf, uG, Cmf)
	
	if Dim == 2:
	 	CMV = gf.MeshFem(Cm, 1)
	 	CMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(2,1)'))
	if Dim == 3:
	 	CMV = gf.MeshFem(Cm, 1)
	 	CMV.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))
	VMC = gf.compute_interpolate_on(GMV, VMG, CMV)
	print(VMC.max(), "---------", "Complementaire VM")
	sl = gf.Slice(('none',), Cm, 1)
	sl.export_to_pos(Path + '/VM' +str(5) + '.pos', CMV, VMC, 'Von Mises Stress')
	sl.export_to_vtk(Path + '/VM' +str(5) + '.vtk', 'ascii', CMV, VMC, 'Von Mises Stress')
	return uC, Cmf, Cmd, Cm

def Solution_export(mf, u, rank):
	mf.export_to_pos(Path + '/u' + str(rank) + '.pos', u, 'U_GL')
	mf.export_to_vtk(Path + '/u' + str(rank) + '.vtk', 'ascii', u, 'U_GL')
	#mf.export_to_vtk(Path +'/alpha' +  str(rank)  +'.vtk',  'ascii', alpha , 'Previous_alpha')
	return 0


