import numpy as np
import getfem as gf
import os
import sys

Path = os.getcwd()

def Gradient_comput(mf, Dm, u, rank):
	DU = gf.compute_gradient(mf, u, Dm)
	Dm.export_to_pos(Path + '/DU' + str(rank) + '.pos',Dm ,DU, 'DUF')
	Dm.export_to_vtk(Path + '/DU' + str(rank) + '.vtk', 'ascii', DU, 'DUF')
	return DU

def Complementary_solution(Mesh, uG, Gmf, Dim, DmG, DUG):
	Cm = gf.Mesh('import', 'gmsh', Mesh)
	Cmf = gf.MeshFem(Cm, 1)
	Dmc = gf.MeshFem(Cm, 1);
	if Dim == 2:
		Cmf.set_fem(gf.Fem('FEM_PK(2,1)'))
		Dmc.set_fem(gf.Fem('FEM_PK(2,0)'));
		Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))
	if Dim == 3:
		Cmf.set_fem(gf.Fem('FEM_PK(3,2)'))
		Dmc.set_fem(gf.Fem('FEM_PK(3,1)'));
		Cmim = gf.MeshIm(Cm, gf.Integ('IM_TETRAHEDRON(5)'))
	Cmd = gf.Model('real')
	uC = gf.compute_interpolate_on(Gmf, uG, Cmf)
	DUC = gf.compute_interpolate_on(DmG, DUG, Dmc)
	return uC, Cmf, DUC, Dmc

def Solution_export(mf, u, rank):
	mf.export_to_pos(Path + '/u' + str(rank) + '.pos', u, 'U_GL')
	mf.export_to_vtk(Path + '/u' + str(rank) + '.vtk', 'ascii', u, 'U_GL')
	return 0


