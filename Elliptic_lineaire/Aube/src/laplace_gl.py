import numpy as np
import getfem as gf
import os
import sys
def FE_Model_Global(mesh, Dim):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh)
	# Create a MeshFem of for field of dimension 1
	mf = gf.MeshFem(m, 1)
	Dm = gf.MeshFem(m, 1);
	if Dim == 2:
		mf.set_fem(gf.Fem('FEM_PK(2,1)'))
		Dm.set_fem(gf.Fem('FEM_PK(2,0)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(4)'))
	if Dim == 3:
		mf.set_fem(gf.Fem('FEM_PK(3,2)'))
		Dm.set_fem(gf.Fem('FEM_PK(3,1)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	md = gf.Model('real')
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf)
	# Bilinear local Form
	#md.add_Laplacian_brick(mim, 'u')
	md.add_linear_term(mim, "1*Grad_u.Grad_Test_u ", -1, True, True)
	# Preparation of coupling
	nbd = mf.nbdof()
#	Fnbd = Fmf.nbdof()
	return(m, mf,  md,  mim,  nbd, Dm)

def FE_Model_local(mesh, Dim):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh)
	# Create a MeshFem of for field of dimension 1
	mf = gf.MeshFem(m, 1)
	Dm = gf.MeshFem(m, 1);
	if Dim == 2:
		mf.set_fem(gf.Fem('FEM_PK(2,1)'))
		Dm.set_fem(gf.Fem('FEM_PK(2,0)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(4)'))
	if Dim == 3:
		mf.set_fem(gf.Fem('FEM_PK(3,2)'))
		Dm.set_fem(gf.Fem('FEM_PK(3,1)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	# Create an empty real model
	md = gf.Model('real')
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf)
	# Bilinear local Form
	#md.add_Laplacian_brick(mim, 'u')
	md.add_linear_term(mim, "1*Grad_u.Grad_Test_u", -1, True, True);
	#md.add_nonlinear_term(md, mim, "sin(X(1) + X(2)) * Grad_u.Grad_Test_u - my_f * Test_u", -1)
	# Preparation of coupling
	nbd = mf.nbdof()
#	Fnbd = Fmf.nbdof()
	return(m, mf,  md,  mim,  nbd, Dm)








