import numpy as np
import getfem as gf
import os
import sys



def Global_Model(mesh, Dim):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh)
	# Create a MeshFem of for field of dimension 2
	if Dim == 2:
		mf = gf.MeshFem(m, 2)  # displacement
		Dm = gf.MeshFem(m, 1);
		mf.set_fem(gf.Fem('FEM_PK(2,1)'))
		Dm.set_fem(gf.Fem('FEM_PK(2,0)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(4)'))
	if Dim == 3:
		mf = gf.MeshFem(m, 3)  # displacement
		Dm = gf.MeshFem(m, 2);
		mf.set_fem(gf.Fem('FEM_PK(3,2)'))
		Dm.set_fem(gf.Fem('FEM_PK(3,1)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	# Create an empty real model
	md = gf.Model('real')
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf)
	E = 1e+8
	Nu = 0.3
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	md.add_initialized_data('cmu', Mu)
	md.add_initialized_data('clambda', Lambda)
	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')
	nbd = mf.nbdof()
	return(m, mf,  md,  mim,  nbd, Dm)

def Local_Model(mesh, Dim):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh)
	# Create a MeshFem of for field of dimension 2
	if Dim == 2:
		mf = gf.MeshFem(m, 2)  # displacement
		Dm = gf.MeshFem(m, 1);
		mf.set_fem(gf.Fem('FEM_PK(2,1)'))
		Dm.set_fem(gf.Fem('FEM_PK(2,0)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(4)'))
	if Dim == 3:
		mf = gf.MeshFem(m, 3)  # displacement
		Dm = gf.MeshFem(m, 2);
		mf.set_fem(gf.Fem('FEM_PK(3,2)'))
		Dm.set_fem(gf.Fem('FEM_PK(3,1)'));
		mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	# Create an empty real model
	md = gf.Model('real')
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf)
	E = 1e+8
	Nu = 0.3
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	md.add_initialized_data('cmu', Mu)
	md.add_initialized_data('clambda', Lambda)
	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')
	nbd = mf.nbdof()
	return(m, mf,  md,  mim,  nbd, Dm)




