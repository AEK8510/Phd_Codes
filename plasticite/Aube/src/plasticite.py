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
		mf.set_fem(gf.Fem('FEM_PK(2,2)'))
		Dm.set_fem(gf.Fem('FEM_PK(2,1)'));
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
	E = 1e+5
	Nu = 0.3
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	#Lambda = 121150
	#Mu = 80769
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
		mim = gf.MeshIm(m, gf.Integ('IM_TRIANGLE(4)'))
		mim_data = gf.MeshImData(mim, -1, [2, 2])
		mim_data2 = gf.MeshImData(mim, -1, [1])
		mfu = gf.MeshFem(m, 2)
		mfxi = gf.MeshFem(m, 1)
		mfd = gf.MeshFem(m)
		mfdu = gf.MeshFem(m)
		mfu.set_fem(gf.Fem('FEM_PK(2,2)'))
		mfxi.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(2,2)'))
		mfd.set_fem(gf.Fem('FEM_PK(2,2)'))
		mfdu.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(2,1)'))
	if Dim == 3:
		mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
		mim_data = gf.MeshImData(mim, -1, [3,3])
		mim_data2 = gf.MeshImData(mim, -1, [1])
		mfu = gf.MeshFem(m, 3)
		mfxi = gf.MeshFem(m, 1)
		mfd = gf.MeshFem(m)
		mfdu = gf.MeshFem(m)
		mfu.set_fem(gf.Fem('FEM_PK(3,2)'))
		mfxi.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,2)'))
		mfd.set_fem(gf.Fem('FEM_PK(3,2)'))
		mfdu.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
	E = 1e+5
	Nu = 0.3
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	#Lambda = 121150
	#Mu = 80769
	#sigma_y = 4000
	sigma_y = 4000
	Hk = 16.
	Hi = 20.
	md = gf.Model('real')
	md.add_fem_variable('u', mfu)
	md.add_fem_data('Previous_u', mfu)
	md.add_im_data('Previous_Ep', mim_data)
	md.add_im_data('Previous_alpha', mim_data2)
	md.add_fem_data('xi', mfxi)
	md.add_fem_data('Previous_xi', mfxi)
	md.add_initialized_data('lambda', Lambda)
	md.add_initialized_data('mu', Mu)
	md.add_initialized_data('H_k', Hk)
	md.add_initialized_data('H_i', Hi)
	md.add_initialized_data('sigma_y', sigma_y)
	md.add_small_strain_elastoplasticity_brick(mim, 'Prandtl Reuss linear hardening', 0, 'u', 'xi',
	 				'Previous_Ep', 'Previous_alpha', 'lambda', 'mu', 'sigma_y', 'H_k', 'H_i', '1', 'timestep')
	#md.add_small_strain_elastoplasticity_brick(mim, 'Prandtl Reuss', 0, 'u', 'xi', 'Previous_Ep', 'lambda', 'mu',
	#                                           'sigma_y', '1', 'timestep')
	nbd = mfu.nbdof()
	return(m, mfu,  md,  mim,  nbd, mfdu)





