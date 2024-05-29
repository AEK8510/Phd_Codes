"""The purpose of this module is to provide functions to generate a finite element model
with getfem"""



import numpy as np
import getfem as gf

def FE_Global_Model(mesh):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh);
	Dm = gf.MeshFem(m, 1);
	Dm.set_fem(gf.Fem('FEM_PK(3,0)'));
	mf = gf.MeshFem(m, 3)  # displacement
	mfp = gf.MeshFem(m, 1)  # pressure
	mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	mf.set_fem(gf.Fem('FEM_PK(3,1)'))
	mfp.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
	# Create an empty real model
	md = gf.Model('real')
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf)
	# Bilinear local Form
	E = 1e4
	Nu = 0.3
	#Lambda = 121150  
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	#Mu = 80769	
	Mu = E / (2 * (1 + Nu))
	#print(Mu)
	#print(Lambda)
	md.add_initialized_data('cmu', Mu)
	md.add_initialized_data('clambda', Lambda)
	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')
	nbd = mf.nbdof()
	return(m, mf, md, mim, nbd, Dm)



def FE_Local_Model(mesh,BORD, j, interface_id):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh);
	# Create a MeshFem  for field of dimension 1

	Dm = gf.MeshFem(m, 1);
	mfxi = gf.MeshFem(m, 1)
	mfd = gf.MeshFem(m)
	mfdu = gf.MeshFem(m)
	mf = gf.MeshFem(m, 3)

	mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	mim_data = gf.MeshImData(mim, -1, [3, 3])
	mim_data2 = gf.MeshImData(mim, -1, [1])

	Dm.set_fem(gf.Fem('FEM_PK(3,0)'));
	mf.set_fem(gf.Fem('FEM_PK(3,1)'))
	mfxi.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
	mfd.set_fem(gf.Fem('FEM_PK(3,1)'))
	mfdu.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

	md = gf.Model('real')
	md.add_fem_variable('u', mf)

	E = 1e4
	Nu = 0.3
	#Lambda = 121150
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	#Mu = 80769
	sigma_y = 4000
	Hk = 16
	Hi = 20

	md.add_fem_data('Previous_u', mf)
	md.add_im_data('Previous_Ep', mim_data)
	md.add_im_data('Previous_alpha', mim_data2)
	md.add_fem_data('xi', mfxi)
	md.add_fem_data('Previous_xi', mfxi)
	md.add_initialized_data('lambda', Lambda)
	md.add_initialized_data('mu', Mu)
	md.add_initialized_data('sigma_y', sigma_y)
	md.add_initialized_data('H_k', Hk)
	md.add_initialized_data('H_i', Hi)

	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'lambda', 'mu', region=m.regions()[0])
	#md.add_small_strain_elastoplasticity_brick(mim,'Prandtl Reuss', 0, 'u', 'xi', 'Previous_Ep', 'lambda', 'mu', 'sigma_y', '1', 'timestep',m.regions()[1])
	md.add_small_strain_elastoplasticity_brick(mim, 'Prandtl Reuss linear hardening', 0, 'u', 'xi', 'Previous_Ep',
	                                           'Previous_alpha', 'lambda', 'mu', 'sigma_y', 'H_k', 'H_i', '1', 'timestep', m.regions()[1])
	#md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda1', 'cmu1', region=m.regions()[0])
	#md.add_finite_strain_elasticity_brick(mim, lawname, 'u', 'params', region=m.regions()[0]);
	nbd = mf.nbdof()

	if BORD:
		id = 1000000 + j
		a = mf.basic_dof_on_region(id);
		b = mf.basic_dof_on_region(interface_id);
		inter = np.argwhere(np.in1d(b, a));
	else:
		inter = []

	return(m, mf, md, mim, nbd, inter, Dm)



