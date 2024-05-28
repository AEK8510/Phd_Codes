"""The purpose of this module is to provide functions to generate a finite element model
with getfem"""



import numpy as np
import getfem as gf
import random
#random.seed(42) 
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
	#E = 1.e6
	E = 934556
	Nu = 0.4
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	#print(Mu)
	#print(Lambda)
	md.add_initialized_data('cmu', Mu)
	md.add_initialized_data('clambda', Lambda)
	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')
	nbd = mf.nbdof()
	return(m, mf, md, mim, nbd, Dm)



def FE_Local_Model(mesh,BORD, j, interface_id):
	#random.seed(42)
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh);
	# Create a MeshFem  for field of dimension 1
	Dm = gf.MeshFem(m, 1);
	# # assign the P2 fem to all convexs of the MeshFEM
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
	E = 1e6
	#random.seed(j)
	#E1 = random.uniform(1, 1e+4)
	#print(E1,"----",j)
	E1 = 1e2
	Nu = 0.4
	Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu = E / (2 * (1 + Nu))
	Lambda1 = E1 * Nu / ((1 + Nu) * (1 - 2 * Nu))
	Mu1 = E1 / (2 * (1 + Nu))
	#print(Mu1)
	#print(Lambda1)
	md.add_initialized_data('cmu', Mu)
	md.add_initialized_data('clambda', Lambda)
	md.add_initialized_data('cmu1', Mu1)
	md.add_initialized_data('clambda1', Lambda1)
	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu', region = m.regions()[0])
	md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda1', 'cmu1', region=m.regions()[1])
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



