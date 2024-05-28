"""The purpose of this module is to provide functions to generate a finite element model
	with getfem
"""



import numpy as np
import getfem as gf

def FE_Global_Model(mesh):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh);
	# Create a MeshFem  for field of dimension 1
	mf = gf.MeshFem(m, 1);
	mp = gf.MeshFem(m, 1);
	Dm = gf.MeshFem(m, 1);
	# assign the P2 fem to all convexs of the MeshFEM
	mf.set_fem(gf.Fem('FEM_PK(3,1)'));
	mp.set_fem(gf.Fem('FEM_PK(3,0)'));
	Dm.set_fem(gf.Fem('FEM_PK(3,0)'));
	# The integration that will be used
	mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	# Create an empty real model
	md = gf.Model('real');
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf);
	# Bilinear local Form
	#md.add_Laplacian_brick(mim, 'u');
	global_rho = '1'
	kappa1 = mp.eval(global_rho)
	md.add_initialized_fem_data('coeff1', mp, kappa1)
	#md.add_generic_elliptic_brick(mim, 'u', 'coeff1', region=-1)
	md.add_linear_term(mim, 'Grad_u.Grad_Test_u ',  region=-1)
	nbd = mf.nbdof();
	return(m, mf, md, mim, nbd, Dm)



def FE_Local_Model(mesh,BORD, j, interface_id):
	# Creation of a simple cartesian mesh
	m = gf.Mesh('import', 'gmsh', mesh);
	# Create a MeshFem  for field of dimension 1
	mf = gf.MeshFem(m, 1);
	mp = gf.MeshFem(m, 1);
	Dm = gf.MeshFem(m, 1);
	# assign the P2 fem to all convexs of the MeshFEM
	mf.set_fem(gf.Fem('FEM_PK(3,1)'));
	mp.set_fem(gf.Fem('FEM_PK(3,0)'));
	Dm.set_fem(gf.Fem('FEM_PK(3,0)'));
	# The integration that will be used
	mim = gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)'));
	# Create an empty real model
	md = gf.Model('real');
	# Declare that "u" is an unknown of the system on the FEM 'Amf'
	md.add_fem_variable('u', mf);
	# Bilinear local Form
	#md.add_Laplacian_brick(mim, 'u');
	matrix_value = '1'
	sphere_value = '1e-1'
	kappa2 = mp.eval(sphere_value)
	md.add_initialized_fem_data('coeff2', mp, kappa2)
	md.add_generic_elliptic_brick(mim, 'u', 'coeff2', region=m.regions()[1])
	#md.add_linear_term(mim, 'Grad_u.Grad_Test_u ', region=m.regions()[1])
	kappa1 = mp.eval(matrix_value)
	md.add_initialized_fem_data('coeff1', mp, kappa1)
	#md.add_nonlinear_term(mim, 'coeff1 * Grad_u.Grad_Test_u + 1*u*u*u*Test_u', region=m.regions()[0])
	md.add_generic_elliptic_brick(mim, 'u', 'coeff1', region=m.regions()[0])
	nbd = mf.nbdof();
	if BORD:
		id = 1000000 + j
		a = mf.basic_dof_on_region(id);
		b = mf.basic_dof_on_region(interface_id);
		inter = np.argwhere(np.in1d(b, a));
	else:
		inter = []
	return(m, mf, md, mim, nbd, inter, Dm)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
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
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

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


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def aux_fin_comput(Fmd, Interpo, uG, Gbound, new_region_100_fin, uFd, C, F_index, FM):
	uFd[new_region_100_fin] = Interpo.dot(uG[Gbound])
	Fmd.set_private_rhs(F_index, uFd[new_region_100_fin]);
	Fmd.set_private_matrix(F_index, C);
	Fmd.solve();
	uF = Fmd.variable('u')

	return(uF,Interpo.transpose().dot(FM.dot(Fmd.variable('Fmult'))))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def Aitken_omega(rG_j_1, rG_j,ci, omega,omega_new):
	if ci >= 2:
		omega_old = np.copy(omega_new)
		deno = (np.linalg.norm(rG_j - rG_j_1)) ** 2
		rG_j_1_T = rG_j_1.T
		nume = rG_j_1_T.dot(rG_j - rG_j_1)
		frac = nume / deno
		omega_new = - omega_old * frac
	else:
		omega_new = omega
	return omega_new

def global_solve (Gmd, pG, pG_index):
	Gmd.set_private_rhs(pG_index, pG)
	Gmd.solve()
	uG = Gmd.variable('u')
	return uG


def interface_identification(nb_patchs, file):
	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
	if nb_patchs ==  64:
		file1 = open(file, 'r')
		Lines = file1.readlines()
		count = 5
		#nb_patchs = 64;
		region_interface = np.empty((nb_patchs, 0)).tolist();
		region = {}
		while(len(Lines[count]) < 24):
			temp = Lines[count][14:-2]
			#print(temp, "---------", len(temp))
			for i in range(0, len(temp)):
				if (temp[i] == '_'):
					c = i
			region_1 = int(temp[0:c])
			region_2 = int(temp[c+1:len(temp)])
			region[count - 5] = int(Lines[count][2:6])
			region_interface[(region_1) - 1].append(region[count - 5])
			region_interface[(region_2) - 1].append(region[count - 5])
			count = count + 1
		#print(region_interface)
		Bord_index = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
		interface_id = 1000
	# +++++++++++++++++++++++ Link between patches and interfaces ++++++++++++++++++++++++++++++++++#
	if nb_patchs == 8:
	#nb_patchs = 8;
		domain_faces_list = np.zeros((2 * 12), dtype=int);
		regions = (111, 112, 114, 122, 124, 131, 134, 144, 151, 152, 162, 171);
		region_interface = np.empty((nb_patchs, 0)).tolist();
		for x in regions:
			region_interface[int((x - 100) % (nb_patchs + 1)) - 1].append(x);
			region_interface[int((x - 100) // (nb_patchs + 1)) - 1].append(x);
		print(region_interface)
		Bord_index = (0, 1, 2, 3)
		interface_id = 100
	return (region_interface, interface_id, Bord_index)


# Aux and Fine Interface Dirichlet BCs
#+++++++++++++++++++++++++++ OLD VERSION OF DIRICHLET_BC_LAGRABGIEN ++++++++++++++++++++++++#
#
# def Dirichlet_BC_lagrangien(Fmf, Amf, Fmd, Amd, Fmim, Amim, Fnbd, Anbd, region_num, BORD, bbord_aux, bbord_fin):
# 	Ad = np.zeros(Anbd)
# 	#Amd.add_initialized_fem_data('InterDiri', Amf, Ad)
# 	#Amd.add_multiplier('Amult', Amf, 'u')
#
# 	if BORD:
# 		region_100_aux = Amf.basic_dof_on_region(100);
# 		new_region_100_aux = np.delete(region_100_aux, bbord_aux, 0);
# 		region_100_fin = Fmf.basic_dof_on_region(100);
# 		new_region_100_fin = np.delete(region_100_fin, bbord_fin, 0);
#
# 		# TEST = Amf.basic_dof_nodes(new_region_100)
# 		# values = np.zeros((len(new_region_100)))
# 		# Amd.add_initialized_data('cpoints', TEST)
# 		# Amd.add_initialized_data('value', values)
# 		# Amd.add_pointwise_constraints_with_multipliers('u','cpoints', 'value')
# 		# print("cpoints----------------------------", len(Amd.variable('cpoints')))
# 		# print("value----------------------------", len(Amd.variable('value')))
# 	else :
# 		new_region_100_aux = Amf.basic_dof_on_region(100);
# 		new_region_100_fin = Fmf.basic_dof_on_region(100);
# 		#Amd.add_Dirichlet_condition_with_multipliers(
# 		#	Amim, 'u', 'Amult', region_num, 'InterDiri')
#
#
#
# 	#Amd.set_private_matrix(Amd.add_constraint_with_multipliers('u', 'Amult', B, values), B)
#
#
#
# 	#Fmd.set_private_matrix(Fmd.add_constraint_with_multipliers('u', 'Fmult', C, values_fin), C)
# 	#Fmd.set_private_rhs(Fmd.add_constraint_with_multipliers('u', 'Fmult', C, values_fin), values_fin)
#
#
# 	uFd = np.zeros(Fnbd)
#
# 	#
# 	# Fmd.add_initialized_fem_data('InterDiri', Fmf, uFd)
# 	#
#
# 	#
# 	# Fmd.add_Dirichlet_condition_with_multipliers(
# 	# 	Fmim, 'u', 'Fmult', region_num, 'InterDiri')
#
# 	return(Ad, uFd, Fmd, Amd, new_region_100_aux, new_region_100_fin)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# def Dirichlet_BC_lagrangien(Fmf, Amf, Fnbd, Anbd, BORD, bbord_aux, bbord_fin):
#
#
#
# 	return(Ad, uFd, new_region_100_aux, new_region_100_fin)




	###############################################################################
#++++++++++++++++++++++++++ OLD VERSION OF AUX_FIN_COMPUT +++++++++++++++++++++++++++++++++++++++++++++++#
#
# def aux_fin_comput(Amf,Amd,Fmd,Fmf,Ad, AO, GO, AM, uFd, Interpo, FM,uG, new_region_100_aux,new_region_100_fin,BORD):
#
# 	#Amd.solve()
# 	print("###########", len(GO));
# 	print(len(AO));
#
# 	#Amd.add_multiplier('Amult', Amf, 'u')
# 	Amd.add_variable('Amult', len(new_region_100_aux));
# 	#Amd.set('resize variable', 'Amult', len(new_region_100_aux))
# 	B = gf.Spmat('empty', len(new_region_100_aux), Amd.nbdof());
# 	B.add(np.arange(len(new_region_100_aux)), new_region_100_aux,
# 	      gf.Spmat('identity', len(new_region_100_aux)));
# 	print(B)
# 	values = np.zeros((len(new_region_100_aux)));
# 	print("..........", B.size())
# 	A_index = Amd.add_constraint_with_multipliers('u', 'Amult', B, values);
# 	Amd.set_private_rhs(A_index, uG[GO]);
# 	Amd.set_private_matrix(A_index, B);
# 	print()
# 	Amd.solve('noisy')
# 	uA = Amd.variable('u');
# 	#values_index = Amd.add_explicit_rhs('u', uG[GO])
# 	## THE OLD METHOD ##
# 	# if BORD:
# 	# 	Ad[new_region_100_aux[AO]] = uG[GO]
# 	# 	print(new_region_100_aux[AO])
# 	# else:
# 	# 	print(Amf.basic_dof_on_region(100)[AO])
# 	# 	Ad[Amf.basic_dof_on_region(100)[AO]] = uG[GO]
# 	#print("-----------", Ad[Amf.basic_dof_on_region(100)[AO]])
# 	#Amd.set_variable('InterDiri', Ad)
# 	#print(len(Amd.variable('DirichletData')))
# 	#print("-------UA---------", uA)
# 	# Locals
# 	# if BORD:
# 	# 	uFd[new_region_100_fin] = Interpo.dot(
# 	# 		Ad[new_region_100_aux])
# 	# else:
# 	# 	uFd[Fmf.basic_dof_on_region(100)] = Interpo.dot(
# 	#  	    Ad[Amf.basic_dof_on_region(100)])
# 	#Fmd.set_variable('InterDiri', uFd)
#
#
# 	Fmd.add_variable('Fmult', len(new_region_100_fin));
# 	#Fmd.add_multiplier('Fmult', Fmf, 'u')
# 	C = gf.Spmat('empty', len(new_region_100_fin), Fmd.nbdof());
# 	C.add(np.arange(len(new_region_100_fin)), new_region_100_fin,
# 	      gf.Spmat('identity', len(new_region_100_fin)));
# 	values_fin = np.zeros((len(new_region_100_fin)));
# 	#print(C.size(), "############", B.size())
# 	Fmd.add_constraint_with_multipliers('u', 'Fmult', C, values_fin);
# 	values_fin_index = Fmd.add_explicit_rhs('u', values_fin);
# 	Fmd.set_private_rhs(values_fin_index, values_fin);
#
# 	Fmd.solve()
# 	uF = Fmd.variable('u')
# 	#Fmd.solve()
#
# 	#print(np.shape(AM))
# 	#print((np.argwhere(AO > 110)))
# 	#print((AO ))
# 	#lambda_a = Amd.variable('Amult')[np.argwhere(AO < 110)]
# 	#print(len(lambda_a))
# 	#print(len([AO]))
# 	#print("--------", len(np.argwhere(uA[AO] < 1e-16)), "--------", np.argwhere(uA[AO] < 1e-16))
# 	#print(uA[AO])
#
# 	print(len(AO))
# 	print(len(GO))
# 	#print(AM.dot(Amd.variable('Amult'))[AO])
# 	# ,Interpo.transpose().dot(FM.dot(Fmd.variable('Fmult'))),
# 	# print((Amf.basic_dof_on_region(1000002)))
# 	# print((Amf.basic_dof_on_region(100)))
# 	#print(AO[0:len(Amd.variable('Amult'))])
#
# 	#print(b)
# 	#print(a)
#
# 	#print(len(Amd.variable('Amult')))
# 	#AM.dot(Amd.variable('Amult'))[AO]
# 	#,Interpo.transpose().dot(FM.dot(Fmd.variable('Fmult')))
# 	return(uF,uA)





	#========== Construction de la matrice pour F.Magoules=====#

	#A = gf.asm('laplacian', Gmim, Gmf ,np.repeat([1], Gnbd),Gnbd)
	#A= gf.asm_laplacian(Gmim , Gmf , Gmf, np.repeat([1], Gnbd) ,Gnbd)
	#
	# B = gf.asm_laplacian(Gmim,  Gmf,  Gmf,  np.ones((1, Gnbd)))
	# #print(A)
	#
	# #gf.Spmat.display(A)
	# gf.Spmat.to_csc(B)
	# print(B)
	# gf.Spmat.display(B)
	# gf.Spmat.save(B,'mm', 'test1')
	# print("--------")
	# #fichier = open("C:/Users/ahmed/untitled/globallocal/results/aubeGL/test", "r")
	# M = np.loadtxt("C:/Users/ahmed/untitled/globallocal/results/aubeGL/test")
	# A = np.zeros((701,701))
	# k = 0
	# for k in range(0,4661):
	#
	#  		#print(int(M[i,0]-1), int(M[i,1]-1))
	#  	A[int(M[k,0]-1),int(M[k,1]-1)] = float(M[k,2])
	#  	#k = k + 1
	# print(A)
	# print(type(M))
	#
	# import matplotlib.pyplot as plt
	# fig = plt.spy(A)
	# plt.show()
	# #print(A)
	# #gf.util_save_matrix(FMT='mm', FILENAME='test', A=B)
	# print("--------")


