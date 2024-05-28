#!/usr/bin/env python3

"""The purpose of this code is to generate a global-local coupling algorithm
for thermals problems in sequential and parallel version using the model laplace_gl."""

"_Authors_ : PG, AE"
import mpi4py
mpi4py.rc.threads=False
import sys
import os
import numpy as np
import getfem as gf
from mpi4py import MPI
gf.util_trace_level(level=0)
gf.util_warning_level(level=0)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
datatype = MPI.FLOAT
itemsize = datatype.Get_size
#----------------------------------------------------------------------------------------------------------------------#
import time
class Timer:
	def __enter__(self):
		self.start = time.time()
		return self

	def __exit__(self, *args):
		self.end = time.time()
		self.interval = self.end - self.start
#----------------------------------------------------------------------------------------------------------------------#
exec(open('data/'+sys.argv[1]+'.py').read())
try:
    os.makedirs('../../Meshes/'+sys.argv[1])
except FileExistsError:
    pass
os.chdir('../../Meshes/'+sys.argv[1])

if rank == 0:
	if rank == 0:
		print("***********************************************************************************************************")
		print('******** Couplage global-local %s avec un solveur %s avec coefficient de relaxation = %s ********' % (sys.argv[3], sys.argv[2], sys.argv[5]))
		print("-----------------------------------------------------------------------------------------------------------")
		print('******************************* Problème resolu : Thermique linèaire **************************************')
		print("-----------------------------------------------------------------------------------------------------------")
		print('************************************ Nombre de patches locaux = 2 *****************************************')
		print("-----------------------------------------------------------------------------------------------------------")
		print('********************************** Nombre de couers de calcul = %d *****************************************' %nprocs)
		print("***********************************************************************************************************")
		print("Reading data ...")
#----------------------------------------------- Create a new mesh ----------------------------------------------------#
# print("Meshing")
# exec(open('../../src/mesh2d.py').read())
# exit(-2)
from src import Coupling_gl
from src import Resolution_gl
from src import laplace_gl
from src import Post_processing
from src import RHS_GBC
#+++++++++++++++++++++++++++++++++++++++ Global finite element model ++++++++++++++++++++++++++++++++++++++++++++++++++#
Dim = int(sys.argv[1])
solveur = sys.argv[2]
model = sys.argv[3]
svd_trunc = int(sys.argv[4])
#++++++++++++++++++++++++ Global finite element model +++++++++++++++++++++++++++++++++++++++++#
if Dim == 2 :
	eps = 1.e-6
if Dim == 3 :
	eps = 1.e-3
(Gm, Gmf, Gmd, Gmim, Gnbd, GDm) = laplace_gl.FE_Model_Global('Global.msh', Dim)
Gintdofs = np.concatenate((Gmf.basic_dof_on_region(Dim - 1), Gmf.basic_dof_on_region(Dim)))
ci = 0
converged = np.arange(1, dtype='i')
converged[0] = 0
loc_iter = 0
Dirichlet_condition = '0.'
RHS_value = '1000.'
rG_win_s = np.zeros((nprocs-1, Gnbd+2))
win_rG_s = {}

if rank == 0:
	(Gm, Gmd, Gmf, Gmim) = RHS_GBC.Global_dirichlet_boundary(Gm, Gmf, Gmd, Gmim, Dirichlet_condition, Dim)
	Gmd = RHS_GBC.load_volumic_data(Gmf, Gmim, Gmd, RHS_value)
	omega_old = 1
	omega_new = float(sys.argv[5])
	rG = np.zeros(len(Gintdofs), dtype=float)
	rG_j = np.zeros(len(Gintdofs), dtype=float)
	rG_j_1 = np.zeros(len(Gintdofs), dtype=float)
	rG_j_2 = np.zeros(len(Gintdofs), dtype=float)
	pG = np.zeros(Gnbd, dtype=float)
	uG = np.zeros(Gnbd, dtype=float)
	P = np.zeros((Gnbd, svd_trunc), dtype=float)
	pG_index = Gmd.add_explicit_rhs('u', pG)
	a = Gmd.interval_of_variable('u')[0]
	Glob_c = np.zeros((nprocs - 1))
	Glob_c_prec = np.zeros((nprocs - 1))
	ind_svd = 0
	for s in range(0, nprocs - 1):
		win_rG_s[s] = MPI.Win.Create(rG_win_s[s, :], comm=MPI.COMM_WORLD)
	win_conv = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
	win_uG = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
	print("---------------------- Start resolution of the Global-Local coupling ----------------------------------")

if rank != 0 :
	#--------------------------------- Fin problem +  volumic load ----------------------------------------------------#
	(Fm, Fmf, Fmd, Fmim, Fnbd, FDm) = laplace_gl.FE_Model_local('Aux_'+str(rank)+'.msh', Dim)
	Fmd = RHS_GBC.load_volumic_data(Fmf, Fmim, Fmd, RHS_value)
	#-------------------------------- Global to local coupling --------------------------------------------------------#
	(uFd, Interpo, FM) = Coupling_gl.Global_to_local_coupling(Fmf, Fmd, Fmim, Fnbd, Gmf, rank + Dim - 2)
	uF = np.zeros(Fnbd, dtype=float)
	LFonG = np.zeros(Gnbd + 2, dtype=float)
	uG_interface = np.zeros(len(Gmf.basic_dof_on_region(rank + Dim - 2))+1)
	loc_G = 0
	loc_G_prec = 0
	count = 0
	for s in range(0, nprocs - 1):
		win_rG_s[s] = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
	win_conv = MPI.Win.Create(converged, comm=MPI.COMM_WORLD)
	win_uG = MPI.Win.Create(uG_interface, comm=MPI.COMM_WORLD)

#------------------------------------ Synchronous version ---------------------------------------------------------#
if model == "Synchrone" :
	with Timer() as t:
		while(converged[0] == 0):
			win_uG.Fence()
			if rank == 0:
				loc_iter += 1
				uG_inter, uG, rG_j, rG_j_1, rG_j_2, omega_new = \
					Resolution_gl.Global_resolution(solveur,rG_j_2, rG_j_1, rG_j, ci, omega_old, omega_new, pG, pG_index, rG, Gintdofs, Gmd, Gmf, loc_iter, Dim)
				#if loc_iter == 1:
					#Post_processing.Solution_export(Gmf, uG, 'G')
					#DUg = Post_processing.Gradient_comput(Gmf, GDm, uG, rank+10)
					#uC, Cmf, DUC, Dmc = Post_processing.Complementary_solution('Complement.msh', uG, Gmf, Dim, GDm, DUg)
					#Post_processing.Solution_export(Cmf, uC, rank+10)
					#DUC = Post_processing.Gradient_comput(Cmf, Dmc, uC, rank+5+10)
				for i in range(0, nprocs - 1):
					win_uG.Put([uG_inter[i], MPI.DOUBLE], i + 1)
				#loc_iter += 1
			win_uG.Fence()
			for s in range(0, nprocs - 1):
				win_rG_s[s].Fence()
			if rank != 0:
				LFonG[Gmf.basic_dof_on_region(rank + Dim - 2)], uF = \
					Resolution_gl.Local_resolution(Fmd, Fmf, uFd, Interpo, FM, rank + Dim - 2, uG_interface[0:len(Gmf.basic_dof_on_region(rank + Dim - 2))])
				win_rG_s[rank - 1].Put([LFonG, MPI.DOUBLE], 0)
				loc_iter += 1
				#if loc_iter == 1:
				#	DUF = Post_processing.Gradient_comput(Fmf, FDm, uF, rank+10)
				#	DUF = Post_processing.Solution_export(Fmf, uF, rank+10)
			for s in range(0, nprocs - 1):
				win_rG_s[s].Fence()

			win_conv.Fence()
			if rank == 0:
				norm_rG, rG = Resolution_gl.Residu_computation(Gmd, Gmim, Gnbd, Gintdofs, nprocs, rG_win_s, a, Dim)
				print("it", ci, "norm r", norm_rG, "rank:", " ", rank)
				ci += 1
				if norm_rG < eps or ci == 1:
					converged[0] = 1
					for i in range(1, nprocs):
						win_conv.Put([converged, MPI.INT], i)
			win_conv.Fence()
	print("loc_iteration:", "---------", loc_iter)
	print("Time spent", t.interval, "------", rank)

#------------------------------ Asynchronous version with wait --------------------------------------------------------#
elif model == "Asynchrone":
	with Timer() as t:
		while(converged[0] == 0):
			if rank == 0:
				if solveur == "Aitken_SVD":
					P[:, ind_svd] = pG.copy()
					ind_svd += 1
					if ind_svd % int(svd_trunc) == 0:
						pG = Resolution_gl.Aitken_Asynchrone(P, ind_svd, svd_trunc)
						ind_svd = 0
				loc_iter += 1
				uG_inter, uG, rG_j, rG_j_1, rG_j_2, omega_new = \
					Resolution_gl.Global_resolution(solveur,rG_j_2, rG_j_1, rG_j, ci, omega_old, omega_new, pG, pG_index, rG, Gintdofs, Gmd, Gmf, loc_iter, Dim)
				for i in range(0, nprocs - 1):
					win_uG.Lock(i + 1)
					win_uG.Put([uG_inter[i], MPI.DOUBLE], i + 1)
					win_uG.Flush(i + 1)
					win_uG.Unlock(i + 1)
				#loc_iter += 1

			if rank != 0:
				while(loc_G == loc_G_prec and converged[0] == 0):
					win_uG.Lock(rank)
					loc_G =  np.copy(uG_interface[len(Gmf.basic_dof_on_region(rank + Dim - 2))])
					count += 1
					win_uG.Unlock(rank)
					ready = False
				loc_G_prec = np.copy(loc_G)
				ready = True
				if ready == True:
					LFonG[Gmf.basic_dof_on_region(rank + Dim - 2)], uF = \
						Resolution_gl.Local_resolution(Fmd, Fmf, uFd, Interpo, FM, rank + Dim - 2, uG_interface[0:len(Gmf.basic_dof_on_region(rank + Dim - 2))])
					loc_iter += 1
					LFonG[Gnbd + rank - 1] = loc_iter
					win_rG_s[rank - 1].Lock(0)
					win_rG_s[rank - 1].Put([LFonG, MPI.DOUBLE], 0)
					win_rG_s[rank - 1].Flush(0)
					win_rG_s[rank - 1].Unlock(0)

			if rank == 0:
				while (np.array_equal(Glob_c, Glob_c_prec)) :
					for i in range(0, nprocs - 1):
						win_rG_s[i].Lock(rank)
						Glob_c[i] = rG_win_s[i, Gnbd + i]
						win_rG_s[i].Unlock(rank)
				Glob_c_prec = Glob_c.copy()
				norm_new, rG = Resolution_gl.Residu_computation(Gmd, Gmim, Gnbd, Gintdofs, nprocs, rG_win_s, a, Dim)
				print("it", ci, "norm r", norm_new, "rank:", " ", rank)
				ci += 1
				if norm_new < eps:
					converged[0] = 1
					for i in range(0, nprocs-1):
						win_conv.Lock(i+1)
						win_conv.Put([converged, MPI.INT], i+1)
						win_conv.Flush(i+1)
						win_conv.Unlock(i+1)

	print("loc_iteration:", "---------", loc_iter)
	print("Time spent", t.interval, "------", rank)

#---------------------------------- Asynchronous version without wait -------------------------------------------------#
elif model == "Asynchrone2":
	if rank == 0:
	 	win_uG.Lock_all()
	 	win_conv.Lock_all()
	if rank != 0:
	 	win_rG_s[rank - 1].Lock(0)
	with Timer() as t:
		while(converged[0] == 0):
			if rank == 0:
				if solveur == "Aitken_SVD":
					P[:, ind_svd] = pG.copy()
					ind_svd += 1
					if ind_svd % int(svd_trunc) == 0:
						pG = Resolution_gl.Aitken_Asynchrone(P, ind_svd, svd_trunc)
						ind_svd = 0
				loc_iter += 1
				uG_inter, uG, rG_j, rG_j_1, omega_new = \
					Resolution_gl.Global_resolution(solveur, rG_j_1, rG_j, ci, omega_old, omega_new, pG, pG_index, rG, Gintdofs, Gmd, Gmf, loc_iter, Dim)
				for i in range(0, nprocs - 1):
					win_uG.Put([uG_inter[i], MPI.DOUBLE], i + 1)
					win_uG.Flush_local(0)
				#win_uG.Flush_local(0)
				#loc_iter += 1

			if rank != 0:
				LFonG[Gmf.basic_dof_on_region(rank + Dim - 2)], uF = \
						Resolution_gl.Local_resolution(Fmd, Fmf, uFd, Interpo, FM, rank + Dim - 2, uG_interface[0:len(Gmf.basic_dof_on_region(rank + Dim - 2))])
				LFonG[Gnbd + rank - 1] = loc_iter
				win_rG_s[rank - 1].Put([LFonG, MPI.DOUBLE], 0)
				win_rG_s[rank - 1].Flush_all()
				loc_iter += 1

			if rank == 0:
				norm_new, rG = Resolution_gl.Residu_computation(Gmd, Gmim, Gnbd, Gintdofs, nprocs, rG_win_s, a, Dim)
				print("it", ci, "norm r", norm_new, "rank:", " ", rank)
				ci += 1
				if norm_new < eps:
					converged[0] = 1
					for i in range(0, nprocs-1):
						win_conv.Put([converged, MPI.INT], i+1)
					win_conv.Flush_all()

	if rank == 0:
	 	win_uG.Unlock_all()
	 	win_conv.Unlock_all()
	if rank != 0:
	 	win_rG_s[rank - 1].Unlock(0)
	print("loc_iteration:", "---------", loc_iter)
	print("Time spent", t.interval, "------", rank)

#---------------------------------- Asynchronous version without wait -------------------------------------------------#
elif model == "Asynchrone3":
	with Timer() as t:
		while(converged[0] == 0):
			if rank == 0:
				if solveur == "Aitken_SVD":
					P[:, ind_svd] = pG.copy()
					ind_svd += 1
					if ind_svd % int(svd_trunc) == 0:
						pG = Resolution_gl.Aitken_Asynchrone(P, ind_svd, svd_trunc)
						ind_svd = 0
				loc_iter += 1
				uG_inter, uG, rG_j, rG_j_1, omega_new = \
					Resolution_gl.Global_resolution(solveur, rG_j_1, rG_j, ci, omega_old, omega_new, pG, pG_index, rG, Gintdofs, Gmd, Gmf, loc_iter, Dim)
				for i in range(0, nprocs - 1):
					win_uG.Lock(i + 1)
					win_uG.Put([uG_inter[i], MPI.DOUBLE], i + 1)
					win_uG.Flush(i + 1)
					win_uG.Unlock(i + 1)
				#loc_iter += 1
			if rank != 0:
				LFonG[Gmf.basic_dof_on_region(rank + Dim - 2)], uF = \
					Resolution_gl.Local_resolution(Fmd, Fmf, uFd, Interpo, FM, rank + Dim - 2, uG_interface[0:len(Gmf.basic_dof_on_region(rank + Dim - 2))])
				loc_iter += 1
				LFonG[Gnbd + rank - 1] = loc_iter
				win_rG_s[rank - 1].Lock(0)
				win_rG_s[rank - 1].Put([LFonG, MPI.DOUBLE], 0)
				win_rG_s[rank - 1].Flush(0)
				win_rG_s[rank - 1].Unlock(0)

			if rank == 0:
				norm_new, rG = Resolution_gl.Residu_computation(Gmd, Gmim, Gnbd, Gintdofs, nprocs, rG_win_s, a, Dim)
				print("it", ci, "norm r", norm_new, "rank:", " ", rank)
				ci += 1
				if norm_new < eps:
					converged[0] = 1
					for i in range(0, nprocs-1):
						win_conv.Lock(i+1)
						win_conv.Put([converged, MPI.INT], i+1)
						win_conv.Flush(i+1)
						win_conv.Unlock(i+1)

	print("loc_iteration:", "---------", loc_iter)
	print("Time spent", t.interval, "------", rank)
#----------------------------------------------------------------------------------------------------------------------#
win_uG.Fence()
win_uG.Free()
win_conv.Fence()
win_conv.Free()
for s in range(0, nprocs - 1):
	win_rG_s[s].Fence()
for s in range(0, nprocs - 1):
	win_rG_s[s].Free()
if rank == 0:
	print("---------------------- End the resolution of the Global-Local coupling --------------------------------")
	Post_processing.Solution_export(Gmf, uG, 'G')
#	DUg = Post_processing.Gradient_comput(Gmf, GDm, uG, rank)
#	uC, Cmf, DUC, Dmc = Post_processing.Complementary_solution('Complement.msh', uG, Gmf, Dim, GDm, DUg)
#	Post_processing.Solution_export(Cmf, uC, rank)
#	DUC = Post_processing.Gradient_comput(Cmf, Dmc, uC, rank+5)
if rank != 0:
#	DUF = Post_processing.Gradient_comput(Fmf, FDm, uF, rank)
	DUF = Post_processing.Solution_export(Fmf, uF, rank)

#
# if rank == 0:
# 	print("---------------------- End the resolution of the Global-Local coupling --------------------------------")
# 	print("The total time:", " ", time2 - time1, "------", rank)
# 	Gmf.export_to_pos('uG.pos', uG[0:Gnbd], 'Global final solution')
# 	DU = gf.compute_gradient(Gmf, uG, GDM)
# 	Cm = gf.Mesh('import', 'gmsh', 'Complement.msh')
# 	Cmf = gf.MeshFem(Cm, 1)
# 	CDM = gf.MeshFem(Cm, 1)
# 	Cmf.set_fem(gf.Fem('FEM_PK(2,1)'))
# 	CDM.set_fem(gf.Fem('FEM_PK(2,0)'))
# 	Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))
# 	Cmd = gf.Model('real')
# 	uC = gf.compute_interpolate_on(Gmf, uG[0:Gnbd], Cmf)
# 	DUC = gf.compute_interpolate_on(GDM, DU, CDM)
# 	Cmf.export_to_pos('uCi.pos', uC, 'Uc')
# 	Cmf.export_to_vtk('uCi.vtk', 'ascii', uC, 'U_GL')
# 	GDM.export_to_pos('DUG' + str(rank) + '.pos',GDM ,DU, 'DUF')
# 	GDM.export_to_vtk('DUG' + str(rank) + '.vtk', 'ascii', DU, 'DUF')
# 	CDM.export_to_pos('DUC' + str(rank) + '.pos', CDM, DUC, 'DUF')
# 	CDM.export_to_vtk('DUC' + str(rank) + '.vtk', 'ascii', DUC, 'DUF')
# # if rank != 0:
# # 	Fmf.export_to_pos('uF'+str(rank)+'.pos', uF, 'U_F' + str(rank))
# if rank != 0:
# 	Fmf.export_to_pos('uF'+str(rank)+'.pos', uF, 'U_F')
# 	Fmf.export_to_vtk('uF'+str(rank)+'.vtk', 'ascii', uF, 'U_GL')
# 	#Amf[i].export_to_pos('uA'+str(ind[i])+'.pos', uAd[i], 'U_A'+str(ind[i]))
# 	DU = gf.compute_gradient(Fmf, uF,  FDM)
# 	Fmf.export_to_pos('DUF' + str(rank) + '.pos',FDM ,DU, 'DUF')
# 	FDM.export_to_vtk('DUF' + str(rank) + '.vtk', 'ascii', DU, 'DUF')
# data = np.loadtxt('convergence.txt')
# print(data)
# x_axis = np.arange(1, ci + 1)
# pyplot.plot(x_axis, data, 'o-')
# pyplot.yscale('log')
# pyplot.xlabel('number of iterations')
# pyplot.ylabel('Relative residual')
# pyplot.show()
	#Amf[i].export_to_pos('uA' + str(ind[i]) + '.pos', uAd[i], 'U_A' + str(ind[i]))

	# Gmf.export_to_pos('uG.pos', uG, 'U_global')
	# ind = str(rank+1)
	# Fmf.export_to_pos('uF'+ind+'.pos', uF, 'U_F'+ind)
	# Cm = gf.Mesh('import', 'gmsh', 'Complement.msh')
	# Cmf = gf.MeshFem(Cm, 1)
	# Cmf.set_fem(gf.Fem('FEM_PK(2,1)'))
	# Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))
	# Cmd = gf.Model('real')
	# uC = gf.compute_interpolate_on(Gmf, uG, Cmf)
	# Cmf.export_to_pos('uCi.pos', uC, 'U_C')

	#time2 = MPI.Wtime()
	#print("Time:", " ", time2 - time1)



#
# elif nprocs == 1 :
#  	time1 = MPI.Wtime()
#  	print("Sequentiel problem")
#  	Am = {}
#  	Amf = {}
#  	Amd = {}
#  	Amim = {}
#  	Anbd = {}
#  	Fm = {}
#  	Fmf = {}
#  	Fmd = {}
#  	Fmim = {}
#  	Fnbd = {}
#  	Ad = {}
#  	AO = {}
#  	GO = {}
#  	AM = {}
#  	uFd = {}
#  	Interpo = {}
#  	FM = {}
# #
#  	for i in range(0,nb_patchs):
# #
#  		ind = str(i + 1)
# #
# # 	# Aux problem +  volumic load
#  		(Am[i], Amf[i], Amd[i], Amim[i], Anbd[i]) = \
#  			laplace_gl.FE_Model('Aux_' + str(i + 1) + '.msh')
#  		Amd[i] = RHS_GBC.load_volumic_data(Amf[i], Amim[i], Amd[i], value)
# # 		# Fin problem +  volumic load
#  		(Fm[i], Fmf[i], Fmd[i], Fmim[i], Fnbd[i]) = \
#  			laplace_gl.FE_Model('Fin' + str(i + 1) + '.msh')
#  		Fmd[i] = RHS_GBC.load_volumic_data(Fmf[i], Fmim[i], Fmd[i], value)
# #
# # 		# ## Global to local coupling
#  		(Ad[i], AO[i], GO[i], AM[i], uFd[i], Interpo[i], FM[i]) = \
#  			laplace_gl.Global_to_local_coupling(Fmf[i], Amf[i], Fmd[i], Amd[i], Fmim[i], Amim[i], Fnbd[i], Anbd[i], Gmf, i+1)
# #
#  	Gnbd = Gmf.nbdof()
#  	Gintdofs = np.concatenate((Gmf.basic_dof_on_region(1), Gmf.basic_dof_on_region(2)))
# # 	# # Global interface corrective load
#  	pG = np.zeros(Gnbd, dtype=float)
#  	pG_index = Gmd.add_explicit_rhs('u', pG)
#  	rG = np.zeros(len(Gintdofs))
#  	LAonG = np.zeros(Gnbd, dtype=float)
#  	LFonG = np.zeros(Gnbd, dtype=float)
#  	ci = 0
#  	converged = False
#  	list = np.zeros(nb_patchs + 1)
#  	for i in range(1, nb_patchs + 1):
#  		list[i] = list[i - 1] + len(Gmf.basic_dof_on_region(i))
# #
#  	ind = 0
# #
#  	while (converged == False):
# #
#  		pG[Gintdofs] += 1.35 * rG
#  		Gmd.set_private_rhs(pG_index, pG)
#  		Gmd.solve()
#  		uG = Gmd.variable('u')
#  		for i in range(0,nb_patchs):
#  			LAonG[Gmf.basic_dof_on_region(i + 1)[GO[i]]], LFonG[Gmf.basic_dof_on_region(i + 1)], uFd[i] = \
#  				laplace_gl.aux_fin_comput(Amf[i], Amd[i], Fmd[i], Fmf[i], Ad[i], AO[i], GO[i], AM[i], \
#  				                          uFd[i], Interpo[i], FM[i], Gmf, i + 1,uG)
#  			rG[int(list[i]): int(list[i + 1])] = -(-LFonG + LAonG + pG)[Gmf.basic_dof_on_region(i + 1)]
#  		print("it", ci, "norm r", np.linalg.norm(rG))
#  		ci += 1
# #
#  		if (np.linalg.norm(rG) < 1.e-10 or ci == 100):
#  			converged = True
# # #Export results
# #
#  	print("rank", rank, "converged",converged)
# #
#  	Gmf.export_to_pos('uG.pos', uG, 'U_global')
#  	for i in range(0,nb_patchs):
#  		ind = str(i+1)
#  		Fmf[i].export_to_pos('uF'+ind+'.pos', uFd[i], 'U_F'+ind)
#  	Cm = gf.Mesh('import', 'gmsh', 'Complement.msh')
#  	Cmf = gf.MeshFem(Cm, 1)
#  	Cmf.set_fem(gf.Fem('FEM_PK(2,1)'))
#  	Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))
#  	Cmd = gf.Model('real')
#  	uC = gf.compute_interpolate_on(Gmf, uG, Cmf)
#  	Cmf.export_to_pos('uCi.pos', uC, 'U_C')
# # #
#  	time2 = MPI.Wtime()
# #
#  	print("Time:", " ", time2 - time1)
#
# else:
# #
#  	print(" For sequentiel : 1 proc")
#  	print(" For parallel : 2 procs")


#from src import laplace_ref
# #import elastiscity_lin
#
#
#
