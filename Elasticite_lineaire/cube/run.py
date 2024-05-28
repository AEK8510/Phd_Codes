#!/usr/bin/env python3

"""The purpose of this code is to generate a global-local coupling algorithm
for thermals problems in sequential and parallel version using the model laplace_gl."""

"_Authors_ : PG, AE"
import mpi4py
mpi4py.rc.threads=False
import sys
import os
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('pb', 'data/cube.py')
cube = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cube)
import getfem as gf
from mpi4py import MPI
gf.util_trace_level(level=0)
gf.util_warning_level(level=0)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
datatype = MPI.FLOAT
itemsize = datatype.Get_size
import random
#random.seed(rank)
#-----------------------------------------------------------------------------------------------------------------------
import time
class Timer:
	def __enter__(self):
		self.start = time.time()
		return self

	def __exit__(self, *args):
		self.end = time.time()
		self.interval = self.end - self.start
#-----------------------------------------------------------------------------------------------------------------------
exec(open('data/cube.py').read())
try:
    os.makedirs('../../Meshes/'+sys.argv[1])
except FileExistsError:
    pass
os.chdir('../../Meshes/'+sys.argv[1])
#-----------------------------------------------------------------------------------------------------------------------
if rank == 0:
	print("***********************************************************************************************************")
	print('***** Couplage global-local %s avec un solveur %s et un coefficient de relaxation = %s *****' % (sys.argv[3], sys.argv[2], sys.argv[5]))
	print("-----------------------------------------------------------------------------------------------------------")
	print('******************************* Problème resolu : Elasticité linèaire **************************************')
	print("-----------------------------------------------------------------------------------------------------------")
	print('************************************ Nombre de patches locaux = %s *****************************************' % sys.argv[1])
	print("-----------------------------------------------------------------------------------------------------------")
	print('********************************** Nombre de couers de calcul = %d *****************************************' % nprocs)
	print("***********************************************************************************************************")
	print("Reading data ...")
# print("Meshing")
# if meshername == "mesh2D":
#  	exec(open('../../src/mesh2d.py').read())
# elif meshername == "mesh3D":
#  	exec(open('../../src/meshcube.py').read())
from src import Coupling_gl
from src import Resolution_gl
from src import Linear_elasticity
from src import Post_processing
from src import Preprocessing_gl
from src import RHS_GBC
#++++++++++++++++++++++++ Global finite element model +++++++++++++++++++++++++++++++++++++++++#
nb_patchs = sys.argv[1]
solveur = sys.argv[2]
model = sys.argv[3]
svd_trunc = sys.argv[4]
(region_interface, interface_id, Bord_index) = Preprocessing_gl.interface_identification_2(int(nb_patchs), nb_patchs)
if rank == 0:
	print(Bord_index)
#exit(-2)
#++++++++++++++++++++++++ Global finite element model +++++++++++++++++++++++++++++++++++++++++#
bbord_glob = [];
(Gm, Gmf,  Gmd, Gmim,  Gnbd, Gmp) = Linear_elasticity.FE_Global_Model('Global.msh');
ci = 0
converged = np.arange(1, dtype='i')
converged[0] = 0
loc_iter = 0
Dirichlet_condition = '[0., 0., 0.]'
RHS_value = '[10., 10., 10.]'
rG_win_s = np.zeros((nprocs - 1, Gnbd + nprocs - 1))
win_rG_s = {}
uG = np.zeros(Gnbd + 1, dtype=float)
#exit(-2)
if rank == 0:
	# Global Dirichlet boundary
	# left {0,1}, right {0,-1}, back {1,1}, face {1,-1}
	# bottom {2,1}, top {2,-1}
	(Gm, Gmd, Gmf, Gmim) = cube.Global_dirichlet_boundary(Gm, Gmf, Gmd, Gmim,  Dirichlet_condition);
	(Gmd, Gmf) = cube.load_volumic_data(Gmf, Gmim, Gmd);
	omega_new = float(sys.argv[5])
	omega_old = 1.
	rG = np.zeros(Gnbd, dtype=float)
	rG_j = np.zeros(Gnbd, dtype=float)
	rG_j_1 = np.zeros(Gnbd, dtype=float)
	pG = np.zeros(Gnbd, dtype=float)
	pG_index = Gmd.add_explicit_rhs('u', pG)
	Glob_c = np.zeros(nprocs - 1, dtype=float)
	Glob_c_prec = np.zeros(nprocs - 1, dtype=float)
	uG_send = np.zeros(Gnbd+1, dtype=float)
	P = np.zeros((Gnbd, int(svd_trunc)), dtype=float)
	rG_win_s_recv = np.zeros((nprocs - 1, Gnbd + nprocs - 1))
	rG_win_buf = np.zeros(Gnbd + nprocs - 1, dtype=float)
	ind_svd = 0
	for s in range(0, nprocs - 1):
		win_rG_s[s] = MPI.Win.Create(rG_win_s[s, :], comm=MPI.COMM_WORLD)
	win_conv = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
	win_uG = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
	win_rG = MPI.Win.Create(rG_win_buf, comm=MPI.COMM_WORLD)
#rand = [0,1,2,3,8]
load = False
if rank != 0:
	# size = rand[rank] - rand[rank - 1]
	#size = int(nb_patchs / (nprocs - 1));
	#ind = np.zeros(size, dtype=int)
	# for i in range(0, size):
	#	ind[i] = rand[rank - 1] + i + 1
	uF, FM, Fmp, DU, Fm, Fmf, Fmd, Fmim, Fnbd, uFd, Interpo, bbord_fin, new_region_100_fin, Gbound, F_index, C, ind, size = \
		Preprocessing_gl.list_initialisation(rank, load, int(nb_patchs), nprocs)
	for i in range(0, size):
		# +++++++++++++++++++++++++++++++++ Fine problem +  volumic load +++++++++++++++++++++++++++++++++++++++++++#
		(Fm[i], Fmf[i], Fmd[i], Fmim[i], Fnbd[i], bbord_fin[i], Fmp[i]) = Linear_elasticity.FE_Local_Model(
			'Fin_' + str(ind[i]) + '.msh', 1, ind[i], interface_id)
		(Fmd[i], Fmf[i]) = cube.load_volumic_data(Fmf[i], Fmim[i], Fmd[i]);
		print(Fnbd[i], "-----", rank)
		# +++++++++++++++++++++++++++++++++ Global to local coupling +++++++++++++++++++++++++++++++++++++++++++++++#
		(Interpo[i], new_region_100_fin[i], Gbound[i], FM[i]) = \
			Coupling_gl.Global_to_local_coupling(Fmim[i], Fmf[i], Gmf, region_interface[ind[i] - 1], 1, bbord_fin[i], ind[i], interface_id);
		(C[i], F_index[i], uFd[i]) = \
			Coupling_gl.Add_Lagrangian_multipliers(Fmd[i], Fmf[i], Fnbd[i], new_region_100_fin[i])
			# ++++++++++++++++++++++++++++++ Take into account the Global Dirichlet condition ++++++++++++++++++++++++++#
		if (ind[i] - 1) in Bord_index:
			# ID of the border to take into account
			BORD_ID = 1000000 + ind[i]
			g_2 = Fmf[i].eval('[0., 0., 0.]');
			Fmd[i].add_initialized_fem_data('DirichletData', Fmf[i], g_2);
			Fmd[i].add_Dirichlet_condition_with_multipliers(Fmim[i], 'u', Fmf[i], BORD_ID, 'DirichletData');
	LFonG = np.zeros(Gnbd + nprocs - 1, dtype=float)
	rG_loc = np.zeros(Gnbd + nprocs - 1, dtype=float)
	loc_G = 0
	loc_G_prec = 0
	count = 0
	for s in range(0, nprocs - 1):
		win_rG_s[s] = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
	win_conv = MPI.Win.Create(converged, comm=MPI.COMM_WORLD)
	win_uG = MPI.Win.Create(uG, comm=MPI.COMM_WORLD)
	win_rG = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
#exit(-2)
#---------------------------------------------- Synchronous Model -----------------------------------------------------#
if model == "Synchrone" :
	loc_iter = 0
	with Timer() as t:
		while ((converged[0] == 0)):
			win_uG.Fence()
			if rank == 0:
				uG, rG_j, rG_j_1, omega_new = \
					Resolution_gl.Global_resolution(solveur, rG_j_1, rG_j, ci, omega_old, omega_new, pG, pG_index, rG, Gmd)
				for i in range(1, nprocs):
					win_uG.Put([uG, MPI.DOUBLE], i)
				loc_iter += 1
			win_uG.Fence()

			for s in range(0, nprocs - 1):
				win_rG_s[s].Fence()
			if rank != 0:
				rG_loc[0:Gnbd] = 0
				for i in range(0, size):
					(uF[i], LFonG[Gbound[i]]) = \
						Resolution_gl.Local_resolution(Fmd[i], Interpo[i], uG[0:Gnbd], Gbound[i], new_region_100_fin[i],
																							uFd[i],C[i], F_index[i], FM[i])
					rG_loc[Gbound[i]] += LFonG[Gbound[i]]
				loc_iter += 1
				win_rG_s[rank - 1].Put([rG_loc, MPI.DOUBLE], 0)

			for s in range(0, nprocs - 1):
				win_rG_s[s].Fence()

			win_conv.Fence()
			if rank == 0:
				norm_rG, rG, iters = Resolution_gl.Residu_computation(Gnbd, nprocs, rG_win_s)
				print("it", ci, "norm r", norm_rG, "rank:", " ", rank)
				ci += 1
				if norm_rG < 1.e-6:
					converged[0] = 1
					for i in range(1, nprocs):
						win_conv.Put([converged, MPI.INT], i)
			win_conv.Fence()
	if rank == 0:
		print("------", iters.min(), "----", iters.max())
		print("loc_iteration:", "---------", loc_iter)
		print("Time spent", t.interval, "------", rank)
#---------------------------------------------- Asynchronous Model with wait-------------------------------------------#
elif model == "Asynchrone" :
	with Timer() as t:
		while converged[0] == 0:
			if rank == 0:
				if solveur == "Aitken_SVD":
					P[:, ind_svd] = pG.copy()
					ind_svd += 1
					if ind_svd % int(svd_trunc) == 0:
						pG = Resolution_gl.Aitken_Asynchrone(P, ind_svd, svd_trunc)
						ind_svd = 0
				uG[0:Gnbd], rG_j, rG_j_1, omega_new = \
					Resolution_gl.Global_resolution(solveur, rG_j_1, rG_j, ci, omega_old,omega_new, pG, pG_index, rG, Gmd)
				loc_iter += 1
				uG[Gnbd] = loc_iter
				for i in range(1, nprocs):
					win_uG.Lock(i,lock_type=MPI.LOCK_EXCLUSIVE)
					win_uG.Put([uG, MPI.DOUBLE], i)
					win_uG.Flush(i)
					win_uG.Unlock(i)

			if rank != 0:
				while loc_G == loc_G_prec and converged[0] == 0:
					win_uG.Lock(rank, lock_type=MPI.LOCK_EXCLUSIVE)
					loc_G = np.copy(uG[Gnbd])
					win_uG.Unlock(rank)
				loc_G_prec = np.copy(loc_G)
				rG_loc[0:Gnbd] = 0
				for i in range(0, size):
					(uF[i], LFonG[Gbound[i]]) = \
						Resolution_gl.Local_resolution(Fmd[i], Interpo[i], uG[0:Gnbd], Gbound[i], new_region_100_fin[i],
						                               uFd[i], C[i], F_index[i], FM[i])
					rG_loc[Gbound[i]] += LFonG[Gbound[i]]
				loc_iter += 1
				rG_loc[Gnbd + rank - 1] = loc_iter
				win_rG_s[rank - 1].Lock(0, lock_type=MPI.LOCK_EXCLUSIVE)
				win_rG_s[rank - 1].Put([rG_loc, MPI.DOUBLE], 0)
				win_rG_s[rank - 1].Flush(0)
				win_rG_s[rank - 1].Unlock(0)

			if rank == 0:
				while np.array_equal(Glob_c, Glob_c_prec) and converged[0] == 0:
					for i in range(0, nprocs - 1):
						win_rG_s[i].Lock(0, lock_type=MPI.LOCK_EXCLUSIVE)
						Glob_c[i] = rG_win_s[i, Gnbd + i]
						win_rG_s[i].Unlock(0)
				Glob_c_prec = Glob_c.copy()
				norm_rG, rG, iters = Resolution_gl.Residu_computation(Gnbd, nprocs, rG_win_s)
				print("it", ci, "norm r", norm_rG, "rank:", " ", rank)
				ci += 1
				if norm_rG < 1.e-6:
					converged[0] = 1
					for i in range(1, nprocs):
						win_conv.Lock(i)
						win_conv.Put([converged, MPI.INT], i)
						win_conv.Flush(i)
						win_conv.Unlock(i)
	print("loc_iteration:", "---------", loc_iter)
	print("Time spent", t.interval, "------", rank)
#------------------------------------------- Asynchronous Model without wait-------------------------------------------#
elif model == "Asynchrone2":
	with Timer() as t:
		if rank == 0:
			win_uG.Lock_all()
			win_conv.Lock_all()
		if rank != 0:
			win_rG_s[rank - 1].Lock(0)
		while converged[0] == 0:
			if rank == 0:
				if solveur == "Aitken_SVD":
					P[:, ind_svd] = pG.copy()
					ind_svd += 1
					if ind_svd % int(svd_trunc) == 0:
						pG = Resolution_gl.Aitken_Asynchrone(P, ind_svd, int(svd_trunc))
						ind_svd = 0
				uG[0:Gnbd], rG_j, rG_j_1, omega_new = \
					Resolution_gl.Global_resolution(solveur, rG_j_1, rG_j, ci, omega_old, omega_new, pG, pG_index, rG, Gmd)
				loc_iter += 1
				uG[Gnbd] = loc_iter
				for i in range(1, nprocs):
					win_uG.Put([uG, MPI.DOUBLE], i)
				win_uG.Flush_local(0)
				#win_uG.Flush_all()

			if rank != 0:
				rG_loc[0:Gnbd] = 0
				for i in range(0, size):
					(uF[i], LFonG[Gbound[i]]) = \
						Resolution_gl.Local_resolution(Fmd[i], Interpo[i], uG[0:Gnbd], Gbound[i],new_region_100_fin[i],
																						uFd[i], C[i],F_index[i], FM[i])
					rG_loc[Gbound[i]] += LFonG[Gbound[i]]
				loc_iter += 1
				rG_loc[Gnbd + rank - 1] = loc_iter
				win_rG_s[rank - 1].Put([rG_loc, MPI.DOUBLE], 0)
				win_rG_s[rank - 1].Flush(0)

			if rank == 0:
				norm_rG, rG, iters = Resolution_gl.Residu_computation(Gnbd, nprocs, rG_win_s)
				print("it", ci, "norm r", norm_rG, "rank:", " ", rank)
				ci += 1
				if norm_rG < 1.e-6 and ci > 50:
					converged[0] = 1
					for i in range(1, nprocs):
						win_conv.Put([converged, MPI.INT], i)
						win_conv.Flush(0)
					#win_conv.Flush_all()

		if rank == 0:
			win_uG.Unlock_all()
			win_conv.Unlock_all()
		if rank != 0:
			win_rG_s[rank - 1].Unlock(0)
	if rank == 0:
		print("-----", iters.min(), "------", iters.max())
		print("loc_iteration:", "---------", loc_iter)
		print("Time spent", t.interval, "------", rank)
# ---------------------------------------------- Asynchronous Model with wait-------------------------------------------#
elif model == "Asynchrone3" :
	with Timer() as t:
		if rank == 0:
			win_uG.Lock_all()
			win_conv.Lock_all()
		if rank  != 0:
		#for i in range(0, nprocs-1):
			win_rG_s[rank - 1].Lock(0)
		while converged[0] == 0:
			if rank == 0:
				if solveur == "Aitken_SVD":
					P[:, ind_svd] = pG.copy()
					ind_svd += 1
					if ind_svd % int(svd_trunc) == 0:
						pG = Resolution_gl.Aitken_Asynchrone(P, ind_svd, svd_trunc)
						ind_svd = 0
				uG[0:Gnbd], rG_j, rG_j_1, omega_new = \
					Resolution_gl.Global_resolution(solveur, rG_j_1, rG_j, ci, omega_old,omega_new, pG, pG_index, rG, Gmd)
				loc_iter += 1
				uG[Gnbd] = loc_iter
				for i in range(1, nprocs):
					#win_uG.Lock(i,lock_type=MPI.LOCK_EXCLUSIVE)
					win_uG.Put([uG, MPI.DOUBLE], i)
				win_uG.Flush_all()
					#win_uG.Unlock(i)

			if rank != 0:
				rG_loc[0:Gnbd] = 0
				for i in range(0, size):
					(uF[i], LFonG[Gbound[i]]) = \
						Resolution_gl.Local_resolution(Fmd[i], Interpo[i], uG[0:Gnbd], Gbound[i], new_region_100_fin[i],
						                               uFd[i], C[i], F_index[i], FM[i])
					rG_loc[Gbound[i]] += LFonG[Gbound[i]]
				loc_iter += 1
				rG_loc[Gnbd + rank - 1] = loc_iter
				#win_rG_s[rank - 1].Lock(0, lock_type=MPI.LOCK_EXCLUSIVE)
				win_rG_s[rank - 1].Put([rG_loc, MPI.DOUBLE], 0)
				win_rG_s[rank - 1].Flush(0)
				#win_rG_s[rank - 1].Unlock(0)

			if rank == 0:
				norm_rG, rG, iters = Resolution_gl.Residu_computation(Gnbd, nprocs, rG_win_s)
				print("it", ci, "norm r", norm_rG, "rank:", " ", rank)
				ci += 1
				if norm_rG < 1.e-6 and ci > 40:
					converged[0] = 1
					for i in range(1, nprocs):
						#win_conv.Lock(i)
						win_conv.Put([converged, MPI.INT], i)
					win_conv.Flush_all()
		if rank == 0 :				#win_conv.Unlock(i)
			win_uG.Unlock_all()
			win_conv.Unlock_all()
		#for i in range(0, nprocs-1):
		if rank != 0:
			win_rG_s[rank -1].Unlock(0)
	if rank == 0:
		print("loc_iteration:", "---------", loc_iter, iters.min(), iters.max())
		print("Time spent", t.interval, "------", rank)
#----------------------------------------------------------------------------------------------------------------------#
win_uG.Free()
win_conv.Free()
for s in range(0, nprocs - 1):
	win_rG_s[s].Free()
#================================================== Export results ====================================================#
#if rank == 0:
#	Gmf.export_to_pos('uG.pos', uG[0:Gnbd], 'U_global')
#if rank != 0:
#	Post_processing.Gradient_comput(Fmf, Fmp, uF, size, ind, DU)
#	Post_processing.Solution_export(Fmf, uF, size, ind)

