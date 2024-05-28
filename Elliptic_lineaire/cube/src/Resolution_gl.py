import numpy as np
import getfem as gf
import os
import sys


def Residu_computation(Gnbd, nprocs, rG_win_s):
	temp_s = np.zeros(Gnbd)
	iters = np.zeros(nprocs-1)
	for s in range(0, nprocs - 1):
		temp_s[:] += rG_win_s[s, 0:Gnbd]
		iters[s] = rG_win_s[s, Gnbd + s]
	#temp_s[:] += rG_win_s[0:Gnbd]
	rG = temp_s.copy()
	norm_new = np.linalg.norm(rG)
	return norm_new, rG, iters

def Local_resolution(Fmd, Interpo, uG, Gbound, new_region_100_fin, uFd, C, F_index, FM):
	uFd[new_region_100_fin] = Interpo.dot(uG[Gbound])
	Fmd.set_private_rhs(F_index, uFd[new_region_100_fin]);
	Fmd.set_private_matrix(F_index, C);
	#Fmd.solve('noisy', 100,  1e-6);
	Fmd.solve();
	uF = Fmd.variable('u')
	return(uF,Interpo.transpose().dot(FM.dot(Fmd.variable('Fmult'))))

def Aitken_Asynchrone(P, ind_svd, svd_trunc):
	#print("HEEEEEEEEEEELLLLLLLLLOOOOOOOO")
	#if solveur == "Aitken":
	U, s, V = np.linalg.svd(P[:, 0:ind_svd])
	frob_norm = np.linalg.norm(P[:, 0:ind_svd], 'fro')
	sin_val_sum = 0
	n_svd = 0
	while (abs((frob_norm) - np.sqrt(sin_val_sum)) > 1e-8 or n_svd == svd_trunc):
		sin_val_sum += s[n_svd] ** 2
		print(abs((frob_norm) - np.sqrt(sin_val_sum)), "-----------", "residu_svd_trunc")
		n_svd += 1
	P_svd = U[:, 0:n_svd].T.dot(P[:, ind_svd - n_svd - 1:ind_svd])
	e_svd = P_svd[:, 1:n_svd] - P_svd[:, 0:n_svd - 1]
	P_chap = e_svd[:, 1: n_svd - 1].dot(np.linalg.pinv(e_svd[:, 0: n_svd - 2]))
	part_1 = U[:, 0:n_svd].dot(np.linalg.pinv(np.eye(n_svd, n_svd) - P_chap))
	part_2 = P_svd[:, n_svd - 1] - P_chap.dot(P_svd[:, n_svd - 2])
	pG = part_1.dot(part_2)
	P *= 0
	return pG

def Global_resolution(solveur, rG_j_1, rG_j, ci, omega, omega_new, pG, pG_index, rG, Gmd):
	if solveur == "Aitken":
		rG_j_1 = rG_j
		rG_j = rG
		if ci >= 2:
			omega_old = np.copy(omega_new)
			deno = (np.linalg.norm(rG_j - rG_j_1)) ** 2
			rG_j_1_T = rG_j_1.T
			nume = rG_j_1_T.dot(rG_j - rG_j_1)
			frac = nume / deno
			omega_new = - omega_old * frac
		else:
			omega_new = omega
	pG += omega_new * rG
	Gmd.set_private_rhs(pG_index, pG)
	Gmd.solve()
	uG = Gmd.variable('u');
	return uG, rG_j, rG_j_1, omega_new


