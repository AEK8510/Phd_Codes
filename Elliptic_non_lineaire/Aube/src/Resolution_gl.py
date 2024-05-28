import numpy as np
import getfem as gf
import os
import sys

def Local_resolution(Fmd, Fmf, uFd, Interpo, FM, region_num, uG):
	uFd[Fmf.basic_dof_on_region(region_num)] = Interpo.dot(uG)
	Fmd.set_variable('InterDiri', uFd)
	Fmd.solve('max iter', 150, 'max res', 1e-12, 'lsearch', 'basic', 'lsolver', 'mumps')
	#Fmd.solve('noisy', 40,  1e-6)
	uF = Fmd.variable('u')
	return (Interpo.transpose().dot(FM.dot(Fmd.variable('Fmult'+str(region_num)))), uF)

def Aitken_Asynchrone(P, ind_svd, svd_trunc):
	U, s, V = np.linalg.svd(P[:, 0:ind_svd])
	frob_norm = np.linalg.norm(P[:, 0:ind_svd], 'fro')
	sin_val_sum = 0
	n_svd = 0
	while (abs((frob_norm) - np.sqrt(sin_val_sum)) > 1e-14 and n_svd < svd_trunc - 1):
		sin_val_sum += s[n_svd] ** 2
		n_svd += 1
	P_svd = U[:, 0:n_svd].T.dot(P[:, ind_svd - n_svd - 1:ind_svd])
	e_svd = P_svd[:, 1:n_svd] - P_svd[:, 0:n_svd - 1]
	P_chap = e_svd[:, 1: n_svd - 1].dot(np.linalg.pinv(e_svd[:, 0: n_svd - 2]))
	part_1 = U[:, 0:n_svd].dot(np.linalg.pinv(np.eye(n_svd, n_svd) - P_chap))
	part_2 = P_svd[:, n_svd - 1] - P_chap.dot(P_svd[:, n_svd - 2])
	pG = part_1.dot(part_2)
	P *= 0
	return pG

def Residu_computation(Gmd, Gmim, Gnbd, Gintdofs, nprocs, rG_win_s, a, Dim, NLCbis, Dm, md):
	temp_s = np.zeros(Gnbd)
	for s in range(0, nprocs - 1):
		temp_s[:] += rG_win_s[s, 0:Gnbd]
	if Dim == 2:
		Reg_ID = 13
	if Dim == 3:
		Reg_ID = 10
	#md.add_initialized_fem_data('coefbis', Dm, NL_Coef1)
	#print(np.linalg.norm(temp_s[Gintdofs]), "-----------------", 2)
	#print(np.linalg.norm(gf.asm_generic(Gmim, 1, 'Grad_u.Grad_Test_u + 1*u*u*u*Test_u - VolumicData*Test_u', Reg_ID, Gmd)[Gintdofs + a]), "------", 1)
	rG = - gf.asm_generic(Gmim, 1, 'Grad_u.Grad_Test_u - VolumicData*Test_u ', Reg_ID , Gmd)[Gintdofs + a]+ temp_s[Gintdofs]
	#print(rG, "-------------------", 3)
	norm_new = np.linalg.norm(rG)
	return norm_new, rG


def Global_resolution(solveur, rG_j_1, rG_j, ci, omega, omega_new, pG, pG_index, rG, Gintdofs, Gmd, Gmf, loc_iter, Dim, NLC):
	uG_inter = {}
	if solveur == "Aitken":
		rG_j_1 = rG_j.copy()
		rG_j = rG.copy()
		if ci >= 2:
			omega_old = np.copy(omega_new)
			deno = (np.linalg.norm(rG_j - rG_j_1)) ** 2
			#print(rG_j , "----------", rG_j_1)
			rG_j_1_T = (rG_j_1.copy()).T
			nume = rG_j_1_T.dot(rG_j - rG_j_1)
			frac = float(nume / deno)
			#print(deno, "---------", nume)
			omega_new = - omega_old * frac
		else:
			omega_new = np.copy(omega)
	pG[Gintdofs] += omega_new * rG.copy()
	Gmd.set_private_rhs(pG_index, pG)
	Gmd.solve('lsolver', 'mumps')
	#Gmd.solve('very noisy', 'max iter', 150, 'max res', 1.e-15, 'lsearch', 'simplest', 'lsolver', 'mumps')
	uG = Gmd.variable('u');
	for i in range(0, 2):
		uG_inter[i] = np.append(uG[Gmf.basic_dof_on_region(i + Dim - 1)], loc_iter)
		#uG_inter[i] = np.append(uG_inter[i], loc_iter)
	return uG_inter, uG, rG_j, rG_j_1, omega_new


