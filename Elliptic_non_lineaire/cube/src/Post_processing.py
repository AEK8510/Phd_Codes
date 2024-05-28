import numpy as np
import getfem as gf
import os
import sys
#os.chdir('./results/cube/')
Path = os.getcwd()

def Gradient_comput(mf, Dm, u, size, ind, DU):
	for i in range(0, size):
		DU[i] = gf.compute_gradient(mf[i], u[i], Dm[i])
		Dm[i].export_to_pos(Path + '/DU' + str(ind[i]) + '.pos',Dm[i] ,DU[i], 'DU_F' + str(ind[i]))
		Dm[i].export_to_vtk(Path + '/DU' + str(ind[i]) + '.vtk', 'ascii', DU[i], 'DUF')
	return 0



def Solution_export(mf, u, size, ind):
	for i in range(0, size):
		mf[i].export_to_pos(Path + '/u' + str(ind[i]) + '.pos', u[i], 'U_F'+str(ind[i]))
		mf[i].export_to_vtk(Path + '/u' + str(ind[i]) + '.vtk', 'ascii', u[i], 'U_GL')
	return 0

