import getfem as gf
import numpy as np

Cm = gf.Mesh('import', 'gmsh', 'Complement.msh')
Cmf = gf.MeshFem(Cm, 1)
Cmf.set_fem(gf.Fem('FEM_PK(2,1)'))
Cmim = gf.MeshIm(Cm, gf.Integ('IM_TRIANGLE(4)'))

Fm1 = gf.Mesh('import', 'gmsh', 'Fin1.msh')
Fmf1 = gf.MeshFem(Fm1, 1)
Fmf1.set_fem(gf.Fem('FEM_PK(2,1)'))
Fmim1 = gf.MeshIm(Fm1, gf.Integ('IM_TRIANGLE(4)'))

Fm2 = gf.Mesh('import', 'gmsh', 'Fin2.msh')
Fmf2 = gf.MeshFem(Fm2, 1)
Fmf2.set_fem(gf.Fem('FEM_PK(2,1)'))
Fmim2 = gf.MeshIm(Fm2, gf.Integ('IM_TRIANGLE(4)'))

####
bbottom = Cmf.dof_nodes()[1] < -1+0.0001
fdown = np.where(bbottom)[0]
####

####################################################

Cnbd = Cmf.nbdof()
Fnbd1 = Fmf1.nbdof()
Fnbd2 = Fmf2.nbdof()

Cnbb1 = len(Cmf.basic_dof_on_region(1))
Cnbb2 = len(Cmf.basic_dof_on_region(2))
Fnbb1 = len(Fmf1.basic_dof_on_region(1))
Fnbb2 = len(Fmf2.basic_dof_on_region(2))

CFdofs = Cnbd + Fnbd1 + Fnbd2
totdofs = CFdofs + Fnbb1 + Fnbb2

fnotdown = np.setdiff1d(range(Cnbd), fdown)

# Global to Fine interpolation
#
Interpo1 = gf.asm_interpolation_matrix(Cmf, Fmf1)[Fmf1.basic_dof_on_region(1)[
    :, np.newaxis], Cmf.basic_dof_on_region(1)]
Interpo2 = gf.asm_interpolation_matrix(Cmf, Fmf2)[Fmf2.basic_dof_on_region(2)[
    :, np.newaxis], Cmf.basic_dof_on_region(2)]

####################################################
# base matrices
KC = gf.asm_laplacian(Cmim,  Cmf,  Cmf,  np.ones((1, Cnbd)))
KF1 = gf.asm_laplacian(Fmim1, Fmf1, Fmf1, np.ones((1, Fnbd1)))
KF2 = gf.asm_laplacian(Fmim2, Fmf2, Fmf2, np.ones((1, Fnbd2)))

FC = gf.asm_volumic_source(Cmim,  Cmf,  Cmf,  np.ones((1, Cnbd)))
Ff1 = gf.asm_volumic_source(Fmim1, Fmf1, Fmf1, np.ones((1, Fnbd1)))
Ff2 = gf.asm_volumic_source(Fmim2, Fmf2, Fmf2, np.ones((1, Fnbd2)))

# Total problem
Ktot = np.zeros((totdofs, totdofs))
Ftot = np.zeros(totdofs)

Ftot[0:Cnbd] = FC
Ftot[Cnbd:Cnbd+Fnbd1] = Ff1
Ftot[Cnbd+Fnbd1:Cnbd+Fnbd1+Fnbd2] = Ff2
Ftot[fdown] = np.zeros(len(fdown))

Ktot[0:Cnbd, 0:Cnbd] = KC.full()
Ktot[fdown[:, np.newaxis], fdown] = np.eye(len(fdown))
Ktot[fdown[:, np.newaxis], fnotdown] = np.zeros((len(fdown), len(fnotdown)))
Ktot[fnotdown[:, np.newaxis], fdown] = np.zeros((len(fnotdown), len(fdown)))

Ktot[Cnbd:Cnbd+Fnbd1, Cnbd:Cnbd+Fnbd1] = KF1.full()
Ktot[Cnbd+Fnbd1:Cnbd+Fnbd1+Fnbd2, Cnbd+Fnbd1:Cnbd+Fnbd1+Fnbd2] = KF2.full()

# Coupling
Ktot[Cmf.basic_dof_on_region(1)[:, np.newaxis], np.arange(
    CFdofs, CFdofs+Fnbb1)] = Interpo1.transpose()
Ktot[Cmf.basic_dof_on_region(2)[:, np.newaxis], np.arange(
    CFdofs+Fnbb1, totdofs)] = Interpo2.transpose()
Ktot[Cnbd + Fmf1.basic_dof_on_region(1)[:, np.newaxis],
     np.arange(CFdofs, CFdofs+Fnbb1)] = -np.eye(Fnbb1)
Ktot[Cnbd + Fnbd1 + Fmf2.basic_dof_on_region(
    2)[:, np.newaxis], np.arange(CFdofs+Fnbb1, totdofs)] = -np.eye(Fnbb2)
#
Ktot[CFdofs:(CFdofs+Fnbb1), Cmf.basic_dof_on_region(1)] = Interpo1
Ktot[(CFdofs+Fnbb1):totdofs, Cmf.basic_dof_on_region(2)] = Interpo2
Ktot[CFdofs:(CFdofs+Fnbb1), Cnbd +
     Fmf1.basic_dof_on_region(1)] = -np.eye(Fnbb1)
Ktot[(CFdofs+Fnbb1):totdofs, Cnbd + Fnbd1 +
     Fmf2.basic_dof_on_region(2)] = -np.eye(Fnbb2)

# Solving
x = np.linalg.solve(Ktot, Ftot)
xC = x[0:Cnbd]
xF1 = x[Cnbd:Cnbd+Fnbd1]
xF2 = x[Cnbd+Fnbd1:CFdofs]

lam1 = x[CFdofs:CFdofs+Fnbb1]
lam2 = x[CFdofs+Fnbb1:totdofs]


Cmf.export_to_pos('xC.pos', xC, 'xC')
Fmf1.export_to_pos('xF1.pos', xF1, 'xF1')
Fmf2.export_to_pos('xF2.pos', xF2, 'xF2')
