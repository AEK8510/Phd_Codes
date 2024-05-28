#import gmsh
import os
import numpy as np
import getfem as gf


meshername = "mesh3D"



# ""
def load_volumic_data(mf, mim, md):
    f = mf.eval('[100.]')
    md.add_initialized_fem_data('VolumicData', mf, f)
    md.add_source_term_brick(mim, 'u', 'VolumicData')
    return (md, mf)

# Global Dirichlet boundary


def Global_dirichlet_boundary(m, mf, md, mim, value, BORD_ID):
    #faces_id = 0
    #face_id = 1
    #border = m.outer_faces()
    #fnor = m.normal_of_faces(border)
    #bord = abs(fnor[faces_id, :]+1*face_id) < 1e-14
    #fbord = np.compress(bord, border, axis=1)
    #m.set_region(BORD_ID, fbord)
    g = mf.eval(value)
    md.add_initialized_fem_data('DirichletData', mf, g)
    md.add_Dirichlet_condition_with_multipliers(mim, 'u', mf, BORD_ID, 'DirichletData')
    return (m, md, mf, mim)
