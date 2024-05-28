#import gmsh
import os
import numpy as np
import getfem as gf


meshername = "mesh3D"

# Data
#    Sides of cuboids
Hx = Hy = Hz = 1.
# Inclusions radius
radius = min(Hx, Hy, Hz)/4

# number of repetition per
nx = 8
ny = 4
nz = 4


# size of elements
h_fin = .05
h_glob = .25

# The gmsh path

#gmsh_path = "C:/Users/pierr/Documents/Codes/gmsh-4.5.6-Windows64/gmsh"
gmsh_path = "gmsh"
if os.getenv("USERNAME") == "ahmed":
    gmsh_path = "C:/Users/ahmed/Downloads/GMSH/gmsh"


# ""
def load_volumic_data(mf, mim, md):
    f = mf.eval('1.')
    md.add_initialized_fem_data('VolumicData', mf, f)
    md.add_source_term_brick(mim, 'u', 'VolumicData')
    return (md, mf)

# Global Dirichlet boundary


def Global_dirichlet_boundary(m, mf, md, mim, value):
    faces_id = 0
    face_id = 1
    BORD_ID = 42
    border = m.outer_faces()
    fnor = m.normal_of_faces(border)
    bord = abs(fnor[faces_id, :]+1*face_id) < 1e-14
    fbord = np.compress(bord, border, axis=1)
    m.set_region(BORD_ID, fbord)
    g = mf.eval(value)
    md.add_initialized_fem_data('DirichletData', mf, g)
    md.add_Dirichlet_condition_with_multipliers(
        mim, 'u', mf, BORD_ID, 'DirichletData')
    return (m, md, mf, mim)
