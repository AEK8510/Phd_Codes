import numpy as np
import getfem as gf


# 2D Global/Local Mesher
# Generates one Global Mesh, a pair of (Aux, Fine) Meshes per Patch, One Interface per Patch

# the mesh is described by (i ranges over patches number):
# BC[] : piecewise boundary of the complement zone
# BA[] : boundary of the Aux zone
# BF[] : boundary of the Fine zone
# OInter[] : Open interface (for patches touching the boundary)
# IInter[] : Internal interface (patches inside the domain)

# ODetail[] : Function to add geometrical details in the patch
# IDetail[] : Function to add geometrical details in the patch

# BC,BA,BF,OInter must have the same length
# The Global boundary is BC[1] + BA[1] + BC[2] + BA[2] ...
# OInter[i] is a physical line in the Global domain
# The Auxiliary domain is bounded by BA[i] + OInter[i]
# The Fine domain is bounded by BF[i] + OInter[i]
# ODetail has the length of OInter, IDetail has the length oh IInter

######


meshername = "mesh2d"

# Global geometrical variables
lu = 1.
lp = .25
lr = lp/4

BC = []
BA = []
BF = []
OInter = []
IInter = []


# User input

BC.append(np.array([[-lp, 0, 0], [-lu, 0, 0], [-lu, -lu, 0], [2*lu, -lu, 0], [2*lu, 0, 0],
                    [lu, 0, 0], [lu, 2*lu, 0], [0, 2*lu, 0], [0, lp, 0]]))

BA.append(np.array([]))

BF.append(np.array([[0, 0, 0]]))

OInter.append(np.array([[-lp, -lp, 0], [lp, -lp, 0], [lp, lp, 0]]))

IInter.append(np.array([[lp, lu, 0], [lp+lu/2, lu, 0],
                        [lp+lu/2, 2*lp+lu, 0], [lp, 2*lp+lu, 0]]))


# The detail functions describe the alterations in the patch, typically holes
# It returns the list of added nodes, lines, and surfaces (negative)
# The detail functions are stored in the array FDetails
# FDetails must return ([],[],[]) if it is empty


def Det1(model, lf, op, ol, os):
    model.geo.addPoint(lp/2, -lp/2, 0., lf, op)
    model.geo.addPoint(lp/2-lr, -lp/2, 0., lf, op+1)
    model.geo.addPoint(lp/2+lr, -lp/2, 0., lf, op+2)
    model.geo.addCircleArc(op+1, op, op+2, ol)
    model.geo.addCircleArc(op+2, op, op+1, ol+1)
    model.geo.addCurveLoop([ol, ol+1], os)
    return([op, op+1, op+2], [ol, ol+1], [-os])


def Det2(model, lf, op, ol, os):
    model.geo.addPoint(lp+lr, lu+lr, 0., lf, op)
    model.geo.addPoint(lp+lu/2-lr, lu+lr, 0., lf, op+1)
    model.geo.addPoint(lp+lu/2-lr, 2*lp+lu-lr, 0., lf, op+2)
    model.geo.addPoint(lp+lr, 2*lp+lu-lr, 0., lf, op+3)
    model.geo.addLine(op, op+1, ol)
    model.geo.addLine(op+1, op+2, ol+1)
    model.geo.addLine(op+2, op+3, ol+2)
    model.geo.addLine(op+3, op, ol+3)
    model.geo.addCurveLoop([ol, ol+1, ol+2, ol+3], os)
    return([op, op+1, op+2, op+3], [ol, ol+1, ol+2, ol+3], [-os])


FDetails = [Det1, Det2]


lcG = .1
lcF = .025

############################################################


def load_volumic_data(mf, mim, md):
    f = mf.eval(1.)
    md.add_initialized_fem_data('VolumicData', mf, f)
    md.add_source_term_brick(mim, 'u', 'VolumicData')
    return md

# Global Dirichlet boundary


def Global_dirichlet_boundary(m, mf, md, mim, value):
    faces_id = 1
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
