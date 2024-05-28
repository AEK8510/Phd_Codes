import gmsh
import os
import math
import re
import numpy as np
import glob


# function to return key for any value
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def meshcubes(withInclusions):
    if withInclusions:
        h = h_fin
        name = "Fin"
    else:
        h = h_glob
        name = "Global"

    npoints = (nx+1)*(ny+1)*(nz+1)
    nedges = nx * (ny+1) * (nz+1) + (nx+1) * ny * (nz+1) + (nx+1) * (ny+1) * nz
    nfaces = nx*ny*(nz+1) + nx*(ny+1)*nz + (nx+1)*ny*nz
    nvol = nx*ny*nz

    # Physical ids :
    # 1..nvol : cubes
    # nvol+1.. 2*nvol : inclusions
    # Shift1 is used to separate interfaces from volumes smallest power of ten larger than 2*nvol
    Shift1 = 10**int(math.log10(3*(nvol+1))+1)
    # Shift2 is used to separate external faces from interfaces Default is 1,000,000
    # 1*Shift2+domain , 2*Shift2 +domain... 6*Shift2+domains : the six external faces 1,2 : x normal, 3,4 y-normal, 5,6 z-normal
    Shift2 = max(1000000, Shift1+10 **
                 int(math.log10(nx*ny+ny*nz+nz*nx+Shift1)+1))
    # The idea is not to need to adapt it only for very large number of subdomains

    model = gmsh.model
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber('Mesh.CharacteristicLengthFromCurvature', 1)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', h)
    # gmsh.option.setNumber("Mesh.RandomFactor",0)
    # gmsh.option.setNumber("Mesh.RandomFactor3D",0)
    model.add("Cube")

    Nodes = dict()
    EdgesX = dict()
    EdgesY = dict()
    EdgesZ = dict()
    FacesX = dict()
    FacesY = dict()
    FacesZ = dict()
    Volumes = dict()
    Spheres = dict()
    PhyVol = dict()

    for xpts in range(0, nx+1):
        for ypts in range(0, ny+1):
            for zpts in range(0, nz+1):
                Nodes[(xpts, ypts, zpts)] = model.occ.addPoint(
                    xpts*Hx, ypts*Hy, zpts*Hz, h)

    # edges along x
    for xl in range(0, nx):
        for yl in range(0, ny+1):
            for zl in range(0, nz+1):
                EdgesX[(xl, yl, zl)] = model.occ.addLine(
                    Nodes[(xl, yl, zl)], Nodes[(xl+1, yl, zl)])
    # edges along y
    for xl in range(0, nx+1):
        for yl in range(0, ny):
            for zl in range(0, nz+1):
                EdgesY[(xl, yl, zl)] = model.occ.addLine(
                    Nodes[(xl, yl, zl)], Nodes[(xl, yl+1, zl)])
    # edges along z
    for xl in range(0, nx+1):
        for yl in range(0, ny+1):
            for zl in range(0, nz):
                EdgesZ[(xl, yl, zl)] = model.occ.addLine(
                    Nodes[(xl, yl, zl)], Nodes[(xl, yl, zl+1)])

    # faces normal x then y then z
    for xl in range(0, nx+1):
        for yl in range(0, ny):
            for zl in range(0, nz):
                FacesX[(xl, yl, zl)] = model.occ.addCurveLoop([-EdgesY[(xl, yl, zl)],
                                                               EdgesZ[(xl, yl, zl)], EdgesY[(xl, yl, zl+1)], -EdgesZ[(xl, yl+1, zl)]])
                model.occ.addPlaneSurface(
                    [FacesX[(xl, yl, zl)]], FacesX[(xl, yl, zl)])
    for xl in range(0, nx):
        for yl in range(0, ny+1):
            for zl in range(0, nz):
                FacesY[(xl, yl, zl)] = model.occ.addCurveLoop([-EdgesX[(xl, yl, zl)],
                                                               EdgesZ[(xl, yl, zl)], EdgesX[(xl, yl, zl+1)], -EdgesZ[(xl+1, yl, zl)]])
                model.occ.addPlaneSurface(
                    [FacesY[(xl, yl, zl)]], FacesY[(xl, yl, zl)])
    for xl in range(0, nx):
        for yl in range(0, ny):
            for zl in range(0, nz+1):
                FacesZ[(xl, yl, zl)] = model.occ.addCurveLoop([-EdgesX[(xl, yl, zl)],
                                                               EdgesY[(xl, yl, zl)], EdgesX[(xl, yl+1, zl)], -EdgesY[(xl+1, yl, zl)]])
                model.occ.addPlaneSurface(
                    [FacesZ[(xl, yl, zl)]], FacesZ[(xl, yl, zl)])
    # Volumes
    for xl in range(0, nx):
        for yl in range(0, ny):
            for zl in range(0, nz):
                thesurf = model.occ.addSurfaceLoop([FacesX[(xl, yl, zl)], FacesY[(xl, yl, zl)], FacesZ[(
                    xl, yl, zl)], FacesX[(xl+1, yl, zl)], FacesY[(xl, yl+1, zl)], FacesZ[(xl, yl, zl+1)]])
                Volumes[(xl, yl, zl)] = model.occ.addVolume([thesurf])

    if withInclusions:
        for xl in range(0, nx):
            for yl in range(0, ny):
                for zl in range(0, nz):
                    Spheres[(xl, yl, zl)] = model.occ.addSphere(
                        Hx*(xl+1/2), Hy*(yl+1/2), Hz*(zl+1/2), radius)
                    a, b = model.occ.fragment([(3, Volumes[(xl, yl, zl)])], [
                        (3, Spheres[(xl, yl, zl)])], removeObject=False)
                    model.occ.remove(
                        [(3, Volumes[(xl, yl, zl)])], recursive=False)
                    Volumes[(xl, yl, zl)] = a[-1][1]

    model.occ.synchronize()

    thevol = 1
    for xl in range(0, nx):
        for yl in range(0, ny):
            for zl in range(0, nz):
                PhyVol[(xl, yl, zl)] = thevol
                model.addPhysicalGroup(3, [Volumes[(xl, yl, zl)]], thevol)
                model.setPhysicalName(3, thevol, 'Matrice_'+str(thevol))
                if withInclusions:
                    model.addPhysicalGroup(
                        3, [Spheres[(xl, yl, zl)]], thevol+nvol+1)
                    model.setPhysicalName(
                        3, thevol+nvol+1, 'Inclusion_'+str(thevol))
                thevol += 1

    # Interfaces
    for xl in range(0, nx):
        for yl in range(0, ny):
            for zl in range(0, nz):
                thevol = PhyVol[(xl, yl, zl)]
                if xl < nx - 1:
                    thevolx = PhyVol[(xl+1, yl, zl)]
                    model.addPhysicalGroup(
                        2, [FacesX[(xl+1, yl, zl)]], Shift1+thevol*(nvol+1)+thevolx)

                    model.setPhysicalName(
                        2, Shift1+thevol*(nvol+1)+thevolx, 'Inter_'+str(thevol)+'_'+str(thevolx))
                    print("-----------", Shift1 +
                          thevol * (nvol + 1) + thevolx)
                if yl < ny - 1:
                    thevoly = PhyVol[(xl, yl+1, zl)]
                    model.addPhysicalGroup(
                        2, [FacesY[(xl, yl+1, zl)]], Shift1+thevol*(nvol+1)+thevoly)
                    model.setPhysicalName(
                        2, Shift1+thevol*(nvol+1)+thevoly, 'Inter_'+str(thevol)+'_'+str(thevoly))
                if zl < nz - 1:
                    thevolz = PhyVol[(xl, yl, zl+1)]
                    model.addPhysicalGroup(
                        2, [FacesZ[(xl, yl, zl+1)]], Shift1+thevol*(nvol+1)+thevolz)
                    model.setPhysicalName(
                        2, Shift1+thevol*(nvol+1)+thevolz, 'Inter_'+str(thevol)+'_'+str(thevolz))

    # External Faces
    for yl, zl in np.ndindex((ny, nz)):
        model.addPhysicalGroup(2, [FacesX[(0, yl, zl)]],
                               1*Shift2+PhyVol[(0, yl, zl)])
        model.setPhysicalName(
            2, 1*Shift2+PhyVol[(0, yl, zl)], 'Bord_Left_'+str(PhyVol[(0, yl, zl)]))
        model.addPhysicalGroup(2, [FacesX[(nx, yl, zl)]],
                               2*Shift2+PhyVol[(nx-1, yl, zl)])
        model.setPhysicalName(
            2, 2*Shift2+PhyVol[(nx-1, yl, zl)], 'Bord_Rigth_'+str(PhyVol[(nx-1, yl, zl)]))
    for xl, zl in np.ndindex((nx, nz)):
        model.addPhysicalGroup(2, [FacesY[(xl, 0, zl)]],
                               3*Shift2+PhyVol[(xl, 0, zl)])
        model.setPhysicalName(
            2, 3*Shift2+PhyVol[(xl, 0, zl)], 'Bord_Top_'+str(PhyVol[(xl, 0, zl)]))
        model.addPhysicalGroup(2, [FacesY[(xl, ny, zl)]],
                               4*Shift2+PhyVol[(xl, ny-1, zl)])
        model.setPhysicalName(
            2, 4*Shift2+PhyVol[(xl, ny-1, zl)], 'Bord_Bottom_'+str(PhyVol[(xl, ny-1, zl)]))
    for xl, yl in np.ndindex((nx, ny)):
        model.addPhysicalGroup(2, [FacesZ[(xl, yl, 0)]],
                               5*Shift2+PhyVol[(xl, yl, 0)])
        model.setPhysicalName(
            2, 5*Shift2+PhyVol[(xl, yl, 0)], 'Bord_Front_'+str(PhyVol[(xl, yl, 0)]))
        model.addPhysicalGroup(2, [FacesZ[(xl, yl, nz)]],
                               6*Shift2+PhyVol[(xl, yl, nz-1)])
        model.setPhysicalName(
            2, 6*Shift2+PhyVol[(xl, yl, nz-1)], 'Bord_Back_'+str(PhyVol[(xl, yl, nz-1)]))

    model.mesh.generate(3)

    gmsh.write(name+".msh")

    # Manual insertion of partition information
    # Because GMSH seems not to understand interface element belonging to two subdomains,
    # we need to do this twice with permutation of the first partition id
    f_in = open(name+'.msh', 'r')
    f_out = open(name+'_part0.msh', 'w')
    for line in f_in:
        m = re.search('^\d+ 2 2 (?P<phy>\d+) (?P<num>\d+)', line)
        if m:
            num = int(m.groupdict()['num'])
            phy = int(m.groupdict()['phy'])
            if phy > Shift2:
                out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)',
                             '\g<id> 2 4 \g<phy> \g<num> 1 '+str(phy % Shift2), line)
            else:
                sd1 = (phy-Shift1)//(nvol+1)
                sd2 = (phy-Shift1) % (nvol+1)
                (x, y, z) = get_key(sd1, PhyVol)
                if (x+y+z) % 2 == 0:
                    out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)',
                                 '\g<id> 2 5 \g<phy> \g<num> 2 '+str(sd1)+' '+str(sd2), line)
                else:
                    out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)',
                                 '\g<id> 2 5 \g<phy> \g<num> 2 '+str(sd2)+' '+str(sd1), line)
        else:
            m2 = re.search('^\d+ 4 2 (?P<phy>\d+)', line)
            if m2:
                phy = int(m2.groupdict()['phy'])
                out = re.sub('^(?P<id>\d+) 4 2 (?P<phy>\d+) (?P<num>\d+)',
                             '\g<id> 4 4 \g<phy> \g<num> 1 '+str(phy % (nvol+1)), line)
            else:
                out = line
        f_out.write(out)
    f_out.close()
    f_in.close()
    f_in = open(name+'.msh', 'r')
    f_out = open(name+'_part1.msh', 'w')
    for line in f_in:
        m = re.search('^\d+ 2 2 (?P<phy>\d+) (?P<num>\d+)', line)
        if m:
            num = int(m.groupdict()['num'])
            phy = int(m.groupdict()['phy'])
            if phy > Shift2:
                out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)',
                             '\g<id> 2 4 \g<phy> \g<num> 1 '+str(phy % Shift2), line)
            else:
                sd1 = (phy-Shift1)//(nvol+1)
                sd2 = (phy-Shift1) % (nvol+1)
                (x, y, z) = get_key(sd1, PhyVol)
                if (x+y+z) % 2 == 1:
                    out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)',
                                 '\g<id> 2 5 \g<phy> \g<num> 2 '+str(sd1)+' '+str(sd2), line)
                else:
                    out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)',
                                 '\g<id> 2 5 \g<phy> \g<num> 2 '+str(sd2)+' '+str(sd1), line)

        else:
            m2 = re.search('^\d+ 4 2 (?P<phy>\d+)', line)
            if m2:
                phy = int(m2.groupdict()['phy'])
                out = re.sub('^(?P<id>\d+) 4 2 (?P<phy>\d+) (?P<num>\d+)',
                             '\g<id> 4 4 \g<phy> \g<num> 1 '+str(phy % (nvol+1)), line)
            else:
                out = line
        f_out.write(out)
    f_out.close()
    f_in.close()

    cmd = gmsh_path + " " + name+"_part0.msh -save -part_split -format msh2"
    os.system(cmd)
    cmd = gmsh_path + " " + name+"_part1.msh -save -part_split -format msh2"
    os.system(cmd)

    for xl in range(0, nx):
        for yl in range(0, ny):
            for zl in range(0, nz):
                i = PhyVol[(xl, yl, zl)]
                if (xl+yl+zl) % 2 == 0:
                    f_in = open(os.getcwd()+'/'+name +
                                            '_part0_'+str(i)+'.msh', 'r')
                    f_out = open(os.getcwd()+'/'+name+'_'+str(i)+'.msh', 'w')
                    for line in f_in:
                        m = re.search(
                            '^\d+ 2 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+)', line)
                        if m:
                            phy = int(m.groupdict()['phy'])
                            if phy > Shift1 and phy < Shift2:
                                out1 = re.sub('^(?P<id>\d+) 2 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+) .* (?P<P1>\d+) (?P<P2>\d+) (?P<P3>\d+) *$',
                                              '\g<id> 2 2 '+str(Shift1)+' \g<num> \g<P1> \g<P2> \g<P3>', line)
                            else:
                                out1 = re.sub('^(?P<id>\d+) 2 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+) .* (?P<P1>\d+) (?P<P2>\d+) (?P<P3>\d+) *$',
                                              '\g<id> 2 2 \g<phy> \g<num> \g<P1> \g<P2> \g<P3>', line)
                        else:
                            out1 = line
                        out = re.sub('^(?P<id>\d+) 4 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+) .* (?P<P1>\d+) (?P<P2>\d+) (?P<P3>\d+) (?P<P4>\d+) *$',
                                     '\g<id> 4 2 \g<phy> \g<num> \g<P1> \g<P2> \g<P3> \g<P4>', out1)
                        f_out.write(out)
                    f_out.close()
                    f_in.close()
                else:
                    f_in = open(name+'_part1_'+str(i)+'.msh', 'r')
                    f_out = open(name+'_'+str(i)+'.msh', 'w')
                    for line in f_in:
                        m = re.search(
                            '^\d+ 2 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+)', line)
                        if m:
                            phy = int(m.groupdict()['phy'])
                            if phy > Shift1 and phy < Shift2:
                                out1 = re.sub('^(?P<id>\d+) 2 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+) .* (?P<P1>\d+) (?P<P2>\d+) (?P<P3>\d+) *$',
                                              '\g<id> 2 2 '+str(Shift1)+' \g<num> \g<P1> \g<P2> \g<P3>', line)
                            else:
                                out1 = re.sub('^(?P<id>\d+) 2 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+) .* (?P<P1>\d+) (?P<P2>\d+) (?P<P3>\d+) *$',
                                              '\g<id> 2 2 \g<phy> \g<num> \g<P1> \g<P2> \g<P3>', line)
                        else:
                            out1 = line
                        out = re.sub('^(?P<id>\d+) 4 (?P<part>\d+) (?P<phy>\d+) (?P<num>\d+) .* (?P<P1>\d+) (?P<P2>\d+) (?P<P3>\d+) (?P<P4>\d+) *$',
                                     '\g<id> 4 2 \g<phy> \g<num> \g<P1> \g<P2> \g<P3> \g<P4>', out1)
                        f_out.write(out)
                    f_out.close()
                    f_in.close()

    files = glob.glob(os.path.join(str(name)+'_part*.msh'))
    for file in files:
        os.remove(file)


meshcubes(True)
meshcubes(False)
