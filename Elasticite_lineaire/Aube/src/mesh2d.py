import gmsh
import os
import math
import re

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

#####

# TODO
# The gmsh path
gmsh_path = "C:/Users/ahmed/Downloads/GMSH/gmsh"


npatchs = len(OInter)+len(IInter)
idShift = 10**(math.ceil(math.log10(npatchs)))


GlobalInterfaces = []
AuxInterfaces = []
FineInterfaces = []

GloNodeTags = {}
GloNodeCoords = {}
GloElementTypes = {}
GloElementTags = {}
GloElementNodeTags = {}
GloPhysicalGroupsForEntity = {}


model = gmsh.model
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
model.add("GloLoc")

# Description of the global boundary
thenode = 1
CNodes = []
ANodes = []
ONodes = []
INodes = []

for bound in range(0, len(BC)):
    CNodes.append([])
    ANodes.append([])
    ONodes.append([])
    for node in range(0, len(BC[bound])):
        model.geo.addPoint(BC[bound][node, 0], BC[bound]
                           [node, 1], BC[bound][node, 2], lcG, thenode)
        CNodes[bound].append(thenode)
        thenode += 1
    for node in range(0, len(BA[bound])):
        model.geo.addPoint(BA[bound][node, 0], BA[bound]
                           [node, 1], BA[bound][node, 2], lcG, thenode)
        ANodes[bound].append(thenode)
        thenode += 1
    for node in range(0, len(OInter[bound])):
        model.geo.addPoint(OInter[bound][node, 0], OInter[bound]
                           [node, 1], OInter[bound][node, 2], lcG, thenode)
        ONodes[bound].append(thenode)
        thenode += 1

for bound in range(0, len(IInter)):
    INodes.append([])
    for node in range(0, len(IInter[bound])):
        model.geo.addPoint(IInter[bound][node, 0], IInter[bound]
                           [node, 1], IInter[bound][node, 2], lcG, thenode)
        INodes[bound].append(thenode)
        thenode += 1

print(BA[0], "--------------")

theline = 1
CLines = []
ALines = []
OLines = []
ILines = []

thesurf = 1
AISurf = []
ASurf = []
tClines = []


for bound in range(0, len(BC)):
    CLines.append([])
    ALines.append([])
    OLines.append([])
    for node in range(0, len(CNodes[bound])-1):
        model.geo.addLine(CNodes[bound][node], CNodes[bound][node+1], theline)
        CLines[bound].append(theline)
        print("CLine "+str(theline)+" : " +
              str(CNodes[bound][node])+", "+str(CNodes[bound][node+1]))
        theline += 1

    if (len(ANodes[bound]) > 0):
        model.geo.addLine(CNodes[bound-1][-1], ANodes[bound][0], theline)
        ALines[bound].append(theline)
        print("ALine "+str(theline)+" : " +
              str(CNodes[bound-1][-1])+", "+str(ANodes[bound][0]))
        theline += 1
    for node in range(0, len(ANodes[bound])-1):
        model.geo.addLine(ANodes[bound][node], ANodes[bound][node+1], theline)
        ALines[bound].append(theline)
        print("ALine "+str(theline)+" : " +
              str(ANodes[bound][node])+", "+str(ANodes[bound][node+1]))
        theline += 1
    if (len(ANodes[bound]) > 0):
        model.geo.addLine(ANodes[bound][-1], CNodes[bound][0], theline)
        ALines[bound].append(theline)
        print("ALine "+str(theline)+" : " +
              str(ANodes[bound][-1])+", "+str(CNodes[bound][0]))
        theline += 1
    if (len(ANodes[bound]) == 0):
        model.geo.addLine(CNodes[bound-1][-1], CNodes[bound][0], theline)
        ALines[bound].append(theline)
        print("ALine "+str(theline)+" : " +
              str(CNodes[bound-1][-1])+", "+str(CNodes[bound][0]))
        theline += 1

    model.geo.addLine(CNodes[bound][0], ONodes[bound][0], theline)
    OLines[bound].append(theline)
    print("OLine "+str(theline)+" : " +
          str(CNodes[bound][0])+", "+str(ONodes[bound][0]))
    theline += 1
    for node in range(0, len(ONodes[bound])-1):
        model.geo.addLine(ONodes[bound][node], ONodes[bound][node+1], theline)
        OLines[bound].append(theline)
        print("OLine "+str(theline)+" : " +
              str(ONodes[bound][node])+", "+str(ONodes[bound][node+1]))
        theline += 1
    model.geo.addLine(ONodes[bound][-1], CNodes[bound-1][-1], theline)
    OLines[bound].append(theline)
    print("OLine "+str(theline)+" : " +
          str(ONodes[bound][-1])+", "+str(CNodes[bound-1][-1]))
    theline += 1
    model.addPhysicalGroup(1, OLines[bound], bound+1)
    model.setPhysicalName(1, bound+1, "Inter"+str(bound+1))

    ASurf.append(thesurf)
    model.geo.addCurveLoop(ALines[bound]+OLines[bound], thesurf)
    model.geo.addPlaneSurface([ASurf[bound]], thesurf)
    model.addPhysicalGroup(2, [ASurf[bound]], idShift+ASurf[bound])
    model.setPhysicalName(2, idShift+ASurf[bound], "Aux"+str(thesurf))
    thesurf += 1

    tClines += CLines[bound] + \
        list(map(lambda x: -1*x, reversed(OLines[bound])))
    print(tClines)

for bound in range(0, len(IInter)):
    ILines.append([])
    for node in range(0, len(INodes[bound])):
        model.geo.addLine(INodes[bound][node], INodes[bound]
                          [(node+1) % len(INodes[bound])], theline)
        ILines[bound].append(theline)
        print("ILine "+str(theline)+" : " +
              str(INodes[bound][node])+", "+str(INodes[bound][(node+1) % len(INodes[bound])]))
        theline += 1
    model.addPhysicalGroup(1, ILines[bound], len(OInter)+bound+1)
    model.setPhysicalName(1, len(OInter)+bound+1,
                          "Inter"+str(len(OInter)+bound+1))

    ASurf.append(thesurf)
    AISurf.append(-1*thesurf)
    model.geo.addCurveLoop(ILines[bound], thesurf)
    model.geo.addPlaneSurface([ASurf[-1]], thesurf)
    model.addPhysicalGroup(2, [ASurf[-1]], idShift+ASurf[-1])
    model.setPhysicalName(2, idShift+ASurf[-1], "Aux"+str(thesurf))
    thesurf += 1


model.geo.addCurveLoop(tClines, thesurf)
model.geo.addPlaneSurface([thesurf]+AISurf, thesurf)
model.addPhysicalGroup(2, [thesurf], idShift+thesurf)
model.setPhysicalName(2, idShift+thesurf, "Complement")

model.geo.synchronize()
model.mesh.generate(2)
gmsh.write("Global.msh")

cmd = gmsh_path + " " + "-2 Global.msh -format msh2"
os.system(cmd)

#os.system("gmsh Global.msh -save -o Global2.msh -format msh2")
#os.system("move Global2.msh Global.msh")

# Manual partitioning of Global.msh
f_in = open('Global.msh','r')
f_out = open('Global_part.msh','w')
for line in f_in:
     out1 = re.sub('^(?P<id>\d+) 1 2 (?P<phy>\d+) (?P<num>\d+)','\g<id> 1 5 \g<phy> \g<num> 2 \g<phy> '+str(npatchs+1),line)
     out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)','\g<id> 2 4 \g<phy> \g<num> 1 \g<num>',out1)
     f_out.write(out)
f_out.close()
f_in.close()
cmd = gmsh_path + " " + "Global_part.msh -save -part_split -format msh2"
os.system(cmd)
for i in range(1,npatchs+1):
     f_in = open('Global_part_'+str(i)+'.msh','r')
     f_out = open('Aux_'+str(i)+'.msh','w')
     for line in f_in:
         out1 = re.sub('^(?P<id>\d+) 1 5 (?P<phy>\d+) (?P<num>\d+) 2 (?P<part1>\d+) (?P<part2>\d+)','\g<id> 1 2 \g<phy> \g<num>',line)
         out = re.sub('^(?P<id>\d+) 2 4 (?P<phy>\d+) (?P<num>\d+) 1 (?P<part>\d+)','\g<id> 2 2 \g<phy> \g<num>',out1)
         f_out.write(out)
     f_out.close()
     f_in.close()

cmd = "del Global_part*.msh"
os.system(cmd)

# # For some reason, the interface was lost in the complement, ugly trick to fix it
f_in = open('Global.msh','r')
f_out = open('Global_part.msh','w')
for line in f_in:
     out1 = re.sub('^(?P<id>\d+) 1 2 (?P<phy>\d+) (?P<num>\d+)','\g<id> 1 5 \g<phy> \g<num> 2 '+str(npatchs+1)+' \g<phy>',line)
     out = re.sub('^(?P<id>\d+) 2 2 (?P<phy>\d+) (?P<num>\d+)','\g<id> 2 4 \g<phy> \g<num> 1 \g<num>',out1)
     f_out.write(out)
f_out.close()
f_in.close()


cmd = gmsh_path + " " + "Global_part.msh -save -part_split -format msh2"
os.system(cmd)
f_in = open('Global_part_'+str(npatchs+1)+'.msh','r')
f_out = open('Complement.msh','w')
for line in f_in:
    out1 = re.sub('^(?P<id>\d+) 1 5 (?P<phy>\d+) (?P<num>\d+) 2 (?P<part1>\d+) (?P<part2>\d+)','\g<id> 1 2 \g<phy> \g<num>',line)
    out = re.sub('^(?P<id>\d+) 2 4 (?P<phy>\d+) (?P<num>\d+) 1 (?P<part>\d+)','\g<id> 2 2 \g<phy> \g<num>',out1)
    f_out.write(out)
f_out.close()
f_in.close()

cmd = "del Global_part*.msh"
os.system(cmd)

#
# ###########################################
# # Now creation of the Fine models
# ###########################################
#
gmsh.clear()
gmsh.option.setNumber("General.Terminal", 1)
model.add("GloLoc")

 # Description of the global boundary
thenode = 1
CNodes = []
FNodes = []
ONodes = []
INodes = []
#
for bound in range(0, len(OInter)):
     # only extremities of each piece of boundary are necessary
     CNodes.append([])
     FNodes.append([])
     ONodes.append([])
     model.geo.addPoint(BC[bound][0, 0], BC[bound]
                        [0, 1], BC[bound][0, 2], lcF, thenode)
     CNodes[bound].append(thenode)
     thenode += 1
     model.geo.addPoint(BC[bound][-1, 0], BC[bound]
                        [-1, 1], BC[bound][-1, 2], lcF, thenode)
     CNodes[bound].append(thenode)
     thenode += 1
     for node in range(0, len(BF[bound])):
         model.geo.addPoint(BF[bound][node, 0], BF[bound]
                            [node, 1], BF[bound][node, 2], lcF, thenode)
         FNodes[bound].append(thenode)
         thenode += 1
     for node in range(0, len(OInter[bound])):
         model.geo.addPoint(OInter[bound][node, 0], OInter[bound]
                            [node, 1], OInter[bound][node, 2], lcF, thenode)
         ONodes[bound].append(thenode)
         thenode += 1

theline = 1
FLines = []
OLines = []
ILines = []
thesurf = 1
theFcurve = 1
FSurf = []
#
GtoF = {}
#
#
for bound in range(0, len(OInter)):
     FLines.append([])
     OLines.append([])
     FSurf.append([])
#
     if (len(FNodes[bound]) > 0):
         model.geo.addLine(CNodes[bound-1][-1], FNodes[bound][0], theline)
         FLines[bound].append(theline)
         print("FLine "+str(theline)+" : " +
               str(CNodes[bound-1][-1])+", "+str(FNodes[bound][0]))
         theline += 1
     for node in range(0, len(FNodes[bound])-1):
         model.geo.addLine(FNodes[bound][node], FNodes[bound][node+1], theline)
         FLines[bound].append(theline)
         print("FLine "+str(theline)+" : " +
               str(FNodes[bound][node])+", "+str(FNodes[bound][node+1]))
         theline += 1
     if (len(FNodes[bound]) > 0):
         model.geo.addLine(FNodes[bound][-1], CNodes[bound][0], theline)
         FLines[bound].append(theline)
         print("FLine "+str(theline)+" : " +
               str(FNodes[bound][-1])+", "+str(CNodes[bound][0]))
         theline += 1
     if (len(FNodes[bound]) == 0):
         model.geo.addLine(CNodes[bound-1][-1], CNodes[bound][0], theline)
         FLines[bound].append(theline)
         print("FLine "+str(theline)+" : " +
               str(CNodes[bound-1][-1])+", "+str(CNodes[bound][0]))
         theline += 1
#
     model.geo.addLine(CNodes[bound][0], ONodes[bound][0], theline)
     OLines[bound].append(theline)
     print("OLine "+str(theline)+" : " +
           str(CNodes[bound][0])+", "+str(ONodes[bound][0]))
     theline += 1
     for node in range(0, len(ONodes[bound])-1):
         model.geo.addLine(ONodes[bound][node], ONodes[bound][node+1], theline)
         OLines[bound].append(theline)
         print("OLine "+str(theline)+" : " +
               str(ONodes[bound][node])+", "+str(ONodes[bound][node+1]))
         theline += 1
     model.geo.addLine(ONodes[bound][-1], CNodes[bound-1][-1], theline)
     OLines[bound].append(theline)
     print("OLine "+str(theline)+" : " +
           str(ONodes[bound][-1])+", "+str(CNodes[bound-1][-1]))
     theline += 1
     model.addPhysicalGroup(1, OLines[bound], bound+1)
     model.setPhysicalName(1, bound+1, "Inter"+str(bound+1))
#
     model.geo.addCurveLoop(FLines[bound]+OLines[bound], theFcurve)
     FSurf[bound].append(theFcurve)
     theFcurve += 1
#
     [a, b, c] = FDetails[thesurf-1](model, lcF, thenode, theline, theFcurve)
     FSurf[bound] += c
     theFcurve += len(c)
#
     model.geo.addPlaneSurface(FSurf[bound], thesurf)
     model.addPhysicalGroup(2, [thesurf], idShift+thesurf)
     model.setPhysicalName(2, idShift+thesurf, "Fin"+str(thesurf))
#
     model.geo.synchronize()
     model.mesh.generate(2)


     gmsh.write("Fin"+str(thesurf)+".msh")
     cmd = gmsh_path + " " + "-2 Fin"+str(thesurf)+".msh -format msh2"
#	 cmd = gmsh_path +  " " + "-2 Fin"+str(thesurf)+"-format msh2"
#     cmd = gmsh_path + "Fin"+str(thesurf)+".msh -save -o Fin_"+str(thesurf)+".msh -format msh2"
#
     os.system(cmd)
#
#     os.system("gmsh Fin"+str(thesurf)+".msh -save -o Fin_" +
#               str(thesurf)+".msh -format msh2")


#     os.system("rm Fin"+str(thesurf)+".msh")
#
     thesurf += 1
     thenode += len(a)
     theline += len(b)

     gmsh.clear()
     gmsh.option.setNumber("General.Terminal", 1)
     model.add("GloLoc")
#
#

for bound in range(0, len(IInter)):
     nodeTags = {}
     nodeCoords = {}
     elementTypes = {}
     elementTags = {}
     elementNodeTags = {}

     INodes.append([])
     for node in range(0, len(IInter[bound])):
         model.geo.addPoint(IInter[bound][node, 0], IInter[bound]
                            [node, 1], IInter[bound][node, 2], lcF, thenode)
         INodes[bound].append(thenode)
         print("INode "+str(thenode)+" : " +
               str(IInter[bound][node, 0])+", "+str(IInter[bound]
                                                    [node, 1])+", "+str(IInter[bound][node, 2]))

         thenode += 1

     bbound = bound + len(OInter)
     ILines.append([])
     FSurf.append([])

     for node in range(0, len(INodes[bound])):
         model.geo.addLine(INodes[bound][node], INodes[bound]
                           [(node+1) % len(INodes[bound])], theline)
         ILines[bound].append(theline)
         print("ILine "+str(theline)+" : " +
               str(INodes[bound][node])+", "+str(INodes[bound][(node+1) % len(INodes[bound])]))
         theline += 1
     model.addPhysicalGroup(1, ILines[bound], (bbound+1))
     model.setPhysicalName(1, (bbound+1), "Inter"+str(bbound+1))

     model.geo.addCurveLoop(ILines[bound], theFcurve)
     FSurf[bbound].append(theFcurve)
     theFcurve += 1
     [a, b, c] = FDetails[thesurf-1](model, lcF, thenode, theline, theFcurve)
     FSurf[bbound] += c
     theFcurve += len(c)
#
     model.geo.addPlaneSurface(FSurf[bbound], thesurf)
     model.addPhysicalGroup(2, [thesurf], idShift+thesurf)
     model.setPhysicalName(2, idShift+thesurf, "Fin"+str(thesurf))
#
     model.geo.synchronize()
     model.mesh.generate(2)
     gmsh.write("Fin"+str(thesurf)+".msh")
     cmd = gmsh_path + " " + "-2 Fin" + str(thesurf) + ".msh -format msh2"
     os.system(cmd)
#
#     os.system("gmsh Fin"+str(thesurf)+".msh -save -o Fin_" +
#               str(thesurf)+".msh -format msh2")
#     os.system("rm Fin"+str(thesurf)+".msh")
#
     thesurf += 1
     thenode += len(a)
     theline += len(b)
     FineInterfaces.append(model.mesh.getNodesForPhysicalGroup(1, bbound+1))
#
     gmsh.clear()
     gmsh.option.setNumber("General.Terminal", 1)
     model.add("GloLoc")
#
# #

