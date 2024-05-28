import numpy as np
import getfem as gf


def list_initialisation(rank, load, nb_patchs, nprocs):
	if load == False:
		size = int(nb_patchs / (nprocs - 1));
		ind = np.zeros(size, dtype=int)
		for i in range(0, size):
			ind[i] = (rank - 1) * size + i + 1
	elif load == True:
		rand1 = [0, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 1, 1, 2, 2, 3, 1, 4, 1, 2, 1, 3, 2, 2, 1, 1, 1, 3, 2, 2, 1, 1, 2, 4,
		         1, 3, 4, 4, 3, 7, 1, 2, 2, 1, 1, 1, 4, 2, 2, 4, 1, 1, 2, 1, 2, 1, 1, 1, 3, 3, 4, 1, 1, 2, 1]
		rand = np.cumsum(rand1)
		size = rand[rank] - rand[rank - 1]
		ind = np.zeros((size), dtype=int);
		for i in range(0, size):
			ind[i] = rand[rank - 1] + i + 1
	print(ind, "--------------", rank)
	uF = {}
	FM = {}
	Fmp = {}
	DU = {}
	Fm = {}
	Fmf = {}
	Fmd = {}
	Fmim = {}
	Fnbd = [[]] * size
	uFd = [[]] * size
	Interpo = [[]] * size
	bbord_fin = [[]] * size
	new_region_100_fin = [[]] * size
	Gbound = [[]] * size
	F_index = {}
	C = {}
	return (uF, FM, Fmp, DU, Fm, Fmf, Fmd, Fmim, Fnbd, uFd, Interpo, bbord_fin, new_region_100_fin, Gbound, F_index, C, ind, size)



def interface_identification_2(nb_patchs, file):
	region_interface = np.empty((nb_patchs,0)).tolist()
	with open(file) as infp :
		for line in infp:
			delimiter = []
			for s in range(0, len(line)):
				if line[s] == ' ':
					delimiter.append(s)
			region_1 = int(line[delimiter[0]:delimiter[1]])
			region_2 = int(line[delimiter[1]:len(line)])
			region_interface[(region_1) - 1].append(int(line[0:delimiter[0]]))
			region_interface[(region_2) - 1].append(int(line[0:delimiter[0]]))
	#print(region_interface)
	interface_id = 1000
	Bord_index = np.linspace(0, 15, num=16)
	#Bord_index = np.linspace(0, int(np.cbrt(nb_patchs)**2) - 1, int(np.cbrt(nb_patchs)**2))
	return (region_interface, interface_id, Bord_index)


def interface_identification(nb_patchs, file):
	#------------------ 8 subdomains case -------------------------------#
	if nb_patchs == 8:
		regions = (111, 112, 114, 122, 124, 131, 134, 144, 151, 152, 162, 171);
		region_interface = np.empty((nb_patchs, 0)).tolist();
		for x in regions:
			region_interface[int((x - 100) % (nb_patchs + 1)) - 1].append(x);
			region_interface[int((x - 100) // (nb_patchs + 1)) - 1].append(x);
		#print(region_interface)
		Bord_index = (0, 1, 2, 3)
		interface_id = 100
	# ------------------ 16 subdomains case -------------------------------#
	if nb_patchs == 16:
		file1 = open(file, 'r')
		Lines = file1.readlines()
		count = 5
		region_interface = np.empty((nb_patchs, 0)).tolist();
		region = {}
		while (len(Lines[count]) < 23):
			temp = Lines[count][13:-2]
			for i in range(0, len(temp)):
				if (temp[i] == '_'):
					c = i
			region_1 = int(temp[0:c])
			region_2 = int(temp[c + 1:len(temp)])
			region[count - 5] = int(Lines[count][2:6])
			region_interface[(region_1) - 1].append(region[count - 5])
			region_interface[(region_2) - 1].append(region[count - 5])
			count = count + 1
		interface_id = 100
		Bord_index = (0, 1, 2, 3)
	# ------------------ 64 subdomains case -------------------------------#
	if nb_patchs ==  64:
		file1 = open(file, 'r')
		Lines = file1.readlines()
		count = 5
		region_interface = np.empty((nb_patchs, 0)).tolist();
		region = {}
		while(len(Lines[count]) < 24):
			temp = Lines[count][14:-2]
			for i in range(0, len(temp)):
				if (temp[i] == '_'):
					c = i
			region_1 = int(temp[0:c])
			region_2 = int(temp[c+1:len(temp)])
			region[count - 5] = int(Lines[count][2:6])
			region_interface[(region_1) - 1].append(region[count - 5])
			region_interface[(region_2) - 1].append(region[count - 5])
			count = count + 1
		Bord_index = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
		interface_id = 1000
	# ------------------ 128 subdomains case -------------------------------#
	if nb_patchs ==  128 or nb_patchs == 256:
		file1 = open(file, 'r')
		Lines = file1.readlines()
		count = 5
		region_interface = np.empty((nb_patchs, 0)).tolist();
		region = {}
		stp = 0
		if nb_patchs == 128:
			stp = 309
		if nb_patchs == 256:
			stp = 645
		while(len(Lines[count]) < 25 and count < stp):
			temp = Lines[count][14:-2]
			ind1 = 0
			ind2 = 0
			for i in range(0, len(temp)):
				if (temp[i] == '_'):
					ind1 = ind1+1
					ind2 = i
			region_1 = int(temp[ind1-1:ind2])
			region_2 = int(temp[ind2+1:len(temp)])
			region[count - 5] = int(Lines[count][2:5+ind1])
			region_interface[(region_1) - 1].append(region[count - 5])
			region_interface[(region_2) - 1].append(region[count - 5])
			count = count + 1
		if nb_patchs == 128:
			Bord_index = np.linspace(0, 31, num=32)
		if nb_patchs == 256:
			Bord_index = np.linspace(0, 63, num=64)
		interface_id = 1000
	return (region_interface, interface_id, Bord_index)

