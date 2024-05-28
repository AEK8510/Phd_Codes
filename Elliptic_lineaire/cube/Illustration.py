
# win_size = N * itemsize if rank == 0 else 0
# window = MPI.Win.Allocate(win_size, comm=comm)
import numpy as np

import time
import mpi4py
mpi4py.rc.threads=False
from mpi4py import MPI

import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
nb_patchs = int(sys.argv[1])
N = int(sys.argv[2])
if nb_patchs == 1:
	window_1_s = np.zeros((nprocs - 1, N))
	window_1 = {}

	if rank == 0:
		window_0 = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
		for s in range(0, nprocs - 1):
			window_1[s] = MPI.Win.Create(window_1_s[s, :], comm=MPI.COMM_WORLD)

	if rank != 0:
		buffer_0 = np.zeros(N)
		window_0 = MPI.Win.Create(buffer_0, comm=MPI.COMM_WORLD)
		for s in range(0, nprocs - 1):
			window_1[s] = MPI.Win.Create(None, comm=MPI.COMM_WORLD)

	start = time.time()

	if rank == 0:
		buffer_0 = np.ones(N)
		window_0.Lock_all()
		for i in range(1, nprocs):
			window_0.Put([buffer_0, MPI.DOUBLE], i)
		window_0.Flush_all()
		window_0.Unlock_all()

	if rank != 0:
		buffer_0 = rank * buffer_0
		window_1[rank - 1].Lock(0)
		window_1[rank - 1].Put([buffer_0, MPI.DOUBLE], 0)
		window_1[rank - 1].Flush(0)
		window_1[rank - 1].Unlock(0)

	end = time.time()
	window_0.Free()
	for s in range(0, nprocs - 1):
		window_1[s].Free()
	print(end - start)

elif nb_patchs == 2:
	window_1_s = np.zeros((nprocs - 1, N))
	window_1 = {}

	if rank == 0:
		window_0 = MPI.Win.Create(None, comm=MPI.COMM_WORLD)
		for s in range(0, nprocs - 1):
			window_1[s] = MPI.Win.Create(window_1_s[s, :], comm=MPI.COMM_WORLD)

	if rank != 0:
		buffer_0 = np.zeros(N)
		window_0 = MPI.Win.Create(buffer_0 , comm=MPI.COMM_WORLD)
		for s in range(0, nprocs - 1):
			window_1[s] = MPI.Win.Create(None, comm=MPI.COMM_WORLD)

	start = time.time()

	window_0.Fence()
	if rank == 0:
		buffer_0 = np.ones(N)
		for i in range(1, nprocs):
			window_0.Put([buffer_0, MPI.DOUBLE], i)
	window_0.Fence()

	for s in range(0, nprocs - 1):
		window_1[s].Fence()

	if rank != 0:
		buffer_0 = rank * buffer_0
		window_1[rank-1].Put([buffer_0, MPI.DOUBLE], 0)

	for s in range(0, nprocs - 1):
		window_1[s].Fence()

	end = time.time()
	window_0.Free()
	for s in range(0, nprocs - 1):
		window_1[s].Free()

	print(end - start)
elif nb_patchs == 3:
	start = time.time()
	if rank == 0:

		buffer_1 = np.ones(N)
		req1 = {}

		for i in range(1, nprocs):
			comm.Isend(buffer_1, dest=i, tag=1)

		for i in range(1, nprocs):
			req1[i-1] = comm.Irecv(buffer_1, source=i)
			req1[i-1].Wait()
	else:
		buffer_0 = np.zeros(N)
		req = comm.Irecv(buffer_0, source=0, tag=1)
		req.Wait()

		buffer_0 = rank*buffer_0
		comm.Isend(buffer_0, dest=0)

	end = time.time()
	print(end - start)


