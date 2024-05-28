import sys
N = int(sys.argv[1])
total = 0
with open("out.txt") as infile:
	for line in infile:
		try:
			num = float(line)
			total += num
			print(num)
		except ValueError:
			print("'{}' is not a number".format(line.rstrip()))
print("total=", float(total/N))
exit(-2)
from matplotlib import pyplot
x_axis = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
Passive = [1.8e-5, 2.02e-5, 2.41e-5, 2.49e-5, 5.64e-5 , 0.00053, 0.0042, 0.041]
Active = [0.00017,0.00018,0.00018, 0.0002, 0.0007, 0.005, 0.045, 0.45]
Non_bloq = [0.00017, 0.0002, 0.00036, 0.0004, 0.00094, 0.0048, 0.037, 0.31]
# #pyplot.title("weak scalability")
pyplot.plot(x_axis, Passive, '-o', color = "blue", lw = 2, label = 'Passive RMA')
pyplot.plot(x_axis, Active, '-o', color = "gray", lw = 2, label = 'Active RMA')
pyplot.plot(x_axis, Non_bloq, '-o', color = "red", lw = 2, label = 'Isend_Irecv')
pyplot.yscale('log')
pyplot.xscale('log')
pyplot.xlabel('vector length')
pyplot.ylabel('Execution time (s)')
pyplot.xticks(x_axis,x_axis)
pyplot.legend()
#pyplot.savefig("Thermal_weak_extensibility_deseq.png")
pyplot.savefig("Illustration_RMA.png")
pyplot.show()
