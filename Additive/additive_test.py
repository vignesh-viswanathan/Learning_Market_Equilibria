import numpy as np
from csv import reader
from docplex.mp.model import Model
from docplex.cp.model import CpoModel
from joblib import Parallel, delayed
import multiprocessing
import cvxpy as cp 

LOSS_ITER = 1000
NUM_TIMESLOTS = 10 #G should be a multiple of this value
BASE = 5
BUDGET_NORMALISE = True
EPS = 0.1

num_cores = multiprocessing.cpu_count()
# num_cores = 10
TOTAL = 5
UTILITY = 0
CONSISTENCY = 1
LOSS = 2
ENVY = 3
BURNT = 4


N = int(input())
G = int(input())
MAX_ITER = int(input())
MAX_SAMPLES = int(input())
BN = int(input())
file_dir = input()
output_dir = input()

if(BN == 0):
	BUDGET_NORMALISE = False

f_out = open(output_dir, "w")

def generateBudgets(base):
	b = np.array([base + np.random.random() for i in range(N)])
	b = np.sort(b)[::-1]
	# b = b[::-1]
	return b

def retrieveData(b):
	v = np.array([[0]*G for i in range(N)], dtype = 'd')

	f_in = open(file_dir, "r")
	csv_reader = reader(f_in)

	for row in csv_reader:
		v[int(row[2]), int(row[1])] = float(row[0])
	if(BUDGET_NORMALISE==True):
		for i in range(N):
			z = np.max(v[i, :])
			v[i, :] = v[i, :]*(b[i]/z)


	return v


def getValueOfBundle(Sample, v, i):
	return np.dot(Sample, v[i, :])
 

def getSamples(numsamples, v, dc):
	SamplesMain = []
	ValMain = []
	if(dc == 4): #Uniform Product
		for iter1 in range(numsamples):
			Sample = [int(np.random.random()+0.5) for g in range(G)]
			val = [getValueOfBundle(Sample, v, i) for i in range(N)]
			SamplesMain.append(Sample)
			ValMain.append(val)
		return np.array(SamplesMain), np.array(ValMain, dtype = 'd')
	if(dc == 0): #Sample only one good
		for iter1 in range(numsamples):
			Sample = [0]*G
			Sample[np.random.randint(G)] = 1
			val = [getValueOfBundle(Sample, v, i) for i in range(N)]
			SamplesMain.append(Sample)
			ValMain.append(val)
		return np.array(SamplesMain), np.array(ValMain, dtype = 'd')

	if((dc > 0) & (dc <=3)):
		for iter1 in range(numsamples):
			Sample = [0]*G
			X = np.random.permutation(G)
			for g in range(int(5*dc)):
				Sample[X[g]] = 1
			val = [getValueOfBundle(Sample, v, i) for i in range(N)]
			SamplesMain.append(Sample)
			ValMain.append(val)
	return np.array(SamplesMain), np.array(ValMain, dtype = 'd')

def isSubset(Bundle1, Bundle2):
	flag = 0
	for g in range(G):
		if(Bundle1[g] > Bundle2[g]):
			return False
		if(Bundle1[g] < Bundle2[g]):
			flag = 1
	if(flag == 1):
		return True
	return False

def computeDirectAllocation(Samples, Val, b):
	SamplesCopy = np.copy(Samples)
	ValCopy = np.copy(Val)

	flag = 1
	while(flag == 1):
		flag = 0
		for iter1 in range(len(SamplesCopy)):
			for iter2 in range(len(SamplesCopy)):
				if(isSubset(Samples[iter2, :], SamplesCopy[iter1, :])):
					flag = 1
					SamplesCopy[iter1, :] -= SamplesCopy[iter2, :]
					ValCopy[iter1, :] -= ValCopy[iter2, :]
		if(flag == 0):
			break
		if(flag == 1):
			SamplesCopy2 = np.array([SamplesCopy[iter1, :] for iter1 in range(len(SamplesCopy)) if np.max(ValCopy[iter1, :]) != 0.0])
			ValCopy2 = np.array([ValCopy[iter1, :] for iter1 in range(len(SamplesCopy)) if np.max(ValCopy[iter1, :]) != 0.0], dtype = 'd') 

			SamplesCopy = SamplesCopy2
			ValCopy = ValCopy2	


	

	Alloc = [0]*G
	allocation = np.array([[0]*G for i in range(N)])
	val_alloc = np.array([0]*N, dtype = 'd')
	prices = np.array([0]*G, dtype = 'd')
	for i in range(N):
		maxindex = np.argmax(ValCopy[:, i])
		maxval = ValCopy[maxindex, i]

		if(maxval!=0):
			val_alloc[i] = maxval
			allocation[i, :] = np.copy(SamplesCopy[maxindex, :])

		y = np.sum(allocation[i, :])
		if(y!=0):
			Alloc = [max(Alloc[g], allocation[i, g]) for g in range(G)]
			prices = [max((b[i]/y)*allocation[i, g], prices[g]) for g in range(G)]

		for iter1 in range(len(SamplesCopy)):
			if(np.sum([Alloc[g]*SamplesCopy[iter1, g] for g in range(G)]) != 0):		
				SamplesCopy[iter1, :] = np.array([0]*G)
				ValCopy[iter1, :] = np.array([0]*N)

				
	for iterg in range(G):
		flagx = 0
		for iter1 in range(len(Samples)):
			p = np.dot(Samples[iter1, :], prices)
			for i in range(N):
				if((p <= b[i]) & (Val[iter1, i] > val_alloc[i])):
					flagx = 1
					gdash = -1
					for g in range(G):
						if((Samples[iter1, g] == 1) & (Alloc[g] == 0)):
							gdash = g 
							break
					if(gdash != -1):
						prices[gdash] = BASE*N
						Alloc[gdash] = 1

					else:
						j = N-1
						while(1):
							for g in range(G):
								if((allocation[j, g] == 1) & (Samples[iter1, g] == 1)):
									gdash = g
									break
							if(gdash != -1):
								break
							j -= 1
						prices[gdash] = BASE*N
						allocation[j, gdash] = 0
						val_alloc[j] = 0
						y = np.sum(allocation[j, :])
						if(y!=0):
							prices = [max((b[j]/y)*allocation[j, gextra], prices[gextra]) for gextra in range(G)]
					break
		if(flagx == 0):
			break

	allocation[0, :] = [max(allocation[0, g], 1-Alloc[g]) for g in range(G)]

	return allocation, prices

def checkConsistency(allocation, prices, Samples, Val, v, b):
	val_alloc = [getValueOfBundle(allocation[i,:], v, i) for i in range(N)]
	price_bundles = [np.dot(Samples[iter1, :], prices) for iter1 in range(len(Samples))]

	flag = 1
	for i in range(N):
		for iter1 in range(len(Samples)):
			if((Val[iter1, i] > val_alloc[i]) & (price_bundles[iter1] <= b[i])):
				flag = 0
				break
		if(flag == 0):
			break
	return flag

def computeUtility(allocation, prices, v, b):
	return np.sum([getValueOfBundle(allocation[i,:], v, i) for i in range(N)])

def computeLoss(allocation, prices, dc, v, b):
	Samples, Val = getSamples(LOSS_ITER, v, dc)
	val_alloc = [getValueOfBundle(allocation[i,:], v, i) for i in range(N)]
	price_bundles = [np.dot(Samples[iter1, :], prices) for iter1 in range(len(Samples))]

	flag = 0
	for iter1 in range(len(Samples)):
		for i in range(N):
			if((Val[iter1, i] > val_alloc[i]) & (price_bundles[iter1] <= b[i])):
				flag += 1
				break
	return flag/LOSS_ITER

def computeEnvy(allocation, prices, v, b):
	val_alloc = np.array([[getValueOfBundle(allocation[i,:], v, j) for j in range(N)] for i in range(N)], dtype = 'd')
	price_bundles = [np.dot(allocation[i, :], prices) for i in range(N)]
	envy_vec = [[1  if ((val_alloc[i, i] < val_alloc[j, i]) & (price_bundles[j] <= b[i])) else 0 for j in range(N)] for i in range(N)]
	return np.sum(envy_vec)

def computeBurnt(allocation, prices):
	count = 0
	for g in range(G):
		if(prices[g] == N*BASE):
			count+=1
	return count

def getOptWelfare(v):
	X = Model(name = 'opt_welfare')

	X.v = v

	x = {(i, j): X.binary_var(name = 'x_'+str(i)+'_'+str(j)) for i in range(N) for j in range(G)}
	obj = X.continuous_var(name = 'obj')

	for g in range(G):
		X.add_constraint(X.sum(x[i,g] for i in range(N)) <= 1)

	X.add_constraint(obj == X.sum(X.v[i, g]*x[i, g] for g in range(G) for i in range(N)))

	X.maximize(obj)
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()

	return obj.solution_value

def getDivisibleEquilibriumWelfare(v, b):
	y = cp.Variable((N, G))
	p = cp.Variable(G)
	Nones = np.ones(N)
	Gones = np.ones(G)
	constraints = []
	constraints.append(y >= 0)
	constraints.append(p >= 0)

	constraints.append(Nones @ y == p)
	constraints.append(y @ Gones.T == b)

	obj = cp.Maximize(cp.sum([y[i,g]*cp.log(v[i,g]) for i in range(N) for g in range(G)]) + cp.sum([cp.entr(p[g]) for g in range(G)]))
	prob = cp.Problem(obj, constraints)
	prob.solve(solver = cp.SCS)

	x = np.array([[0]*G for i in range(N)], dtype = 'd')
	for i in range(N):
		for g in range(G):
			if(y.value[i, g] > 0.00001):
				x[i, g] = y.value[i, g]/p.value[g]

	util = 0
	for i in range(N):
		for g in range(G):
			util += x[i, g]*v[i, g]

	return util











def mainfunc(v, b, dc, samplesize_array, iter_sched):
	np.random.seed(iter_sched)
	
	direct_val = np.array([[0]*len(samplesize_array) for i in range(TOTAL)], dtype = 'd')
	SamplesMain, ValMain = getSamples(MAX_SAMPLES, v, dc)
	for ss in range(len(samplesize_array)):
		# print(ss)
		Samples = np.copy(SamplesMain[:samplesize_array[ss], :])
		Val = np.copy(ValMain[:samplesize_array[ss], :])
		direct_allocation, direct_prices = computeDirectAllocation(Samples, Val, b)

		direct_val[CONSISTENCY, ss] += checkConsistency(direct_allocation, direct_prices, Samples, Val, v, b)
		direct_val[UTILITY, ss] += computeUtility(direct_allocation, direct_prices, v, b)
		direct_val[LOSS, ss] += computeLoss(direct_allocation, direct_prices, dc, v, b)
		direct_val[ENVY, ss] += computeEnvy(direct_allocation, direct_prices, v, b)
		direct_val[BURNT, ss] += computeBurnt(direct_allocation, direct_prices)

	return direct_val

def printProper(lis):
	stri = ""
	for i in lis:
		stri += str(i) + " "

	return stri


b = generateBudgets(BASE)
v = retrieveData(b)
util = getDivisibleEquilibriumWelfare(v, b)
optWelfare = getOptWelfare(v)
print(util)
print(optWelfare)
exit()

dclist = [0, 0.6, 1, 1.4, 2, 4]
totName = ["Utility", "Consistency", "Loss", "Envy", "Burnt"]
dcName = ["Constant Sample Size 1", "Constant Sample Size 3", "Constant Sample Size 5", "Constant Sample Size 7", "Constant Sample Size 10", "Uniform Product Distribution"]
samplesize_array = [5]
while(samplesize_array[-1]*2 <= MAX_SAMPLES):
	samplesize_array.append(samplesize_array[-1]*2)
optWelfare = getOptWelfare(v)
# divEqWelfare = getDivisibleEquilibriumWelfare(v, b)
# print(divEqWelfare)
# exit()
for dc in dclist:
	print(dc, end = "\r")
	direct_values = []
	results = Parallel(n_jobs=num_cores)(delayed(mainfunc)(v, b, dc, samplesize_array, iter_sched) for iter_sched in range(MAX_ITER))
	for iter_sched in range(MAX_ITER):
		direct_values.append(results[iter_sched])

	f_out.write(dcName[dclist.index(dc)] + ":\n")
	f_out.write("Opt_avg = " + str(optWelfare) + "\n")
	f_out.write("Opt_var = " + str(0) + "\n")
	for t in range(TOTAL):
		temp_avg = []
		temp_var = []
		for ss in range(len(samplesize_array)):
			temp = []
			for iter_sched in range(MAX_ITER):
				temp.append(direct_values[iter_sched][t, ss])
			temp_avg.append(np.mean(temp))
			temp_var.append(np.var(temp, ddof = 1))
		f_out.write(totName[t]+"_avg = " + printProper(temp_avg) + "\n")
		f_out.write(totName[t]+"_var = " + printProper(temp_var) + "\n")

	f_out.write("\n")

f_out.close()








