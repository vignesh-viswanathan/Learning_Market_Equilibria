import numpy as np
from csv import reader
from docplex.mp.model import Model
from joblib import Parallel, delayed
import multiprocessing

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
	return np.max([Sample[g]*v[i,g] for g in range(G)])
 

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
	Alloc = [0]*G
	allocation = np.array([[0]*G for i in range(N)])
	prices = np.array([0]*G, dtype = 'd')

	for i in range(N):
		SamplesCopy = np.copy(Samples)
		ValCopy = np.copy(Val)
		flag = 0
		while(flag == 0):
			maxindex = np.argmax(ValCopy[:,i])
			maxval = ValCopy[maxindex, i]
			Bi = SamplesCopy[maxindex, :]
			for iter1 in range(len(SamplesCopy)):
				if(ValCopy[iter1, i] == maxval):
					Bi = np.array([Bi[g]*SamplesCopy[iter1, g] for g in range(G)])
				if(ValCopy[iter1, i] < maxval):
					Bi = np.array([Bi[g]*(1-SamplesCopy[iter1, g]) for g in range(G)])
			if(np.sum([Bi[g]*Alloc[g] for g in range(G)]) != 0):
				for iter1 in range(len(SamplesCopy)):
					if(ValCopy[iter1, i] == maxval):
						SamplesCopy[iter1, :] = np.array([0]*G)
						ValCopy[iter1, :] = np.array([0]*N)
			else:
				flag = 1
				y = np.sum(Bi)
				if(y!= 0):
					Alloc = [max(Alloc[g], Bi[g]) for g in range(G)]
					allocation[i, :] = np.copy(Bi)
					prices = [max((b[i]/y)*allocation[i, g], prices[g]) for g in range(G)]

	allocation[-1, :] = [max(allocation[-1, g], 1-Alloc[g]) for g in range(G)]

	return allocation, prices

def computeIndirectAllocation(Samples, Val, b):
	v = np.array([[N*BASE]*G for i in range(N)], dtype = 'd')
	for iter1 in range(len(Samples)):
		for g in range(G):
			if(Samples[iter1, g] == 1):
				for i in range(N):
					v[i, g] = min(v[i, g], Val[iter1, i])

	for i in range(N):
		for g in range(G):
			if(v[i,g] == BASE*N):
				v[i,g] = 0.1

	Alloc = [0]*G
	allocation = np.array([[0]*G for i in range(N)])
	prices = np.array([0]*G, dtype = 'd')
	for i in range(N):
		maxindex = np.argmax(v[i, :])
		maxval = v[i, maxindex]
		if(maxval!= 0):
			Alloc[maxindex] = 1
			prices[maxindex] = b[i]
			allocation[i, maxindex] = 1
			v[:, maxindex] = np.array([0]*N)

	allocation[-1, :] = [max(allocation[-1, g], 1-Alloc[g]) for g in range(G)]

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

	for i in range(N):
		X.add_constraint(X.sum(x[i,g] for g in range(G)) <= 1)

	X.add_constraint(obj == X.sum(X.v[i, g]*x[i, g] for g in range(G) for i in range(N)))

	X.maximize(obj)
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()

	return obj.solution_value


def getOptEqWelfare(v, b):
	vdash = np.copy(v)
	Alloc = [0]*G
	util = 0
	for i in range(N):
		maxindex = np.argmax(vdash[i, :])
		maxval = vdash[i, maxindex]
		util += maxval	
		vdash[:, maxindex] = np.array([0]*N)

	return util 




def mainfunc(v, b, dc, samplesize_array, iter_sched):
	np.random.seed(iter_sched)
	
	direct_val = np.array([[0]*len(samplesize_array) for i in range(TOTAL)], dtype = 'd')
	indirect_val = np.array([[0]*len(samplesize_array) for i in range(TOTAL)], dtype = 'd')
	SamplesMain, ValMain = getSamples(MAX_SAMPLES, v, dc)

	for ss in range(len(samplesize_array)):
		Samples = np.copy(SamplesMain[:samplesize_array[ss], :])
		Val = np.copy(ValMain[:samplesize_array[ss], :])
		direct_allocation, direct_prices = computeDirectAllocation(Samples, Val, b)
		indirect_allocation, indirect_prices = computeIndirectAllocation(Samples, Val, b)

		direct_val[CONSISTENCY, ss] += checkConsistency(direct_allocation, direct_prices, Samples, Val, v, b)
		direct_val[UTILITY, ss] += computeUtility(direct_allocation, direct_prices, v, b)
		direct_val[LOSS, ss] += computeLoss(direct_allocation, direct_prices, dc, v, b)
		direct_val[ENVY, ss] += computeEnvy(direct_allocation, direct_prices, v, b)
		direct_val[BURNT, ss] += computeBurnt(direct_allocation, direct_prices)

		indirect_val[CONSISTENCY, ss] += checkConsistency(indirect_allocation, indirect_prices, Samples, Val, v, b)
		indirect_val[UTILITY, ss] += computeUtility(indirect_allocation, indirect_prices, v, b)
		indirect_val[LOSS, ss] += computeLoss(indirect_allocation, indirect_prices, dc, v, b)
		indirect_val[ENVY, ss] += computeEnvy(indirect_allocation, indirect_prices, v, b)
		indirect_val[BURNT, ss] += computeBurnt(indirect_allocation, indirect_prices)


	return direct_val, indirect_val

def printProper(lis):
	stri = ""
	for i in lis:
		stri += str(i) + " "

	return stri


b = generateBudgets(BASE)
v = retrieveData(b)

dclist = [0, 0.6, 1, 1.4, 2, 4]
totName = ["Utility", "Consistency", "Loss", "Envy", "Burnt"]
dcName = ["Constant Sample Size 1", "Constant Sample Size 3", "Constant Sample Size 5", "Constant Sample Size 7", "Constant Sample Size 10", "Uniform Product Distribution"]
samplesize_array = [5]
while(samplesize_array[-1]*2 <= MAX_SAMPLES):
	samplesize_array.append(samplesize_array[-1]*2)
optWelfare = getOptWelfare(v)
optEqWelfare = getOptEqWelfare(v, b)
for dc in dclist:
	print(dc, end = "\r")
	direct_values = []
	indirect_values = []
	results = Parallel(n_jobs=num_cores)(delayed(mainfunc)(v, b, dc, samplesize_array, iter_sched) for iter_sched in range(MAX_ITER))
	for iter_sched in range(MAX_ITER):
		direct_values.append(results[iter_sched][0])
		indirect_values.append(results[iter_sched][1])

	f_out.write(dcName[dclist.index(dc)] + ":\n")
	f_out.write("Opt_avg = " + str(optWelfare) + "\n")
	f_out.write("Opt_var = " + str(0) + "\n")
	f_out.write("OptEq_avg = " + str(optEqWelfare) + "\n")
	f_out.write("OptEq_var = " + str(0) + "\n")
	f_out.write("Direct Learning: \n")
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

	f_out.write("Indirect Learning: \n")
	for t in range(TOTAL):
		temp_avg = []
		temp_var = []
		for ss in range(len(samplesize_array)):
			temp = []
			for iter_sched in range(MAX_ITER):
				temp.append(indirect_values[iter_sched][t, ss])
			temp_avg.append(np.mean(temp))
			temp_var.append(np.var(temp, ddof = 1))
		f_out.write(totName[t]+"_avg = " + printProper(temp_avg) + "\n")
		f_out.write(totName[t]+"_var = " + printProper(temp_var) + "\n")

	f_out.write("\n")

f_out.close()








