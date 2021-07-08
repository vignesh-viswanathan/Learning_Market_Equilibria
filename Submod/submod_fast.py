import numpy as np
from csv import reader
from docplex.mp.model import Model

LOSS_ITER = 1000
NUM_TIMESLOTS = 10 #G should be a multiple of this value
BASE = 5
BUDGET_NORMALISE = True

TOTAL = 5
UTILITY = 0
CONSISTENCY = 1
LOSS = 2
ENVY = 3
BURNT = 4


N = int(input())
G = int(input())
MAX_ITER = int(input())
MAX_SCHED = int(input())
MAX_SAMPLES = int(input())
BN = int(input())
Kdash = int(input())
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
	v = np.array([[0]*G for i in range(N)], dtype = 'f')

	f_in = open(file_dir, "r")
	csv_reader = reader(f_in)

	for row in csv_reader:
		v[int(row[2]), int(row[1])] = float(row[0])
	if(BUDGET_NORMALISE==True):
		for i in range(N):
			z = np.max(v[i, :])
			v[i, :] = v[i, :]*(b[i]/z)


	return v

def createSchedule():
	X = np.random.permutation(G)
	sched = [[int(i * G/NUM_TIMESLOTS + j) for j in range(int(G/NUM_TIMESLOTS))] for i in range(NUM_TIMESLOTS)]
	return np.array(sched)

def getValueOfBundle(Sample, v, sched, i):
	slot_vals = np.array([np.max([v[i, sched[iter1, iter2]]*Sample[sched[iter1, iter2]] for iter2 in range(int(G/NUM_TIMESLOTS))]) for iter1 in range(NUM_TIMESLOTS)])
	slot_vals = np.sort(slot_vals)[::-1]
	val = np.sum(slot_vals[:Kdash])
	return val
 

def getSamples(numsamples, v, sched, dc):
	SamplesMain = []
	ValMain = []
	if(dc == 4): #Uniform Product
		for iter1 in range(numsamples):
			Sample = [int(np.random.random()+0.5) for g in range(G)]
			val = [getValueOfBundle(Sample, v, sched, i) for i in range(N)]
			SamplesMain.append(Sample)
			ValMain.append(val)
		return np.array(SamplesMain), np.array(ValMain)
	if(dc == 0): #Sample only one good
		for iter1 in range(numsamples):
			Sample = [0]*G
			Sample[np.random.randint(G)] = 1
			val = [getValueOfBundle(Sample, v, sched, i) for i in range(N)]
			SamplesMain.append(Sample)
			ValMain.append(val)
		return np.array(SamplesMain), np.array(ValMain)

	if((dc > 0) & (dc <=3)):
		for iter1 in range(numsamples):
			Sample = [0]*G
			X = np.random.permutation(G)
			for g in range(int(5*dc)):
				Sample[X[g]] = 1
			val = [getValueOfBundle(Sample, v, sched, i) for i in range(N)]
			SamplesMain.append(Sample)
			ValMain.append(val)
	return np.array(SamplesMain), np.array(ValMain)

def isSubset(Bundle1, Bundle2):
	flag = 0
	for g in range(G):
		if(Bundle1[g] > Bundle2[g]):
			return False
		if(Bundle1[g] < Bundle2[g]):
			flag = 1
	if(flag == 1):
		return True

def computeDirectAllocation(Samples, Val, b, c):
	SamplesCopy = np.copy(Samples)
	ValCopy = np.copy(Val)

	for iter1 in range(len(SamplesCopy)):
		for iter2 in range(len(Samples)):
			if(isSubset(Samples[iter2, :], SamplesCopy[iter1, :])):
				SamplesCopy[iter1, :] -= Samples[iter2, :]
				ValCopy[iter1, :] -= Val[iter2, :]


	for iter1 in range(len(SamplesCopy)):
		for iter2 in range(len(SamplesCopy)):
			if(isSubset(SamplesCopy[iter1, :], SamplesCopy[iter2, :])):
				SamplesCopy[iter1, :] = np.array([0]*G)
				ValCopy[iter1, :] = np.array([0]*N)
				break

	SamplesCopy2 = np.array([SamplesCopy[iter1, :] for iter1 in range(len(SamplesCopy)) if np.max(ValCopy[iter1, :]) != 0.0])
	ValCopy2 = np.array([ValCopy[iter1, :] for iter1 in range(len(SamplesCopy)) if np.max(ValCopy[iter1, :]) != 0.0]) 

	SamplesCopy = SamplesCopy2
	ValCopy = ValCopy2

	F = []
	Gminus = [0 if np.sum([Samples[iter1, g] for iter1 in range(len(Samples))]) != 0 else 1 for g in range(G)]
	F = [[1 if (((np.sum([Samples[iter1, g] if Val[iter1, i] >= c[i] else 0 for iter1 in range(len(Samples))]) != 0) & (np.sum([Samples[iter1, g] if Val[iter1, i] < c[i] else 0 for iter1 in range(len(Samples))]) == 0 ))| Gminus[g] == 1) else 0 for g in range(G)] for i in range(N)]

	Alloc = [0]*G
	allocation = np.array([[0]*G for i in range(N)])
	val_alloc = np.array([0]*N, dtype = 'f')
	prices = np.array([0]*G, dtype = 'f')
	for i in range(N):
		if((np.max(ValCopy[:, i]) < c[i]) & (np.sum([F[i][g]*Alloc[g] for g in range(G)]) == 0)):
			val_alloc[i] = c[i]
			allocation[i, :] = np.array(F[i])
		else:
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

def checkConsistency(allocation, prices, Samples, Val, sched, v, b):
	val_alloc = [getValueOfBundle(allocation[i,:], v, sched, i) for i in range(N)]
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

def computeUtility(allocation, prices, sched, v, b):
	return np.sum([getValueOfBundle(allocation[i,:], v, sched, i) for i in range(N)])

def computeLoss(allocation, prices, dc, sched, v, b):
	Samples, Val = getSamples(LOSS_ITER, v, sched, dc)
	val_alloc = [getValueOfBundle(allocation[i,:], v, sched, i) for i in range(N)]
	price_bundles = [np.dot(Samples[iter1, :], prices) for iter1 in range(len(Samples))]

	flag = 0
	for iter1 in range(len(Samples)):
		for i in range(N):
			if((Val[iter1, i] > val_alloc[i]) & (price_bundles[iter1] <= b[i])):
				flag += 1
				break
	return flag/LOSS_ITER

def computeEnvy(allocation, prices, sched, v, b):
	val_alloc = np.array([[getValueOfBundle(allocation[i,:], v, sched, j) for j in range(N)] for i in range(N)])
	price_bundles = [np.dot(allocation[i, :], prices) for i in range(N)]
	envy_vec = [[1  if ((val_alloc[i, i] < val_alloc[j, i]) & (price_bundles[j] <= b[i])) else 0 for j in range(N)] for i in range(N)]
	return np.sum(envy_vec)

def computeBurnt(allocation, prices):
	count = 0
	for g in range(G):
		if(prices[g] == N*BASE):
			count+=1
	return count

def getOptWelfare(sched, v):
	X = Model(name = 'opt_welfare')

	X.v = v
	X.sched = sched
	X.Kdash = Kdash

	x = {(i, j): X.binary_var(name = 'x_'+str(i)+'_'+str(j)) for i in range(N) for j in range(G)}
	obj = X.continuous_var(name = 'obj')

	for i in range(N):
		X.add_constraint(X.sum(x[i,g] for g in range(G)) <= X.Kdash)

	for g in range(G):
		X.add_constraint(X.sum(x[i,g] for i in range(N)) <= 1)

	for i in range(N):
		for t in range(NUM_TIMESLOTS):
			X.add_constraint(X.sum(x[i, X.sched[t, j]] for j in range(int(G/NUM_TIMESLOTS))) <= 1)

	X.add_constraint(obj == X.sum(X.v[i, g]*x[i, g] for g in range(G) for i in range(N)))

	X.maximize(obj)
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()

	return obj.solution_value


b = generateBudgets(BASE)
v = retrieveData(b)
dclist = [0, 0.6, 1, 1.4, 2, 4]
dcName = ["Constant Sample Size 1", "Constant Sample Size 3", "Constant Sample Size 5", "Constant Sample Size 7", "Constant Sample Size 10", "Uniform Product Distribution"]
samplesize_array = [5]
while(samplesize_array[-1]*2 <= MAX_SAMPLES):
	samplesize_array.append(samplesize_array[-1]*2)
for dc in dclist:
	optWelfare = 0
	direct_values = np.array([[0]*len(samplesize_array) for i in range(TOTAL)], dtype = 'f')
	for iter_sched in range(MAX_SCHED):
		sched = createSchedule()
		optWelfare += getOptWelfare(sched, v)/MAX_SCHED
		for iter_val in range(MAX_ITER):
			SamplesMain, ValMain = getSamples(MAX_SAMPLES, v, sched, dc)
			for ss in range(len(samplesize_array)):
				# print(dc, MAX_SCHED, MAX_ITER, ss, "\t\t", end = '\r')
				Samples = np.copy(SamplesMain[:samplesize_array[ss], :])
				Val = np.copy(ValMain[:samplesize_array[ss], :])

				c = np.array([0]*N)
				if(BUDGET_NORMALISE == True):
					c = np.copy(b)
					for i in range(N):
						c[i] = c[i] - 0.0001
				
				
				direct_allocation, direct_prices = computeDirectAllocation(Samples, Val, b, c)

				direct_values[CONSISTENCY, ss] += checkConsistency(direct_allocation, direct_prices, Samples, Val, sched, v, b)/(MAX_ITER*MAX_SCHED)
				direct_values[UTILITY, ss] += computeUtility(direct_allocation, direct_prices, sched, v, b)/(MAX_ITER*MAX_SCHED)
				direct_values[LOSS, ss] += computeLoss(direct_allocation, direct_prices, dc, sched, v, b)/(MAX_ITER*MAX_SCHED)
				direct_values[ENVY, ss] += computeEnvy(direct_allocation, direct_prices, sched, v, b)/(MAX_ITER*MAX_SCHED) 
				direct_values[BURNT, ss] += computeBurnt(direct_allocation, direct_prices)/(MAX_ITER*MAX_SCHED)

	f_out.write(dcName[dclist.index(dc)] + ":\n")
	f_out.write("Average Optimal Welfare: " + str(optWelfare) + "\n")
	f_out.write("Direct Learning Consistency: " + str(direct_values[CONSISTENCY, :]) + "\n")
	f_out.write("Direct Learning Welfare: " + str(direct_values[UTILITY, :]) + "\n")
	f_out.write("Direct Learning Loss: " + str(direct_values[LOSS, :]) + "\n")
	f_out.write("Direct Learning Envy: " + str(direct_values[ENVY, :]) + "\n")
	f_out.write("Direct Learning Burn: " + str(direct_values[BURNT, :]) + "\n")
	f_out.write("\n")

f_out.close()








