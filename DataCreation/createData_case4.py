import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


N = 2000 #Number of people in each set
G = 30
S = 50
b = [1000 + np.random.random() for i in range(N)]
b.sort(reverse = True)

print("Loading data...", end = '\r')
df = pd.read_csv('../ratings.csv')
gbobject = df.groupby('movieId')
data = df.values
print("Data loading done.")

print("Collecting relavant data..", end = '\r')
count = gbobject.count()
countnew = count.where(count['userId']>N).dropna()
y = np.array(countnew.index)
iter1 = 0
while(1):
    inter = np.array(gbobject.get_group(y[iter1]).userId)
    movielist = [y[iter1]]
    while(1):
        #Find Max intersection which is not in movielist
        val = -1
        movietemp = -1
        intertempmax = np.array([])
        for i in range(len(y)):
            if(y[i] in movielist):
                continue
            #print("Inner Loop: "+str(i)+"/"+str(len(y)), end = '\r')
            intertemp = np.intersect1d(inter, np.array(gbobject.get_group(y[i]).userId))
            if(len(intertemp) > val):
                val = len(intertemp)
                intertempmax = intertemp
                movietemp = y[i]
        z = np.intersect1d(inter, intertempmax)
        if(len(z)>=N):
            movielist.append(movietemp)
            inter = z
        else:
            break
    iter1 += 1
    if(len(movielist)>=30):
        break

print("Relavant data collection done.")        
print("Number of movies: "+ str(len(movielist)))
print("Number of people: " + str(len(inter)))

print("Preprocessing data...", end = '\r')
movielist2 = movielist[:G]
inter2 = list(inter[:N])
df2 = df[df.userId.isin(inter2)]
df3 = df2[df2.movieId.isin(movielist2)]
df11 = df3.copy()
df11['seat'] = 0
for i in range(1,50):
    df12 = df3.copy()
    df12['seat'] = i
    df11 = df11.append(df12)
df11.rating += [np.random.random()/10 for i in range(S*N*G)]
df11 = df11.reset_index(drop = True)
df11['good_index'] = [movielist2.index(df11.loc[i, 'movieId'])*S + df11.loc[i, 'seat'] for i in range(len(df11))]
df11['player_index'] = [int(inter2.index(df11.loc[i, 'userId'])) for i in range(len(df11))]
del df11['seat']
del df11['userId']
del df11['movieId']
del df11['timestamp']
print("Data Preprocessing done.")

df11.to_csv("../data4.csv", sep = ',', index = False, header = False)
