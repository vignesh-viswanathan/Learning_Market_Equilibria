import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L = 2000
N = 30 #Number of people in each set
G = 50 #Number of Movies
S = 1 #Number of seats in a movie

print("Loading data...", end = '\r')
df = pd.read_csv('../ratings.csv')
gbobject = df.groupby('movieId')
data = df.values
print("Data loading done.")

print("Collecting relavant data...", end = '\r')
count = gbobject.count()
countnew = count.where(count['userId']>L).dropna()
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
    if(len(movielist)>=G):
        break

print("Relavant Data Collection Done.")        
print("Number of movies: "+ str(len(movielist)))
print("Number of people: " + str(len(inter)))

movielist2 = movielist[:G]
inter2 = list(inter[:N])
df2 = df[df.userId.isin(inter2)]
df3 = df2[df2.movieId.isin(movielist2)]
df4 = df3.copy()
df4.rating += [np.random.random()/10 for i in range(N*G)]
df4 = df4.reset_index(drop = True)
df4['good_index'] = [int(movielist2.index(df4.loc[i, 'movieId'])) for i in range(len(df4))]
df4['player_index'] = [int(inter2.index(df4.loc[i, 'userId'])) for i in range(len(df4))]
del df4['userId']
del df4['movieId']
del df4['timestamp']
print("Data Preprocessing done.")

df4.to_csv("../data2.csv", sep = ',', index = False, header = False)
