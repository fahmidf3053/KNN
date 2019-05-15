def Dis(data1, data2, length):
    distance = 0
    for x in range(length):
        distance = distance+np.square(data1[x] - data2[x])
       # print(distance)

    return np.sqrt(distance)




def knn(trainingSet, testInstance, k):
    distances = list()
    tTestInstance=testInstance.drop(["whether he/she donated blood in March 2007"]) 
    length = len(tTestInstance)
    tTrainingSet = trainingSet.drop(["whether he/she donated blood in March 2007"],axis=1)
    
    #print(len(tTrainingSet))
    #print(tTestInstance)
    #print(tTrainingSet)
    
    for x in range(len(tTrainingSet)):
        dist = Dis(tTestInstance, tTrainingSet.iloc[x], length)
        distances.append(dist) 
    
    trainingSet["Distance"]=distances
    sortSet=trainingSet.sort_values('Distance')
    
    cnt=[0,0]
    #print(cnt[0])
    #print(cnt)
    for x in range(k):
        ind=int(sortSet.iloc[x][4])
        cnt[ind]+=1
   # print(cnt)
    
    index, value = max(enumerate(cnt), key=operator.itemgetter(1))
    
    #print(index)
    return index



import pandas as pan
import numpy as np
import math
import operator

data = pan.read_csv("transfusion.data")

#data.head()
#tData = data.drop(["whether he/she donated blood in March 2007"],axis=1)
#tData.head()
index = len(data)

par=int(.1*index)-1
parcent=0;
#print(par)
for x in range(10):
    
    testSegment = data.iloc[x*par:(x*par+(par-1))]
    
    trainSegement = data.drop(data.index[x*par:(x*par+(par-1))]) 
    #print(len(testSegment))
    match=0
    for y in range(len(testSegment)):  
        cls=knn(trainSegement, testSegment.iloc[y], 3)
        #print(cls)
        #print(testSegment.iloc[y][4])
        if cls==testSegment.iloc[y][4]:
            #print("ami asi ",y," number e")
            match+=1
            
        
        #print(testSegment.iloc[0])
    print(match)
    parcent+=match/(par*10)


print("Result is: ",parcent*100,"%")