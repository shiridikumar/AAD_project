import algospackage
import algotools
import random
from algotools import GeneticAlgo
from random import randint
n=int(input())
weights=[[0 for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        num=randint(100,200)
        weights[i][j]=num

import skpalgotools
from skpalgotools import GeneticAlgo
obj=GeneticAlgo(n,weights,population=1000,mutation_prob=0.4,max_iter=100)
state,cost=obj.solve()
print(state,cost)


