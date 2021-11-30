from skpalgotools import algos
a=algos.kthrank([4,20,7,3,6,10],6,3)
print(a)

import skpalgotools
from skpalgotools import algos
a=algos.coinchange([2,5,3,6],4,10)
print(a)

import skpalgotools
from skpalgotools import algos
b=algos.knapSack(4,[4,5,1],[1,2,3],3)
print(b)

import skpalgotools
from skpalgotools import algos
b=algos.matrixMultiplication(5, [40,20,30,10,30])
print(b)

b=algos.nCr( 8,2)
print(b)

b= algos.editDistance("shiridi", "thrdi")
print(b)


b= algos.equalPartition(5, [10,6,6,3,7])
print(b)


b= algos.lcs(7,6,"shiridikumar","akanksha")
print(b)

b= algos.maximizeTheCuts(4,2,1,1)
print(b)


b= algos.LongestRepeatingSubsequence("ahshaoashddfnkdjsvs")
print(b)


b= algos.longestSubsequence([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15],16)
print(b)

e=algos.largest_subarray_sum([1,2,3-2,5],5)
print(e)

b= algos.dijikstra(4 ,[[1, 2, 24], [1, 4, 20], [3, 1, 3], [4, 3, 12]] ,1)
print(b)

b= algos.prims(4 ,[[1, 2, 24], [1, 4, 20], [3, 1, 3], [4, 3, 12]] ,1)
print(b)


a=algos.bfs(4 ,[[1, 2], [2, 4], [3, 1], [4, 3]], 1)
print(a)

a=algos.dfs(4 ,[[1, 2], [2, 4], [3, 1], [4, 3]], 1)
print(a)
from random import randint
print()
print("genetic algorithm function")
n=int(input("enter number of nodes"))
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



import skpalgotools
class node:
    def __init__(self,val):
        self.val=val
        self.right=None
        self.left=None
print()
print("diameter tree function")

from skpalgotools import diameter_tree
n=int(input("enter number of nodes"))
g=[node(0)]
for i in range(n):
    g.append(node(i+1))
for i in range(n):
    u,v,w=map(int,input().split())
    if(w!=-1):
        g[u].right=g[w]
    if(v!=-1):
        g[u].left=g[v]

a=diameter_tree(g[0])
print(a)

print()
print("binary tree function")
import skpalgotools
class node:
    def __init__(self,val):
        self.val=val
        self.right=None
        self.left=None

from skpalgotools import binary_tree_height
n=int(input("enter number of nodes"))
g=[node(0)]
for i in range(n):
    g.append(node(i+1))
for i in range(n):
    u,v,w=map(int,input().split())
    if(w!=-1):
        g[u].right=g[w]
    if(v!=-1):
        g[u].left=g[v]

a=binary_tree_height(g[0])
print(a)


a=algos.kthrank([4,20,7,3,6,10],6,3)
print(a)

