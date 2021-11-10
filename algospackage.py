import numpy as np

#kth smallest
def median(me,num):
    me.sort()
    return me[num//2]

def kthmaximum(a,l,r,k):
    a=np.array(a)
    n=r-l+1
    ind=0
    if(r<l):
        return
    med=[0]*((n+4)//5)
    for i in range(n//5):
        med[i]=median(a[l+i*5:],5)
        ind+=1
    if(ind*5<n):
        med[ind]=median(a[l+ind*5:],n%5)
        ind+=1
    overall_med=med[0] if (ind==1) else kthmaximum(med,0,ind-1,ind//2)
    for j in range(l,r):
        if(a[j]==overall_med):
            temp=a[j]
            a[j]=a[r]
            a[r]=temp
    pivot=a[r]
    i=l-1
    for j in range(l,r+1):
        if(a[j]<pivot):
            i+=1
            temp=a[i]
            a[i]=a[j]
            a[j]=temp
    i+=1
    temp=a[i]
    a[i]=a[r]
    a[r]=temp
    if(i+1>k):
        r=i-1
        return kthmaximum(a,l,r,k)
    elif(i+1<k):
        l=i+1
        return kthmaximum(a,l,r,k-(i+1-l))
    else:
        return a[i]

"""n=int(input())
a=[int(i) for i in input().split()]
k=int(input())
ans=kthmaximum(a,0,n-1,k)
print(ans)"""



"----------------------------------------------------------------------------------------------------------------"
# find the contigous sub array with largest sum
def largest_subarray_sum(self,a,N):
    temp=0
    curr=a[0]
    for i in range(len(a)):
        temp+=a[i]
        if(temp>curr):
            curr=temp
        if(temp<0):
            temp=0

    return curr

"""tree and graph algorithms
1)level order or breadth first
2)depth first
3) height of a tree
4) breadth of a tree
8) merge two bsts
9) topological sort

greedy()
9) huffman coding
10) activity selection
11) job sequencing
12) k centers problem
13) dijikstras
14) prims
26) maximum product subset of an array

dp()

15) coin change problem
16)knapsack problem
17)binomial coefficient
18)matric chain multiplication
19) edit distance
20) longest common subsequence
21) Subset sum problem
22) max-min =k
23) derangements
24) largest rectangular submatrix
25)floyd warshall algorithm


"""
"--------------------------------------------------------------------------------------------------------------------"

def bfs(a,n,source=0):
    q=[source]
    l=[]
    visited=[False]*(n+1)
    while(len(q)!=0):
        s=q.pop(0)
        print(s)
        l.append(s)
        visited[s]=True
        for i in a[s]:
            if(not(visited[i])):
                q.append(i)
    return l

def dfs(a,n,source=0):
    q=[source]
    visited=[False]*(n+1)
    l=[]
    dfs_helper(a,n,source,visited,l)
    return l

def dfs_helper(a,n,source,visited,l):
    s=source
    visited[s]=True
    l.append(s)
    for i in a[s]:
        if(not(visited[i])):
            dfs_helper(a,n,i,visited,l)
    return l

def coinchange(self, S, m, n): 
    dp=[[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(len(dp)):
        dp[i][0]=1
    for i in range(1,m+1):
        for j in range(1,n+1):
            if(S[i-1]>j):
                dp[i][j]=dp[i-1][j]
            else:
                dp[i][j]=dp[i-1][j]+dp[i][j-S[i-1]]
    return dp[m][n]

def tree_height(a,n,source):
    if(len(a[source])==0):
        return 0
    m=0
    for i in a[source]:
        l=tree_height(a,n,i)
        if(l>m):
            m=l
    height=m+1
    return height

def binary_tree_height(root):
    if(root==None):
        return 0
    height=max(binary_tree_height(root.left),binary_tree_height(root.right))+1
    return height

global maxi
maxi=0
def diameter_tree(root):
    ans=diameter_tree_helper(root)
    return maxi


def diameter_tree_helper(root):
    global maxi
    if(root==None):
        return 0
    lh=diameter_tree_helper(root.left)
    rh=diameter_tree_helper(root.right)
    height=max(lh,rh)+1
    maxi=max(lh+rh,maxi)
    return height

def knapSack(self,W, wt, val, n):
    dp=[[0 for i in range(W+1)] for j in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,W+1):
            temp=0
            if(wt[i-1]<=j):
                temp=dp[i-1][j-wt[i-1]]+val[i-1]
            dp[i][j]=max(temp,dp[i-1][j])
    return dp[n][W]
def nCr(self, n, r):
    dp=[[0 for i in range(r+1)] for j in range(n+1)]
    for i in range(n+1):
        dp[i][0]=1
    for i in range(1,n+1):
        for j in range(1,r+1):
            dp[i][j]=(dp[i-1][j]+dp[i-1][j-1])%((10**9)+7)
    return dp[n][r]

def matrix_multiplication(self,s,n,arr,dp):
    if(dp[s][n]==-1):
        if(n==s):
            return 0
        mi=float("inf")
        for i in range(s,n):
            put=self.matrix_multiplication(s,i,arr,dp)+self.matrix_multiplication(i+1,n,arr,dp)+arr[s-1]*arr[i]*arr[n]
            mi=min(mi,put)
        dp[s][n]=mi
    return dp[s][n]

"______________________________Driver code_______________________________________________"


from collections import defaultdict
class Graph:
    def __init__(self,n):
        self.nodes=n
        self.graph=defaultdict(list)
    def addedge(self,u,v):
        self.graph[u].append(v)   


class node:
    def __init__(self,val):
        self.val=val
        self.right=None
        self.left=None

n=int(input())
g=[node(0)]
for i in range(n):
    g.append(node(i+1))
for i in range(n):
    u,v,w=map(int,input().split())
    if(w!=-1):
        g[u].right=g[w]
    if(v!=-1):
        g[u].left=g[v]

a=diameter_tree(g[1])
print(a)









    









            




