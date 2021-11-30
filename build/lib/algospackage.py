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
class algos:
    def __init__(self,p):
        parm =p
    def knapSack(self,W, wt, val, n):
        dp=[[0 for i in range(W+1)] for j in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,W+1):
                temp=0
                if(wt[i-1]<=j):
                    temp=dp[i-1][j-wt[i-1]]+val[i-1]
                dp[i][j]=max(temp,dp[i-1][j])
        return dp[n][W]

    def matrixMultiplication(self, N, arr):
        dp=[[0 for i in range(N+1)] for j in range(N+1)]
        for j in range(N-1,0,-1):
            for k in range(j,N):
                mi=float("inf")
                if(j==k):
                    dp[j][k]=0
                    continue
                for i in range(j,k):
                    temp=dp[j][i]+dp[i+1][k]+arr[j-1]*arr[i]*arr[k]
                    mi=min(temp,mi)
                dp[j][k]=mi
        return dp[1][N-1]


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

    def editDistance(self, s, t):
        dp=[[0 for i in range(len(t)+1)] for j in range(len(s)+1)]
        for i in range(len(dp)):
            dp[i][0]=i
        dp[0]=[i for i in range(len(dp[0]))]
        for i in range(1,len(dp)):
            for j in range(1,len(dp[i])):
                dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(0 if s[i-1]==t[j-1] else 1))
        return dp[len(s)][len(t)]

    def equalPartition(self, N, arr):
        dp=[[False for i in range(sum(arr)//2+1)] for j in range(N+1)]
        if(sum(arr)%2==1):
            return 0
        dp[0]=[False for i in range(len(dp[0]))]
        for i in range(len(dp)):
            dp[i][0]=True
        for i in range(1,len(dp)):
            for j in range(1,len(dp[i])):
                temp=dp[i-1][j-arr[i-1]] if(arr[i-1]<=j) else False
                dp[i][j]=temp or dp[i-1][j]
        return dp[N][sum(arr)//2]

    def lcs(self,x,y,str1,str2):
        dp=[[0 for i in range(len(str2)+1)] for j in range(len(str1)+1)]
        for i in range(1,len(str1)+1):
            for j in range(1,len(str2)+1):
                if(str1[i-1]==str2[j-1]):
                    dp[i][j]=1+dp[i-1][j-1]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return dp[len(str1)][len(str2)]

    def maximizeTheCuts(self,n,x,y,z):
        dp=[-1 for i in range(n+1)]
        dp[0]=0
        for i in range(1,n+1):
            p,q,r=-1,-1,-1
            if(i-x>=0):
                p=(1+dp[i-x]) if dp[i-x]!=-1 else -1
            if(i-y>=0):
                q=(1+dp[i-y]) if dp[i-y]!=-1 else -1
            if(i-z>=0):
                r=(1+dp[i-z]) if dp[i-z]!=-1 else -1
            dp[i]=max(p,q,r)
        return max(0,dp[n])

    def LongestRepeatingSubsequence(self, str):
        dp=[[0 for i in range(len(str)+1)] for j in range(len(str)+1)]
        for i in range(1,len(str)+1):
            for j in range(1,len(str)+1):
                if(str[i-1]==str[j-1] and i!=j):
                    dp[i][j]=1+dp[i-1][j-1]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return dp[len(str)][len(str)]


    def longestSubsequence(self,a,n):
        dp=[1 for i in range(len(a))]
        for i in range(len(a)):
            ma=0
            for j in range(i):
                if(a[i]>a[j]):
                    ma=max(dp[j],ma)
            dp[i]=1+ma
        return max(dp)


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

#n=int(input())
#g=[node(0)]
"""for i in range(n):
    g.append(node(i+1))
for i in range(n):
    u,v,w=map(int,input().split())
    if(w!=-1):
        g[u].right=g[w]
    if(v!=-1):
        g[u].left=g[v]

a=diameter_tree(g[1])
print(a)"""


from collections import defaultdict
class Graph1:
    def __init__(self,vertices):
        self.vertices=vertices
        self.graph=defaultdict(list)

    def addedge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def prims(self,l,source,edges):
        exp=[source];queue=[];nodes=[]
        count=0;cost=0
        vis=[float("inf")]*(self.vertices+1)
        visited=[False]*(self.vertices+1);vis[source]=0
        while(count!=self.vertices-1):
            for i in self.graph[source]:
                if(l[source][i]<vis[i] and visited[i]!=True):
                    vis[i]=l[source][i]
            ind=0;m=float("inf")
            visited[source]=True
            for i in range(len(vis)):
                if(not(visited[i]) and vis[i]<m):
                    m=vis[i];ind=i
            source=ind
            count+=1
        return vis
    def dijikstra(self,d,l,source):
        vis=[False]*(self.vertices+1)
        count=1
        while(count!=self.vertices):
            for i in self.graph[source]:
                if(d[source]+l[source][i]<d[i]):
                    d[i]=d[source]+l[source][i]
            ind=0;m=float("inf");vis[source]=True
            count+=1
            for i in range(len(vis)):
                if(not(vis[i])):
                    if(d[i]<m):
                        m=d[i]
                        ind=i
            source=ind
        for i in range(len(d)):
                if(d[i]==float("inf")):
                    d[i]=-1
        return d
        
def prims(n, edges, start):
    g=Graph1(n)
    l=[ [i for i in range(n+1)] for j in range(n+1)]
    for i in range(len(edges)):
        u,v,cost=edges[i][0],edges[i][1],edges[i][2]
        l[u][v]=cost;l[v][u]=cost
        g.addedge(u,v)
    ans=g.prims(l,start,edges)
    return sum(ans[1:])


def dijikstra(n, edges, s):
    g=Graph1(n)
    l=[]
    for i in range(n+1):
        s1=[]
        for j in range(n+1):
            s1.append(float("inf"))
        l.append(s1)
    for i in range(len(edges)):
        u,v,cost=edges[i][0],edges[i][1],edges[i][2]
        if(cost<l[u][v]):
            l[u][v]=cost;l[v][u]=cost
            g.addedge(u,v)
    source=s
    d=[float("inf")]*(n+1)
    d[source]=0
    ans=g.dijikstra(d,l,source)
    p=ans[1:source]
    p.extend(ans[source+1:])
    return (p)









    









            




