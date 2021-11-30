import numpy as np
import random
from random import randint, sample, seed
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.fromnumeric import swapaxes

class GeneticAlgo:
    def __init__(self,n,weights,population=100,mutation_prob=0.5,max_iter=100):
        self.nodes=n
        self.weights=weights
        self.population=population
        self.child=self.initialize()
        self.max_iter=max_iter
        self.mutation_prob=mutation_prob
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if(self.weights[i][j]==0):
                    self.weights[i][j]=float("inf")
    

    def fitness(self,state):
        diff=0
        for i in range(len(state)-1):
            diff+=self.weights[state[i]][state[i+1]]
        diff+=self.weights[state[-1]][state[0]]

        return diff

    def initialize(self):
        pop=self.population
        child=set()
        for i in range(pop):
            child.add(str(sample(range(self.nodes),self.nodes)))
        child=list(map(lambda x:np.array(eval(x)),child))
        return child

    def print_population(self):
        for i in range(len(self.child)):
            #print(self.child[i])
            flag=1

    def solve(self):
        iter=0
        states=[]
        fit_evals=list(map(lambda x:self.fitness(x),self.child))

        for i in range(len(fit_evals)):
            states.append((self.child[i],fit_evals[i]))
        states1=sorted(states,key=lambda kv:kv[1])

        select=len(states1)//5
        start=select;end=5*select
        if(select==0):
            select=len(states1)
            end=len(states1);start=0
        self.iterations=0
        iter=self.iterations
        cont=0
        prev=states1[0][1]
        go_forward=states1[:select]
        generate=states1[start:end]

        fitness=[go_forward[0][1]]

        while(iter<1000):

            if(len(go_forward)>0):
                g=self.swap(go_forward.copy())
                if(g!=-1):
                    new_states=g
                else:
                    new_states=go_forward.copy()

            for j in range(len(generate)):
                prob=randint(1,100)
                if(prob%2==0):
                    s2=randint(0,len(generate)-1)
                    state_one,state_two=self.ordered_corssing(generate[j][0],generate[s2][0])
                    eval1,eval2=self.fitness(state_one),self.fitness(state_two)
                    generate[j]=(state_one,eval1);generate[s2]=(state_two,eval2)

            new_states.extend(generate)
            new_states=sorted(new_states,key=lambda x:x[1])

            if(new_states[0][1]==prev):
                cont+=1
            else:
                prev=new_states[0][1]
                cont=0

            fitness.append(str(new_states[0][1]))
            go_forward=new_states[:select].copy()
            generate=new_states[start:end].copy()
            iter+=1

            if(cont==self.max_iter):
                break
        
        fitness=list(map(int,fitness))
        sns.set()
        plt.plot(range(len(fitness)),fitness)
        plt.show()
        return new_states[0]

    def swap(self,states):
        initial=str(states[0][1])
        mi=str(states[0][1])
        for i in range(len(states)):
            prob=self.mutation_prob
            k=randint(1,100)
            if(k<=100*prob):
                pos1=randint(0,len(states[i][0])-1)
                pos2=randint(0,len(states[i][0])-1)
                temp=states[i][0][pos1]
                states[i][0][pos1]=states[i][0][pos2]
                states[i][0][pos2]=temp
                new=(states[i][0],self.fitness(states[i][0]))
                if(new[1]<int(mi)):
                    mi=str(new[1])
                    states[i]=new
            

        if(int(mi)>int(initial)):
            return -1

        return states

    def ordered_corssing(self,a,b):
        segment_size=len(a)//3
        a1=a.copy()
        b1=b.copy()
        a=np.array(a)
        b=np.array(b)
        a1=np.array(a1)
        b1=np.array(b1)
        k=random.randint(0,len(a)-segment_size-1)
        d=np.array([False]*len(a))
        for i in range(k,k+segment_size):
            d[a[i]]=True
        count=0
        c=a.copy()
        count=0
        i=0
        while(i<len(b) and count<len(a)):
            if(d[a[count]]):
                count+=1
            else:
                if(not(d[b[i]])):
                    a[count]=b[i]
                    count+=1
                i+=1      

        d1=[False]*len(b)
        for i in range(k,k+segment_size):
            d1[b[i]]=True
        i=0
        count=0
        while(i<len(a1) and count<len(b)):
            if(d1[b[count]]):
                count+=1
            else:
                if(not(d1[a1[i]])):
                    b[count]=a1[i]
                    count+=1
                i+=1 
        return (a,b)


    def mutate(self,a,b):
        i=randint(0,len(a)-1)
        temp=a[i]
        a[i]=b[i]
        b[i]=temp
        return (a,b)

        
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
    def knapSack(W, wt, val, n):
        dp=[[0 for i in range(W+1)] for j in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,W+1):
                temp=0
                if(wt[i-1]<=j):
                    temp=dp[i-1][j-wt[i-1]]+val[i-1]
                dp[i][j]=max(temp,dp[i-1][j])
        return dp[n][W]

    def matrixMultiplication( N, arr):
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


    def nCr( n, r):
        dp=[[0 for i in range(r+1)] for j in range(n+1)]
        for i in range(n+1):
            dp[i][0]=1
        for i in range(1,n+1):
            for j in range(1,r+1):
                dp[i][j]=(dp[i-1][j]+dp[i-1][j-1])%((10**9)+7)
        return dp[n][r]

    def coinchange(S, m, n): 
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

    def editDistance( s, t):
        dp=[[0 for i in range(len(t)+1)] for j in range(len(s)+1)]
        for i in range(len(dp)):
            dp[i][0]=i
        dp[0]=[i for i in range(len(dp[0]))]
        for i in range(1,len(dp)):
            for j in range(1,len(dp[i])):
                dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(0 if s[i-1]==t[j-1] else 1))
        return dp[len(s)][len(t)]

    def equalPartition( N, arr):
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

    def lcs(x,y,str1,str2):
        dp=[[0 for i in range(len(str2)+1)] for j in range(len(str1)+1)]
        for i in range(1,len(str1)+1):
            for j in range(1,len(str2)+1):
                if(str1[i-1]==str2[j-1]):
                    dp[i][j]=1+dp[i-1][j-1]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return dp[len(str1)][len(str2)]

    def maximizeTheCuts(n,x,y,z):
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

    def LongestRepeatingSubsequence( str):
        dp=[[0 for i in range(len(str)+1)] for j in range(len(str)+1)]
        for i in range(1,len(str)+1):
            for j in range(1,len(str)+1):
                if(str[i-1]==str[j-1] and i!=j):
                    dp[i][j]=1+dp[i-1][j-1]
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return dp[len(str)][len(str)]


    def longestSubsequence(a,n):
        dp=[1 for i in range(len(a))]
        for i in range(len(a)):
            ma=0
            for j in range(i):
                if(a[i]>a[j]):
                    ma=max(dp[j],ma)
            dp[i]=1+ma
        return max(dp)

    def largest_subarray_sum(a,N):
        temp=0
        curr=a[0]
        for i in range(len(a)):
            temp+=a[i]
            if(temp>curr):
                curr=temp
            if(temp<0):
                temp=0
        return curr

    def bfs(n,edges,source=0):
        g=Graph(n)
        for i in range(len(edges)):
            g.addedge(edges[0],edges[1])
        a=g.graph
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

    def dfs(n,edges,source=0):
        g=Graph(n)
        for i in range(len(edges)):
            g.addedge(edges[0],edges[1])
        a=g.graph
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

    def median(self,me,num):
        me.sort()
        return me[num//2]

    def kthmaximum(self,a,l,r,k):
        a=np.array(a)
        n=r-l+1
        ind=0
        if(r<l):
            return
        med=[0]*((n+4)//5)
        for i in range(n//5):
            med[i]=self.median(a[l+i*5:],5)
            ind+=1
        if(ind*5<n):
            med[ind]=self.median(a[l+ind*5:],n%5)
            ind+=1
        overall_med=med[0] if (ind==1) else self.kthmaximum(med,0,ind-1,ind//2)
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
            return self.kthmaximum(a,l,r,k)
        elif(i+1<k):
            l=i+1
            return self.kthmaximum(a,l,r,k-(i+1-l))
        else:
            return a[i]


from collections import defaultdict
class Graph:
    def __init__(self,n):
        self.nodes=n
        self.graph=defaultdict(list)
    def addedge(self,u,v):
        self.graph[u].append(v)   
        self.graph[v].append(u)


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






