import random
from random import randint, sample, seed
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.fromnumeric import swapaxes

class GeneticAglo:
    def __init__(self,n,weights,population=100):
        self.nodes=n
        self.weights=weights
        self.population=population
        print(population)
        self.child=self.initialize()
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

            if(cont==100):
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


#n=int(input())
"""weights=[[0 for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        num=randint(100,200)
        weights[i][j]=num


obj=GeneticAglo(n,weights,1000)
state,cost=obj.solve()
print(state,cost)"""