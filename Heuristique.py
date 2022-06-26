from random import random
import random
from tracemalloc import start
import numpy

import streamlit as st
import time

import pandas as pd
# import matplotlib.pyplot as plt

class Project:
    # This class defines a project and its attributes
    def __init__(self, l, np, w):
        # l= project ID
        # np= Total tasks in each project
        # w= Project weight
        self.l = l
        self.np = np
        self.w = w
        self.C_max = 0

    def get_task(self):
        self.T = numpy.empty(self.np, dtype=object)
        n = self.np
        for i in range(n):
            d = processing_time[i]
            # skill = input("skill : ")
            self.T[i] = Task(self.l, num[self.l] + i + 1, d)

    def set_C_max(self, C):
        self.C_max = C


class Task:
    def __init__(self, p, i, d):
        # p =  project
        # i = ID
        # d = processing time
        # skill = ID of skill needed
        self.p = p
        self.i = i
        self.d = d
        self.finish_time = 0

    def set_finish_time(self, t):
        self.finish_time = t + self.d


# Instance definition
st.sidebar.header("Paramétres du modèle")
n = st.sidebar.slider("Nombre de projets",1,100,2,1)
m = st.sidebar.slider("Nombre de tache",1,100,11,1)
task_per_project=[]
task_per_project_text = st.sidebar.text_input('Nombre de taches par projet séparer par virgule', '')
task_per_project_text=task_per_project_text.split(",")

#n = 4
#m = 16
#task_per_project = [7, 5, 2, 2]
#R = 4
#Days = 15
for i in task_per_project_text:
    task_per_project.append(int(i))

if sum(task_per_project) > m or sum(task_per_project)< m :
    st.sidebar.warning("Error to many or less tasks per project")


num = [0]
for i in task_per_project:
    num.append(i)

R = st.sidebar.slider("Nombre de programmeurs",1,100,4,1)

Days = st.  sidebar.slider("Nombre de jours",1,500,15,1)

st.header("Project management informations system")

# Skills_map = matrix of zeros and ones that shows if a certain programmer can do a given task

Agenda = numpy.zeros((R, Days))

P = numpy.empty(n, dtype=object)

st.header("Matrice de compétence")
skills_map=[]

for i in range(R):
    buffer_skills=[random.randint(0, 1) for x in range(m)]
       
    skills_map.append(buffer_skills)

skills_map=numpy.asarray(skills_map)
st.write(skills_map)
st.header("Matrice de précédence")

precedence=[]

for i in range(m):
    buffer_precedence=[0]*m
    index=random.randint(0,m-1)
    buffer_precedence[index]=1

            
    precedence.append(buffer_precedence)
    

precedence=numpy.asarray(precedence)
precedence=precedence.T
st.write(precedence)
#project_importance = [4, 3, 2, 1]

#processing_time = [2, 3, 3, 3, 2, 3, 2, 2, 1, 2, 1, 3, 3, 2, 6]
project_importance = random.sample(range(0,n),n)
for i in range(len(project_importance)):
    project_importance[i]=project_importance[i]+1



#processing_time = [2, 3, 3, 3, 2, 3, 2, 2, 1, 2, 1]
st.header("Temps de traitement")
processing_time= [random.randrange(1,5) for i in range(m)]

st.write(processing_time)

makespan_p = [0]*n

for i in range(len(P)):
    np = task_per_project[i]
    w = project_importance[i]
    P[i] = Project(i, np, w)
    P[i].get_task()

#paramètre

makespan_p = [0] * n

Objective_function = 0
num = [0]
s = 0
for i in range(n):
    s = s + task_per_project[i]
    num.append(s)

for i in range(len(P)):
    np = task_per_project[i]
    w = project_importance[i]
    P[i] = Project(i, np, w)
    P[i].get_task()

# wee zin

# Functions
def find_resources(T):
    F = []
    for i in range(R):
        if skills_map[i][T] == 1:
            F.append(i)
    return F


def check_if_available(t, r, d):
    available = 1
    j = t
    for i in range(d):
        if Agenda[r][j] != 0:
            available = 0
        j = j + 1
    return available


def schedule_task(t, r, T):
    p = T.d
    j = t
    for i in range(p):
        Agenda[r][j] = T.i
        j = j + 1


def check_precedence(T):
    for j in range(m):
        if precedence[j][T] != 0:
            return j


def makespan(p):
    for i in range(P[p].np):
        makespan_p[p] = makespan_p[p] + P[p].T[i].finish_time

#start=time.time()
# Heuristic
P = sorted(P, key=lambda x: x.w, reverse=True)
for p in range(n):
    finish_time_task = [0]*P[p].np
    for a in range(P[p].np):
        find_resources(a)
        scheduled = 0
        f = 0
        if check_precedence(a) != None:
            t = P[p].T[a - 1].finish_time
        else:
            t = 0
        while scheduled == 0:
            r = find_resources(a)[f]
            if check_if_available(t, r, P[p].T[a].d) == 1:
                schedule_task(t, r, P[p].T[a])
                scheduled = 1
                P[p].T[a].set_finish_time(t)
                finish_time_task[a] = P[p].T[a].finish_time
            else:
                f = f + 1
            if f == len(find_resources(a)):
                f = 0
                t = t + 1
    C = numpy.max(finish_time_task)
    P[p].set_C_max(C)

P = sorted(P, key=lambda x: x.l, reverse=False)

for i in range(n):
    makespan_p[i] = P[i].C_max

for i in range(len(P)):
    Objective_function = Objective_function + makespan_p[i] * project_importance[i]
#time.sleep(1)
#end=time.time()
st.write(Agenda)
st.header("Makesapan")
st.write(makespan_p) 
st.header("Fonction Objectif")
st.write(Objective_function)   
#st.write(a)

#a=print(f"Runtime of the program is {end - start}")
print(Agenda)
print(makespan_p)
print(Objective_function)

