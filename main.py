from random import random
import random
import numpy

import streamlit as st


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

    def get_task(self):
        self.T = numpy.empty(self.np, dtype=object)
        n = self.np
        for i in range(n):
            d = processing_time[i]
            # skill = input("skill : ")
            self.T[i] = Task(self.l, num[self.l] + i + 1, d)
    
    



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



st.sidebar.header("Paramétres du modèle")
n = st.sidebar.slider("Nombre de projets",1,100,2,1)
m = st.sidebar.slider("Nombre d'activités",1,100,11,1)
task_per_project=[]
task_per_project_text = st.sidebar.text_input('Nombre de taches par personne séparer par virgule', '5,6')
task_per_project_text=task_per_project_text.split(",")



for i in task_per_project_text:
    task_per_project.append(int(i))

if sum(task_per_project) > m or sum(task_per_project)< m :
    st.sidebar.warning("Error to many or less tasks per project")


num = [0]
for i in task_per_project:
    num.append(i)

R = st.sidebar.slider("Nombre de programmeurs",1,100,4,1)

Days = st.  sidebar.slider("Nombre de jours",1,365,15,1)

st.header("Programme de ASMA")

#uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:
#    dataframe = pd.read_csv(uploaded_file)
#    st.write(dataframe)

# Instance definition
#n = 2
#m = 11
#num = [0, 5, 6]
#task_per_project = [5, 6]
P = numpy.empty(n, dtype=object)
#R = 4
#Days = 15

Agenda = numpy.zeros((R, Days))


st.header("Skills Map")
skills_map=[]

for i in range(R):
    buffer_skills=[random.randint(0, 1) for x in range(m)]
       
    skills_map.append(buffer_skills)

skills_map=numpy.asarray(skills_map)
st.write(skills_map)


st.header("Precedence")

precedence=[]

for i in range(m):
    buffer_precedence=[0]*m
    index=random.randint(0,m-1)
    buffer_precedence[index]=1

            
    precedence.append(buffer_precedence)
    

precedence=numpy.asarray(precedence)
precedence=precedence.T
st.write(precedence)

        


# Skills_map = matrix of zeros and ones that shows if a certain programmer can do a given task
'''
skills_map = [[0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
              [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
              [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]]

precedence = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
'''
project_importance = random.sample(range(0,n),n)
for i in range(len(project_importance)):
    project_importance[i]=project_importance[i]+1



#processing_time = [2, 3, 3, 3, 2, 3, 2, 2, 1, 2, 1]

processing_time= [random.randrange(1,10) for i in range(m)]



makespan_p = [0]*n

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
    makespan_p[p] = P[p].T[task_per_project[p] - 1].finish_time




# Heuristic
for p in range(n):
    for a in range(task_per_project[p]):
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
            else:
                f = f + 1
            if f == len(find_resources(a)):
                f = 0
                t = t + 1



makespan(0)
makespan(1)

st.header("Agenda")
for n, i in enumerate(Agenda):
    for k, j in enumerate(i):
        Agenda[n][k] = int(j)
    
st.write(Agenda)
st.header("Makesapan")
st.write(makespan_p)
print(Agenda)
print(makespan_p)





st.balloons()