#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: prototype_selection_matrix.py
# Version 1.0  
# Author : Prashan Wanigasekara (prashanw@mit.edu) 
# ---------------------------------------------------------------------------

import cplex
from cplex.exceptions import CplexError
import numpy as np
from scipy import linalg as LA
from scipy import spatial
import matplotlib
#matplotlib.use('Agg')
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import math
import collections
import argparse
import itertools
import os
import time
import pprint
import cPickle as pickle
import numpy.linalg as linalg
import re
import random
from mpl_toolkits.mplot3d import Axes3D
from itertools import product 
import glob
import math
from decimal import Decimal
import pandas
set_seed=210
random.seed(set_seed)
pp = pprint.PrettyPrinter(indent=4)

#alternate minimization
alt_runs=1
#version
version='March18'
#constants
EPSILON=0.000001
w1_min=1e-8
w1=0
w2=1e6
M=4*w2
#objective coefficents
C_k=1000
C_n=1000
C_alpha=.0001
C_beta=.0001
lambda_array=[.0001]
C_v=0.0001

#non objective coefficients
C_cijb=0
C_d=0

#plotting and printing
autoaxis=False
axequal=True
plot_one=True
print_result_variables=False
supress_write=True #True for the cluster
warm_start_flag=True
apply_run_parametrs=True
print_to_report=True
cm = plt.get_cmap('gist_rainbow')

random.seed(1)

#Data
HEADER=1

def set_MIP_run_parameters(my_prob):
    time_limit,tl=True,5*60  
    emphasis,emp=True,0
    max_num_sol,sol=False,5
    max_search_nodes,n=False,3
    aggregator_flag,agg=False,0 
    tolerane_flag,tolerance_value=True,0
    Integrality,i_value=True,0
    presolve_ignore,presolve_value=False,0
    advance_start,advance_start_value=True,1
    repair_tries,repair_tries_value=True,10
    conflict_display,conflict_value=True,2
    numerical_precisin,numerical_precision_value=True,1
    
    parameter_str=''
    
    if time_limit==True:
        my_prob.parameters.timelimit.set(tl)
        parameter_str+='time_limit= '+str(tl)+'s | '
    if emphasis==True:
        my_prob.parameters.emphasis.mip.set(emp)
        parameter_str+='emphasis= '+str(emp)+' | ' 
    if max_num_sol==True:
        my_prob.parameters.mip.limits.solutions.set(sol)
        parameter_str+='max_num_sol= '+str(sol)+' | ' 
    if max_search_nodes==True:
        my_prob.parameters.mip.limits.nodes.set(n)
        parameter_str+='max_search_nodes= '+str(n)+' | '
    if aggregator_flag==True:   
        my_prob.parameters.preprocessing.aggregator.set(agg)
    if tolerane_flag==True:          
        my_prob.parameters.mip.tolerances.absmipgap.set(tolerance_value)
        my_prob.parameters.mip.tolerances.mipgap.set(tolerance_value)
    if Integrality==True:
        my_prob.parameters.mip.tolerances.integrality.set(i_value)
    if presolve_ignore==True:
        my_prob.parameters.preprocessing.presolve.set(presolve_value)
    if advance_start==True:
        my_prob.parameters.advance.set(advance_start_value) 
    if repair_tries==True:
        my_prob.parameters.mip.limits.repairtries.set(repair_tries_value) 
    if conflict_display==True:
        my_prob.parameters.conflict.display.set(conflict_value)        
    if numerical_precisin==True:
        my_prob.parameters.emphasis.numerical.set(numerical_precision_value)        
     
        
        
        
    return parameter_str

class MyCallback(cplex.callbacks.MIPInfoCallback):    
    
    def __call__(self):
        if self.has_incumbent():
            self.incobjval.append(self.get_incumbent_objective_value())
            self.bestobjval.append(self.get_best_objective_value())
            self.times.append(self.get_time())
            self.dettimes.append(self.get_time())
            self.start_dettime.append(self.get_start_time())




class Solution:  
    
    def __init__(self,N,B,a,b,omega,features,point_fn,report_file):
        self.N=N
        self.B=B
        self.non_zero_a=a
        self.non_zero_b=b
        self.non_zero_omega=omega
        self.non_zero_features=features
        self.complete_solution=False
        self.variable_dict=[]
        self.point_fn=point_fn
        self.report_file=report_file
        

    def write(self):
        with open('./output/'+version+'_plots/sols/'+str(self.report_file)+'_'+str(self.B)+'balls.pkl','wb') as object_write:
            pickle.dump(self,object_write, -1)
    
    def write_2(self,iteration,alt_type,population,dimension):
        with open('./output/'+version+'_plots/saved_iteration_sols/'+str(self.report_file)+'_iter_'+str(iteration)+'_'+alt_type+'.pkl','wb') as object_write:
                pickle.dump(self,object_write, -1)
                self.create_complete_solution(population)
                print 'after writing ws objective :',self.calculate_objective(population,dimension)
                        
            
    def __str__(self):
        print '\n-----------------------------------------------\nprinting solution\n'
        print 'N',self.N,'B',self.B,'\nnon_zero_a\n',self.non_zero_a,'\nnon_zero_b\n',self.non_zero_b,"\nnon_zero_omega\n",pp.pprint(self.non_zero_omega.items()),"\nself.non_zero_features\n",self.non_zero_features
        print '-----------------------------------------------'
        
    def change(self,mapping,new_N,new_B):
        '''Given a mapping change the objective variables'''        
        new_a,new_b=[],[]
        self.N=new_N
        self.B=new_B
        self.original_point_file_name='None'
        
        for elm in self.non_zero_a:            
            new_a.append((''.join([str(int(map_item[1])) for map_item in mapping if map_item[0]==int(elm[0])]),elm[1]))
        self.non_zero_a=new_a
                
        for elm in self.non_zero_b:
            new_b.append(''.join(['b'+str(int(map_item[1])) for map_item in mapping if map_item[0]==int(elm[1:])]))
        self.non_zero_b=new_b               
        return
        
    def create_complete_solution(self,population):
        '''Given a solution create other related objective variables d, ksi, eta ... based on self.N, self.B
           order k,n,a,b,c,omega,d,features''' 
        
        variable_dict=collections.OrderedDict()
        k_dict,n_dict,a_dict,b_dict,c_dict,omega_dict,d_dict,v_dict=[collections.OrderedDict() for _ in range(8)]
        
        a_dict={'a'+str(j)+'_'+str(b):1 if ((str(j),str(b))) in self.non_zero_a else 0 for j,b in product(range(1,self.N+1),range(1,self.B+1))  }        
        b_dict={'b'+str(j):1 if ('b'+str(j)) in self.non_zero_b else 0 for j in range(1,self.N+1)}        
           
        for i,j,b in product(range(1,self.N+1),range(1,self.N+1),range(1,self.B+1)):
                d_dict['d'+str(int(i))+'_'+str(int(j))+'_'+str(int(b))]=d(int(i),j,b,population,self.non_zero_omega)
                c_dict['c'+str(i)+'_'+str(j)+'_'+str(b)]=int(d_dict['d'+str(i)+'_'+str(j)+'_'+str(b)]<1)*a_dict['a'+str(j)+'_'+str(b)]

        
        for i in range(1,self.N+1):
            c_similar_sum=0
            c_diff_sum=0
            for b in range(1,self.B+1):
                center=find_center(b,a_dict,self.N)
                if center==-1:
                    continue
                if funtion_ij(population,i,center):
                    c_similar_sum+=c_dict['c'+str(i)+'_'+str(center)+'_'+str(b)]
                else:
                    c_diff_sum+=c_dict['c'+str(i)+'_'+str(center)+'_'+str(b)]
                    
            if c_similar_sum==0:
                k_dict['k'+str(i)]=1
            else:
                k_dict['k'+str(i)]=0
                
            if c_diff_sum>=1:
                n_dict['n'+str(i)]=1
            else:
                n_dict['n'+str(i)]=0
        
        omega_dict={'o'+str(b)+'_'+str(dim):value for b,tup in self.non_zero_omega.items() for dim,value in tup}
        dim=len(self.non_zero_omega.values()[0])   
        v_dict={'v'+str(v):1 if 'v'+str(v) in self.non_zero_features else 0 for v in range(1,dim+1) }
        dict_list=[k_dict,n_dict,a_dict,b_dict,c_dict,omega_dict,d_dict,v_dict]
        [variable_dict.update(di) for di in dict_list]
        
        #print pp.pprint(variable_dict.items())
        print 'num warm start variables =',len(variable_dict.keys())
        self.variable_dict=variable_dict   
        self.complete_solution=True
        return variable_dict       
    
    def calculate_objective(self,population,dimension):
        if self.complete_solution==True:
            k_list,n_list,ajb_list,beta_list,c_list,v_list,omega,centers,sum_omega=extract(self.variable_dict,population,dimension)
#            print 'k_list in objective',k_list
#            print 'n_list in objective',n_list  
#            print 'c_list in objective',c_list              
#            self.__str__()            
            return calculate_objective(k_list,n_list,ajb_list,beta_list,c_list,v_list,omega,centers,sum_omega,lambda_array[0])
        else:
            return -1

    
 


def d(i,j,b,points,omega):
    '''Given alpha_jb and omega_b calculate d_ijb'''
    difference=np.matrix(points[i][1:])-np.matrix(points[j][1:]) #row vector
    dim=len(omega[str(b)])  
    omega_b=np.zeros(shape=(dim,dim))
    if b not in omega.keys():
        return 10e6
    
    omega_list= omega[str(b)]
    for l,value in omega_list:
        omega_b[int(l)-1][int(l)-1]=value
    d=float(difference*np.matrix(omega_b)*difference.transpose())  
    return d

def find_center(ball,a_dict,N):
    center=''.join(str(j) for j in range(1,N+1) if a_dict['a'+str(j)+'_'+str(ball)]==1)    
    if center!='':
        return int(center)
    else:
        return -1
    

def funtion_ij(points,i,j):
    '''Return true if i and j have the same class.'''
    return int(points[i][0]==points[j][0])   
    
def f(x):
    if x==0:
        return 100000000000
    else:
        return 2*1/math.sqrt(x)

def create_points(points_file,test_file):
    '''Read the points and create a data structure'''    
    points=collections.OrderedDict()
    names_dict={}
    
    #train
    with open(points_file,'r') as f:
        content=f.readlines()         
        header=content[0:HEADER]
        header=','.join(header)
        header_items=header.strip().split(',')
        dimensions=len(header_items)-2 
        header=header_items[2:]
        
        point_number=1
        for line in content[HEADER:]:           
            if line is not ' ':
                point_info=line.strip().split(',')
                if 'NA' in point_info:
                    continue
                data=map(float,point_info[1:])
                data=[round(num,3) for num in data]
                names_dict[point_number]=point_info[0]
                points[point_number]=np.array(data) 
                point_number+=1

    N=len(points.keys())
    
    
    df=pandas.DataFrame.from_dict(points,'index')
    cols_to_norm = [i for i in xrange(1,dimensions+1)]
    print 'mean,max,min : ',df.mean(),df.max(),df.min()
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    points=df.transpose().to_dict('list')        
        
    #test
    test_points=collections.OrderedDict()
    
    with open(test_file,'r') as f:
        content=f.readlines()         
        
        point_number=1
        for line in content[HEADER:]:           
            if line is not ' ':
                point_info=line.strip().split(',')
                if 'NA' in point_info:
                    continue
                data=map(float,point_info[1:])
                data=[round(num,3) for num in data]
                test_points[point_number]=np.array(data) 
                point_number+=1

    df=pandas.DataFrame.from_dict(test_points,'index')
    cols_to_norm = [i for i in xrange(1,dimensions+1)]
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    test_points=df.transpose().to_dict('list')
             
    return points,N,dimensions,names_dict,header,test_points
    
def create_fold_points(data_set,fold):
    '''Read the points and create a data structure
    run this for each fold to normalize    
    ''' 
    train_file='data/'+data_set+'/fold'+str(fold)+'/train/fold_'+str(fold)+'_train.csv'
    test_file='data/'+data_set+'/fold'+str(fold)+'/test/fold_'+str(fold)+'_test.csv'
    HEADER=1
    
    #train
    train_points=collections.OrderedDict()
    names_dict={}
    with open(train_file,'r') as f:
        content=f.readlines()         
        header=content[0:HEADER]
        write_header=header
        header=','.join(header)
        header_items=header.strip().split(',')
        dimensions=len(header_items)-2 
        header=header_items[2:]
        
        point_number=1
        for line in content[HEADER:]:           
            if line is not ' ':
                point_info=line.strip().split(',')
                if 'NA' in point_info:
                    continue
                data=map(float,point_info[1:])
                data=[round(num,3) for num in data]
                names_dict[point_number]=point_info[0]
                train_points[point_number]=np.array(data) 
                point_number+=1

    N=len(train_points.keys())
    
    #normalize train
    df_train=pandas.DataFrame.from_dict(train_points,'index')
    df_train_normalized=pandas.DataFrame.from_dict(train_points,'index')
    cols_to_norm = [i for i in xrange(1,dimensions+1)]
    df_train_normalized[cols_to_norm] = df_train[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    train_points=df_train_normalized.transpose().to_dict('list')
    
    with open('data/'+data_set+'/fold'+str(fold)+'/train/fold_'+str(fold)+'_train_normalized.csv','w') as f:
            f.write(','.join(write_header))            
            for k in train_points.keys():
                f.write(names_dict[k]+','+','.join(map(str,train_points[k]))+'\n')
          
    
    #test
    test_points=collections.OrderedDict()
    names_dict_train={}
    with open(test_file,'r') as f:
        content=f.readlines()         
        point_number=1
        for line in content[HEADER:]:           
            if line is not ' ':
                point_info=line.strip().split(',')
                if 'NA' in point_info:
                    continue
                data=map(float,point_info[1:])
                names_dict_train[point_number]=point_info[0]
                data=[round(num,3) for num in data]
                test_points[point_number]=np.array(data) 
                point_number+=1

     
    #normalize test
    df_test=pandas.DataFrame.from_dict(test_points,'index')
    df_test_normalized=pandas.DataFrame.from_dict(test_points,'index')
    df_test_normalized[cols_to_norm] = (df_test[cols_to_norm]-df_train.mean()[cols_to_norm])/(df_train.max()[cols_to_norm]-df_train.min()[cols_to_norm])    
    test_points=df_test_normalized.transpose().to_dict('list')
    
    with open('data/'+data_set+'/fold'+str(fold)+'/test/fold_'+str(fold)+'_test_normalized.csv','w') as f:
        f.write(','.join(write_header))        
        for k in test_points.keys():
            f.write(names_dict_train[k]+','+','.join(map(str,test_points[k]))+'\n')
              
    return train_points,N,dimensions,names_dict,header,test_points
    
   
def populate_objective(prob,lambda_o,B,N,dimensions):
    '''Populate objective values'''
    #objective coeffcients    
    k_coefficeints=[C_k]*N
    n_coefficeints=[C_n]*N
    a_coefficeints=[C_alpha]*N*B
    b_coefficeints=[C_beta]*N
    c_coefficeints=[C_cijb]*N*N*B
    omega_coefficeints=[lambda_o] *dimensions*B
    d_coefficeints=[C_d]*N*N*B
    feature_coefficients=[C_v]*dimensions  
    #uppder bounds
    k_upper=[1.0]*N
    n_upper=[1.0]*N
    a_upper=[1.0]*N*B
    b_upper=[1.0]*N
    c_upper=[1.0]*N*N*B
    omega_upper=[cplex.infinity] *dimensions*B
    d_upper=[cplex.infinity]*N*N*B 
    feature_coefficients_upper=[1.0]*dimensions   
    #lower bounds
    k_lower=[0.0]*N
    n_lower=[0.0]*N
    a_lower=[0.0]*N*B
    b_lower=[0.0]*N
    c_lower=[0.0]*N*N*B
    omega_lower=[0.0] *dimensions*B
    d_lower=[0.0]*N*N*B
    feature_coefficients_lower=[0.0]*dimensions
    #type
    ctype='I'*len(k_coefficeints)+'I'*len(n_coefficeints)+\
          'I'*len(a_coefficeints)+'I'*len(b_coefficeints)+\
          'I'*len(c_coefficeints)+\
          'C'*len(omega_coefficeints)+'C'*len(d_coefficeints)+\
          'I'*len(feature_coefficients)
          
    k,n,a,b,c,omega,d,features=([] for _ in range(8))

    
    for i in xrange(N):
        k.append('k'+str(i+1))
        n.append('n'+str(i+1))
        b.append('b'+str(i+1))
        
        for ball in xrange(B):
            a.append('a'+str(i+1)+'_'+str(ball+1))
            for j in xrange(N):
                c.append('c'+str(i+1)+'_'+str(j+1)+'_'+str(ball+1))
                d.append('d'+str(i+1)+'_'+str(j+1)+'_'+str(ball+1))
                                
    for ball in xrange(B):
        for dim in xrange(dimensions):
            omega.append('o'+str(ball+1)+'_'+str(dim+1))
            
    for dim in xrange(dimensions):
        features.append('v'+str(dim+1))
                           
    columns=k+n+a+b+c+omega+d+features
        
    name_index_dict=dict(zip(columns,[i for i in xrange(len(columns))]))
        
    coefficiants=k_coefficeints+n_coefficeints+a_coefficeints+b_coefficeints\
                 +c_coefficeints+omega_coefficeints+d_coefficeints+feature_coefficients
    upper=k_upper+n_upper+a_upper+b_upper+c_upper+omega_upper+d_upper+feature_coefficients_upper
    lower=k_lower+n_lower+a_lower+b_lower+c_lower+omega_lower+d_lower+feature_coefficients_lower 

    prob.objective.set_sense(prob.objective.sense.minimize) 
        
    
    prob.variables.add(obj = coefficiants, lb = lower, ub = upper, types = ctype,names = columns) #set objective related values                        
    return name_index_dict
        

def generate_constraints(prob,points,N,dimensions,B,name_index_dict,alt_type,omega_dict,v_list,alpha_prev,beta_prev,feature_dict):
    '''Create constraints'''    
    print 'generating constraints for '
    print 'N=',N,',dimensions=',dimensions,',balls=',B
    d_s,d_coefficients=[],[]
    alphas,alpha_co=[],[]
    beta,beta_co=[],[]
    omega,omega_co=[],[]
    ksi,ksi_co=[],[]
    eta,eta_co=[],[]
    alpha_d_c,alpha_d_c_co=[],[]
    alpha_c,alpha_c_co=[],[]
    epsilon,epsilon_co=[],[]
    ellipse_size,ellipse_size_co=[],[]
    features,features_co=[],[]
    my_sense=''
    
    alpha_constraints_value=1.0
    beta_constraints_value=0.0
    #omega_constraints_value=1000000.0
    d_constraints_value=0.0
    ksi_constraints_value=-1.0
    eta_constraints_value=0.0
    alpha_d_c_constraints_value=0.0
    alpha_c_constraints_value=0.0
    epsilon_constraints_value=1-EPSILON+M
    ellipse_constraints_w2_value=w2   
    ellipse_constraints_w1_value=w1
    feature_constraints_value=w1*B
    #alternating minimization
    
    random_sample=[]
    #step 0
    if not alpha_prev and not beta_prev:        
        random_sample=random.sample([i for i in xrange(1,N+1)],B) 
        print 'initialize alpha beta by randomly sampling',random_sample
    print 
    
    t1=time.time()
    #alpha constraints  
    alpha_constraints_1=[]
    alpha_constraints_2=[]
    if alt_type=='proto':
        for b in xrange(B):
            temp_alphas=[]
            temp_alpha_co=[]
            for j in xrange(N):
                temp_alphas.append(name_index_dict['a'+str(j+1)+'_'+str(b+1)])
                temp_alpha_co.append(1.0)
            alphas.append(temp_alphas)
            alpha_co.append(temp_alpha_co)
        alpha_constraints=map(list,zip(alphas,alpha_co))
        for i in xrange(len(alpha_constraints)):         
            my_sense+='L'    
    elif alt_type=='space' and random_sample:    #step 0                           
        for p,ball in product(xrange(N),xrange(B)):           
            my_sense+='E'
            point_index=p+1
            if point_index in random_sample and random_sample.index(point_index)==ball:
                alpha_constraints_1.append([[name_index_dict['a'+str(point_index)+'_'+str(random_sample.index(point_index)+1)]],[1.0]])                
            else:
                alpha_constraints_2.append([[name_index_dict['a'+str(point_index)+'_'+str(ball+1)]],[1.0]]) 
    elif alt_type=='space' and not random_sample:  
        print 'previous alpha',alpha_prev
        for p,ball in product(xrange(N),xrange(B)):           
            my_sense+='E'
            point_index=p+1
            ball_index=ball+1
            if tuple([str(point_index),str(ball_index)]) in alpha_prev:
                print 'found previous alpha'
                alpha_constraints_1.append([[name_index_dict['a'+str(point_index)+'_'+str(ball_index)]],[1.0]])                
            else:
                alpha_constraints_2.append([[name_index_dict['a'+str(point_index)+'_'+str(ball_index)]],[1.0]]) 
  
    #betas constraints
    beta_constraints_1=[]
    beta_constraints_2=[]
    if alt_type=='proto':
        for j in xrange(N):
            temp_beta=[]
            temp_beta_co=[]
            for b in xrange(B):
                temp_beta.append(name_index_dict['a'+str(j+1)+'_'+str(b+1)])
                temp_beta_co.append(1.0)
            temp_beta.append(name_index_dict['b'+str(j+1)])
            temp_beta_co.append(-1.0*B)
            beta.append(temp_beta)
            beta_co.append(temp_beta_co) 
        beta_constraints=map(list,zip(beta,beta_co)) 
        for i in xrange(len(beta_constraints)): 
            my_sense+='L'    
    elif alt_type=='space' and random_sample:                                
        for p in xrange(N):            
            my_sense+='E'
            point_index=p+1
            if point_index in random_sample:
                beta_constraints_1.append([[name_index_dict['b'+str(point_index)]],[1.0]])
            else:
                beta_constraints_2.append([[name_index_dict['b'+str(point_index)]],[1.0]])
    elif alt_type=='space' and not random_sample: 
        print 'previous beta',beta_prev                 
        for p in xrange(N):            
            my_sense+='E'
            point_index=p+1
            if ('b'+str(point_index)) in beta_prev:
                print 'found previous beta'
                beta_constraints_1.append([[name_index_dict['b'+str(point_index)]],[1.0]])
            else:
                beta_constraints_2.append([[name_index_dict['b'+str(point_index)]],[1.0]])
         
    
    
    omega_constraints=[]
    d_constraints=[]
    new_omega_constraints=[]
    new_omega_value=[]
    
    #d constraints
    d_value=[]
    if alt_type=='space':
        for i in xrange(N):
            for j in xrange(N):
                difference=np.matrix(points[i+1][1:])-np.matrix(points[j+1][1:])
                for b in xrange(B):
                    temp_d_s=[]
                    temp_d_coefficients=[]
                    
                    
                    for dim in xrange(dimensions):
                        name=name_index_dict["o"+str(b+1)+'_'+str(dim+1)]
                        temp_d_coefficients.append(difference[0,dim]**2)                       
                        temp_d_s.append(name)
                    temp_d_s.append(name_index_dict['d'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])
                    temp_d_coefficients.append(-1)   
                    d_s.append(temp_d_s)
                    d_coefficients.append(temp_d_coefficients)
        d_constraints=map(list,zip(d_s,d_coefficients))
        for i in xrange(len(d_constraints)):
            my_sense+='E'
    elif alt_type=='proto':
        print  'previous omega',pp.pprint(omega_dict.items())                       
        for i,j,b in product(xrange(N),xrange(N),xrange(B)): 
            ball_idx=b+1            
            #if ball_idx in omega_dict.keys():                        
            d_constraints.append([[name_index_dict['d'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)]],[1.0]])
            d_value.append(d(i+1,j+1,str(b+1),points,omega_dict))
#            else:
#                d_constraints.append([[name_index_dict['d'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)]],[1.0]])
#                d_value.append(0.0)
            
        for b,dim in product(xrange(B),xrange(dimensions)):
#            ball_idx=b+1
#            if ball_idx in omega_dict.keys():       
            for tup in omega_dict[str(b+1)]:
                if tup[0]==dim+1:
                    new_omega_constraints.append([[name_index_dict["o"+str(b+1)+'_'+str(dim+1)]],[1.0]])
                    new_omega_value.append(tup[1])
            
    #ksi,eta,(alpha,d,c) constraints   
    for i in xrange(N):
        temp_ksi=[]
        temp_ksi_co=[]
        temp_eta=[]
        temp_eta_co=[]
        for b in xrange(B):
            for j in xrange(N):
                temp_alpha_d_c=[]
                temp_alpha_d_c_co=[]
                temp_alpha_c=[]
                temp_alpha_c_co=[]
                temp_epsilon=[]
                temp_epsilon_co=[]
                
                #ksi
                temp_ksi.append(name_index_dict['c'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])
                temp_ksi_co.append(-1*funtion_ij(points,i+1,j+1))
                
                #eta
                temp_eta.append(name_index_dict['c'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])
                temp_eta_co.append(int(not(funtion_ij(points,i+1,j+1))))
                #(alpha,d,c)
                temp_alpha_d_c.append(name_index_dict['a'+str(j+1)+'_'+str(b+1)])
                temp_alpha_d_c.append(name_index_dict['d'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])
                temp_alpha_d_c.append(name_index_dict['c'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])
                temp_alpha_d_c_co.append(1.0)
                temp_alpha_d_c_co.append(-1.0)
                temp_alpha_d_c_co.append(-1.0)
                #(alpha,d,c)
                alpha_d_c.append(temp_alpha_d_c)
                alpha_d_c_co.append(temp_alpha_d_c_co)
                #(alpha,c)
                temp_alpha_c.append(name_index_dict['a'+str(j+1)+'_'+str(b+1)])
                temp_alpha_c.append(name_index_dict['c'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])
                temp_alpha_c_co.append(-1.0)
                temp_alpha_c_co.append(1.0)
                alpha_c.append(temp_alpha_c)
                alpha_c_co.append(temp_alpha_c_co)
                # epsilon, M constraints
                temp_epsilon.append(name_index_dict['d'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])  
                temp_epsilon.append(name_index_dict['c'+str(i+1)+'_'+str(j+1)+'_'+str(b+1)])              
                temp_epsilon_co.append(1.0)
                temp_epsilon_co.append(M)
                epsilon.append(temp_epsilon)
                epsilon_co.append(temp_epsilon_co)
             
        #ksi
        temp_ksi.append(name_index_dict['k'+str(i+1)])
        temp_ksi_co.append(-1.0)
        
        ksi.append(temp_ksi)
        ksi_co.append(temp_ksi_co)
        #eta
        temp_eta.append(name_index_dict['n'+str(i+1)])
        temp_eta_co.append( -1.0*B)
        eta.append(temp_eta)
        eta_co.append(temp_eta_co)    
        
    ksi_constraints=map(list,zip(ksi,ksi_co))
    eta_constraints=map(list,zip(eta,eta_co))    
    alpha_d_c_constraints=map(list,zip(alpha_d_c,alpha_d_c_co))   
    alpha_c_constraints=map(list,zip(alpha_c,alpha_c_co))    
    epsilon_constraints=map(list,zip(epsilon,epsilon_co))
    
    for i in xrange(len(ksi_constraints+eta_constraints+alpha_d_c_constraints+alpha_c_constraints+epsilon_constraints)):
        my_sense+='L'
        
    #ellipse constraints I 
    for b in xrange(B):
        for dim in xrange(dimensions):
            temp_ellipse_size=[]
            temp_ellipse_size_co=[]
            temp_ellipse_size.append(name_index_dict['o'+str(b+1)+'_'+str(dim+1)])
            temp_ellipse_size_co.append(1.0)
            ellipse_size.append(temp_ellipse_size)
            ellipse_size_co.append(temp_ellipse_size_co)
    
    #ellipse_constraints=map(list,zip(ellipse_size,ellipse_size_co))        
    #ellipse constraints II
    for b in xrange(B):
        for dim in xrange(dimensions):
            temp_ellipse_size=[]
            temp_ellipse_size_co=[]
            temp_ellipse_size.append(name_index_dict['o'+str(b+1)+'_'+str(dim+1)])
            temp_ellipse_size_co.append(1.0)
            ellipse_size.append(temp_ellipse_size)
            ellipse_size_co.append(temp_ellipse_size_co)
            
    ellipse_constraints=map(list,zip(ellipse_size,ellipse_size_co))   
    for i in xrange(len(ellipse_constraints)/2):
        my_sense+='L'
    for i in xrange(len(ellipse_constraints)/2):
        my_sense+='G'
            
    #feature constraints
    feature_constraints_1=[]
    feature_constraints_2=[]
    if alt_type=='space':
        for b,l in product(xrange(B),xrange(dimensions)):
            temp_features=[]
            temp_features_co=[]
            
            temp_features.append(name_index_dict['o'+str(b+1)+'_'+str(l+1)])            
            temp_features_co.append(1.0)
            
            temp_features.append(name_index_dict['v'+str(l+1)])
            temp_features_co.append(-1.0*w2)
            
            features.append(temp_features)
            features_co.append(temp_features_co)
        feature_constraints=map(list,zip(features,features_co))
        for i in xrange(len(feature_constraints)):
            my_sense+='L'  
    elif alt_type=='proto':
        print 'previous v',[feature_dict[v] for v in v_list]
        for l in xrange(dimensions):
            my_sense+='E'            
            if ('v'+str(l+1)) in v_list: 
                feature_constraints_1.append([[name_index_dict['v'+str(l+1)]],[1.0]])                                
            else:
                feature_constraints_2.append([[name_index_dict['v'+str(l+1)]],[1.0]])                
                              
    
    my_rownames=[]  
    rows=[]
    value_tuple=[]
    my_rhs=[]
   
    
    if alt_type=='space':              
        for row in xrange(len(alpha_constraints_1+alpha_constraints_2+beta_constraints_1+beta_constraints_2+omega_constraints+d_constraints+ksi_constraints+eta_constraints+alpha_d_c_constraints+alpha_c_constraints+epsilon_constraints+ellipse_constraints+feature_constraints)):
            my_rownames.append('r'+str(row+1))
        
        rows=[cplex.SparsePair(elm[0],elm[1]) for elm in alpha_constraints_1+alpha_constraints_2+beta_constraints_1+beta_constraints_2+omega_constraints+d_constraints+ksi_constraints+eta_constraints+alpha_d_c_constraints+alpha_c_constraints+epsilon_constraints+ellipse_constraints+feature_constraints]             
        value_tuple=[(alpha_constraints_1,1.0),(alpha_constraints_2,0.0),
                    (beta_constraints_1,1.0),(beta_constraints_2,0.0),
                    (d_constraints,d_constraints_value),
                    (ksi_constraints,ksi_constraints_value),(eta_constraints,eta_constraints_value),
                    (alpha_d_c_constraints,alpha_d_c_constraints_value),(alpha_c_constraints,alpha_c_constraints_value),
                    (epsilon_constraints,epsilon_constraints_value),(ellipse_constraints[:len(ellipse_constraints)/2],ellipse_constraints_w2_value),
                    (ellipse_constraints[len(ellipse_constraints)/2:],ellipse_constraints_w1_value),
                    (feature_constraints,feature_constraints_value)]   
        my_rhs=list(itertools.chain.from_iterable([[b]*len(a) for a,b in value_tuple]))      
    elif alt_type=='proto':
        for row in xrange(len(alpha_constraints+beta_constraints+omega_constraints+ksi_constraints+eta_constraints+alpha_d_c_constraints+alpha_c_constraints+epsilon_constraints+ellipse_constraints+feature_constraints_1+feature_constraints_2)):
            my_rownames.append('r'+str(row+1))          
        rows=[cplex.SparsePair(elm[0],elm[1]) for elm in alpha_constraints+beta_constraints+omega_constraints+ksi_constraints+eta_constraints+alpha_d_c_constraints+alpha_c_constraints+epsilon_constraints+ellipse_constraints+feature_constraints_1+feature_constraints_2]             
        value_tuple=[(alpha_constraints,alpha_constraints_value),(beta_constraints,beta_constraints_value),
                    (ksi_constraints,ksi_constraints_value),(eta_constraints,eta_constraints_value),
                    (alpha_d_c_constraints,alpha_d_c_constraints_value),(alpha_c_constraints,alpha_c_constraints_value),
                    (epsilon_constraints,epsilon_constraints_value),
                    (ellipse_constraints[:len(ellipse_constraints)/2],ellipse_constraints_w2_value),
                    (ellipse_constraints[len(ellipse_constraints)/2:],ellipse_constraints_w1_value),
                    (feature_constraints_1,1.0),(feature_constraints_2,0.0)]                   
        my_rhs=list(itertools.chain.from_iterable([[b]*len(a) for a,b in value_tuple]))
        
        for row in xrange(len(d_constraints+new_omega_constraints)):
            my_sense+='E'
            my_rownames.append('r'+str(len(my_rownames)+row+1)) 
        
        d_rows=[cplex.SparsePair(elm[0],elm[1]) for elm in d_constraints+new_omega_constraints]  
        rows=rows+d_rows  
                
        my_rhs=my_rhs+d_value+new_omega_value
        
#        for i in xrange(len(my_rhs)):
#            print rows[i],my_sense[i],my_rhs[i]
                              
    
    t2=time.time()
    time_in_loop=t2-t1
    a=time.time()
    prob.linear_constraints.add(names = my_rownames,lin_expr = rows,
                                rhs = my_rhs,senses = my_sense) 
    b=time.time()
    add_time=round(b-a,2)
    
    return add_time,time_in_loop   
    
def extract(result,points,dimensions):
    '''Takes in a result dictionary,points and extracts the variable values into lists'''
    k_list,n_list,ajb_list,beta_list,c_list,v_list=([] for _ in range(6)) 
    d_list=[]
    omega=collections.defaultdict(list)    
    centers={}#collections.defaultdict(list)    
    
    #extract values from the solved problem    
    for key,value in result.items():
        if key[0]=='a'and abs(value-1.0)<.001:
            objective_variables=key[1:].split('_')                       
            j=objective_variables[0]
            b=objective_variables[1]
            ajb_list.append((j,b))
            for d in xrange(1,3):
                centers[b]=tuple(list(points[int(j)][1:])+[j,b])            
        if key[0]=='o':
            objective_variables=key[1:].split('_') 
            ball=objective_variables[0]
            L=int(objective_variables[1])
            omega[ball].append((L,value))
                 
        if key[0]=='k' and abs(value-1.0)<.001:
            k_list.append(key)
        if key[0]=='n' and abs(value-1.0)<.001:
            n_list.append(key)
        if key[0]=='b' and abs(value-1.0)<.001:
            beta_list.append(key)
        if key[0]=='c' and abs(value-1.0)<.001:
            c_list.append(key)
        if key[0]=='v' and abs(value-1.0)<.001:
            #print 'v',value
            v_list.append(key)
        if key[0]=='d':
            d_list.append(value)
    #print 'centers',centers.items()        
    print 'max d',max(d_list)
    sum_omega=0
    sum_omega=sum([tup[1] for item in omega.values() for tup in item])
    
    #append a 0 to missing dimensions
    for ball,omega_list in omega.items():
        if len(omega_list)!=dimensions:
            non_zero_d=zip(*omega_list)[0]
            missing_d= set([i for i in xrange(1,dimensions+1)]).difference(non_zero_d)
            for d in missing_d:
                omega_list.append((d,0))
             
    return k_list,n_list,ajb_list,beta_list,c_list,v_list,omega,centers,sum_omega

def calculate_objective(k_list,n_list,ajb_list,beta_list,c_list,v_list,omega,centers,sum_omega,lambda_o):
    objective_val=len(k_list)*C_k+len(n_list)*C_n+len(ajb_list)*C_alpha+len(beta_list)*C_beta+sum_omega*lambda_o+len(v_list)*C_v    
    return objective_val
    

def extract_and_output_solution(points_fn,report_file,points,N,dimensions,my_prob,B,lambda_o,pdf_pages,time_to_solve,parameter_str,names_dict,header,iteration,alt_type,feature_dict,class_dict):
    '''Extract variables,values from solved MIP and create output'''    
    num_variables = my_prob.variables.get_num()
    values     = my_prob.solution.get_values()   
    result={}
    
    
              
    result=dict(zip(my_prob.variables.get_names(),values))
    
    if print_result_variables==True:
        write_file=open('./output/'+version+'_plots/sols/'+report_file+'_'+str(B)+'balls.csv','w') 
        types      = my_prob.variables.get_types()    
        for j in range(num_variables):
            write_file.write("{},{},{},{}\n".format(j,my_prob.variables.get_names()[j],values[j],types[j]))
            
    
    print 'start extract'
    k_list,n_list,ajb_list,beta_list,c_list,v_list,omega,centers,sum_omega=extract(result,points,dimensions)   
    print 'end extract'    
    sol=Solution(N,B,ajb_list,beta_list,omega,v_list,points_fn,report_file)
         
#    if supress_write==False:    
#        sol.write() 
#        if alt_type=='proto':
#            sol.write_2(iteration,alt_type,points,dimensions)
#            print 'the objective of the solution is ',sol.calculate_objective(points,dimensions)
#            print 'writing mst'
#            my_prob.MIP_starts.write('./test_all_'+str(iteration)+'.mst')
            

    print_solution(my_prob,points,dimensions,sol,k_list,n_list,c_list,centers,sum_omega,lambda_o,pdf_pages,time_to_solve,parameter_str,names_dict,header,iteration,alt_type,report_file,feature_dict,class_dict)  
    print 'end extract_and_output_solution'    
    return pdf_pages,omega,v_list,ajb_list,beta_list,k_list,n_list,sum_omega,my_prob,centers
    

def setup_objective_and_contraints(prob,B,lambda_o,points,N,dimensions,alt_type,omega,v_list,alpha_prev,beta_prev,feature_dict,iteration):
    '''Add objective, constraints to MIP'''
    obj1=time.time()
    name_index_dict=populate_objective(prob,lambda_o,B,N,dimensions) #set objective  
    obj2=time.time()
    obj_time=round(obj2-obj1,2)
        
    const1=time.time()
    add_time,time_in_loop=generate_constraints(prob,points,N,dimensions,B,name_index_dict,alt_type,omega,v_list,alpha_prev,beta_prev,feature_dict) #set constraints                                   
    const2=time.time()
    const_time=round(const2-const1,2)   
    if supress_write!=True:    
        prob.write('./output/'+version+'_plots/'+version+'_'+report_file+'_'+str(iteration)+'_'+alt_type+'.lp') #write constraint file
        
    return obj_time,const_time,add_time,time_in_loop

def create_MIP_start_sparsePair(ws_solution,sampled_points,population,N,B,dimension): 
    '''Take a previous warm start solution and create a cplex sparse pair'''
    mapping=[(index,pop_index) for index,point in sampled_points.items() for pop_index,pop_point in population.items() if list(point)==list(pop_point)]          
    ws_solution.change(mapping,N,B)
    warm_start_dict=ws_solution.create_complete_solution(population)
    print 'ws objective :',ws_solution.calculate_objective(population,dimension)
    
    return cplex.SparsePair(warm_start_dict.keys(),warm_start_dict.values())

def sample_points(sample_point_names):
    '''create a list of points based on the names in sample_point_names'''

    sampled_points_list=[]
    print sample_point_names
    for name in list(sample_point_names):
        points={}
        with open('./data/'+name,'r') as f:
            content=f.readlines()  
            point_number=1
            for line in content[HEADER:]:           
                if line is not ' ':
                    point_info=line.strip().split(',')
                    if 'NA' in point_info:
                        continue
                    data=map(float,point_info[1:])
                    data=[round(num,3) for num in data]
                    points[point_number]=np.array(data) 
                    point_number+=1 
            sampled_points_list.append(points)
            
    
            
    
    return sampled_points_list

def read(ws_sol_names):
    '''Create a list of previously solved solutions'''
    warm_start_solution_list=[]
    
    for sol_name in ws_sol_names:    
        with open('./output/'+version+'_plots/'+sol_name,'rb') as object_read:
            try :            
                obj=pickle.load(object_read)
                warm_start_solution_list.append(obj)
            except :
                print 'error reading warm start solution'
                pass
        
    return warm_start_solution_list
    
def read_2():
    '''Create a list of previously solved solutions'''
    warm_start_solution_list=[]
    
    saved_sols=os.listdir('./output/'+version+'_plots/saved_iteration_sols/')
    
    for sol_name in saved_sols:    
        with open('./output/'+version+'_plots/saved_iteration_sols/'+sol_name,'rb') as object_read:
            try :            
                obj=pickle.load(object_read)
                warm_start_solution_list.append(obj)
            except :
                print 'error reading warm start solution'
                pass
        
    return warm_start_solution_list

def warm_start(my_prob,points,N,B,sample_point_names,ws_sol_names,dimension):   
    '''Read a previously saved solution, modify it to fit the current problem and add it'''
    population=points    
    sampled_points_list=sample_points(sample_point_names)    
    warm_start_solution_list=read(ws_sol_names)         
    
    if len(sampled_points_list)!=len(warm_start_solution_list):
        raise 'warm start read exception'
        
    index=1
    for ws_sol,sample in zip(warm_start_solution_list,sampled_points_list):   
        warm_start_sp=create_MIP_start_sparsePair(ws_sol,sample,population,N,B,dimension)
        print 'adding warm start'        
        my_prob.MIP_starts.add(warm_start_sp, my_prob.MIP_starts.effort_level.check_feasibility, "ws"+str(index))
        index+=1
 

def create_ws_lists(N):   
    sample_point_names=[]
    ws_sol_names=[]
    with open('./output/'+version+'_plots/ws_info.txt','r') as f:
        for line in f:
            sample_point_names.append(line.strip().split(',')[0])
            ws_sol_names.append(line.strip().split(',')[1])
    
    return sample_point_names,ws_sol_names

def run_MIP(points_fn,report_file,points,N,dimensions,B,lambda_o,pdf_pages,names_dict,header,test_points):
    time_str='\n' 
    parameter_str=''
    class_dict={1:'Breakfast Cereals',2:'Snacks',3:'Sweets'}
    feature_dict={'v1':'Sugar','v2':'Iron','v3':'Cholestrl'}
    
    #    class_dict={1:'High Gain',2:'Low Gain',3:'no change',4:'Low Loss',5:'High Loss'}
    #    feature_dict={'v1':'rank momentum','v2':'pit time','v3':'tire age'} 
    omega={}
    v_list=[]
    alpha=[]
    beta=[]
    
    alternate_iterations=collections.defaultdict(list)
    
    iteration=0
    my_prob_prev=[]
    ws_list=[]
    prev_iter=0
    prev_alt_type='none'  
    cumulative_objective=[]
    cumulative_time=[]
    objective_anno={}
    
    for alt_run in xrange(2*alt_runs): 
        error_flag=0
        if alt_run%2==0:
            alt_type='space'
        else:
            alt_type='proto'
        print '\n----------------------------------------------'   
        iteration=(alt_run+2)/2
        print 'iteration: ',iteration,', type: ',alt_type
        try:
            my_prob = cplex.Cplex()
            
            obj_time,const_time,add_time,time_in_loop=setup_objective_and_contraints(my_prob,B,lambda_o,points,N,dimensions,alt_type,omega,v_list,alpha,beta,feature_dict,iteration)
            
            num_variables=my_prob.variables.get_num()      
            num_constraints=my_prob.linear_constraints.get_num()
            time_str+='Run '+str(iteration)+' | '+alt_type+'\n'
            time_str+='obj setup time= '+str(obj_time)+'s | constr setup time= '+str(const_time)+'s (add time= '+str(add_time)+'s | loop= '+str(round(time_in_loop,2))+'s)\n'
            
            print 'time to setup objective and constraints',obj_time+const_time,'sec'        
            print 'num_variables for original problem : ',my_prob.variables.get_num()
           
            if apply_run_parametrs==True:        
                parameter_str=set_MIP_run_parameters(my_prob)
                
            if warm_start_flag==True: 
                
                if my_prob_prev!=[]:
                    a=time.time()  
                    print 'starting warm start'
                    print 'length of ws varibales',len(my_prob_prev.variables.get_names()) 
                    #print 'does ws and current variable names match',my_prob_prev.variables.get_names()==my_prob.variables.get_names()
                    sp=cplex.SparsePair(my_prob_prev.variables.get_names(),my_prob_prev.solution.get_values())
                    ws_list.append(sp)
                    #for i,pair in enumerate(ws_list):
                    my_prob.MIP_starts.add(sp,4,"ws"+str(prev_iter)+'_'+prev_alt_type)
                    #my_prob.MIP_starts.add(sp,4,"ws"+str(i))
                    b=time.time()
                    ws_time=round(b-a,2)
                    time_str+='ws_setup_time= '+str(ws_time)+'s\n'
                    print 'final warm starts: ',my_prob.MIP_starts.get_num()             
                    print 'final ws names: ',my_prob.MIP_starts.get_names()
                    print 'final ws effort elevels: ',my_prob.MIP_starts.get_effort_levels()                
                    print 'warm start setup time',ws_time,'sec'    
               
            print '\n'
            print 'start minimization','iteration: ',iteration,', type: ',alt_type+'\n'
            cb = my_prob.register_callback(MyCallback)
            cb.incobjval, cb.bestobjval,cb.times,cb.dettimes, cb.start_dettime = [], [], [], [], []
            
            start=time.time()
            #solve the optimization problem            
            my_prob.solve()
            end=time.time()
            time_to_solve=round(end-start,2)
            print 'cumulative_objective',cumulative_objective
            print 'cb.incobjval',cb.incobjval
            print 'cb.bestobjval',cb.bestobjval
            
            time_str+='MIP solve time= '+str(time_to_solve)+'s\n'
            print '\nstop minimization','iteration: ',iteration,', type: ',alt_type,', time elapsed: ',time_to_solve
            print "Solution value  = ", my_prob.solution.get_objective_value()
            time_str+='#variables= '+str(num_variables)+' | #constraints= '+str(num_constraints)+' \n'
            #append the solution to incumbent objective            
            
            if error_flag!=1:
                cumulative_objective,objective_anno,cumulative_time=plot_objective(cb,pdf_pages,iteration,alt_type,cumulative_objective,objective_anno,my_prob.solution.get_objective_value(),cumulative_time)
            
            
        except CplexError, exc:
            print exc 
            my_prob.conflict.refine_MIP_start(0,my_prob.conflict.all_constraints())
            my_prob.conflict.get()
             
            error_flag=1                       
            continue
        a=time.time()       
        #omega is sent to generate constraints               
        pdf_pages,omega,v_list,alpha,beta,k_list,n_list,sum_omega,my_prob_prev,centers = extract_and_output_solution(points_fn,report_file,points,N,dimensions,my_prob,B,lambda_o,pdf_pages,time_to_solve,parameter_str,names_dict,header,iteration,alt_type,feature_dict,class_dict)
        prev_iter=iteration
        prev_alt_type=alt_type        
        b=time.time()
        extraction_time=round(b-a,2)
       
        alternate_iterations['iter_type'].append(alt_type+'_'+str(iteration))
        alternate_iterations['objective'].append(my_prob.solution.get_objective_value()) 
        alternate_iterations['sol_type'].append(my_prob.solution.status[my_prob.solution.get_status()])           
        alternate_iterations['ksi'].append(len(k_list))
        alternate_iterations['eta'].append(len(n_list))
        alternate_iterations['alpha'].append(len(alpha))
        alternate_iterations['beta'].append(len(beta))
        alternate_iterations['omega'].append(sum_omega)
        alternate_iterations['v'].append(len(v_list))        
        
        
        time_str+='time for extraction= '+str(extraction_time)+'s | '+'\n'            
        print 'time for extraction',extraction_time,'sec'    
        print "Solution value  = ", my_prob.solution.get_objective_value()
        compute_test_performance(alpha,omega,v_list,points,test_points,centers,B,feature_dict,pdf_pages,class_dict,dimensions,lambda_o,iteration,alt_type,header,names_dict)
     
            
    print alternate_iterations.items()
    fig=plt.figure()
    ax = fig.add_subplot(111)    
    ax.plot(xrange(1,len(alternate_iterations['objective'])+1),[round(obj,2) for obj in alternate_iterations['objective']],label='objective')
    ax.legend(fontsize=8)
    plt.xticks(np.arange(1, len(alternate_iterations['objective'])+1, 1.0),alternate_iterations['iter_type'],size=7)
    
    for i,tup in enumerate(zip(alternate_iterations['sol_type'],alternate_iterations['objective'],alternate_iterations['iter_type'])):
        print i+1,tup
        ax.annotate(tup[0], xy=(i+1,tup[1]),  xycoords='data',textcoords='offset points', ha='center', va='bottom',arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'),size=5)
               
    if print_to_report==True:                      
        plt.title('objective')
        plt.xlabel('iterations')
        plt.ylabel('objective value')
        plt.grid(True)
        pdf_pages.savefig(fig)
        plt.close()
       
        fig=plt.figure()
        plt.plot(xrange(1,len(alternate_iterations['ksi'])+1),alternate_iterations['ksi'],label='ksi')
        plt.plot(xrange(1,len(alternate_iterations['eta'])+1),alternate_iterations['eta'],label='eta')
        plt.legend(fontsize=8)
        plt.xticks(np.arange(1, len(alternate_iterations['ksi'])+1, 1.0),alternate_iterations['iter_type'],size=7)
        
        plt.title('alternating iterations')
        plt.ylabel('unweighted contribution to objective')
        plt.xlabel('iterations')
        plt.grid(True)
        pdf_pages.savefig(fig)
        plt.close()
    
        
        fig=plt.figure()
        plt.plot(xrange(1,len(alternate_iterations['alpha'])+1),alternate_iterations['alpha'],label='alpha')
        plt.plot(xrange(1,len(alternate_iterations['beta'])+1),alternate_iterations['beta'],label='beta')
        plt.plot(xrange(1,len(alternate_iterations['v'])+1),alternate_iterations['v'],label='v')
        plt.legend(fontsize=8)
        plt.xticks(np.arange(1, len(alternate_iterations['alpha'])+1, 1.0),alternate_iterations['iter_type'],size=7)       
        plt.title('alternating iterations')
        plt.ylabel('unweighted contribution to objective')
        plt.xlabel('iterations')
        plt.grid(True)
        pdf_pages.savefig(fig)
        plt.close()
        
        fig=plt.figure()
        plt.plot(xrange(1,len(alternate_iterations['omega'])+1),alternate_iterations['omega'],label='sum of omega')
        plt.legend(fontsize=8)
        plt.xticks(np.arange(1, len(alternate_iterations['omega'])+1, 1.0),alternate_iterations['iter_type'],size=7)   
        plt.title('sum of omega')
        plt.ylabel('unweighted contribution to objective')
        plt.xlabel('iterations')
        plt.grid(True)
        pdf_pages.savefig(fig)
        plt.close()    
        plt.close("all")  
    return time_str 


def compute_test_performance(alpha,non_zero_omega,non_zero_features,train_points,test_points,centers,B,feature_dict,pdf_pages,class_dict,dimensions,lambda_o,iteration,alt_type,header,names_dict) :
    
    test_k_list=[]
    test_n_list=[]
    pop_new={}
        
    append_points_dict={int(j):train_points[int(j)]for (j,b) in alpha}   
    for i,v in test_points.items():
        #for each point
        i_label=int(test_points[i][0])
        pop_new[1]=test_points[i]
        temp_ksi=1
        temp_eta=0
        #go through the balls
        for (j,b) in alpha:          
            j=int(j)
            pop_new[2]=train_points[j]
            j_label=int(train_points[j][0])
            dis=1.0-EPSILON
            if i_label==j_label and (d(1,2,b,pop_new,non_zero_omega)<dis or abs(d(1,2,b,pop_new,non_zero_omega)-dis)<1e-15):
                temp_ksi=0                
                
            if i_label!=j_label and (d(1,2,b,pop_new,non_zero_omega)<dis or abs(d(1,2,b,pop_new,non_zero_omega)-dis)<1e-15):
                temp_eta=1
        #update the count 
        if temp_ksi==1:
            test_k_list.append(i)
        if temp_eta==1:
            test_n_list.append(i)
            
    appended_points=dict(test_points)
    
    print 'test k: ',len(test_k_list),' | ',','.join(map(str,test_k_list))+'\n'
    print 'test n: ',len(test_n_list),' | ',','.join(map(str,test_n_list))+'\n'
    
    test_str='test k:'+str(len(test_k_list))+' | '+','.join(map(str,test_k_list))+'\n'
    test_str+='test n:'+str(len(test_n_list))+' | '+','.join(map(str,test_n_list))+'\n'
    
    draw_plot(centers,non_zero_omega,non_zero_features,feature_dict,B,appended_points,pdf_pages,class_dict,dimensions,lambda_o,test_str,iteration,alt_type,header,train_points,'Test',names_dict)
       
    return test_k_list,test_n_list

def draw_plot(centers,omega,v_list,feature_dict,B,points,pdf_pages,class_dict,dimensions,lambda_o,report_str,iteration,alt_type,header,train_points,split,names_dict):
    
    if len(v_list)==2 and print_to_report==True:
        new_points,new_omega,new_centers=convert_dimensions_d_to_Nd(v_list,points,omega,centers,dimensions,2)
        plot_2d(B,new_points,new_centers,new_omega,lambda_o,len(v_list),pdf_pages,report_str,class_dict,v_list,feature_dict,iteration,alt_type,train_points,split,names_dict) 
    if len(v_list)==3 and print_to_report==True:
        new_points,new_omega,new_centers=convert_dimensions_d_to_Nd(v_list,points,omega,centers,dimensions,3)
        plot_3d(B,new_points,new_centers,new_omega,class_dict,pdf_pages,report_str,header,iteration,alt_type,feature_dict,v_list,train_points,split,names_dict)
    else:
        #save text1
        fig=plt.figure()
        plt.axis('off')
        plt.title('Run '+str(iteration)+' | '+alt_type+' | MIP parameters and results')
        plt.text(0, 0.3,report_str,fontsize=8)
        pdf_pages.savefig(fig)

def create_report(B,lambda_array,points_file,report_file,data_set,fold):
    '''Run the MIP and creates a pdf report'''
    t1=time.time()
    train_file='./data/'+str(data_set)+'/fold'+str(fold)+'/train/fold_'+str(fold)+'_train.csv'
    test_file='./data/'+str(data_set)+'/fold'+str(fold)+'/test/fold_'+str(fold)+'_test.csv'

    points,N,dimensions,names_dict,header,test_points=create_fold_points(data_set,fold)
    print 'dimensions',dimensions
    points_fn=str(data_set)+'_'+str(fold)
    pdf_pages = PdfPages('./output/'+version+'_plots/sols/'+version+'_'+report_file+'.pdf')    
         
    if plot_one==False: 
        for b,lam in product(xrange(1,B+1),lambda_array):
                                         
            time_str=run_MIP(points_fn,report_file,points,N,dimensions,b,lam,pdf_pages,names_dict,header,test_points)  #run the MIP for each ball size                    
    else: 
        for lam in lambda_array:
            time_str=run_MIP(points_fn,report_file,points,N,dimensions,B,lam,pdf_pages,names_dict,header,test_points)                     
    
    t2=time.time() 
    total_time=round(t2-t1,2)
    try:
        time_str+='total time= '+str(total_time)+'s'
    except :
        print 'CPLEX Error  1217: No solution exists.'
        pass
    write_time(pdf_pages,time_str)
    pdf_pages.close()

    return time_str

def plot_objective(cb,pdf_pages,iteration,alt_type,cumulative_objective,objective_anno,final_obj,cumulative_time):
     # plot obj value   
    if print_to_report==True:
           
        fig=plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('on')
        plt.xlabel('time (sec)')
        plt.ylabel('Objective value')
        plt.title('Run '+str(iteration)+' | '+alt_type+' | Objective value as search progressed')
    
    cumulative_objective.extend(cb.incobjval) 
    cumulative_objective.append(final_obj)
    new_dettimes=[]
    if cumulative_time:
        new_dettimes=[cumulative_time[-1]+item-cb.start_dettime[0] for item in cb.dettimes]
    else:
        new_dettimes=[item-cb.start_dettime[0] for item in cb.dettimes]
        
    cumulative_time.extend(new_dettimes)    
    
    if new_dettimes:
        cumulative_time.append(new_dettimes[-1]+1)
    else:
        cumulative_time.append(1)
    
    
        
        
    size = len(cumulative_objective)
    print 'times match',len(cumulative_objective),len(cumulative_time)
    new_times=[int(t-cumulative_time[1]) for t in cumulative_time[1:]]
    

    if print_to_report==True:
        plt.plot(cumulative_time, cumulative_objective,'g') 
        
    objective_anno[size-1]=(alt_type+str(iteration),cumulative_objective[-1],cumulative_time[-1])
    
    if print_to_report==True:
        for k,val in objective_anno.items():
            ax.annotate(val[0], xy=(val[2],val[1]),  xycoords='data',textcoords='data', ha='center', va='bottom',size=7,rotation=90,arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='red'))    
                      
        pdf_pages.savefig(fig)
        plt.close()
    return cumulative_objective,objective_anno,cumulative_time

def write_time(pdf_pages,time_str):
    
    fig=plt.figure()
    plt.axis('off')
    plt.title('Timing parmaters for the experiment')
    plt.text(0,0.1,time_str,fontsize=8)
    pdf_pages.savefig(fig)
    

def plot_points_table(points_file,ball_range,fig,ax,pdf_pages,N,dimensions):
    '''PLots the points to the pdf'''
    f=open(points_file,'r')
    content=f.readlines()     
    write_str=''  
    plt.title('Raw data for '+str(len(content)-1)+' points',fontsize=8)
    wrs_list=[]
    wrs_list.append(','.join(content[0].split(','))+str('\n'))
    for i in xrange(3):
        for index,value in enumerate(content[1:]):
            if index%80==0:
                plt.text(0+index*(.33),1,write_str,fontsize=4,va='top')  
                wrs_list.append(write_str)
                write_str=''
            write_str+=','.join(value.split(','))+str('\n')
    wrs_list.append(write_str)        
    ax.axis('off')
    plt.tight_layout()      
    pdf_pages.savefig(fig) 
    plt.close()
    
def print_solution(my_prob,points,dimensions,sol,k_list,n_list,c_list,centers,sum_omega,lambda_o,pdf_pages,time_to_solve,parameter_str,names_dict,header,iteration,alt_type,report_file,feature_dict,class_dict):
    '''Print and create output to the console'''    
    B=sol.B
    N=sol.N
    ajb_list=sol.non_zero_a
    beta_list=sol.non_zero_b
    v_list=sol.non_zero_features
    omega=sol.non_zero_omega
#    class_dict={1:'Breakfast Cereals',2:'Snacks',3:'Sweets'}
#    feature_dict={'v1':'Water','v2':'Energ','v3':'Protein','v4':'Lipid','v5':'Carbohydrt','v6':'Sugar','v7':'Calcium','v8':'Iron','v9':'Sodium','v10':'Cholestrl'}    
    #feature_dict={'v1':'Sugar','v2':'Iron','v3':'Cholestrl'}    
    omega_str=''									
    
    selected_centers=[b[1:] for b in beta_list]
        
    
    print 'version: ',version,'| filename: ',os.path.realpath(__file__).split('\\')[-1],'\n'
    print 'B:',B,',N:',N
    print 'number of balls selected: ',len(ajb_list),'\n'
    print 'objective ksi:',len(k_list)*C_k,'| eta:',len(n_list)*C_n,'| alpha:',len(ajb_list)*C_alpha,'| beta:',len(beta_list)*C_beta,'| omega',sum_omega*lambda_o,'| v:',len(v_list)*C_v
    print "Solution value  = ", my_prob.solution.get_objective_value()    
    print "Solution status = " , my_prob.solution.get_status(), ":",my_prob.solution.status[my_prob.solution.get_status()],'\n'
    
    print 'nonzero k',k_list,'contribution to objective:',len(k_list)*C_k,'\n'
    print 'nonzero eta',n_list,'contribution to objective:',len(n_list)*C_n,'\n'
    print 'nonzero beta',beta_list,'contribution to objective:',len(beta_list)*C_beta,'\n'
    print 'alpha(j,b)',ajb_list,'contribution to objective:',len(ajb_list)*C_alpha,'\n'
    print 'nonzero v','[',','.join([feature_dict[v_] for v_ in map(str,v_list)]),']contribution to objective:',len(v_list)*C_v,'\n'
    print 'omega contribution to objective:',sum_omega*lambda_o,'\n'
    
    report_str=''
    report_str2=''
    report_str+='Run '+str(iteration)+','+alt_type+', seed: '+str(set_seed)+'\n'
    report_str2+='Run '+str(iteration)+','+alt_type+'\n'
    report_str+="B:"+str(B)+" | dimensions:"+str(dimensions)+" | N:"+str(N)+"\n"
    report_str+='number of balls selected: '+str(len(ajb_list))+'\n'  
    report_str+='features selected ['+', '.join([feature_dict[v_] for v_ in map(str,v_list)])+']    | contribution to objective: '+str(len(v_list)*C_v)+'\n\n'
    
    report_str+="Objective value = "+str(my_prob.solution.get_objective_value())+'\n'
    report_str+="Solution status = "+str(my_prob.solution.get_status())+":"+str(my_prob.solution.status[my_prob.solution.get_status()])+'\n\n'    
    
    report_str+='coefficients: C_ksi='+str(C_k)+' | C_eta='+str(C_n)+' | C_alpha='+str(C_alpha)+' | C_beta='+str(C_beta)+' | C_omega='+str(lambda_array[0])+' | C_v='+str(C_v)+'\n'
    report_str+='variable contributions = ksi:'+str(len(k_list)*C_k)+' |  eta:'+str(len(n_list)*C_n)+' |  alpha:'+str(len(ajb_list)*C_alpha)+' |  beta:'+str(len(beta_list)*C_beta)+' |  omega:'+str(sum_omega*lambda_o)+' |  v:'+str(len(v_list)*C_v)+'\n\n'    
    
    report_str+='run parameters: '+str(parameter_str)+'\n'    
    report_str+='time to solve MIP = '+str(round(float(time_to_solve),2))+' sec\n\n'
        
    report_str+='nonzero k ['+str(len(k_list))+' points not correctly covered]    | contribution to objective: '+str(len(k_list)*C_k)+'\n'
    report_str+='nonzero eta ['+str(len(n_list))+' points wrongly covered]    | contribution to objective: '+str(len(n_list)*C_n)+'\n'
    report_str+='nonzero beta ['+','.join(map(str,beta_list))+']    | contribution to objective: '+str(len(beta_list)*C_beta)+'\n'
    report_str+='alpha(j,b) ['+','.join(map(str,ajb_list))+']    | contribution to objective: '+str(len(ajb_list)*C_alpha)+'\n'
    report_str+='omega contribution to objective: '+str(sum_omega*lambda_o)+'\n\n'
    report_str+='w1= '+str(w1)+', w2= '+str(w2)+', epsilon= '+str(EPSILON)+', M= '+str(M) 
    #report_str+='warm start status:'+str(warm_start_flag)+' | apply_run_parametrs:'+str(apply_run_parametrs)+'\n\n'        
    
    report_str2+='1:Water, 2:Energ, 3:Protein, 4:Lipid_Tot, 5:Carbohydrt, 6:Sugar, 7:Calcium, 8:Iron, 9:Sodium, 10:Cholestrl\n\n'
    report_str2+='Omega for all balls \nball : (dim,omega)\n' 
    try :    
        for b,value in omega.items():
            #omega_str+=str(iteration)+','+str(alt_type)+',omega,'+str(b)+','+','.join(map(str,zip(*sorted(value,key=lambda x: x[0]))[1]))+'\n'
            print 'ball',b,': (dim, omega)\n',[(item[0],round(item[1],7)) for item in sorted(value,key=lambda x: x[0])]
            report_str2+='ball'+str(b)+':'+str([(item[0],round(item[1],7)) for item in sorted(value,key=lambda x: x[0])]).strip('[]')+'\n'
    except :
        pass

    
    report_str+='\n'
    print '\nCenters of balls'    
    print 'ball: (['+','.join(['feature'+str(i+1) for i in xrange(dimensions)])+'],(j,b),proto_name'
    modified_omega=collections.defaultdict(list)    
    for ball,elm in centers.items():
        for tup in omega[ball]:
            modified_omega[ball].append((tup[0],Decimal(repr(tup[1]))))
            
            
        omega_str+=str(iteration)+','+str(alt_type)+','+str(ball)+','+str(int(elm[-2]))+','+str(int(points[int(elm[-2])][0]))+','+','.join(map(str,elm[:-2]))+','+','.join(map(str,zip(*sorted(modified_omega[ball],key=lambda x: x[0]))[1]))+'\n'
        print ball,':',[round(item,2) for item in elm[:-2]],elm[-2:],names_dict[int(elm[-2])]
         
    print '\nSelected prototypes'
    report_str2+='\nSelected Prototypes\n'
    
    try:     
        for ball,elm in centers.items():
            if elm[-2] in selected_centers:
                print ball,elm
                print 'Ball/Cluster:',ball,' = (class -> prototype) : (',class_dict[points[int(elm[-2])][0]],' -> ',names_dict[int(elm[-2])],')'
        print '\n' 
        
    except:
        pass
    
    draw_plot(centers,omega,v_list,feature_dict,B,points,pdf_pages,class_dict,dimensions,lambda_o,report_str,iteration,alt_type,header,{},'Train',names_dict)
 
        
    misclassified_dict=collections.defaultdict(list)
    classified_dict=collections.defaultdict(list)
    correct=collections.defaultdict(list)
    incorrect=collections.defaultdict(list)
    
    for tup in ajb_list:
        j=tup[0]
        b=tup[1]
        class_j=points[int(j)][0]
        for misclassified_point in n_list:
            i=misclassified_point[1:]
            c_string='c'+i+'_'+j+'_'+b
            if c_string in c_list:
                class_i=points[int(i)][0]
                if class_i!=class_j:
                    misclassified_dict[(j,b)].append(i)
                else:
                    classified_dict[(j,b)].append(i)
    missclassified_str=''
    classified_str=''
    incorrect_str=''
    correct_str=''
    print 'Cluster misclassifications (j,b) : points '   
    for tup,value in misclassified_dict.items():
        #print '(j,ball,class):',tup,class_dict[int(points[int(tup[0])][0])],' -> i:',[names_dict[int(val)] for val in value]
        missclassified_str+='(j,ball,class):'+str(tup)+str(class_dict[int(points[int(tup[0])][0])])+' cluster \n'+str([names_dict[int(val)] for val in value])+'\n\n'
    print 'Misclassified by one but correctly classfied by another[ due to overlap] (j,b)    : point '  
    for tup,value in classified_dict.items():
        #print '(j,ball,class):',tup,class_dict[int(points[int(tup[0])][0])],' -> i:',[names_dict[int(val)] for val in value]          
        classified_str+='(j,ball,class):'+str(tup)+str(class_dict[int(points[int(tup[0])][0])])+' cluster \n'+str([names_dict[int(val)] for val in value])+'\n\n'
        
    class_count=collections.defaultdict(int)
    for tup in ajb_list:
        j=tup[0]
        b=tup[1]        
        class_j=points[int(j)][0]
        class_count[class_j]=class_count[class_j]+1
        for point_p in xrange(1,N+1):
            i=str(point_p)
            class_i=points[int(i)][0]
            c_string='c'+i+'_'+j+'_'+b
            if c_string in c_list:                
                if class_i!=class_j:
                    incorrect[(j,b)].append(i)
                    
                else:
                    correct[(j,b)].append(i) 
    
    print class_count 
    
    
    complete_str=''  
    complete_str+='class summaries\n'+'\n'.join([str(class_dict[class_b])+' -> '+str(count)+' balls'for class_b,count in class_count.items()])
    complete_str+='\n\n'
    for k,val in class_dict.items():
        print 'class: ',class_dict[k],' '+str(class_count[k])+'/'+str(len(ajb_list))+' balls'
        complete_str+='class: '+class_dict[k]+' '+str(class_count[k])+'/'+str(len(ajb_list))+' balls'+'\n'       
        no_miss=set(correct.keys())-set(incorrect.keys())
        no_miss_dict={}
        
        for v in no_miss:
            no_miss_dict[v]=[]        
        
        for (j,b),i_list in incorrect.items()+no_miss_dict.items():
            j=int(j)  
            if points[j][0]!=k:
                continue
            print '     ball: ',b
            print '         proto: ',names_dict[j],'->',class_dict[points[j][0]],' :',[feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[j][1:])]
            print '         omega(from largest): ',pp.pprint([feature_dict['v'+str(tup[0])]+': '+str(tup[1])+' ' for tup in sorted(omega[b],key=lambda x: x[1],reverse=True)])
            print '         axis(from smallest): ',pp.pprint([feature_dict['v'+str(tup[0])]+': '+str(1/math.sqrt(tup[1]))+' ' if tup[1]>1e-8 else feature_dict['v'+str(tup[0])]+': '+str(1/math.sqrt(w1_min))+' ' for tup in sorted(omega[b],key=lambda x: x[1],reverse=True)])
            print '         #correct: ',len(correct[(str(j),b)]),' / #incorrect: ',len(incorrect[(str(j),b)])
            print '\n      incorrectly classifies'            
            
            complete_str+='     ball: '+str(b)+' ('+class_dict[points[j][0]]+')\n'
            complete_str+='         proto #: '+str(j)+'\n'
            complete_str+='         proto: '+names_dict[j]+' -> '+class_dict[points[j][0]]+' : \n           '+'\n           '.join([feature_dict['v'+str(index+1)]+'  : '+str(v)+' ' for index,v in enumerate(points[j][1:])])+'\n'          
            complete_str+='         omega(from largest):  \n           '+'\n           '.join([feature_dict['v'+str(tup[0])]+' : '+str(tup[1]) for tup in sorted(omega[b],key=lambda x: x[1],reverse=True)])+'\n'
            complete_str+='         axis(from smallest):  \n           '+'\n           '.join([feature_dict['v'+str(tup[0])]+' : '+str(1/math.sqrt(tup[1])) if tup[1]>1e-8 else feature_dict['v'+str(tup[0])]+': '+str(1/math.sqrt(w1_min)) for tup in sorted(omega[b],key=lambda x: x[1],reverse=True)])+'\n\n'
            complete_str+='         #correct: '+str(len(correct[(str(j),b)]))+' / #incorrect: '+str(len(incorrect[(str(j),b)]))+'\n'          
            
            complete_str+='\n         correctly classifies'+'\n'
            append_str=''            
            
            maha_inv_omega=np.diag([tup[1] for tup in sorted(omega[b],key=lambda x:x[0])])            
                        
            for count,i in enumerate(correct[(str(j),b)]):
                i=int(i)
                md_correct=spatial.distance.mahalanobis(points[j][1:],points[i][1:],maha_inv_omega)
                append_str+='         #'+str(i)+' ('+names_dict[i]+', '+class_dict[points[i][0]]+', MD:'+str(md_correct)+'), \n'
                append_str+='         '+','.join([feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[i][1:])])+'\n'
                if (count+1)%1==0:
                    complete_str+=append_str
                    append_str=''
            if append_str!='':
                complete_str+='         '+append_str            
            
            complete_str+='\n         incorrectly classifies'+'\n'           
                        
            if i_list==[]:
                print '         none'
                complete_str+='         none'+'\n'
                        
            md_list=[]                        
            for i in i_list:
                i=int(i)
                print '        ',names_dict[i],'->',class_dict[points[i][0]],':',[feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[i][1:])]
                print '\n'
                md_list.append((spatial.distance.mahalanobis(points[j][1:],points[i][1:],maha_inv_omega),'#'+str(i)+' '+names_dict[i]+'->'+class_dict[points[i][0]]+':\n'+'          '+','.join([feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[i][1:])])+'\n'))
            
            for tup in sorted(md_list,key=lambda x:x[0]):            
                complete_str+='          '+'[MD:'+str(tup[0])+'] | '+tup[1]
                print '          '+'[MD:'+str(tup[0])+'] | '+tup[1]+'\n'
            complete_str+='\n'
            
            
                
                        
            
        print '         points not covered'
        complete_str+='       points not covered'+'\n'
        for ksi in k_list:
            ksi=int(ksi[1:])
            class_ksi=points[ksi][0]

            ball_of_class_k=[]                
            for tup in ajb_list:
                j=int(tup[0])
                b=int(tup[1])
                if points[j][0]==class_ksi:
                    ball_of_class_k.append((j,b))
            
            if class_ksi==k:
                print '        ',names_dict[ksi],'->',class_dict[points[ksi][0]],':',[feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[ksi][1:])]
                complete_str+='         #'+str(ksi)+' '+names_dict[ksi]+'->'+class_dict[points[ksi][0]]+': \n         '+','.join([feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[ksi][1:])])+'\n'
                for tup_k in ball_of_class_k:                   
                    maha_inv_omega=np.diag([tup[1] for tup in sorted(omega[str(tup_k[1])],key=lambda x:x[0])])                                       
                    md_not_covered=spatial.distance.mahalanobis(points[tup_k[0]][1:],points[ksi][1:],maha_inv_omega)                    
                    complete_str+='              #'+str(ksi)+': MD form ball '+str(tup_k[1])+' -> '+str(md_not_covered)+'\n'
    
        print '\n'
        complete_str+='\n\n'
    
    print 'incorrect str'                
    for tup,value in incorrect.items():
        print '(j,ball,class):',tup,class_dict[int(points[int(tup[0])][0])],' -> i:',[names_dict[int(val)] for val in value]
        incorrect_str+='(j,ball,class):'+str(tup)+str(class_dict[int(points[int(tup[0])][0])])+' cluster: omega'+str(sorted(omega[tup[1]],key=lambda x: x[1],reverse=True))+' \n'+str([names_dict[int(val)] for val in value])+'\n\n'
    print 'correct str '  
    for tup,value in correct.items():
        print '(j,ball,class):',tup,class_dict[int(points[int(tup[0])][0])],' -> i:',[names_dict[int(val)] for val in value]          
        correct_str+='(j,ball,class):'+str(tup)+str(class_dict[int(points[int(tup[0])][0])])+' cluster: omega'+str(sorted(omega[tup[1]],key=lambda x: x[1],reverse=True))+' \n'+str([names_dict[int(val)] for val in value])+'\n\n'
        
    with open('./output/'+version+'_plots/sols/'+report_file+'_ellipse_plot3d.csv','a') as f:
        f.write(omega_str)
    
    
    if print_to_report==True:
        page_size=46
        all_lines=complete_str.splitlines()
        pages=len(all_lines)/page_size 
        remainder=len(all_lines)%page_size
        
        for i in xrange(pages):    
            fig=plt.figure()
            if i==0:
                plt.title('Prototype Cluster Analysis selected b='+str(len(ajb_list))+' out of B='+str(B),fontsize=8)
            plt.axis('off')
            [fig.text(0.08,.9-.9*float(i)/page_size,l,fontsize=7) for i,l in enumerate(all_lines[i*page_size:(i+1)*page_size])]
            pdf_pages.savefig(fig)
            
        fig=plt.figure()
        plt.axis('off')
        [fig.text(0.08,.9-.9*float(i)/page_size,l,fontsize=7) for i,l in enumerate(all_lines[pages*page_size:pages*page_size+remainder])]
        pdf_pages.savefig(fig)
    
    
    
    with open('./output/'+version+'_plots/'+report_file+'_classification_file_'+str(iteration)+'_'+alt_type+'_correct.txt','w') as f:
        f.write('correctly classified 1-Sugar_Tot_.g., 2-Iron_.mg., 3-Cholestrl_.mg1\n')        
        f.write(correct_str)
    
    with open('./output/'+version+'_plots/'+report_file+'_classification_file_'+str(iteration)+'_'+alt_type+'_incorrect.txt','w') as f:    
        f.write('incorrectly classified 1-Sugar_Tot_.g., 2-Iron_.mg., 3-Cholestrl_.mg1\n')
        f.write(incorrect_str)
    plt.close("all")
    
    return
 


   
def get_mahalanobis_string(i_list,omega,j,b,names_dict,class_dict,feature_dict,points,complete_str):
    maha_inv_omega=np.diag([tup[1] for tup in sorted(omega[b],key=lambda x:x[0])])
            
    md_list=[]            
    
    for i in i_list:
        i=int(i)
        print '        ',names_dict[i],'->',class_dict[points[i][0]],':',[feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[i][1:])]
        print '\n'
        md_list.append((spatial.distance.mahalanobis(points[j][1:],points[i][1:],maha_inv_omega),'#'+str(i)+' '+names_dict[i]+'->'+class_dict[points[i][0]]+':'+','.join([feature_dict['v'+str(index+1)]+': '+str(v)+' ' for index,v in enumerate(points[i][1:])])+'\n'))
    
    for tup in sorted(md_list,key=lambda x:x[0]):            
        complete_str+='          '+'[MD:'+str(tup[0])+'] | '+tup[1]
        print '          '+'[MD:'+str(tup[0])+'] | '+tup[1]+'\n'
    complete_str+='\n'
    
    return complete_str    
    
def plot_ellipsoid(color,c,ax,center_x,center_y,center_z,m_x,m_y,m_z,j,names_dict,mode):
    # your ellispsoid and center in matrix form

      

    A = np.array([[m_x,0,0],[0, m_y,0],[0,0,m_z]])
    center = [center_x,center_y,center_z]
    
    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)
    
    if radii[0]>2:
        radii[0]=2
    if radii[1]>2:
        radii[1]=2
    if radii[2]>2:
        radii[2]=2
    
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
#    ax.scatter(center_x,center_y,center_z,color=color,marker='s')
    name_c=names_dict[j]
    
    ax.plot([center_x],[center_y],[center_z],'s',color=color,label=str(j)+':'+name_c)
    
    ax.text(center_x, center_y, center_z, str(j),None,size=7)
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=5, bbox_to_anchor=(0, 0))    
    
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center  
    if mode!='centers':
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)
  
def plot_3d(balls,points,centers,omega_matrix,class_names,pdf_pages,report_str,header,iteration,alt_type,feature_dict,v_list,train_points,split,names_dict):
       
    NUM_COLORS = balls
    fig = plt.figure() 
    ax = plt.subplot(111, projection='3d')
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])   
    scatter_values=[tuple(val) for val in points.values()] 
    #marker_list=['^','o','+','1']
    marker_list=['.','.','.','.','.']
    color_list=['r','b','g','y','m']   
    classes=list(set(list(zip(*scatter_values)[0])))   
    class_dict={}
    for cl in xrange(len(classes)):
        class_dict[classes[cl]]=(color_list[cl],marker_list[cl])
    
    for point,value in points.items():
        name_c=names_dict[point]
        ax.scatter(value[1],value[2],value[3],color=class_dict[value[0]][0], marker=class_dict[value[0]][1])
        

    for ball,values in centers.items():
        x,y,z=values[:3]
        j=int(values[-2])
        omega=sorted(omega_matrix[ball],key=lambda x:x[0])
        m_x,m_y,m_z=zip(*omega)[1]
         
        if split=='Train':
            plot_ellipsoid(class_dict[points[int(values[-2])][0]][0],int(ball),ax, x,y,z,m_x,m_y,m_z,j,names_dict,'all')
        if split=='Test':
            plot_ellipsoid(class_dict[train_points[int(values[-2])][0]][0],int(ball),ax, x,y,z,m_x,m_y,m_z,j,names_dict,'all')

    
       
    plt.title(split+' | Run '+str(iteration)+' | choose '+alt_type+' | balls(selected/total)= '+str(len(centers.keys()))+'/'+str(len(omega_matrix.keys())))
    ax.set_xlabel(v_list[0]+' | '+feature_dict[v_list[0]])
    ax.set_ylabel(v_list[1]+' | '+feature_dict[v_list[1]])
    ax.set_zlabel(v_list[2]+' | '+feature_dict[v_list[2]])    
#    plt.draw()
    plt.show()   
    pdf_pages.savefig(fig) 
    
    #-------------------------------------------------------------------------
    #only points
    fig = plt.figure()
    plt.ion() 
    ax = plt.subplot(111, projection='3d')
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])   
    scatter_values=[tuple(val) for val in points.values()] 
    #marker_list=['^','o','+','1']
    marker_list=['.','.','.','.','.']
    color_list=['r','b','g','y','m']   
    classes=list(set(list(zip(*scatter_values)[0])))   
    class_dict={}
    for cl in xrange(len(classes)):
        class_dict[classes[cl]]=(color_list[cl],marker_list[cl])
    
    for point,value in points.items():
        name_c=names_dict[point]
        ax.scatter(value[1],value[2],value[3],color=class_dict[value[0]][0], marker=class_dict[value[0]][1])
        

    for ball,values in centers.items():
        x,y,z=values[:3]
        j=int(values[-2])
        omega=sorted(omega_matrix[ball],key=lambda x:x[0])
        m_x,m_y,m_z=zip(*omega)[1]
         
        if split=='Train':
            plot_ellipsoid(class_dict[points[int(values[-2])][0]][0],int(ball),ax, x,y,z,m_x,m_y,m_z,j,names_dict,'centers')
        if split=='Test':
            plot_ellipsoid(class_dict[train_points[int(values[-2])][0]][0],int(ball),ax, x,y,z,m_x,m_y,m_z,j,names_dict,'centers')

    
       
    plt.title(split+' | Run '+str(iteration)+' | choose '+alt_type+' | balls(selected/total)= '+str(len(centers.keys()))+'/'+str(len(omega_matrix.keys())))
    ax.set_xlabel(v_list[0]+' | '+feature_dict[v_list[0]])
    ax.set_ylabel(v_list[1]+' | '+feature_dict[v_list[1]])
    ax.set_zlabel(v_list[2]+' | '+feature_dict[v_list[2]])    
       
    pdf_pages.savefig(fig) 
    plt.close()

     
    
    #save text
    fig=plt.figure()
    plt.axis('off')
    plt.title('Ellipsoid | Run '+str(iteration)+' | '+alt_type+' | MIP parameters and results')
    plt.text(0, 0,report_str,fontsize=8)
    pdf_pages.savefig(fig)
    plt.close()


def convert_dimensions_d_to_Nd(v_list,points,omega,centers,dimensions,Nd=2):
    
    selected_features=[int(v[1:]) for v in v_list]
    selected_features.sort()
    
    new_points={}    
    for p,value in points.items():
        feature_values=[value[f] for f in selected_features]
        feature_values.insert(0,value[0])
        new_points[p]=feature_values
        
    new_omega={}
    for ball,o in omega.items():
        o_values=[tup  for tup in o if tup[0] in selected_features]         
        new_o_values=[]   
        sorted_o=sorted(o_values,key=lambda x:x[0])
        #make the dimesions 1,2 instead of other numbers
        for i in xrange(1,Nd+1):
            new_o_values.append((i,sorted_o[i-1][1]))        
        new_omega[ball]=new_o_values
    
    new_centers={}
    for ball,c in centers.items():
        new_center_list=[c[f-1] for f in selected_features]
        new_center_list=new_center_list+[c[-2],c[-1]]   
        new_centers[ball]=new_center_list
     
    return new_points,new_omega,new_centers   
    
   
   
    
def plot_2d(B,points,centers,omega,lambda_o,dimensions,pdf_pages,report_str,class_dict,v_list,feature_dict,iteration,alt_type,train_points,split,names_dict):
    new_omega=collections.defaultdict()
    for index,tuple_array in omega.items():
        o_array=[[0,0],[0,0]]
        for p in xrange(dimensions):
            extracted_omega=tuple_array[p][1]
            if extracted_omega<.25:
                extracted_omega=.25
            
            o_array[tuple_array[p][0]-1][tuple_array[p][0]-1]=extracted_omega
       
        e_vals, e_vecs = LA.eig(o_array)
        angle=0
        max_eigenvalue_index=np.where(e_vals==max(e_vals))[0][0]
        if e_vecs[max_eigenvalue_index].all()!=0:         
            angle= np.arctan(e_vecs[max_eigenvalue_index][1]/e_vecs[max_eigenvalue_index][0])
        else:
            angle=90
        try : 
            new_omega[index]=((map(f,e_vals.real)[0],map(f,e_vals.real)[1],angle,centers[index][0],centers[index][1]))
        except KeyError:
            pass
    
    print '\nball: (axis1,axis2,angle,feature1,feature2)'
    for ball,elm in new_omega.items():
        print ball,elm
        
    print '\n--------------------------------------------------------------\n'  
        
    fig=plt.figure()
    ax = plt.gca()  
    
    color_list=['r','b','g','y']
    #marker_list=['^','o','+','1']
    dict_features=collections.defaultdict(list)
    for point,value in points.items():
        dict_features[int(value[0])].append((value[1],value[2]))        
    
    print 'dict_features',dict_features.items()    
    plot_ellipse=[]
    for ball,item in new_omega.items():
                
        
        if split=='Train': 
            print 'train selected as centers of balls ',points[int(centers[ball][2])]
            ellipse=Ellipse(xy=(item[-2],item[-1]), height=item[1], width=item[0], 
                        edgecolor=color_list[int(points[int(centers[ball][2])][0]-1)],fc='none',  angle=-90+int(item[2]))
        if split=='Test':
            print 'test selected as centers of balls ',train_points[int(centers[ball][2])]
            ellipse=Ellipse(xy=(item[-2],item[-1]), height=item[1], width=item[0], 
                        edgecolor=color_list[int(train_points[int(centers[ball][2])][0]-1)],fc='none',  angle=-90+int(item[2]))
            ax.scatter(item[-2],item[-1],marker='s',c=color_list[int(train_points[int(centers[ball][2])][0]-1)]) 
       
        
        plot_ellipse.append(ellipse)
        ax.add_patch(ellipse) 
        
    x_val,y_val,point_color,label=[],[],[],[]    
    
    
    for key,value in dict_features.items():
        print 'plotting dict_features values ',value[0][0],value[0][1]
        x_val.append(value[0][0])
        y_val.append(value[0][1])
        point_color.append(color_list[key-1])
        label.append(class_dict[key])
    plt.scatter(x_val,y_val,c=point_color,marker='o',alpha=0.5)  

    
    for p,p_values in points.items():
        plt.text(p_values[1],p_values[2],p,fontsize=8)
    
    plt.legend( scatterpoints=1,fontsize=8)   
    plt.xlabel(v_list[0]+' : '+feature_dict[v_list[0]])
    plt.ylabel(v_list[1]+' : '+feature_dict[v_list[1]])
    
    plt.title(split+' | 2d | Run '+str(iteration)+' | choose '+alt_type+' | balls(selected/total)= '+str(len(centers.keys()))+'/'+str(len(new_omega.keys())))
    plt.grid(True)    
        
    
#    ax.relim()
#    ax.autoscale_view() 
    plt.show()
    pdf_pages.savefig(fig)  
    plt.close()
    
    #save text    
    fig=plt.figure()
    plt.axis('off')
    plt.title('MIP parameters and results')
    plt.text(0, 0.3,report_str,fontsize=8)
    pdf_pages.savefig(fig)
    plt.close()

  

    
if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='prototype selection')
    parser.add_argument('-o','--out',type = str, dest = 'report_file', help='output file',default='report')
    parser.add_argument('-i','--in',type = str, dest = 'points_file', help='input file')
    parser.add_argument('-b','--balls',type = int, dest = 'B', help='the number of balls')
    parser.add_argument('-d','--data_set',type = str, dest = 'data_set', help='data set')
    parser.add_argument('-f','--folds',type = int, dest = 'fold', help='folds')
    
    
    args = parser.parse_args()
    
    report_file=args.report_file
    points_file=args.points_file
    B=args.B
    data_set=args.data_set
    fold=args.fold
    print 'proto_ellipse_alt: '+str(B)+','+str(fold)+','+report_file+', seed'+str(set_seed)
    create_report(B,lambda_array,points_file,report_file,data_set,fold)
    print 'output:', report_file,' | warm start status:',warm_start_flag,' | apply_run_parametrs:',apply_run_parametrs
      