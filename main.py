import csv
import random
import os
import sys
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

T = 1000 #horizon 

regret =  np.zeros((T,)) #regret for round t
alg_score = np.zeros((T,)) #cumulative loss for round t
opt_alg_score = np.zeros((T,)) #cumulative loss for round t

##list that stores experts
experts_list=[]

#for use in UCB
bandit_score = np.zeros((30,)) #total score of each arm for first N rounds
pulls = np.zeros((30,))

## class that defines a expert and its basic functions
class expert:  
  ##constructor with error list as input
  
  def __init__(self,value_array):
    self.value_array = value_array
    self.weight=1  
  
  def get_weight(self):
    return self.weight
  ## returns error value of expert at a given time 
  def get_value(self,i):
    return float(self.value_array[i])
  ## a hello message from our dear Mr. expert
  def print_info(self):
    print(f"Hi im a expert with ={self.weight} yoroshiku!!.")

 
## create the bounds for each expert
def init_expert():
  expert_value_array = []
  with open(os.path.join(sys.path[0], "Milano_timeseries.csv"), "r") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        expert_value_array.append(row)
  
  for i in range(0,30):   
    experts_list.append(expert(expert_value_array[i]))
    experts_list[i].print_info()
    
def expert_reset_weight():
  for expert in experts_list:
    expert.weight=1

## returns the total expert weight
def total_expert_weight():
    total_weight=0    
    for i in range(30):
      total_weight=total_weight + experts_list[i].get_weight()
      
    return total_weight

#returns the choose probabilities of all experts
def create_probabilities_array():
  probablilties_array=[]
  total_weight=total_expert_weight()
  for i in range(0,30):
    probablilties_array.append(experts_list[i].get_weight()/total_weight)         
  return probablilties_array

def epsilon():
  #epsilon=0.01
  epsilon=heta_bandit()
  return epsilon
def heta(time):  
    #heta=np.sqrt(np.log(time)/(time+1))
    #print(heta)
    heta=math.sqrt(math.log(30)/T)   
    #heta=0.001*time
    #heta=1/math.sqrt(time+1)
    #heta=0.5
    return heta
    
def heta_bandit():
  heta_bandit=math.pow(30*math.log(30)/T,1/3)  
  return heta_bandit

def discount_weights(time):
  for i in range(0,30):
    old_weight=experts_list[i].get_weight()
    loss=experts_list[i].get_value(time)      
    new_weight= pow(1-heta(time),loss)*old_weight
    experts_list[i].weight=new_weight

def discount_weights_bandit(time,chosen_expert):
  for i in range(0,30):
    old_weight=experts_list[i].get_weight()
    chosen_value=chosen_expert.get_value(time)
    
    q_expert=create_probabilities_array_bandits()[i]
    
    #print(old_weight)
    if(experts_list[i]==chosen_expert):
      loss=chosen_value/q_expert
      new_weight= pow(1-heta_bandit(),loss)*old_weight
      experts_list[i].weight=new_weight
      
    #else:  
    #  if(p_expert==0) :
    #    loss=0
    #  else:
    #    loss=chosen_value/p_expert
     

    
    #if(new_weight==0):
      #print(f"{new_weight} ;; {old_weight} ;;{loss}")






#returns the chosen expert
def choose_expert():
  return random.choices(experts_list, weights = create_probabilities_array(), k = 1)[0]


def create_probabilities_array_bandits():
  prob_array=create_probabilities_array()
  for p in prob_array:
    q= (1-epsilon())*p + (epsilon()/30)
    p=q
  return prob_array

def choose_expert_bandits():
  return random.choices(experts_list, weights = create_probabilities_array_bandits(), k = 1)[0]



#returns the value of the expert with the smallest loss for a specific time from the inputed data
def minimum(time):
  min=math.inf
  for i in range(0,30):
    exp_val=experts_list[i].get_value(time)
    if(exp_val<min):min=exp_val
  return min



## WMR
def WMR(plot_placement):
  for i in range(1,T):    
    chosen_expert=choose_expert()      
    ## calculate regret    
    value=chosen_expert.get_value(i)  
    minimum_value=minimum(i)  

    if i > 1: alg_score[i] = alg_score[i-1] + value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: alg_score[i] = value 
    if i > 1: opt_alg_score[i] = opt_alg_score[i-1] + minimum_value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: opt_alg_score[i] = minimum_value 
    regret[i] = (alg_score[i]-opt_alg_score[i-1])/(i+1) 
    #print(regret[i]) 
    discount_weights(i)
    #discount_weights_bandit(i,chosen_expert)
  plt.subplot(2, 3, plot_placement)
  plt.title(f"MW Performance T={T} ") 
  plt.xlabel("Round T") 
  plt.ylabel("Total score") 
  plt.plot(np.arange(1,T+1),regret) 
  


#estimates the loos for the not used experts
def WMR_bandit(plot_placement):
  for i in range(1,T):    
    chosen_expert=choose_expert_bandits()      
    ## calculate regret    
    value=chosen_expert.get_value(i)  
    minimum_value=minimum(i)  #get the best case for regret calculation

    if i > 1: alg_score[i] = alg_score[i-1] + value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: alg_score[i] = value 
    if i > 1: opt_alg_score[i] = opt_alg_score[i-1] + minimum_value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: opt_alg_score[i] = minimum_value 
    regret[i] = (alg_score[i]-opt_alg_score[i-1])/(i+1) 
    #print(regret[i]) 
    #discount_weights(i)
    discount_weights_bandit(i,chosen_expert)

  plt.subplot(2, 3, plot_placement)
  plt.title(f"MW Bandit Performance T={T} ") 
  plt.xlabel("Round T") 
  plt.ylabel("Total score") 
  plt.plot(np.arange(1,T+1),regret) 
  


 
## UCB returns index of MAB to select
def UCB(time):
  best_ucb=0
  selected_MAB_index=-1
  for i in range(30):
    ucb=(bandit_score[i]/(pulls[i]+1)) + math.sqrt(np.log(time)/(pulls[i]+1)) ## the formula from the class powerpoint
    if(ucb>best_ucb):
      best_ucb=ucb
      selected_MAB_index=i
  return selected_MAB_index


 

## UCB
def UCB_algorithm(plot_placement):
  for i in range(1,T):
      ## pick best arm
      best_arm_index=UCB(i) ## get the bandit index to pull
      turn_value = experts_list[best_arm_index].get_value(i) ##pull bandit
      bandit_score[best_arm_index] = (1-turn_value) + bandit_score[best_arm_index] ##add up score of bandit
      pulls[best_arm_index] = pulls[best_arm_index] +1 ##add up pulls
      minimum_value=minimum(i) 

      if i > 1: alg_score[i] = alg_score[i-1] + (1-turn_value) #vector keeping track of cummulative explore-then-eploit reward at all times 
      else: alg_score[i] = (1-turn_value)
      if i > 1: opt_alg_score[i] = opt_alg_score[i-1] + (1-minimum_value) #vector keeping track of cummulative explore-then-eploit reward at all times 
      else: opt_alg_score[i] = (1-minimum_value)
      regret[i] = (opt_alg_score[i]-alg_score[i])/(i+1)       
      
  plt.subplot(2, 3, plot_placement)
  plt.title(f"UCB Performance T={T} ")  
  plt.xlabel("Round T") 
  plt.ylabel("Total score") 
  plt.plot(np.arange(1,T+1),regret) 
  



  
  







def main():  
  init_expert() 

  WMR(1) 
  for i in range(0,30):
    print(experts_list[i].weight)
  expert_reset_weight()
  
  WMR_bandit(2)
  for i in range(0,30):
    print(experts_list[i].weight)    
  expert_reset_weight()

  UCB_algorithm(3)

  global T
  T=7000
  global regret 
  regret =  np.zeros((T,)) #regret for round t
  global alg_score 
  alg_score = np.zeros((T,)) #cumulative loss for round t
  global opt_alg_score 
  opt_alg_score = np.zeros((T,)) #cumulative loss for round t

  WMR(4) 
  for i in range(0,30):
    print(experts_list[i].weight)
  expert_reset_weight()
  
  WMR_bandit(5)
  for i in range(0,30):
    print(experts_list[i].weight)    
  expert_reset_weight()

  UCB_algorithm(6)
  plt.show()

if __name__=="__main__":
   main()



