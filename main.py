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

##list that stores experts or bandits
experts_list=[]

#for use in UCB
bandit_score = np.zeros((30,)) #total score of each arm for first N rounds
pulls = np.zeros((30,)) #number of pulls of the respective bandit

## class that defines a expert (also bandit) and its basic functions
class expert:  
  ##constructor with error list from the csv file as input  
  def __init__(self,value_array):
    self.value_array = value_array
    self.weight=1  
  ## returns expert weight
  def get_weight(self):
    return self.weight
  ## returns error value of expert at a given time 
  def get_value(self,i):
    return float(self.value_array[i])
  ## a hello message from our dear Mr. expert
  def print_info(self):
    print(f"Hi im a expert with ={self.weight} yoroshiku!!.")

 
## create the bounds for each expert
## reads the csv file and brakes it up into 30 sub-lists each corresponding to an expert
## each sub-list is then assigned to the corresponding expert
def init_expert():
  expert_value_array = [] # 30 rows in the end
  with open(os.path.join(sys.path[0], "Milano_timeseries.csv"), "r") as file: ## read file 
    csvreader = csv.reader(file)
    for row in csvreader:
        expert_value_array.append(row)
  
  for i in range(0,30):   
    experts_list.append(expert(expert_value_array[i]))
    experts_list[i].print_info()

## set all expert weights to 1 to run another algorithm
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

#define epsilon for the MW bandit case based on the course notes for sublinear regret
def epsilon():  
  epsilon=heta_bandit()
  return epsilon

#define heta for the MW case based on the course notes for sublinear regret
def heta(time):     
    heta=math.sqrt(math.log(30)/T)       
    return heta

#define heta for the MW bandit case based on the course notes for sublinear regret
def heta_bandit():
  heta_bandit=math.pow(30*math.log(30)/T,1/3)  
  return heta_bandit

#discount the weight of each expert relevant to their loss in the way described in the course notes
def discount_weights(time):
  for i in range(0,30):
    old_weight=experts_list[i].get_weight() #store old weight for use in calculations
    loss=experts_list[i].get_value(time)      
    new_weight= pow(1-heta(time),loss)*old_weight #calculate new weight based on the course notes
    experts_list[i].weight=new_weight #assign new weight

#discount the weight of the chosen bandit relevant to their loss in the way described in the course notes
def discount_weights_bandit(time,chosen_expert):
  for i in range(0,30):
    old_weight=experts_list[i].get_weight() #store old weight for use in calculations
    chosen_value=chosen_expert.get_value(time)
    
    q_expert=create_probabilities_array_bandits()[i]    #q probabillity
    
    if(experts_list[i]==chosen_expert):
      loss=chosen_value/q_expert #calculated adjusted loss to ensure E[loss]=chosen_value
      new_weight= pow(1-heta_bandit(),loss)*old_weight #calculate new weight based on the course notes
      experts_list[i].weight=new_weight #assign new weight
      
   

#returns the chosen expert based on the probabilities matrix
def choose_expert():
  return random.choices(experts_list, weights = create_probabilities_array(), k = 1)[0]

#returns q probability array adjusted to work with bandits
def create_probabilities_array_bandits():
  prob_array=create_probabilities_array()
  for p in prob_array:
    q= (1-epsilon())*p + (epsilon()/30) #formula from course notes
    p=q
  return prob_array

#choose bandit based on q probability array
def choose_expert_bandits():
  return random.choices(experts_list, weights = create_probabilities_array_bandits(), k = 1)[0]



#returns the value of the expert with the smallest loss for a specific time from the inputed data
def minimum(time):
  min=math.inf
  for i in range(0,30):
    exp_val=experts_list[i].get_value(time)
    if(exp_val<min):min=exp_val
  return min



## MW algorithm
def MW_algorithm(plot_placement):
  for i in range(1,T):    #for the defined time horizon T do:
    chosen_expert=choose_expert()     #choose expected optimal expert 
    ## calculate regret    
    value=chosen_expert.get_value(i)  
    minimum_value=minimum(i)  #find the index of the expert with the actuall lowest loss (for the calculation of the regret)

    if i > 1: alg_score[i] = alg_score[i-1] + value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: alg_score[i] = value 
    if i > 1: opt_alg_score[i] = opt_alg_score[i-1] + minimum_value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: opt_alg_score[i] = minimum_value 
    regret[i] = (alg_score[i]-opt_alg_score[i-1])/(i+1) 
    
    discount_weights(i)
  #plot regret in subplot
  plt.subplot(2, 3, plot_placement)
  plt.title(f"MW Performance T={T} ") 
  plt.xlabel("Round T") 
  plt.ylabel("Total score") 
  plt.plot(np.arange(1,T+1),regret) 
  


## MW bandit algorithm 
def MW_algorithm_bandit(plot_placement):
  for i in range(1,T):    
    chosen_expert=choose_expert_bandits()       #choose expected optimal bandit 
       
    value=chosen_expert.get_value(i)  
    minimum_value=minimum(i)  #find the index of the expert with the actuall lowest loss (for the calculation of the regret)

    if i > 1: alg_score[i] = alg_score[i-1] + value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: alg_score[i] = value 
    if i > 1: opt_alg_score[i] = opt_alg_score[i-1] + minimum_value #vector keeping track of cummulative explore-then-eploit reward at all times 
    else: opt_alg_score[i] = minimum_value 
    regret[i] = (alg_score[i]-opt_alg_score[i-1])/(i+1) 
   
    discount_weights_bandit(i,chosen_expert)
  #plot regret in subplot
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
  #plot regret in subplot
  plt.subplot(2, 3, plot_placement)
  plt.title(f"UCB Performance T={T} ")  
  plt.xlabel("Round T") 
  plt.ylabel("Total score") 
  plt.plot(np.arange(1,T+1),regret) 
  



  
  







def main():  
  init_expert() #read file and initialize environment

  MW_algorithm(1) #run multiplicative weights algorithm T=1000  
  expert_reset_weight() #reset weights
  
  MW_algorithm_bandit(2) #run multiplicative weights bandit algorithm T=1000   
  expert_reset_weight() #reset weights

  UCB_algorithm(3) #run adjusted UCB algorithm T=1000 

  global T
  T=7000 #change horizon 
  #reinitialize required arrays adjusted for the new horizon
  global regret 
  regret =  np.zeros((T,)) #regret for round t
  global alg_score 
  alg_score = np.zeros((T,)) #cumulative loss for round t
  global opt_alg_score 
  opt_alg_score = np.zeros((T,)) #cumulative loss for round t

  MW_algorithm(4)   #run multiplicative weights algorithm T=7000 
  expert_reset_weight() #reset weights
  
  MW_algorithm_bandit(5)   #run multiplicative weights bandit algorithm T=7000     
  expert_reset_weight() #reset weights

  UCB_algorithm(6) #run adjusted UCB algorithm T=7000 


  plt.show()

if __name__=="__main__":
   main()



