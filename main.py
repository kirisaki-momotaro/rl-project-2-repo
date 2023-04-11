import csv
import random
import os
import sys


##list that stores experts
experts_list=[]

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
    return self.value_array[i]
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
    
    





## returns index of expert with best mui
def total_expert_weight():
    total_weight=0    
    for i in range(30):
      total_weight=total_weight + experts_list[i].get_weight()
      
    return total_weight
def create_probabilities_array():
  probablilties_array=[]
  total_weight=total_expert_weight()
  for i in range(0,30):
    probablilties_array.append(experts_list[i].get_weight()/total_weight)
  return probablilties_array




def choose_expert():
  return random.choices(experts_list, weights = create_probabilities_array(), k = 1)[0]

def main():
  
  init_expert()
  print ("heyyy")
  choose_expert().print_info()
  
 
  
  
  



if __name__=="__main__":
   main()



