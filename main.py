import csv
import random

##list that stores experts
experts_list=[]

## class that defines a expert and its basic functions
class expert:  
  ##constructor with error list as input
  
  def __init__(self,value_array):
    self.value_array = value_array
    self.weight=1  
  
  def get_weight():
    return self.weight
  ## returns error value of expert at a given time 
  def get_value(i):
    return self.value_array[i]
  ## a hello message from our dear Mr. expert
  def print_info(self):
    print(f"Hi im a expert with ={self.weight} yoroshiku!!.")




    


## create the bounds for each expert
def init_expert():
  expert_value_array = []
  with open("Milano_timeseries.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        expert_value_array.append(row)
  
  for i in range(0,29):   
    experts_list.append(expert(expert_value_array[i]))
    experts_list[i].print_info()

## returns index of expert with best mui
def find_optimal_expert_index():
    max_value=0
    max_index=-1
    for i in range(30):
      m=experts_list[i].mu_i()
      if(m>max_value):
        max_value=m
        max_index=i
    return max_index


def main():
  
  init_expert()
 # optimal_MAB_index= find_optimal_expert_index() #index of the arm with best returns
  
  
  



if __name__=="__main__":
   main()



