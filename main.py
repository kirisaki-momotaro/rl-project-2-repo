import csv
import random

##list that stores experts
experts_list=[]

## class that defines a expert and its basic functions
class expert:  
  ##constructor with error list as input
  def __init__(self,err):
    self.err = err
    
  ## midle value of the bounds
  def mu_i(self):
    return (self.a_i+self.b_i)/2
  ## returns random value between the bounds
  def get_value(i):
    return self.err[i]
  ## a hello message from our dear Mr. expert
  def print_info(self):
    print(f"Hi im a expert with a_i={self.a_i},b_i={self.b_i} and mu_i = {self.mu_i()}  yoroshiku!!.")


## create the bounds for each expert
def init_expert():
  for i in range(0,29):
    new_bound1 ,new_bound2 =random.uniform(0, 1) ,random.uniform(0, 1)
    ## inserts bounds in the correct order smaller->bigger
    if(new_bound1>new_bound2):
      experts_list.append(expert(new_bound2,new_bound1))
    else:
      experts_list.append(expert(new_bound1,new_bound2))

    experts_list[i].print_info()

## returns index of expert with best mui
def find_optimal_expert_index():
    max_value=0
    max_index=-1
    for i in range(k):
      m=experts_list[i].mu_i()
      if(m>max_value):
        max_value=m
        max_index=i
    return max_index


def main():
  
  init_expert()
  optimal_MAB_index= find_optimal_expert_index() #index of the arm with best returns
  
  
  



if __name__=="__main__":
   main()


experts = []
with open("Milano_timeseries.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        experts.append(row)

print(experts[1][2])

