
import pandas as pd
import time
from random import randint

# read in dataset
dataset = pd.read_csv("C:/Users/Andrew/Documents/school stuff/Machine Learning/HW 2/Knapsack Problem/knapsack_data.csv")
# set max capacity
max_capacity = 50

items = list(dataset["items"])
item_values = list(dataset["item_values"])
item_weights = list(dataset["item_weights"])

# Define fitness function
def get_fitness(sstr):
    
    fitness = sum([x * y for x,y in zip(item_values , sstr)])
    fitness_penalty = sum(item_weights) * abs(sum([x * y for x,y in zip(item_weights , sstr)]) - max_capacity)

    return fitness - fitness_penalty


# Function to mutate a gene to find neighbors
def mutation(sstr):
    pos = randint(0,len(sstr) - 1)
    sstr[pos] = 0 if sstr[pos] else 1
    return sstr
    
def initialize():
    return [randint(0,1) for index in items]
    
def hill_climbing(sstr,depth):
    best_sstr = sstr
    best_fit = get_fitness(best_sstr)
    num_eval = 1
    for iter in range(depth):
        neig_sstr = best_sstr[:]
        neig_sstr = mutation(neig_sstr)
        neig_fit = get_fitness(neig_sstr)
        num_eval += 1
        if neig_fit >= best_fit:
            best_sstr = neig_sstr[:]
            best_fit = neig_fit
    return (best_fit,best_sstr,num_eval)
    
 
def restart_hill_climbing(sstr,num_restarts,depth):
    best_fit , best_sstr, total_eval = hill_climbing(sstr , depth)
    for index in range(num_restarts):
        solution = initialize()
        fitness, solution, num_eval = hill_climbing(solution , depth)
        total_eval += num_eval
        if fitness > best_fit:
            best_sstr = solution[:]
            best_fit = fitness
    return {"best_fit" : best_fit, "best_sstr" : best_sstr, "total_eval" : total_eval}   
    


# Get hill climbing result by using search depth levels
depth_list = [100,300,500,1000]
starter = initialize()
result_by_depth = map(lambda num: restart_hill_climbing(starter , 500, num) , depth_list)

[x["best_fit"] for x in result_by_depth]


# Get hill climbing result by varying the number of restarts
num_restarts_list = [10,100,300,1000]
starter2 = initialize()
result_by_num_restarts = map(lambda num: restart_hill_climbing(starter , num, 1000) , num_restarts_list)

[x["best_fit"] for x in result_by_num_restarts]



start = time.time()
restart_hill_climbing(starter , 1000, 1000)
end = time.time() - start

####################################################################################################
#num_restarts_list = [10,100,500,1000,5000,10000]
#starter2 = initialize()
#result_by_num_restarts = map(lambda num: restart_hill_climbing(starter , num, 1000) , num_restarts_list)
#
#[x["best_fit"] for x in result_by_num_restarts]













#starter = initialize()
#result = restart_hill_climbing(starter,500,5000)
#print result
#
#
#################################################################################
#
#
#
#
#
#
#
#
#
#
#starter = initialize()
#hill_climbing(starter,1000)


