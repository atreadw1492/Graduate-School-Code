
### Script to perform the Hill Climbing Implementation for the 4 Peaks Problem
################################################################################


# Load packages
from random import randint
import time


# define variation of the accumulation function from itertools -- use for get_fitness
def accumulate(iterable):
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total += element
        yield total

## define 4 peaks fitness function
def get_fitness(input_vector):
    
    input_vector = list(input_vector)
    
    T = math.floor(0.1 * len(input_vector))
    
    all_sums = [x for x in accumulate(input_vector)]
    if sum(input_vector) == len(input_vector):
        trailing_zeros = 0

    elif sum(input_vector) == 0:
        trailing_zeros = len(input_vector)

    else: 
        trailing_zeros = len(input_vector) - all_sums.index(max(all_sums)) - 1

    # leading ones
    if sum(input_vector) == len(input_vector):
        leading_ones = len(input_vector)

    elif sum(input_vector) == 0:
        leading_ones = 0

    else: 
        leading_ones = input_vector.index(min(input_vector))

                        
    if trailing_zeros > T and leading_ones > T:
        R = len(input_vector)
    else:
        R = 0

    return R + max((leading_ones , trailing_zeros))



# Function to mutate a gene to find neighbors
def mutation(sstr):
    pos = randint(0,len(sstr) - 1)
    sstr[pos] = 0 if sstr[pos] else 1
    return sstr
    
def initialize():
    return [randint(0,1) for index in range(0,150)]
    
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

result3000 = restart_hill_climbing(starter , 500, 3000)



# Get hill climbing result by varying the number of restarts
#num_restarts_list = range(100,1000,100)
#starter2 = initialize()
#result_by_num_restarts = map(lambda num: restart_hill_climbing(starter , num, 1000) , num_restarts_list)
#
#[x["best_fit"] for x in result_by_num_restarts]

####################################################################################################
num_restarts_list = [10,100,1000,3000]
starter2 = initialize()
result_by_num_restarts = map(lambda num: restart_hill_climbing(starter2 , num, 1000) , num_restarts_list)

[x["best_fit"] for x in result_by_num_restarts]



start = time.time()
result = restart_hill_climbing(starter2 , 1000, 1000)
end = time.time() - start







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


