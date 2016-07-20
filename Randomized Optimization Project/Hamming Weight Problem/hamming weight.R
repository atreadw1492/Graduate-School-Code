
### Hamming Weight Optimization
### This script contains code for genetic algorithm and simulated annealing implementations to optimize the Hamming Weights problem
#############################################################################################################################################################################################

# Load packages
require(GA) # for genetic algorithms
require(GenSA)

## Define fitness function for Hamming Weight problem
get_fitness <- function(input_vector)
{

      sum(input_vector)  

}

## Define f for use in the GenSA simulated annealing implmentation; since GenSA searches for a global minimum, we need
## to turn our problem into a minimization problem for this function; the negative of the min value will be the positive max value
# achieved for our actual fitness function, get_fitness
f <- function(input_vector)
{
  
    length(input_vector) - sum(input_vector)
  
}



########################### Genetic Algorithms Implmentation ###############################################################################################################################
##############################################################################################################################################################################################################################################################

# Implment genetic algorithm using GA package
start_time <- proc.time()
genetic_eval <- ga(type = "binary", fitness = get_fitness, nBits = 50,maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE,maxfitness = 50)
end_time <- proc.time() - start_time

genetic_eval@solution

x <- summary(genetic_eval)
x$fitness


pop_sizes <- c(10,50,100,300,500)
ga_by_population <- lapply(pop_sizes , function(size) ga(type = "binary", fitness = get_fitness, nBits = 50,maxiter = 1000, run = 200, popSize = size,keepBest = TRUE,maxfitness = 50) )

# Get max fitness and number of iterations by population size
fitness_levels <- sapply(ga_by_population , function(x) summary(x)$fitness)
fitness_iterations <- sapply(ga_by_population , function(x) summary(x)$iter)



######## Simulated Annealing Implementation ##############################################################################################################################################################
##########################################################################################################################################################################################################
limit = 50
# Implmenent simulated annealing algorithm
start_time <- proc.time()
sim_anneal_model <- GenSA(par = rep(0,limit) , lower = rep(0,limit), upper = rep(1,limit), fn = f)
end_time <- proc.time() - start_time


# Test simulated annealing model with varying temperature values
temperature_values <- c(100,500,1000,2000,3000,5000,7500,10000)
sim_anneal_by_temp <- lapply(temperature_values , function(temp) GenSA(par = rep(0,50) , lower = rep(0,50), upper = rep(1,50), fn = f, control = list(temperature = temp)))
sim_anneal_by_temp_results <- 50 - sapply(sim_anneal_by_temp , function(x) x$value)

# Get number of iterations required to hit optimum value
temp <- lapply(sim_anneal_by_temp , function(x) data.frame(x$trace.mat))
sim_anneal_by_temp_iters <- sapply(sim_anneal_by_temp , function(x) subset(data.frame(x$trace.mat),current.minimum == 00)$nb.steps[1])















