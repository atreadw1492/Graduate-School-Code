
### 4 Peaks Problem
### This script contains code for genetic algorithm and simulated annealing implementations to optimize the 4 Peaks problem
###########################################################################################################################

# Load packages
require(GenSA) # simulated annealing
require(GA) # genetic algorithms


## Define fitness function for 4 Peaks problem
get_fitness <- function(input_vector)
{
    T <- floor(0.1 * length(input_vector))
  
    all_sums <- Reduce(sum,input_vector,accumulate = TRUE)
    trailing_zeros <- length(input_vector) - which.max(all_sums)
    
    if(sum(input_vector) == length(input_vector))
    {
      trailing_zeros = 0
    }
    else if(sum(input_vector) == 0)
    {
      trailing_zeros = length(input_vector)
    }
    else{ trailing_zeros = length(input_vector) - which.max(all_sums) }
    
    
    # leading ones
    if(sum(input_vector) == length(input_vector))
    {
      leading_ones = length(input_vector)
    }
    else if(sum(input_vector) == 0)
    {
      leading_ones = 0
    }
    
    else{ leading_ones <- which.min(input_vector) - 1 }
    
    
    R <- if((trailing_zeros > T) & (leading_ones > T) ) length(input_vector) else 0
    
    # return
    max(trailing_zeros , leading_ones) + R
  
}



# Set limit = length of input vector
limit <- 150

# Using domain knowledge, define function to get the global optimum value of the 4 Peaks problem 
# given length of a bit string as input
get_opt <- function(limit)
{
    temp <- limit / 10
    x1  = rep(1,temp + 1)
    x2 = rep(0,limit - length(x1))
    
    get_fitness(c(x1,x2))
    
}

global_opt <- get_opt(limit)

####################################################################################################################################################################################################
####################################################################################################################################################################################################

### Genetic Algorithms Implementation
#####################################

# Implment genetic algorithm using GA package
start_time <- proc.time()
genetic_eval <- ga(type = "binary", fitness = get_fitness, nBits = limit,maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE, maxfitness = global_opt)
end_time <- proc.time() - start_time

genetic_eval@solution

x <- summary(genetic_eval)
x$fitness


# Test out the GA algorithm with varying population sizes
pop_sizes <- c(100,300,500)
ga_by_population <- lapply(pop_sizes , function(size) ga(type = "binary", fitness = get_fitness, nBits = limit,maxiter = 1000, run = 200, popSize = size,keepBest = TRUE, maxfitness = global_opt) )

sapply(ga_by_population , function(x) x@fitnessValue)

ga_by_population[[3]]@iter



####################################################################################################################################################################################################
####################################################################################################################################################################################################

### Simulated Annealing Implementation
#######################################

# Since the GenSA function searches for a global minimum, we need to get the negative of our fitness function
# to work in the GenSA implementation.  The negative of its result will be the optimum value found by the SA algorithm.
f <- function(input)
{
  input <- round(input)
  temp = get_fitness(input)
  -temp
  
}


# Implmenent simulated annealing algorithm -- Initial Run
start_time <- proc.time()
sim_anneal_model <- GenSA(par = rep(0,150) , lower = rep(0,150), upper = rep(1,150), fn = f)
end_time <- proc.time() - start_time

# get fitness value
-sim_anneal_model$value

# Define a starting vector of all zeros for the SA implementations
starter <- rep(0,150)

# Test simulated annealing model with varying temperature values
temperature_values <- c(100,500,1000,2000,3000,5000,7500,10000)
sim_anneal_by_temp <- lapply(temperature_values , function(temp) GenSA(par = starter , lower = rep(0,150), upper = rep(1,150), fn = f, control = list(temperature = temp)))

# get fitness value performance by temperature parameter choice
-(sapply(sim_anneal_by_temp,function(x) x$value))


sim_anneal_by_temp <- lapply(rep(5000,5) , function(temp) GenSA(par = starter , lower = rep(0,150), upper = rep(1,150), fn = f, control = list(temperature = temp)))

# Test with low temperature values
lower_temp_values <- c(0.01 , .05, .075, 0.1, 1, 10, 30, 50)
sim_anneal_by_lower_temp <- lapply(lower_temp_values , function(temp) GenSA(par = starter , lower = rep(0,150), upper = rep(1,150), fn = f, control = list(temperature = temp)))

# Get fitness values of the lower temperature ranges
-(sapply(sim_anneal_by_lower_temp,function(x) x$value))

# Test varying number of iterations for simulated annealing
max_iteration_values <- c(1,10,100,1000)
sim_anneal_by_maxit <- lapply(max_iteration_values , function(elt) GenSA(par = starter , lower = rep(0,150), upper = rep(1,150), fn = f, control = list(maxit = elt,temperature = 5000)))
-(sapply(sim_anneal_by_maxit,function(x) x$value))



x <- GenSA(par = starter , lower = rep(0,150), upper = rep(1,150), fn = f, control = list(maxit = 10000,temperature = 5000))
-x$value


# max_iteration_values <- c(1,10,100,1000)
# sim_anneal_by_maxit <- lapply(max_iteration_values , function(elt) GenSA(par = starter , lower = rep(0,150), upper = rep(1,150), fn = f, control = list(maxit = elt,temperature = 5000)))
# -(sapply(sim_anneal_by_maxit,function(x) x$value))











