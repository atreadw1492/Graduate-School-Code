
### Knapsack Problem
### This script contains code for genetic algorithm and simulated annealing implementations to optimize the Knapsack problem
### The code reads in data from a csv file in the current working directory; change directory to where the file is located
### for the code to work properly.
###########################################################################################################################

# Load packages
require(GA) # for genetic algorithms
require(GenSA)


### Knapsack Problem
####################

# Set parameters
# items <- 1:50
# item_weights <- sample(10,length(items),replace = TRUE)
# item_values <- sample(15,length(items),replace = TRUE)
# 
# # Set max weight
# max_weight <- 100
# 
# # Create dataset containing the weight and value data for the items
# dataset <- data.frame(items = items, item_weights = item_weights, item_values = item_values)

# item_weights <- as.numeric(readLines("http://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p07_w.txt"))
# item_values <- as.numeric(readLines("http://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p07_w.txt"))

# items <- 1:length(item_weights)


# dataset <- data.frame(items = items, item_weights = item_weights, item_values = item_values)
max_capacity <- 50

dataset <- read.csv("knapsack_data.csv")



starter <- c(1,rep(0,49))

# Create function to calculate total value and weight given an input vector of bits
# Each bit of the input vector represents whether the corresponding item of the index of that bit is selected to be
# in the knapsack by having a value of 1 or 0 (selected or not-selected) e.g. if the 5th element of the input vector is 1,
# then the the 5th item is selected etc.
get_fitness <- function(input_vector)
{
    
    fitness <- sum(input_vector * dataset$item_values)
    fitness_penalty <- sum(dataset$item_weights) * abs(sum(input_vector * dataset$item_weights) - max_capacity)
    
    adj_fitness <- fitness - fitness_penalty
    
    return(adj_fitness)
}


############################################################################################################################################################################################
############################################################################################################################################################################################

########################### Genetic Algorithms Implmentation ###############################################################################################################################


# Implment genetic algorithm using GA package
start_time <- proc.time()
genetic_eval <- ga(type = "binary", fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE)
end_time <- proc.time() - start_time

genetic_eval@solution

x <- summary(genetic_eval)
x$fitness


# Run genetic algorithm by varying the population size
# pop_sizes <- seq(500,10000,500)
# ga_by_population <- lapply(pop_sizes , function(size) ga(type = "binary", fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = size) )



smaller_pop_sizes <- seq(100,1000,100)
smaller_ga_by_population <- lapply(smaller_pop_sizes , function(size) ga(type = "binary", fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = size,keepBest = TRUE) )

# Get max fitness and number of iterations by population size
fitness_levels <- sapply(smaller_ga_by_population , function(x) summary(x)$fitness)
fitness_iterations <- sapply(smaller_ga_by_population , function(x) summary(x)$iter)



plot(smaller_pop_sizes , fitness_iterations,type="l",col="blue")
par(bg="lightgrey")
title("Number of Iterations by Population Size")



get_plot <- function(X , fitness_levels, fitness_iterations, TITLE, XLAB)
{
  
  Y_AXIS_LIMITS <- range(c(train_performances,test_performances)) * c(0.95 , 1.05)
  
  # Plot train and test performances as measured against size of training set data points
  plot(X , train_performances,type="l",col="darkblue", ylab = "Performance" , xlab = XLAB, ylim = Y_AXIS_LIMITS)
  par(bg="darkgrey")
  lines(X, test_performances , col = "red", ylab= "" , lty = 2 , xlab = "", ylim = Y_AXIS_LIMITS)
  par(bg="darkgrey")
  title(TITLE)
  
  legend("topright" , c("Train Performance" , "Test Performance") , lty = c(1,2) , col = c("darkblue" , "red") , bty="n", cex = .75)
  
}








# Perform sensitivity analysis of the number of best fitness individuals to survive each generation
top_individuals <- c(10,30,50,75,100)
ga_by_elite_size <- lapply(top_individuals , 
                           function(size) ga(type = "binary", fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE, elitism = size))

  

# Get max fitness value achieved and total number of iterations for each run of algorithm varying by elite size
sapply(ga_by_elite_size , function(x) summary(x)$fitness)
sapply(ga_by_elite_size , function(x) summary(x)$iter)


# Test effect of various crossover functions
ga_uniform_crossover <- ga(type = "binary", crossover = "gabin_uCrossover" ,fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE)


summary(ga_uniform_crossover)




# Test different selection operators
selection_choices <- c("gabin_lrSelection" , "gabin_nlrSelection","gabin_rwSelection","gabin_tourSelection")

ga_by_selection <- lapply(selection_choices , 
                          function(choice) ga(type = "binary", crossover = "gabin_uCrossover", selection = choice , fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE))


# Get max fitness values and total number of iterations 
sapply(ga_by_selection , function(x) summary(x)$fitness)
sapply(ga_by_selection , function(x) summary(x)$iter)


# Test roulette wheel selection further, as it had the worst results of the selection operators
pop_sizes <- c(1000,3000,5000,8000,10000)
roulette_wheel_by_pop_size <- lapply(pop_sizes, function(size) {ga(type = "binary", crossover = "gabin_uCrossover", selection = "gabin_rwSelection" , fitness = get_fitness, 
                                                                   nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE)})

sapply(roulette_wheel_by_pop_size , function(x) summary(x)$fitness)


sp_roulette_wheel_by_pop_size <- lapply(pop_sizes, function(size) {ga(type = "binary", crossover = "gabin_uCrossover", selection = "gabin_rwSelection" , fitness = get_fitness, 
                                                                      nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE)})

sapply(sp_roulette_wheel_by_pop_size , function(x) summary(x)$fitness)



roulette_runs <- lapply(1:10 , try(ga(type = "binary", crossover = "gabin_uCrossover", selection = "gabin_rwSelection" , fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE),silent=TRUE ))





get_rw <- function() ga(type = "binary", crossover = "gabin_uCrossover", selection = "gabin_rwSelection" , fitness = get_fitness, nBits = nrow(dataset),maxiter = 1000, run = 200, popSize = 1000, keepBest = TRUE)

roulette_runs <- lapply(1:20 , function(index) get_rw())


roulette_runs_fitness <- sapply(roulette_runs , function(x) summary(x)$fitness)
summary(roulette_runs_fitness)


par(bg="lightgrey")
plot(roulette_runs_fitness,type="l",col="blue",xlab="Implementation Index",ylab = "Fitness")
title("Max Value Achieved by Wheel Roulette Runs")





# Test effect of various mutation functions
gaControl("binary" = list(selection = "ga_rwSelection",
                          crossover = "gabin_uCrossover"))




############################################################################################################################################################################################




### Simulated Annealing
####################################################################################

# Define function to get the negative return value of get_fitness.  This is done because the GenSA function tries to minimize its function--thus, by minmizing f,
# we maximize our actual fitness function, get_fitness
# This function also ensures the parameters are discrete (rounds them to nearest integer)
f <- function(input)
{
    input <- round(input)
    temp = get_fitness(input)
    -temp
  
}



# Implmenent simulated annealing algorithm
start_time <- proc.time()
sim_anneal_model <- GenSA(par = starter , lower = rep(0,50), upper = rep(1,50), fn = f)
end_time <- proc.time() - start_time


# Test simulated annealing model with varying temperature values
temperature_values <- c(100,500,1000,2000,3000,5000,7500,10000)
sim_anneal_by_temp <- lapply(temperature_values , function(temp) GenSA(par = starter , lower = rep(0,50), upper = rep(1,50), fn = f, control = list(temperature = temp)))

# get fitness value performance by temperature parameter choice
-(sapply(sim_anneal_by_temp,function(x) x$value))

# Test with low temperature values
lower_temp_values <- c(0.01 , .05, .075, 0.1, 1, 10, 30, 50)
sim_anneal_by_lower_temp <- lapply(lower_temp_values , function(temp) GenSA(par = starter , lower = rep(0,50), upper = rep(1,50), fn = f, control = list(temperature = temp)))

-(sapply(sim_anneal_by_lower_temp,function(x) x$value))



# Test varying number of iterations for simulated annealing
max_iteration_values <- c(1,10,100,1000,10000,100000)
sim_anneal_by_maxit <- lapply(max_iteration_values , function(elt) GenSA(par = starter , lower = rep(0,50), upper = rep(1,50), fn = f, control = list(maxit = elt)))
-(sapply(sim_anneal_by_maxit,function(x) x$value))


start <- proc.time()
x <- GenSA(par = starter , lower = rep(0,50), upper = rep(1,50), fn = f, control = list(maxit = 100000))
end <- proc.time() - start

# 
# 
# 
# 
# x <- optim(par=c(0,0), fn=get_fitness, gr = NULL,
#              method = c("SANN"),
#              lower = -Inf, upper = Inf,
#              control = list(fnscale=-1), hessian = T)

# 
# 
# 
# values <- dataset$item_values
# weights <- dataset$item_weights
# beta <- seq(0,1,.01)
# 
# sim_anneal <- function(values , weights, max_capacity, beta, n)
# {
#   
#   v <- length(values)
#   W <- rep(0,v)
#   VW <- 0
#   a <- length(beta)
#   nn <- n * rep(1,a)
#   for(i in 1:a)
#   {
#     b <- beta(i)
#     for(j in 2:nn[i])
#     {
#       
#       c = ceil
#       
#     }
#     
#   }
#   
#   
# }

