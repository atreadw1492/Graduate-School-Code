###########################################################################################################################################################################################
## Script to implement five supervised machine learning algorithms
## Algorithms are implemented on adult social and economic data with the goal of predicting whether an individual earns a salary in excess of $50K
###########################################################################################################################################################################################

###########################################################################################################################################################################################

# Load packages
require(RWeka) # decision trees
require(neuralnet) # neural nets
require(nnet) # neural nets
require(kknn) # k-nearest-neighbor
require(knncat) # k-nearest-neighbor
require(class) # k-nearest neighbor
require(ada) # For boosting
require(caret) # for k-fold cross validation
require(pROC) # roc curves and AUC metrics
require(e1071) # support vector machines


# Read in data
train_income <- read.csv("train_income.csv")
test_income <- read.csv("test_income.csv")

# Drop records with missing values
train_income <- na.omit(train_income)
test_income <- na.omit(test_income)

## Clean up data
#####################################
train_income$DataSet_Type <- "Train"
test_income$DataSet_Type <- "Test"

# Format labels accordingly
dataSet <- data.frame(rbind(train_income,test_income))
dataSet$class <- as.factor(gsub("\\.","",as.character(dataSet$class)))


dataSet <- droplevels(subset(dataSet , native_country != "Holand-Netherlands"))


train_income <- subset(dataSet , DataSet_Type == "Train")
test_income <- subset(dataSet, DataSet_Type == "Test")

train_income$DataSet_Type <- test_income$DataSet_Type <- NULL

train_income = droplevels(train_income)


train_set <- train_income
test_set <- test_income

# Normalize the data using Min-Max approach
for(field in names(train_income))
{
  
    if(is.numeric(train_income[[field]]))
    {
      
          train_income[[field]] <- ( train_income[[field]] - min(train_income[[field]]) ) / ( max(train_income[[field]]) - min(train_income[[field]]) )
      
          test_income[[field]] <- ( test_income[[field]] - min(test_income[[field]]) ) / ( max(test_income[[field]]) - min(test_income[[field]]) )
      
    }
  
}

## Define function to get predicted values, confusion matricies, and accuracy performances 
## of a model on the train and test sets
get_model_info <- function(model,test = test_income, train = train_income)
{
  
      predicted_values <- predict(model,test,type="class")
      
      
      test_confusion_matrix <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))
      
      
      train_confusion_matrix <- table(predict(model,train,type="class"),train[,"class"],dnn=list('predicted','actual'))
      
      
      
      test_income_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
      
      train_income_performance <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)
      
      
      return(environment())
  
}

## Define function to generate plot of train performance versus test performance by some input category, X
get_train_test_plot <- function(X , train_performances, test_performances, TITLE, XLAB)
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


# Get feature names and create a formula object based off the features--to be used by the various algorithms
feature_names <- names(train_income)[names(train_income) != "class"]
model_formula <- as.formula(paste("class ~",paste(feature_names,collapse = "+")))


################################################## Implement the machine learning algorithms ##############################################################################################


# Implement C4.5 Decision Tree Algorithm for entire train set, train_income (using the RWeka package)
start <- proc.time()
decision_tree_fit <- J48(model_formula , data = train_set)
tree_run_time <- proc.time() - start

# Get performance, confusion matrix, and test prediction values
decision_tree_info <- get_model_info(decision_tree_fit,test = test_set, train = train_set)

# Print decision treeperformance and confusion matrix
print(decision_tree_info$test_income_performance)

print(decision_tree_info$test_confusion_matrix)


# Get various metrics for decision tree classifier
decision_tree_eval <- evaluate_Weka_classifier(decision_tree_fit,newdata = test_set,complexity = TRUE, class= TRUE)



## Full train_income dataset is 32,561 records
## Take varying subsets of train_income to examine training / testing performance as training dataset increases

# Create vector of subset-sizes
sizes <- seq(500,32500,500)

all_trees <- lapply(sizes, function(size) J48(model_formula , data = train_set[1:size,]))
all_trees_info <- lapply(all_trees , function(elt) get_model_info(elt,test = test_set, train = train_set))

# Same as above, except use uniformly random subsets
all_trees_random <- lapply(sizes, function(size) J48(model_formula , data = train_set[sample(1:nrow(train_set) , size = size),]))
all_trees_random_info <- lapply(all_trees_random , get_model_info(elt, test = test_set , train = train_set))



train_performances_by_size <- sapply(all_trees_info , function(elt) elt$train_income_performance)
test_performances_by_size <- sapply(all_trees_info , function(elt) elt$test_income_performance)

train_performances_random_by_size <- sapply(all_trees_random , function(elt) elt$train_income_performance)

# Generate plot of train versus test performance by number of training data points
get_train_test_plot(sizes , train_performances_by_size , test_performances_by_size, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")



## Examine Pruning Confidence

# Create vector of confidence levels; the default use in the J48 function is 0.25; using confidence levels beyond 0.5 may cause errors
confidence_levels <- seq(0.05 , 0.5, .05)

# Generate decision trees using the C4.5 algorithm with various confidence level choices
trees_by_confidence <- lapply(confidence_levels , function(elt) J48(model_formula , data = train_set, control = Weka_control(C = elt)))
info_by_confidence <- lapply(trees_by_confidence , function(elt) get_model_info(elt , test = test_set, train = train_set))

# Get train and test performance results based off the various confidence levels for the C4.5 algorithm
train_performances_by_confidence <- sapply(info_by_confidence , function(elt) elt$train_income_performance)
test_performances_by_confidence <- sapply(info_by_confidence , function(elt) elt$test_income_performance)



# Test changing Number of Instances per leaf in decision tree:
various_instances = lapply(1:10 , function(num_instances) J48(model_formula , data = train_set, control = Weka_control(M = num_instances)))
various_instances_info <- lapply(various_instances , function(elt) get_model_info(elt , test = test_set, train = train_set))

# Get test and train performance by instance change
test_performance_by_instance <- sapply(various_instances_info , function(elt) elt$test_income_performance)
train_performance_by_instance <- sapply(various_instances_info , function(elt) elt$train_income_performance)


## Decision Tree Model comparison by Number of Attributes
single_feature_trees <- lapply(feature_names , function(elt) J48(paste("class ~",elt) , train_set))
single_feature_info <- lapply(single_feature_trees , function(elt) get_model_info(elt, test = test_set, train = train_set))

names(single_feature_trees) <- names(single_feature_info) <- feature_names

single_tree_train_performance <- sapply(single_feature_info, function(elt) elt$train_income_performance)
single_tree_test_performance <- sapply(single_feature_info, function(elt) elt$test_income_performance)


# Testing Decision Tree model performance by adding features, one at a time

# feature_orders <- lapply(1:100 , function(index) sample(feature_names,length(feature_names)))

feature_formulas <- paste("class ~",gsub(" ","+",Reduce(function(a,b) paste(a,b),feature_names,accumulate = TRUE)))

one_by_one_feature_trees <- lapply(feature_formulas , function(MODEL) J48(MODEL , train_set))
one_by_one_feature_info <- lapply(one_by_one_feature_trees,function(elt) get_model_info(elt, test = test_set, train = train_set))


one_by_one_feature_train_perfomance <- sapply(one_by_one_feature_info , function(elt) elt$train_income_performance)
one_by_one_feature_test_perfomance <- sapply(one_by_one_feature_info , function(elt) elt$test_income_performance)



# Now test the effect on the decision tree classifier when an attribute is left out of the model
all_but_one_features <- lapply(1:length(feature_names) , function(index) feature_names[-index])
names(all_but_one_features) <- feature_names

all_but_one_trees <- lapply(all_but_one_features , function(MODEL) J48(as.formula(paste("class ~",paste(MODEL,collapse = "+"))) , train_set) )
all_but_one_info <- lapply(all_but_one_trees , function(elt) get_model_info(elt, test = test_set, train = train_set))

all_but_one_train_performance <- sapply(all_but_one_info , function(elt) elt$train_income_performance)
all_but_one_test_performance <- sapply(all_but_one_info , function(elt) elt$test_income_performance)


# Pruning vs. Non-Pruning

# Generate an un-pruned decision tree on the train_income dataset
non_pruned_tree <- J48(formula = model_formula , data = train_set, control = Weka_control(U = TRUE))
non_pruned_info <- get_model_info(non_pruned_tree, test = test_set, train = train_set)

print(non_pruned_info$test_income_performance)
print(non_pruned_info$train_income_performance)


## Cross-Validation


get_tree_with_folds <- function(num_folds)
{
  
      fit <- J48(model_formula , data = train_set, control = Weka_control(R = TRUE , N = num_folds))
      info <- get_model_info(fit, test = test_set, train = train_set)
 
      return(environment()) 
}

folded_fits <- lapply(2:10 , get_tree_with_folds)

sapply(folded_fits , function(elt) elt$info$test_income_performance)
sapply(folded_fits , function(elt) elt$info$train_income_performance)


info$test_income_performance


##########################################################################################################################################################################################

### Boosting Implementation
###########################

# Implement AdaBoost boosting algorithm
boosting_start <- proc.time()
boosting_model <- ada(model_formula,train_income)
boosting_runtime <- proc.time() - boosting_start

# Get test results and confusion matrix of boosting algorithm
boosting_info <- get_model_info(boosting_model)

# Get ROC curve and AUC of boosting algorithm
roc_curve <- roc(ifelse(test_income$class == ">50K",1,0) , ifelse(boosting_info$predicted_values == ">50K",1,0))
auc(roc_curve)


# Get accuracy by below / above 50K subsets of the data -- specificity and sensitivity
f <- test_set
f$predicted_values <- boosting_info$predicted_values

nrow(subset(f , (class == predicted_values) & (class == "<=50K") )) / nrow(subset(f , (class == "<=50K")))
nrow(subset(f , (class == predicted_values) & (class == ">50K") )) / nrow(subset(f , (class == ">50K")))

# Get precision
( boosting_precision <- nrow(subset(f , (class == ">50K") & (class == predicted_values) )) / nrow(subset(f, predicted_values == ">50K")) )






# Run discrete, real, and gentle boosting algorithms
all_boosting_types <- lapply(c("discrete","real","gentle") , function(TYPE) ada(model_formula,train_income,type = TYPE))
names(all_boosting_types) <- c("discrete","real","gentle")

all_boosting_types_info <- lapply(all_boosting_types,get_model_info)

# Get test performance for boosting type
sapply(all_boosting_types_info,function(elt) elt$test_income_performance)

# Get AUCs for each AdaBoost type algorithm
all_boosting_types_aucs <- sapply(all_boosting_types_info , function(elt) auc(roc(ifelse(test_income$class == ">50K.",1,0) , ifelse(elt$predicted_values == ">50K",1,0))))

# Examine boosting performance by adjusting the learning rate parameter, nu   
boosting_models_learning_rates <- lapply(c(0.01,0.05,.1,.12,.15) , function(elt) ada(model_formula , train_income, nu = elt ))
boosting_models_learning_rates_info <- lapply(boosting_models_learning_rates,get_model_info)

# Get accuracies of the different learning rate models
sapply(boosting_models_learning_rates_info,function(elt) elt$test_income_performance)

# Get AUCs of the different learning rate models
sapply(boosting_models_learning_rates_info , function(elt) auc(roc(ifelse(test_income$class == ">50K.",1,0) , ifelse(elt$predicted_values == ">50K",1,0))))


# Examine max depth on trees
tree_depths <- seq(5,50,5)
boosting_by_tree_depths <- lapply(tree_depths, function(elt) ada(model_formula, train_income, rpart.control(maxdepth = elt)))
boosting_by_tree_depths_info <- lapply(boosting_by_tree_depths,get_model_info)

# Get accuracy and AUC by the different max depth tree levels
sapply(boosting_by_tree_depths_info,function(elt) elt$test_income_performance)
aucs_by_boosting_tree_depth <- sapply(boosting_by_tree_depths_info , function(elt) auc(roc(ifelse(test_income$class == ">50K.",1,0) , ifelse(elt$predicted_values == ">50K",1,0))))

# Test varying the number of boosting iterations
num_iterations <- c(10,30,50,75,100)
boosting_by_num_iterations <- lapply(num_iterations, function(num) ada(model_formula, train_income, iter = num))
boosting_by_num_iterations_info <- lapply(boosting_by_num_iterations, get_model_info)


# Get accuracy and AUC by number of boosting iterations
sapply(boosting_by_num_iterations_info,function(elt) elt$test_income_performance)
aucs_by_boosting_tree_depth <- sapply(boosting_by_num_iterations_info , function(elt) auc(roc(ifelse(test_income$class == ">50K.",1,0) , ifelse(elt$predicted_values == ">50K",1,0))))


# Run boosting algorithms with increasingly larger subsets of the training data
sizes <- seq(500,32500,500)
boosting_by_size <- lapply(sizes, function(size) ada(model_formula,train_income[1:size,]))

boosting_by_size_performances <- lapply(boosting_by_size , function(elt) {result = get_model_info(elt) ; return(list(result$test_income_performance , result$train_income_performance))  })

temp <- lapply(boosting_by_size_performances,unlist)

boosting_training_performances <- sapply(temp , function(elt) elt[2])
boosting_testing_performances <- sapply(temp , function(elt) elt[1])

# Get Train / Test accuracy performance plot for boosting algorithm
get_train_test_plot(sizes , boosting_training_performances , boosting_testing_performances, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")


# boosting_by_tree_depth <- ada(model_formula, train_income, rpart.control(maxdepth = 5))





# # Look at different loss functions for the boosting model
# ada(model_formula , train_income, loss =  )
# 
# 
# 
# roc_curve <- roc(ifelse(test_income$class == ">50K",1,0) , ifelse(boosting_info$predicted_values == ">50K",1,0))
# 
# plot(roc_curve)
# 
# 
# auc <- auc(roc_curve)
# 
# 
# 
# roc_curve <- roc(ifelse(test_income$class == ">50K",1,0) , ifelse(decision_tree_info$predicted_values == ">50K",1,0))
# 
# plot(roc_curve)
# 
# 
# auc <- auc(roc_curve)

##########################################################################################################################################################################################

### K-Nearest Neighbor Approach
###############################

## Define function go get test results of knn algorithm implementation
get_knn_info <- function(model)
{
  
  
  predicted_values <- predict(model , train_income, test_income, train.classcol = 15, test.classcol = 15)
  
  
  test_confusion_matrix <- table(predicted_values,test_income[,"class"],dnn=list('predicted','actual'))
  
  test_income_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
  
  
  return(environment())
  
}


# Run knn algorithm
# Algorithm performs feature selection via permutation tests and an improvement cutoff
# Cross-validation is done with three folds in the algorithm
knn_start <- proc.time()
knn_model <- knncat(train_income , classcol = 15 , k = 3, permute = 5, xvals = 3)
knn_end <- proc.time()

knn_values_start <- proc.time()
values <- predict(knn_model , train_income, test_income, train.classcol = 15, test.classcol = 15)
knn_values_end <- proc.time()

# Total knn runtime
(knn_values_end - knn_values_start) + (knn_end - knn_start)

# Get test performance info
test_confusion_matrix <- table(values,test_income[,"class"],dnn=list('predicted','actual'))
test_income_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)

# Test out different values of k in algorithm
k_values <- c(3,5,11,15,21,25)
knn_models_by_k <- lapply(k_values , function(elt) knncat(train_income , classcol = 15 , k = elt, permute = 5, xvals = 3))


# Test a set of values for k to find the optimal choice
knn_model_find_optimal_k <- knncat(train_income , classcol = 15 , k = c(3,5,11,21,31,51), permute = 5, xvals = 3)

# Get info for the optimally chosen value of k -- 21 in this case
knn_model_find_optimal_k_info <- get_knn_info(knn_model_find_optimal_k)


# Get accuracy by below / above 50K subsets of the data -- specificity and sensitivity
f <- test_set
f$predicted_values <- knn_model_find_optimal_k_info$predicted_values

nrow(subset(f , (class == predicted_values) & (class == "<=50K") )) / nrow(subset(f , (class == "<=50K")))
nrow(subset(f , (class == predicted_values) & (class == ">50K") )) / nrow(subset(f , (class == ">50K")))

# Get precision
( knn_precision <- nrow(subset(f , (class == ">50K") & (class == predicted_values) )) / nrow(subset(f, predicted_values == ">50K")) )


# Get roc curve and AUC info for knn algorithm
roc_curve <- roc(ifelse(test_income$class == ">50K",1,0) , ifelse(knn_model_find_optimal_k_info$predicted_values == ">50K",1,0))
auc(roc_curve)




# Look at K-NN by training size ~ Algorithm is much slower than decision trees or boosting, so the analysis will look at fewer subsets
# Since some the categories' values will not show up in the smaller subsets of the training set, one hot encoding is performed to transform
# the categorical data into numerical bits.  


sizes <- seq(500,25000,3000)

train_labels <- train_income$class

knn_by_size <- lapply(sizes, function(size) knn(train_encoded[1:size,-1] , test_encoded[,-1] , train_labels[1:size]))

knn_confusion_by_size <- lapply(knn_by_size , function(elt) table(elt,test_income$class))
knn_test_performances_by_size <- sapply(knn_confusion_by_size , function(elt) sum(diag(elt)) / sum(elt))

plot(sizes , knn_test_performances_by_size , type="l", col = "red" , xlab = "Test Performance by Number of Training Data Points", ylab = "Performance")
title("Test Performance by Number of Training Data Points")


##########################################################################################################################################################################################

### Neural Network Implementation
##########################################################################################################################################################################################




###########################################################################################################################################################################################

# Read in data
train_income <- read.csv("C:/Users/Andrew/Documents/school stuff/Machine Learning/HW 1/train_income.csv")
test_income <- read.csv("C:/Users/Andrew/Documents/school stuff/Machine Learning/HW 1/test_income.csv")

# Drop records with missing values
train_income <- na.omit(train_income)
test_income <- na.omit(test_income)


train_income$DataSet_Type <- "Train"
test_income$DataSet_Type <- "Test"

dataSet <- data.frame(rbind(train_income,test_income))
dataSet$class <- as.factor(gsub("\\.","",as.character(dataSet$class)))


dataSet <- subset(dataSet , native_country != "Holand-Netherlands")


##########################################################################################################################################################################################




## Function to get confusion matrix and prediction results for neural network models
get_net_model_info <- function(model , fields = NULL)
{
  if(is.null(fields))
    {
        net_results <- compute(model,test_encoded[,-1])
        net_train_results <- compute(model,train_encoded[,-1])
        
    }
  else
   {
        net_results <- compute(model,test_encoded[,fields])
        net_train_results <- compute(model,train_encoded[,fields])
   }
  
    results <- data.frame(actual = test_encoded$class, prediction = net_results$net.result)
    results$prediction <- round(results$prediction)
    
    predicted_values <- results$prediction
    
    test_confusion_matrix <- table(predicted_values,test_encoded[,"class"],dnn=list('predicted','actual'))
    
    test_income_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
    
    
    train_results <- data.frame(actual = train_encoded$class, prediction = net_train_results$net.result)
    train_results$prediction <- round(train_results$prediction)
    
    train_predicted_values <- train_results$prediction
    
    train_confusion_matrix <- table(train_predicted_values,train_encoded[,"class"],dnn=list('predicted','actual'))
    
  
    train_income_performance <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)
  
  
  return(environment())
  
  
}

### Transform training and testing data using one hot encoding; necessary for neural net algorithms
##########################################################################################################################################################################################

# Create vector of class to be predicted
dataSet_y_pred <- dataSet$class

# Get dataset containing only the features to be used by the algorithms
dataSet$class <- NULL

# Split the dataset into the fields that are categorical and the ones that are numeric
dataSet_factors <- dataSet[,names(which(sapply(dataSet, function(elt) is.factor(elt))))]
dataSet_non_factors <- dataSet[,names(which(sapply(dataSet, function(elt) !is.factor(elt))))]


# One hot encoding on the categorical data
temp <- as.data.frame(model.matrix(~ . + 0, data=dataSet_factors, contrasts.arg = lapply(dataSet_factors, contrasts, contrasts=FALSE)))

dataSet_encoded <- data.frame(temp,dataSet_non_factors)
dataSet_encoded <- data.frame(class = dataSet_y_pred , dataSet_encoded)
dataSet_encoded$class <- ifelse(dataSet_y_pred == ">50K",1,0)

# Get the transformed train and test sets based off one hot encoding
train_encoded <- subset(dataSet_encoded , DataSet_Type == "Train")
test_encoded <- subset(dataSet_encoded , DataSet_Type == "Test")

train_encoded$DataSet_Type <- test_encoded$DataSet_Type <- NULL

# Clear up data issue
train_encoded$native_countryHoland.Netherlands <- test_encoded$native_countryHoland.Netherlands <- NULL


# Min-Max approach for normaliztion
for(field in names(train_encoded))
{
  
  if(is.numeric(train_encoded[[field]]))
  {
    
    train_encoded[[field]] <- ( train_encoded[[field]] - min(train_encoded[[field]]) ) / ( max(train_encoded[[field]]) - min(train_encoded[[field]]) )
    
    test_encoded[[field]] <- ( test_encoded[[field]] - min(test_encoded[[field]]) ) / ( max(test_encoded[[field]]) - min(test_encoded[[field]]) )
    
  }
  
}



net_start <- proc.time()
net_model <- neuralnet(formula(train_encoded) , data = train_encoded)
net_runtime <- proc.time() - net_start



# Initial run of the neural net algorithm    
net_info <- get_net_model_info(net_model)

# Get ROC curve and AUC

net_roc <- roc(test_encoded$class , net_info$predicted_values)
auc(net_roc)


# Get accuracy by below / above 50K subsets of the data
f <- test_encoded
f$predicted_values <- net_info$predicted_values

nrow(subset(f , (class == predicted_values) & (class == 0) )) / nrow(subset(f , (class == 0)))
nrow(subset(f , (class == predicted_values) & (class == 1) )) / nrow(subset(f , (class == 1)))

# Get precision
( net_precision <- nrow(subset(f , (class == 1) & (class == predicted_values) )) / nrow(subset(f, predicted_values == 1)) )


##########################################################################################################################################################################################



# Run algorithm with reduced features
fields <- rev(names(dataSet_non_factors))[-1]

reduced_net <- neuralnet(formula(train_encoded[,c("class",fields)]) , data = train_encoded[,c("class",fields)])

e <- get_net_model_info(reduced_net,fields)


reduced_net_2_layers <- neuralnet(formula(train_encoded[,c("class",fields)]) , data = train_encoded[,c("class",fields)], hidden = 2)
e2 <- get_net_model_info(reduced_net_2_layers,fields)

    

## Sensitivity analysis on the Neural Nets for the threshold levels
x <- neuralnet(formula(train_encoded) , data = train_encoded, threshold = .001)
info <- get_net_model_info(x)
info$test_income_performance


x <- neuralnet(formula(train_encoded) , data = train_encoded, threshold = 1000)
info <- get_net_model_info(x)
info$test_income_performance



thresholds <- c(10,100,1000)
net_by_threshold <- lapply(thresholds , function(elt) neuralnet(formula(train_encoded) , data = train_encoded, threshold = elt))
net_by_threshold_info <- lapply(net_by_threshold,get_net_model_info)

sapply(net_by_threshold_info , function(elt) elt$test_income_performance)
sapply(net_by_threshold_info , function(elt) elt$train_income_performance)

# Get AUCs of nets_by_threshold's
sapply(net_by_threshold_info , function(info) auc(roc(test_encoded$class , info$predicted_values)))


  


nets_by_hidden_layers <- lapply(1:3, function(num) neuralnet(formula(train_encoded) , data = train_encoded, hidden = num))



two_layers <- neuralnet(formula(train_encoded) , data = train_encoded, hidden = 2)



start <- system.time()
net_model_backprop <- neuralnet(formula(train_encoded) , data = train_encoded,algorithm = "backprop",learningrate = .01,act.fct = "tanh")
end <- system.time()


# net_model_slr <- neuralnet(formula(train_encoded) , data = train_encoded,algorithm = "slr")


# Train neural networks on varying sizes of training points
sizes <- seq(500,32500,500)
nets_by_size <- lapply(sizes, function(elt) neuralnet(formula(train_encoded) , data = train_encoded[1:elt,]))

# save neural network algorithms to R Data file
save(nets_by_size,file = "C:/Users/Andrew/Documents/school stuff/Machine Learning/HW 1/nets_by_size.RData")


net_test_by_size <- sapply(nets_by_size,function(elt) get_net_model_info(elt)$test_income_performance)
net_train_by_size <- sapply(nets_by_size,function(elt) get_net_model_info(elt)$train_income_performance)


get_train_test_plot(sizes , train_performances = net_train_by_size , net_test_by_size, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")


# net_test_performance_by_size <- sapply(net_info_by_size,function(elt) elt$test_income_performance)




## Analysis by examining the number of inputs in hidden layer
# Neural net with one hidden layer


nets_by_num_inputs_in_hidden_layer <- lapply(1:9 , function(num) nnet(model_formula,train_income,size = num, softmax = FALSE) )

nets_info_by_num_inputs_in_hidden_layer <- lapply(nets_by_num_inputs_in_hidden_layer , get_model_info)


sapply(nets_info_by_num_inputs_in_hidden_layer,function(elt) elt$test_income_performance)



x <- lapply(1:100 , function(num) nnet(formula(train_encoded),train_encoded,size = num, softmax = FALSE) )




##########################################################################################################################################################################################
############


### Support Vector Machines
##########################################################################################################################################################################################


svm_start <- proc.time()
svm_model <- svm(model_formula , train_income)
svm_runtime <- proc.time() - svm_start


svm_info <- get_model_info(svm_model)

svm_info$test_income_performance


svm_roc_curve <- roc(ifelse(test_income$class == ">50K",1,0) , ifelse(svm_info$predicted_values == ">50K",1,0))
auc(svm_roc_curve)


f <- test_income
f$predicted_values <- svm_info$predicted_values

nrow(subset(f , (class == predicted_values) & (class == "<=50K") )) / nrow(subset(f , (class == "<=50K")))
nrow(subset(f , (class == predicted_values) & (class == ">50K") )) / nrow(subset(f , (class == ">50K")))


# Get precision
( svm_precision <- nrow(subset(f , (class == ">50K") & (class == predicted_values) )) / nrow(subset(f, predicted_values == ">50K")) )




# Train support vector machines on varying sizes of training points
sizes <- seq(1000,32500,2500)
svms_by_size <- lapply(sizes, function(elt) svm(model_formula , data = train_income[1:elt,]))

svms_by_size_info <- lapply(svms_by_size , get_model_info)

svm_train_by_size_performances <- sapply(svms_by_size_info , function(elt) elt$train_income_performance)
svm_test_by_size_performances <- sapply(svms_by_size_info , function(elt) elt$test_income_performance)


get_train_test_plot(sizes , svm_train_by_size_performances , svm_test_by_size_performances, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")


# svm_start <- proc.time()
# svm_model <- svm(formula(train_encoded) , train_encoded)
# svm_runtime <- proc.time() - svm_start




# predicted_values <- predict(svm_model,test_encoded)
# predicted_values <- round(predicted_values)
# 
# test_confusion_matrix <- table(test_encoded$class , predicted_values)
# sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)




# Test out different kernels

kernel_types <- c("radial","linear","sigmoid","polynomial")

svm_by_kernel <- lapply(kernel_types , function(elt) svm(model_formula , train_income , kernel = elt))

svm_kernel_info <- lapply(svm_by_kernel,get_model_info)
names(svm_kernel_info) <- kernel_types

sapply(svm_kernel_info , function(elt) elt$test_income_performance)


# svm_kernel_roc_curves <- lapply(svm_by_kernel,roc)


svm_kernel_roc_curves <- lapply(svm_kernel_info, function(info) roc(ifelse(test_income$class == ">50K",1,0) , ifelse(info$predicted_values == ">50K",1,0)))

sapply(svm_kernel_roc_curves , auc)

# Test different degree of polynomial kernel ~ Default is 3rd degree polynomial
polynomial_kernel_svms <- lapply(2:5 , function(degree) svm(model_formula , train_income , kernel = "polynomial" , degree = degree))
polynomial_kernel_svms_info <- lapply(polynomial_kernel_svms , get_model_info)

# Get accuracies of the polynomial kernels
sapply(polynomial_kernel_svms_info , function(elt) elt$test_income_performance)


# Get ROC curve info of the different ordered polynomial kernels
svm_polynomial_kernel_roc_curves <- lapply(polynomial_kernel_svms_info, function(info) roc(ifelse(test_income$class == ">50K",1,0) , ifelse(info$predicted_values == ">50K",1,0)))

# Get AUCs of the different ordered polynomial kernels
sapply(svm_polynomial_kernel_roc_curves,auc)



# Picking the linear kernel, as it had the best performance of the kernels--test adjusting the termination tolerance level

x <- svm(model_formula , train_income , kernel = "linear" , tolerance = .1)

x_info <- get_model_info(x)



###########################################################################################################################################################################################





# 
# ######################################################################################################################################################
# 
# 
# 
# 
# 
# ###################################################################################################################################################
# 
# 
# 
# 
# 









