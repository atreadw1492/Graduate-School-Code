###########################################################################################################################################################################################
## Script to implement five supervised machine learning algorithms
## Algorithms are implemented on exercise workout-related data with the goal of predicting whether an individual is performing a specific workout movement correctly
###########################################################################################################################################################################################

###########################################################################################################################################################################################

# Load packages
require(RWeka) # decision trees
require(neuralnet) # neural nets
require(nnet) # neural nets
require(knncat) # k-nearest-neighbor
require(class) # k-nearest neighbor
require(ada) # For boosting
require(caret) # for k-fold cross validation
require(pROC) # roc curves and AUC metrics
require(e1071) # support vector machines


# Read in data
dataSet <- read.csv("weight_lifting_data.csv")


# Get rid of unnecessary fields and any records with missing values
dataSet <- dataSet[,-(1:6)]
dataSet <- na.omit(dataSet)

# Construct the core forumla to be used for the machine learning models
fields <- names(dataSet)
model_formula <- as.formula(paste("class ~",paste(fields[fields != "class"],collapse = "+")))

# Generate random sample of row indicies; use to create train and test sets
train_samples <- sample(1:nrow(dataSet),nrow(dataSet) / 2)

# Create train and test sets
train <- dataSet[train_samples,]
test <- dataSet[-train_samples,]


# Normalize feauture values
train_norm <- as.data.frame(lapply(train , function(value) {if(is.factor(value)) value else (value - min(value)) / (max(value) - min(value))}))
test_norm <- as.data.frame(lapply(test , function(value) {if(is.factor(value)) value else (value - min(value)) / (max(value) - min(value))}))


# Use this for taking varying subsets of the training data to compare how fast each algorithm learns the training data
random_ordered_train_index <- sample(1:nrow(train) , 19000)


###########################################################################################################################################################################################

## Define function to get train and test performance info for the algorithms implemnted in the script
get_model_info <- function(model)
{
  
  predicted_values <- predict(model,test,type="class")
  
  train_predicted_values <- predict(model,train,type="class")
  
  test_confusion_matrix <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))
  
  
  train_confusion_matrix <- table(predict(model,train,type="class"),train[,"class"],dnn=list('predicted','actual'))
  
  
  
  test_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
  
  train_performance <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)
  
  
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

###########################################################################################################################################################################################

# Implement C4.5 Decision Tree Algorithm for entire train set, train (using the RWeka package)
tree_start <- proc.time()
decision_tree_fit <- J48(model_formula , data = train)
tree_end <- proc.time()

tree_end - tree_start

decision_tree_info <- get_model_info(decision_tree_fit)


# Get various metrics for decision tree classifier
decision_tree_eval <- evaluate_Weka_classifier(decision_tree_fit,newdata = test,complexity = TRUE, class= TRUE)



# Get ROC curve and AUC
roc_curve <- multiclass.roc(match(test$class,LETTERS) , match(decision_tree_info$predicted_values,LETTERS))
as.numeric(auc(roc_curve))

###########################################################################################################################################################################################

# Experiment with varying the number of folds for cross-fold validation via reduced error pruning
decision_tree_fit_adj <- J48(model_formula , data = train, control = Weka_control(R= TRUE, N=5))
decision_tree_fit_adj_info <- get_model_info(decision_tree_fit_adj)

# Do reduced error pruning with cross-validaton
num_folds <- c(3,5,7,10)
decision_tree_by_fold <- lapply(num_folds , function(elt) J48(model_formula , data = train, control = Weka_control(R= TRUE, N=elt)))
decision_tree_by_fold_info <- lapply(decision_tree_by_fold , get_model_info)

# Get test and train performance by number of folds
sapply(decision_tree_by_fold_info,function(elt) elt$test_performance)
sapply(decision_tree_by_fold_info,function(elt) elt$train_performance)

# Get ROC curves and AUC
decision_tree_fold_roc_curves <- lapply(decision_tree_by_fold_info , function(info) multiclass.roc(match(test$class,LETTERS) , match(info$predicted_values,LETTERS)))
decision_tree_fold_auc <-sapply(decision_tree_fold_roc_curves , auc)


decision_tree_fold_train_roc_curves <- lapply(decision_tree_by_fold_info , function(info) multiclass.roc(match(train$class,LETTERS) , match(info$train_predicted_values,LETTERS)))
decision_tree_fold_train_auc <-sapply(decision_tree_fold_train_roc_curves , auc)

###########################################################################################################################################################################################



###########################################################################################################################################################################################
## Take varying subsets of train to examine training / testing performance as training dataset increases

# Create vector of subset-sizes
sizes <- seq(1000,nrow(train),1000)

# Generate decision trees for increasingly larger subsets of the training data, starting with 1000 data points up to 19,000
all_trees <- lapply(sizes, function(size) J48(model_formula , data = train[random_ordered_train_index[1:size],]))
all_trees_info <- lapply(all_trees , get_model_info)

# Get train and test accuracies by number of training data points
tree_train_performances_by_size <- sapply(all_trees_info , function(elt) elt$train_performance)
tree_test_performances_by_size <- sapply(all_trees_info , function(elt) elt$test_performance)

# Get Learning Curve Plot comparing train vs. test accuracy against number of training data points
get_train_test_plot(sizes , tree_train_performances_by_size , tree_test_performances_by_size, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")

#####################################################################################################################################################################################


## Examine Pruning Confidence

# Create vector of confidence levels; the default use in the J48 function is 0.25; using confidence levels beyond 0.5 may cause errors
confidence_levels <- seq(0.05 , 0.5, .05)

# Generate decision trees using the C4.5 algorithm with various confidence level choices
trees_by_confidence <- lapply(confidence_levels , function(elt) J48(model_formula , data = train, control = Weka_control(C = elt)))
info_by_confidence <- lapply(trees_by_confidence , get_model_info)

# Get train and test performance results based off the various confidence levels for the C4.5 algorithm
train_performances_by_confidence <- sapply(info_by_confidence , function(elt) elt$train_performance)
test_performances_by_confidence <- sapply(info_by_confidence , function(elt) elt$test_performance)


get_train_test_plot(confidence_levels , train_performances_by_confidence , test_performances_by_confidence, "Train / Test Performance by Pruning Confidence Level", 
                                                                                                            "Pruning Confidence Level")


#####################################################################################################################################################################################

# Pruning vs. Non-Pruning

# Generate an un-pruned decision tree on the train_income dataset
non_pruned_tree <- J48(formula = model_formula , data = train, control = Weka_control(U = TRUE))
non_pruned_info <- get_model_info(non_pruned_tree)

non_pruned_info$train_performance


tree <- J48(formula = model_formula , data = train, control = Weka_control(R = TRUE))
info <- get_model_info(tree)


###########################################################################################################################################################################################

### Support Vector Machine
##########################

start <- proc.time()
svm_model <- svm(model_formula,data=train, scale = TRUE)
run_time <- proc.time() - start

svm_model_info <- get_model_info(svm_model)


# Get SVM's by switching out the kernel functions
kernel_types <- c("radial","linear","sigmoid","polynomial")

svm_by_kernel <- lapply(kernel_types , function(elt) svm(model_formula , train , kernel = elt, scale = TRUE))

svm_kernel_info <- lapply(svm_by_kernel,get_model_info)
names(svm_kernel_info) <- kernel_types

sapply(svm_kernel_info , function(elt) elt$test_performance)



svm_kernel_roc_curves <- lapply(svm_kernel_info , function(info) multiclass.roc(match(test$class,LETTERS) , match(info$predicted_values,LETTERS)) )
sapply(svm_kernel_roc_curves,auc)




##############

# Sigmoid and Linear SVM's had the worse performance among the kernels
# Adjust parameters in this kernels

# Sigmoid first
###############

# Create vector of different cost levels
costs <- 10 ** (-3:3)

# Generate SVM's varying based upon the values in the costs vector
svm_sigmoid_by_cost <- lapply(costs, function(elt) svm(model_formula , train , kernel = "sigmoid", cost = elt , scale = TRUE))
svm_sigmoid_by_cost_info <- lapply(svm_sigmoid_by_cost,get_model_info)


# Get ROC curves and AUC values for the cost-varied SVM's
svm_sigmoid_cost_roc_curves <- lapply(svm_sigmoid_by_cost_info , function(info) multiclass.roc(match(test$class,LETTERS) , match(info$predicted_values,LETTERS)) )
sapply(svm_sigmoid_cost_roc_curves,auc)



model <- svm(model_formula , data = train, cost = .01, nu = 1, kernel = "sigmoid" , scale = TRUE)
info <- get_model_info(model)


# Cross validation
svm_cross_model <- svm(model_formula , data = train, cost = .01, cross = 3, kernel = "sigmoid" , scale = TRUE)
svm_cross_info <- get_model_info(svm_cross_model)


svm_cross10_model <- svm(model_formula , data = train, cost = .01, cross = 10, kernel = "sigmoid" , scale = TRUE)
svm_cross10_info <- get_model_info(svm_cross10_model)


# Create vector of subset-sizes
sizes <- c(1000,5000,10000,15000,19000)


# Generate svms for increasingly larger subsets of the training data, starting with 1000 data points up to 19,000
all_svms <- lapply(sizes, function(size) svm(model_formula,data=train[random_ordered_train_index[1:size],], scale = TRUE))
all_svms_info <- lapply(all_svms , get_model_info)

# Get train and test accuracies by number of training data points
svm_train_performances_by_size <- sapply(all_svms_info , function(elt) elt$train_performance)
svm_test_performances_by_size <- sapply(all_svms_info , function(elt) elt$test_performance)

# Get Learning Curve Plot comparing train vs. test accuracy against number of training data points
get_train_test_plot(sizes , svm_train_performances_by_size , svm_test_performances_by_size, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")







# # Get accuracy by below / above 50K subsets of the data -- specificity and sensitivity
# f <- test
# f$predicted_values <- svm_model_info$predicted_values
# 
# nrow(subset(f , (class == predicted_values) & (class == "<=50K") )) / nrow(subset(f , (class == "<=50K")))
# nrow(subset(f , (class == predicted_values) & (class == ">50K") )) / nrow(subset(f , (class == ">50K")))
# 
# # Get precision
# ( knn_precision <- nrow(subset(f , (class == ">50K") & (class == predicted_values) )) / nrow(subset(f, predicted_values == ">50K")) )


###########################################################################################################################################################################################

# # Test different degree of polynomial kernel ~ Default is 3rd degree polynomial
# polynomial_kernel_svms <- lapply(2:5 , function(degree) svm(model_formula , train_income , kernel = "polynomial" , degree = degree))
# polynomial_kernel_svms_info <- lapply(polynomial_kernel_svms , get_model_info)
# 
# # Get accuracies of the polynomial kernels
# sapply(polynomial_kernel_svms_info , function(elt) elt$test_income_performance)
# 
# 
# # Get ROC curve info of the different ordered polynomial kernels
# svm_polynomial_kernel_roc_curves <- lapply(polynomial_kernel_svms_info, function(info) roc(ifelse(test_income$class == ">50K",1,0) , ifelse(info$predicted_values == ">50K",1,0)))
# 
# # Get AUCs of the different ordered polynomial kernels
# sapply(svm_polynomial_kernel_roc_curves,auc)

###########################################################################################################################################################################################


### K-NN Implementation
###########################################################################################################################################################################################



get_knn_info <- function(model)
{
  
  
  predicted_values <- predict(model , train, test, train.classcol = 15, test.classcol = 15)
  
  
  test_confusion_matrix <- table(predicted_values,test_income[,"class"],dnn=list('predicted','actual'))
  
  test_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
  
  
  return(environment())
  
}




knn_model <- knncat(train_income , classcol = 15 , k = 3, permute = 5, xvals = 3)

values <- predict(knn_model , train_income, test_income, train.classcol = 15, test.classcol = 15)


test_confusion_matrix <- table(predicted_values,test_income[,"class"],dnn=list('predicted','actual'))

test_income_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)








###########################################################################################################################################################################################


### Neural Network Implmentation
###########################################################################################################################################################################################


get_net_model_info <- function(model , fields = NULL)
{
  if(is.null(fields))
    net_results <- compute(model,test_norm[,-53])
  else
    net_results <- compute(model,test_norm[,fields])
  
  results <- data.frame(actual = test_norm$class, prediction = net_results$net.result)
  results$prediction <- round(results$prediction)
  
  predicted_values <- results$prediction
  
  test_confusion_matrix <- table(predicted_values,test_norm[,"class"],dnn=list('predicted','actual'))
  
  
  # train_confusion_matrix <- table(predict(model,train_income,type="class"),train_income[,"class"],dnn=list('predicted','actual'))
  
  
  
  test_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
  
  # train_income_performance <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)
  
  
  return(environment())
  
  
}




# train_norm$class <- match(train_norm$class,LETTERS)

train_labels <- train_norm$class

train_norm <- data.frame(train_norm,class.ind(train_norm$class))
train_norm$class <- NULL

net_forumla <- paste(paste(LETTERS[1:5],collapse="+") ,"~", paste(names(train_norm[,1:(ncol(train_norm) - 5)]),collapse = "+"))


start <- proc.time()
net_model <- neuralnet(net_forumla , data = train_norm)
end <- proc.time()


net_info <- get_net_model_info(net_model)


# Sensitivity analysis on different number of inputs in hidden layer
train_norm$class <- train$class
test_norm$class <- test$class

x <- nnet(model_formula,data=train_norm,size=60 , MaxNWts = 100000)

num_inputs <- seq(10,50,10)

nets_by_num_inputs <- lapply(num_inputs , function(num) nnet(model_formula,data=train_norm,size=num , MaxNWts = 100000) )
nets_by_num_inputs_info <- lapply(nets_by_num_inputs,get_net_model_info)


more_inputs <- seq(60,100,10)
nets_by_more_inputs <- lapply(more_inputs , function(num) nnet(model_formula,data=train_norm,size=num , MaxNWts = 100000) )
nets_by_more_inputs_info <- lapply(nets_by_more_inputs,get_net_model_info)

# Get ROC curves and AUCs of neural nets
roc_curves <- lapply(nets_by_num_inputs_info , function(info) multiclass.roc(match(test_norm$class,LETTERS) , match(info$predicted_values,LETTERS)))
sapply(roc_curves , auc)


roc_curves <- lapply(nets_by_more_inputs_info , function(info) multiclass.roc(match(test_norm$class,LETTERS) , match(info$predicted_values,LETTERS)))
sapply(roc_curves , auc)


# 
# 
# get_net_model_info <- function(model)
# {
#   
# 
#     predicted_values <- predict(model,test_norm)
#     predicted_values <- LETTERS[apply(predicted_values,1,which.max)]
#     
#     
#     
#     test_confusion_matrix <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))
#     test_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
# 
# 
#     return(environment())
#     
# }

# train_predicted_values <- predict(model,train,type="class")
train_confusion_matrix <- table(predict(model,train,type="class"),train[,"class"],dnn=list('predicted','actual'))

train_performance <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)



# Create vector of subset-sizes
sizes <- c(1000,5000,10000,15000)

# Run neural network algorithms by increasingly larger subsets of the training data
nets_by_size <- lapply(sizes, function(size) nnet(model_formula,data=train_norm[random_ordered_train_index[1:size],],size=50 , MaxNWts = 100000))
nets_size_info <- lapply(nets_by_size,get_model_info)

net_train_by_size <- sapply(nets_size_info , function(elt) elt$train_performance)
net_test_by_size <- sapply(nets_size_info , function(elt) elt$test_performance)

get_train_test_plot(sizes , net_train_by_size , net_test_by_size, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")






# Get neural nets of increasingly larger subsets
all_nets <- lapply(sizes, function(size) neuralnet(net_forumla , data = train_norm[random_ordered_train_index[1:size],]))
all_nets_info <- lapply(all_trees , get_model_info)

# Get train and test accuracies by number of training data points
tree_train_performances_by_size <- sapply(all_trees_info , function(elt) elt$train_performance)
tree_test_performances_by_size <- sapply(all_trees_info , function(elt) elt$test_performance)

# Get Learning Curve Plot comparing train vs. test accuracy against number of training data points
get_train_test_plot(sizes , tree_train_performances_by_size , tree_test_performances_by_size, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")





##########################################################################################################################################################################################


### Boosting Implementation
##########################################################################################################################################################################################

# Using RWeka and the C4.5 implemented decision trees as weak learners, train "train" data with boosting algorithm
#

ada_start <- proc.time()
ada_boosted_model <- AdaBoostM1(model_formula,data = train,control = Weka_control(W = list(J48, M = 30)))
ada_end <- proc.time()

ada_end - ada_start

predicted_values <- predict(ada_boosted_model, test, type = "class")

test_confusion <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))

test_performance <- sum(diag(test_confusion)) / sum(test_confusion)


roc_curve <- multiclass.roc(match(test$class,LETTERS) , match(predicted_values,LETTERS))
as.numeric(auc(roc_curve))



get_boosting_info <- function(model)
{
      predicted_values <- predict(model, test, type = "class")
      
      test_confusion <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))
      
      test_performance <- sum(diag(test_confusion)) / sum(test_confusion)

      
      train_predicted_values <- predict(model, train, type = "class")
      
      train_confusion <- table(train_predicted_values,train[,"class"],dnn=list('predicted','actual'))
      
      train_performance <- sum(diag(train_confusion)) / sum(train_confusion)
      
      
      return(environment())
      
}


## Get boosting train / test accuracy performances by utilizing increasingly larger subsets of the training data
sizes <- seq(1000,nrow(train),1000)

random_ordered_train_index <- sample(1:nrow(train) , max(sizes))

# Generate decision trees for increasingly larger subsets of the training data, starting with 1000 data points up to 19,000
boosting_by_size <- lapply(sizes, function(size) AdaBoostM1(model_formula,data = train[random_ordered_train_index[1:size],],control = Weka_control(W = list(J48, M = 30)) ))
boosting_by_size_info <- lapply(boosting_by_size , get_boosting_info)

boosting_test_performances <- sapply(boosting_by_size_info , function(elt) elt$test_performance)
boosting_train_performances <- sapply(boosting_by_size_info , function(elt) elt$train_performance)


get_train_test_plot(sizes , boosting_train_performances , boosting_test_performances, "Train / Test Performance by Number of Training Data Points", "Number of Training Data Points")


#########################################################################################################################################################################################

















# 
# # require(ada)
# 
# require(adabag)
# 
# 
# # boosting_model <- ada(model_formula,train_norm)
# # 
# # boosting_info <- get_model_info(boosting_model)
# # 
# # roc_curve <- roc(boosting_model)
# # auc <- auc(roc)
# 
# start <- proc.time()
# boosting_model <- boosting(model_formula , data = train_norm)
# end <- proc.time()
# 
# 
# 
# predicted_values <- predict(boosting_model,test_norm,type="class")
# 
# test_confusion <- predicted_values$confusion
# 
# test_performance <- sum(diag(test_confusion)) / sum(test_confusion)
# 
# 
# 
# 
# 
# 
# train_predicted_values <- predict(boosted_model, train, type = "class")
# train_confusion <- table(train_predicted_values,train$classe,dnn=c("predicted","actual"))
# 
# train_performance <- sum(diag(train_confusion)) / sum(train_confusion)
# 
# 
# 
# # train <- na.omit(train)
# # test <- na.omit(test)
# 
# 
# 
# 
# 
# 
# 
# ada_boosted_model <- AdaBoostM1(model_formula,data = train_norm,control = Weka_control(W = list(J48, M = 30)))
# 
# predicted_values <- predict(ada_boosted_model, test_norm, type = "class")
# 
# test_confusion <- table(predicted_values,test_norm[,"class"],dnn=list('predicted','actual'))
# 
# test_performance <- sum(diag(test_confusion)) / sum(test_confusion)
# 


###########################################################################################################################################################################################

### K-NN Algorithm Implementation
#################################

get_knn_info <- function(model)
{
  
  
  predicted_values <- model
  test_confusion_matrix <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))
  test_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
  
  
  return(environment())
  
}


# Run k-nn algorithm with k = 1 first
train_labels <- train$class

start <- proc.time()
knn_model <- knn(train[,-ncol(train)] , test[,-ncol(test)] , train_labels)
end <- proc.time()

end - start


# Get k-nn test confusion matrix and accuracy performance
predicted_values <- knn_model
test_confusion_matrix <- table(predicted_values,test[,"class"],dnn=list('predicted','actual'))
test_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)

# Get ROC curve and AUC
knn_roc_curve <- multiclass.roc(match(test$class,LETTERS) , match(predicted_values,LETTERS))
as.numeric(auc(knn_roc_curve))


# Run k-nn algorithm with varying values of k
k_choices <- c(3,5,7,15,21)
knn_by_k_value <- lapply(k_choices , function(elt) knn(train[,-ncol(train)] , test[,-ncol(test)] , train_labels, k = elt))
knn_by_k_value_info <- lapply(knn_by_k_value , get_knn_info)

sapply(knn_by_k_value_info,function(elt) elt$test_performance)


knn_by_k_roc <- lapply(knn_by_k_value_info, function(info) multiclass.roc(match(test$class,LETTERS) , match(info$predicted_values,LETTERS)))
sapply(knn_by_k_roc,auc)




k_values <- c(5,11,15,21,25)
knn_models_by_k <- lapply(k_values , function(elt) knncat(train_income , classcol = 15 , k = elt, permute = 5, xvals = 3))


# Test a set of values for k to find the optimal choice
knn_model_find_optimal_k <- knncat(train_income , classcol = 15 , k = c(3,5,11,21,31,51), permute = 5, xvals = 3)



# knn_model_find_optimal_k <- knncat(train_income , classcol = 15 , k = 101, permute = 5, xvals = 3)

knn_model_find_optimal_k_info <- get_knn_info(knn_model_find_optimal_k)

# Get roc curve and AUC info for knn algorithm
roc_curve <- roc(ifelse(test_income$class == ">50K",1,0) , ifelse(decision_tree_info$predicted_values == ">50K",1,0))
auc(roc_curve)


sizes <- c(1000,5000,10000,15000,19000)



random_ordered_train_index <- sample(1:nrow(train) , max(sizes))

# Generate decision trees for increasingly larger subsets of the training data, starting with 1000 data points up to 19,000
all_knns <- lapply(sizes, function(size) knn(train[random_ordered_train_index[1:size],-ncol(train)] , test[,-ncol(test)] , train_labels[random_ordered_train_index[1:size]]))


knn_test_performances_by_size <- sapply(all_knns , function(elt) sum(diag(table(elt , test$class))) / sum(table(elt , test$class)))


plot(sizes , knn_test_performances_by_size , type="l", col = "red" , xlab = "Test Performance by Number of Training Data Points", ylab = "Performance")
title("Test Performance by Number of Training Data Points")

# Get Learning Curve Plot comparing train vs. test accuracy against number of training data points
get_train_test_plot(sizes , tree_train_performances_by_size , tree_test_performances_by_size, "Train / Test Performance by Number of Training Data Points", 
                                                                                              "Number of Training Data Points")




