
## Script to implement a collection of unsupervised learning algorithms on income-related data
############################################################################################################################################################################################

# devtools::install_github("kassambara/factoextra")

# Load packages
require(stats) # for k-means clustering and PCA
require(fpc) # for clustering metrics
require(mclust) # for EM clustering
require(factoextra) # for PCA plotting
require(ica) # independent components analysis
require(moments) # for calculating kurtosis for ICA
require(neuralnet) # neural nets
require(nnet) # neural nets
require(pROC) # roc curves and AUC metrics
require(randomProjection) # for Random Projection
require(FSelector) # provides information gain functionality for feature selection
require(plyr) # for utility functions

# switch to current working directory
setwd("C:/Users/Andrew/Documents/school stuff/Machine Learning/HW 3")

# define function to get sum of squared error between two matrices ~ use later for reconstruction error for PCA, RCA etc.
sse <- function(a,b) sum((a - (a + b) / 2)^2 + (b - (a + b) / 2)^2)

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
train_income <- droplevels(subset(train_income , native_country != "Holand-Netherlands"))


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

# define dataset without labels attached
train <- train_encoded[,-1]



# for(field in names(train_encoded))
# {
#   
#   if(is.numeric(train_encoded[[field]]))
#   {
#     
#     train_encoded[[field]] <- ( train_encoded[[field]] - min(train_encoded[[field]]) ) / ( max(train_encoded[[field]]) - min(train_encoded[[field]]) )
#     
#     test_encoded[[field]] <- ( test_encoded[[field]] - min(test_encoded[[field]]) ) / ( max(test_encoded[[field]]) - min(test_encoded[[field]]) )
#     
#   }
#   
# }






##########################################################################################################################################################################################
##########################################################################################################################################################################################
########## Implement K-means Clustering ##################################################################################################################################################
############################################################################################################################################################################################

start <- proc.time()
kmeans_model <- kmeans(train_encoded[,-1],5)
end <- proc.time() - start


nrow(train_income)

# create field in train dataset for cluster labels
train_encoded$kmeans_cluster <- kmeans_model$cluster

# Get distribution of data points by cluster
percent_pos <- with(train_encoded , tapply(class, kmeans_cluster, sum)) / with(train_encoded , tapply(kmeans_cluster, kmeans_cluster, length))
percent_neg <- 1 - percent_pos


ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}


get_cluster_distribution <- function(model)
{
  
    train_encoded$kmeans_cluster <- model$cluster
    
    # Get distribution of data points by cluster
    cluster_sizes <- with(train_encoded , tapply(kmeans_cluster, kmeans_cluster, length))
    percent_pos <- with(train_encoded , tapply(class, kmeans_cluster, sum)) / cluster_sizes
    percent_neg <- 1 - percent_pos
    
    total_percent_pos <- with(train_encoded , tapply(class, kmeans_cluster, sum)) / sum(train_encoded$class)
    total_percent_neg = (with(train_encoded , tapply(class, kmeans_cluster, length)) - with(train_encoded , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(train_encoded , class == 0))

    return(environment())
    
}




# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)

kmeans_by_k <- lapply(k_values , function(k) kmeans(train_encoded[,-1] , k))

sapply(kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(kmeans_by_k , function(x) get_cluster_distribution(x,train_encoded))
kmeans_by_k_dist <- dist_by_k

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)

# test kmeans by different algorithms
algorithm_list <- c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen")

kmeans_by_algoritm <- lapply(algorithm_list , function(x)  kmeans(train_encoded[,-1] , 3 , algorithm = x))

dist_by_algorithm <-  lapply(kmeans_by_algoritm , get_cluster_distribution)

# get proportion of postive labels by cluster and size of cluster for the different algorithms
lapply(dist_by_algorithm , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_algorithm , function(x) x$total_percent_pos)

lapply(dist_by_algorithm , function(x) x$total_percent_neg)


#################

## Examine the contents of the clusters

train_income$cluster <- kmeans_by_k[[1]]$cluster
train_encoded$cluster <- kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * kmeans_by_k_dist[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * kmeans_by_k_dist[[1]]$percent_neg,2),"%")
                     ,kmeans_by_k_dist[[1]]$cluster_sizes
                     ,paste0(round(100 * kmeans_by_k_dist[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * kmeans_by_k_dist[[1]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

train_income$cluster <- kmeans_by_k[[2]]$cluster
train_encoded$cluster <- kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * kmeans_by_k_dist[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * kmeans_by_k_dist[[2]]$percent_neg,2),"%")
                     ,kmeans_by_k_dist[[2]]$cluster_sizes
                     ,paste0(round(100 * kmeans_by_k_dist[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * kmeans_by_k_dist[[2]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)



train_encoded$cluster <- train_income$cluster <- NULL



# 
# train_income$cluster <- kmeans_by_k[[2]]$cluster
# train_encoded$cluster <- kmeans_by_k[[2]]$cluster
# 
# tapply(train_income$age , train_income$cluster, mean)
# tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)
# 
# 
# tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)
# 
# tapply(train_income$capital_gain , train_income$cluster, mean)
# 
# 
# tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)
# 
# train_income$cluster <- train_encoded$cluster <- NULL

#######################################################

# # test by differing number of iterations
# iter_values <- c(10 , 30, 50, 75, 100)
# 
# kmeans_by_iter <- lapply(iter_values , function(iter) kmeans(train_encoded[,-1] , 20 ,iter.max = iter))
# 
# 
# 
# # test by differing number of iterations
# num_sets <- c(1 , 5, 10, 15, 20)
# 
# kmeans_by_sets <- lapply(num_sets , function(n) kmeans(train_encoded[,-1] , 20 ,nstart =n ))1
# 
# 



# Determine number of clusters by sums of squares
wss <- (nrow(train_encoded[,-1])-1)*sum(apply(train_encoded[,-1],2,var))
for (i in 2:15) wss[i] <- sum(kmeans(train_encoded[,-1], 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")





##################################################################################################################################################################################################################
##### Expectation Maximiation Clustering ##################################################################################################################################################
###########################################

train <- train_encoded[,-1]

# start <- proc.time()
# em_model <- Mclust(train_encoded[,-1] , initialization=list(subset=sample(1:nrow(train_encoded[,-1]), size=16125)))
# end <- proc.time() - start


train_dataset <- train_encoded[,intersect(names(train) , names(train_income))]
start <- proc.time()
# exp_model <- Mclust(train_dataset , initialization=list(subset=sample(1:nrow(train_dataset), size=16125)))
starter_exp_model <- Mclust(train_dataset , initialization=list(subset=sample(1:nrow(train_dataset), size=16125)))
end <- proc.time() - start

# add cluster assignements based of which cluster has the highest probability amoung the choices
train_encoded$cluster <- apply(exp_model$z , 1 , which.max)
train_income$cluster <- train_encoded$cluster

cluster_sizes <- with(train_encoded , tapply(cluster, cluster, length))
percent_pos <- with(train_encoded , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- with(train_encoded , tapply(class, cluster, sum)) / sum(train_encoded$class)
total_percent_neg = (with(train_encoded , tapply(class, cluster, length)) - with(train_encoded , tapply(class, cluster, sum)) ) / nrow(subset(train_encoded , class == 0))


# rbind(percent_pos , cluster_sizes)
# exp_model$BIC



## Examine the contents of the clusters

result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)




# 
# 
# train_income$cluster <- kmeans_by_k[[2]]$cluster
# train_encoded$cluster <- kmeans_by_k[[2]]$cluster
# 
# tapply(train_income$age , train_income$cluster, mean)
# tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)
# 
# 
# tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)
# 
# tapply(train_income$capital_gain , train_income$cluster, mean)
# 
# 
# tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)
# 
# train_income$cluster <- train_encoded$cluster <- NULL


#******************************************************************************************************************************************************************************************
#******************************************************************************************************************************************************************************************


##################################################################################################################################################################################################################
##### Principle Components Analysis ##################################################################################################################################################
###########################################


# Implement Pricipal Components Analysis
train <- train_encoded[,-1]
start <- proc.time()
fit <- prcomp(train, cor=TRUE)
end <- proc.time() - start
summary(fit) # print variance accounted for 

# get eigenvalues of the components
eigenvalues <- fit$sdev**2
summary(eigenvalues)
sd(eigenvalues)

length(which(eigenvalues < 10^(-6)))


# generate scree plot based on variances
fviz_screeplot(fit, ncp=40)

# generate scree plot based on eigenvalues
fviz_screeplot(fit, ncp=10,choice = "eigenvalue")


# Examine what variables are most important to top components
sort(fit$rotation[,1],decreasing = TRUE)[1:5]
sort(fit$rotation[,2])
sort(fit$rotation[,3])



####
var <- get_pca_var(fit)
fviz_pca_var(fit)



# cumulative variances

# Eigenvalues
eig <- (fit$sdev)^2

# Variances in percentage
variance <- eig*100/sum(eig)

# Cumulative variances
cumvar <- cumsum(variance)

#############################################################################################################################################

fviz_pca_var(fit, col.var="contrib")+
  scale_color_gradient2(low="white", mid="blue", 
                        high="red", midpoint=5) + theme_minimal()


#############################################################################################################################################

biplot(fit, cex = 0.8, col = c("black", "red") )



# loadings(fit) # pc loadings 
# plot(fit,type="lines") # scree plot 
# fit$scores # the principal components
# biplot(fit)



## Revisit neural network models using PCA
###########################################################################################################################################################################################

## Function to get confusion matrix and prediction results for neural network models
get_net_model_info <- function(model , train_set , test_set)
{
  
  net_results <- compute(model,test_set[,-1])
  
  results <- data.frame(actual = test_set$class, prediction = net_results$net.result)
  results$prediction <- round(results$prediction)
  
  predicted_values <- results$prediction
  
  test_confusion_matrix <- table(predicted_values,test_set[,"class"],dnn=list('predicted','actual'))
  
  test_income_performance <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
  
  net_train_results <- compute(model,train_set[,-1])
  train_results <- data.frame(actual = train_set$class, prediction = net_train_results$net.result)
  train_results$prediction <- round(train_results$prediction)
  
  train_predicted_values <- train_results$prediction
  
  train_confusion_matrix <- table(train_predicted_values,train_set[,"class"],dnn=list('predicted','actual'))
  
  
  train_income_performance <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)
  
  
  return(environment())
  
  
}

f <- function(num_components)
{
  
  train_pca <- data.frame(cbind(class = train_encoded$class[1:20000], fit$x[1:20000,1:num_components]))
  test_pca <- data.frame(cbind(class = train_encoded$class[-c(1:20000)], fit$x[-c(1:20000),1:num_components]))
  feature_names <- colnames(fit$x)
  model_formula <- as.formula(paste("class ~",paste(feature_names,collapse = "+")))
  
  return(environment())
  
}

get_precision <- function(x)
  
{
  temp <- x$test_set
  temp$predicted_values <- x$predicted_values
  
  # Get precision
  ( net_precision <- nrow(subset(temp , (class == 1) & (class == predicted_values) )) / nrow(subset(temp, predicted_values == 1)) )
  
  net_precision
}

get_metrics <- function(info)
{
  
  f <- info$test_set
  f$predicted_values <- info$predicted_values
  
  sens <- nrow(subset(f , (class == predicted_values) & (class == 1) )) / nrow(subset(f , (class == 1)))
  spec <- nrow(subset(f , (class == predicted_values) & (class == 0) )) / nrow(subset(f , (class == 0)))
  
  return(environment())
  
}


## Implement neural networks using components from PCA
##########################################################################################################################################################################################

train_encoded$em_cluster <- NULL
# Get feature names and create a formula object based off the features--to be used by the various algorithms
train_pca <- data.frame(cbind(class = train_encoded$class[1:20000], fit$x[1:20000,]))
test_pca <- data.frame(cbind(class = train_encoded$class[-c(1:20000)], fit$x[-c(1:20000),]))
feature_names <- colnames(fit$x)
model_formula <- as.formula(paste("class ~",paste(feature_names,collapse = "+")))


net_start <- proc.time()
net_model <- neuralnet(formula(train_pca) , data = train_pca)
net_runtime <- proc.time() - net_start



net_start <- proc.time()
control_net_model <- neuralnet(formula(train_encoded) , data = train_encoded)
net_runtime <- proc.time() - net_start


m <- get_net_model_info(control_net_model,train_set = train_encoded[1:20000,], train_encoded[-c(1:20000),])
m$test_income_performance
m$train_income_performance
auc(roc(train_encoded[-c(1:20000),]$class , m$predicted_values))
get_precision(m)
get_metrics(m)$spec
get_metrics(m)$sens



# run neural network backpropagation algorithm for varying number of components
nc <- c(2,3,5,10,30,50)
nets_by_num_components <- lapply(nc , function(elt) neuralnet(formula(f(elt)$train_pca) , data = f(elt)$train_pca) )


net_times_by_num_components <- lapply(nc , function(elt) system.time(neuralnet(formula(f(elt)$train_pca) , data = f(elt)$train_pca) ))

net_components_results <- Map(function(model , N) get_net_model_info(model , train_set = f(N)$train_pca , test_set = f(N)$test_pca) , nets_by_num_components , nc)

# get test and train performance
sapply(net_components_results , function(x) x$test_income_performance)
sapply(net_components_results , function(x) x$train_income_performance)


sapply(net_components_results , function(x) auc(roc(x$test_set$class , x$predicted_values)))

net_roc <- roc(test_encoded$class , net_info$predicted_values)
auc(net_roc)



# get precision by number of components
sapply(net_components_results , get_precision)


# get specificity and sensitivity
sapply(net_components_results , function(x) get_metrics(x)$spec)
sapply(net_components_results , function(x) get_metrics(x)$sens)



##########################################################################################################################################################################################
##########################################################################################################################################################################################


## Implement k-means clustering on PCA components
#################################################

get_cluster_dist <- function(model , train_set)
{
  
  train_set$kmeans_cluster <- model$cluster
  
  # Get distribution of data points by cluster
  cluster_sizes <- with(train_set , tapply(kmeans_cluster, kmeans_cluster, length))
  percent_pos <- with(train_set , tapply(class, kmeans_cluster, sum)) / cluster_sizes
  percent_neg <- 1 - percent_pos
  
  total_percent_pos <- with(train_set , tapply(class, kmeans_cluster, sum)) / sum(train_set$class)
  total_percent_neg = (with(train_set , tapply(class, kmeans_cluster, length)) - with(train_set , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(train_set , class == 0))
  
  return(environment())
  
}

train_pca <- data.frame(class = train_encoded$class , fit$x)

# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)

# run k-means on dataset with first 50 principle components
pca_kmeans_by_k <- lapply(k_values , function(k) kmeans(train_pca[,2:51] , k))

pca_kmeans_by_k_dist <- lapply(pca_kmeans_by_k,function(x) get_cluster_dist(x,train_pca))


# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(pca_kmeans_by_k_dist , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(pca_kmeans_by_k_dist , function(x) x$total_percent_pos)

lapply(pca_kmeans_by_k_dist , function(x) x$total_percent_neg)


start <- proc.time()
m <- kmeans(train_pca[,-1],3)
end <- proc.time() - start



## Examine the contents of the clusters
train_income$cluster <- pca_kmeans_by_k[[1]]$cluster
train_encoded$cluster <- pca_kmeans_by_k[[1]]$cluster


# Cluster	% Positive	% Negative	Cluster Size	% TP	% TN	Mean Age	% Civ. Spouse	% Husband	Cap. Gain	% M
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[1]]$percent_neg,2),"%")
                     ,pca_kmeans_by_k_dist[[1]]$cluster_sizes
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[1]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     )

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

train_income$cluster <- pca_kmeans_by_k[[2]]$cluster
train_encoded$cluster <- pca_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[2]]$percent_neg,2),"%")
                     ,pca_kmeans_by_k_dist[[2]]$cluster_sizes
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * pca_kmeans_by_k_dist[[2]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)










tapply(train_income$age , train_income$cluster, mean)
tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)


tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)

tapply(train_income$capital_gain , train_income$cluster, mean)


tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length)

train_income$cluster <- train_encoded$cluster <- NULL





## EM clustering on PCA-reduced data
####################################

pca_dataset <- data.frame(class = train_encoded$class , fit$x)

# run EM clustering on dataset with first five PCA components
pca_dataset <- pca_dataset[,1:6]
start <- proc.time()
pca_exp_model <- Mclust(pca_dataset[,-1] , initialization=list(subset=sample(1:nrow(pca_dataset), size=16125)))
end <- proc.time() - start

# add cluster assignements based of which cluster has the highest probability amoung the choices
pca_dataset$cluster <- apply(pca_exp_model$z , 1 , which.max)
pca_dataset$class <- train_encoded$class

cluster_sizes <- with(pca_dataset , tapply(cluster, cluster, length))
percent_pos <- with(pca_dataset , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- with(pca_dataset , tapply(class, cluster, sum)) / sum(pca_dataset$class)
total_percent_neg = (with(pca_dataset , tapply(class, cluster, length)) - with(pca_dataset , tapply(class, cluster, sum)) ) / nrow(subset(pca_dataset , class == 0))


# rbind(percent_pos , cluster_sizes)


train_encoded$cluster <- train_income$cluster <- pca_dataset$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

pca_dataset$class = NULL



x  <- pca_dataset
x$cluster_class <- ifelse(x$em_cluster %in% c(3,6) , 1, 0)

# get accuracy
nrow(subset(x,class == cluster_class)) / nrow(x)

# get recall
nrow(subset(x,class == cluster_class & class == 1)) / nrow(subset(x,class == 1))

# get sensitivity
nrow(subset(x,class == cluster_class & class == 0)) / nrow(subset(x,class == 0))






##################################################################################################################################################################################################################
##### Independent Components Analysis ##################################################################################################################################################
###########################################

start <- proc.time()
ica_model <- icafast(train , 3)
end <- proc.time() - start

num_components <- c(3,5,10,15,20,30,50,ncol(train))
ica_by_num_components <- lapply(num_components , function(elt) icafast(train , elt))


lapply(ica_by_num_components , function(x) x$vafs)

ica_dataset <- data.frame(class = train_encoded$class , ica_by_num_components[[7]]$S)


x <- icafast(train , 20)

icafast(X,nc,center=TRUE,maxit=100,tol=1e-6,Rmat=diag(nc),
        alg=c("par","def"),fun=c("logcosh","exp","kur"),alpha=1)



ica_all_vars <- icafast(train , ncol(train))
names(ica_all_vars$vafs) <- 1:length(ica_all_vars$vafs)
barplot(ica_all_vars$vafs[1:20] , col="darkblue", main="Variance Explained by Components in ICA")
lines(ica_all_vars$vafs[1:20],col="red",type="o")


ica_by_algorithm <- lapply(c("logcosh","exp","kur"), function(alg) icafast(train , 40, alg = alg))

# look at kurtosis of independent components
all_kurtosis <- lapply(ica_by_num_components , function(x) kurtosis(x$S))
lapply(all_kurtosis,summary)

control_k <- sort(kurtosis(train))
summary(control_k)

k <- sort(kurtosis(ica_all_vars$S))
summary(k)

Reduce(rbind,lapply(all_kurtosis,summary))


######################################################################################################################################################################################################
######################################################################################################################################################################################################

### Kmeans clustering on ICA-reduced dataset
############################################



# 
# nrow(train_income)
# 
# # create field in train dataset for cluster labels
# train_encoded$kmeans_cluster <- kmeans_model$cluster
# 
# # Get distribution of data points by cluster
# percent_pos <- with(train_encoded , tapply(class, kmeans_cluster, sum)) / with(train_encoded , tapply(kmeans_cluster, kmeans_cluster, length))
# percent_neg <- 1 - percent_pos


ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}


get_cluster_distribution <- function(model)
{
  
  ica_dataset$kmeans_cluster <- model$cluster
  
  # Get distribution of data points by cluster
  cluster_sizes <- with(ica_dataset , tapply(kmeans_cluster, kmeans_cluster, length))
  percent_pos <- with(ica_dataset , tapply(class, kmeans_cluster, sum)) / cluster_sizes
  percent_neg <- 1 - percent_pos
  
  total_percent_pos <- with(ica_dataset , tapply(class, kmeans_cluster, sum)) / sum(ica_dataset$class)
  total_percent_neg = (with(ica_dataset , tapply(class, kmeans_cluster, length)) - with(ica_dataset , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(ica_dataset , class == 0))
  
  return(environment())
  
}




# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)

ica_means_by_k <- lapply(k_values , function(k) kmeans(ica_dataset[,-1] , k))

sapply(ica_kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(ica_kmeans_by_k , function(x) get_cluster_distribution(x,ica_dataset))
ica_means_by_k_dist <- dist_by_k

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)


## Examine the contents of the clusters

train_income$cluster <- ica_means_by_k[[1]]$cluster
train_encoded$cluster <- ica_means_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * ica_means_by_k_dist[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * ica_means_by_k_dist[[1]]$percent_neg,2),"%")
                     ,ica_means_by_k_dist[[1]]$cluster_sizes
                     ,paste0(round(100 * ica_means_by_k_dist[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * ica_means_by_k_dist[[1]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

train_income$cluster <- ica_means_by_k[[2]]$cluster
train_encoded$cluster <- ica_means_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * ica_means_by_k_dist[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * ica_means_by_k_dist[[2]]$percent_neg,2),"%")
                     ,ica_means_by_k_dist[[2]]$cluster_sizes
                     ,paste0(round(100 * ica_means_by_k_dist[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * ica_means_by_k_dist[[2]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)



start <- proc.time()
kmeans_model <- kmeans(ica_dataset_5[,-1],5)
end <- proc.time() - start




# # test kmeans by different algorithms
# algorithm_list <- c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen")
# 
# kmeans_by_algoritm <- lapply(algorithm_list , function(x)  kmeans(train_encoded[,-1] , 3 , algorithm = x))
# 
# dist_by_algorithm <-  lapply(kmeans_by_algoritm , get_cluster_distribution)
# 
# # get proportion of postive labels by cluster and size of cluster for the different algorithms
# lapply(dist_by_algorithm , function(x) rbind(x$percent_pos , x$cluster_sizes))
# 
# 
# lapply(dist_by_algorithm , function(x) x$total_percent_pos)
# 
# lapply(dist_by_algorithm , function(x) x$total_percent_neg)

# 
# # test by differing number of iterations
# iter_values <- c(10 , 30, 50, 75, 100)
# 
# kmeans_by_iter <- lapply(iter_values , function(iter) kmeans(train_encoded[,-1] , 20 ,iter.max = iter))
# 
# 
# 
# # test by differing number of iterations
# num_sets <- c(1 , 5, 10, 15, 20)
# 
# kmeans_by_sets <- lapply(num_sets , function(n) kmeans(train_encoded[,-1] , 20 ,nstart =n ))1
# 
# 
# 
# 
# 
# # Determine number of clusters
# wss <- (nrow(train_encoded[,-1])-1)*sum(apply(train_encoded[,-1],2,var))
# for (i in 2:15) wss[i] <- sum(kmeans(train_encoded[,-1], 
#                                      centers=i)$withinss)
# plot(1:15, wss, type="b", xlab="Number of Clusters",
#      ylab="Within groups sum of squares")




## EM clustering on ICA-reduced data
####################################

# # run EM clustering on dataset with first five PCA components
ica_dataset_5 <- ica_dataset[,1:6]
start <- proc.time()
ica_exp_model <- Mclust(ica_dataset_5[,-1] , initialization=list(subset=sample(1:nrow(ica_dataset_5), size=16125)))
end <- proc.time() - start



# add cluster assignements based of which cluster has the highest probability amoung the choices
ica_dataset_5$cluster <- apply(ica_exp_model$z , 1 , which.max)
ica_dataset_5$class <- train_encoded$class

cluster_sizes <- with(ica_dataset_5 , tapply(cluster, cluster, length))
percent_pos <- with(ica_dataset_5 , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- (with(ica_dataset_5 , tapply(class, cluster, sum)) / sum(ica_dataset_5$class))
total_percent_neg = (with(ica_dataset_5 , tapply(class, cluster, length)) - with(ica_dataset_5 , tapply(class, cluster, sum)) ) / nrow(subset(ica_dataset_5 , class == 0))


# rbind(percent_pos , cluster_sizes)

train_encoded$cluster <- train_income$cluster <- ica_dataset_5$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

ica_dataset_5$class <- NULL


x  <- ica_dataset_5
x$cluster_class <- ifelse(x$em_cluster %in% c(2) , 1, 0)


# get accuracy
nrow(subset(x,class == cluster_class)) / nrow(x)

# get recall
nrow(subset(x,class == cluster_class & class == 1)) / nrow(subset(x,class == 1))

# get sensitivity
nrow(subset(x,class == cluster_class & class == 0)) / nrow(subset(x,class == 0))



ica_dataset_5$em_cluster <- NULL




#******************************************************************************************************************************************************************************************

## Implement neural networks using components from ICA
##########################################################################################################################################################################################

train_encoded$em_cluster <- NULL
ica_data <- data.frame(ica_all_vars$S)

# Get feature names and create a formula object based off the features--to be used by the various algorithms
train_ica <- data.frame(cbind(class = train_encoded$class[1:20000], ica_data[1:20000,]))
test_ica <- data.frame(cbind(class = train_encoded$class[-c(1:20000)], ica_data[-c(1:20000),]))
feature_names <- names(ica_data)
model_formula <- as.formula(paste("class ~",paste(feature_names,collapse = "+")))


f <- function(num_components)
{
  
  train_ica <- data.frame(cbind(class = train_encoded$class[1:20000], ica_data[1:20000,1:num_components]))
  test_ica <- data.frame(cbind(class = train_encoded$class[-c(1:20000)], ica_data[-c(1:20000),1:num_components]))
  feature_names <- names(ica_data)
  model_formula <- as.formula(paste("class ~",paste(feature_names,collapse = "+")))
  
  return(environment())
  
}


net_start <- proc.time()
net_model <- neuralnet(formula(train_ica) , data = train_ica)
net_runtime <- proc.time() - net_start


# run neural network backpropagation algorithm for varying number of components
nc <- c(2,3,5,10,30,50,99)
nets_by_num_components <- lapply(nc , function(elt) neuralnet(formula(f(elt)$train_ica) , data = f(elt)$train_ica) )


net_times_by_num_components <- lapply(nc , function(elt) system.time(neuralnet(formula(f(elt)$train_ica) , data = f(elt)$train_ica) ))

net_components_results <- Map(function(model , N) get_net_model_info(model , train_set = f(N)$train_ica , test_set = f(N)$test_ica) , nets_by_num_components , nc)

# get test and train performance
sapply(net_components_results , function(x) x$test_income_performance)
sapply(net_components_results , function(x) x$train_income_performance)


sapply(net_components_results , function(x) auc(roc(x$test_set$class , x$predicted_values)))

# net_roc <- roc(test_encoded$class , net_info$predicted_values)
# auc(net_roc)



# get precision by number of components
sapply(net_components_results , get_precision)


# get specificity and sensitivity
sapply(net_components_results , function(x) get_metrics(x)$spec)
sapply(net_components_results , function(x) get_metrics(x)$sens)




#******************************************************************************************************************************************************************************************

##################################################################################################################################################################################################################
##### Random Projection ##################################################################################################################################################
###########################################

# https://github.com/chappers/randomProjection


start <- proc.time()
rp_model = RandomProjection(train, n_features=50, eps=0.1)
end <- proc.time() - start

# rca_dataset <- data.frame(class = train_encoded$class , rp_model$RP)


# Implement random projection matrices by various number of components
rp_models_by_nc <- lapply(num_components , function(elt) RandomProjection(train, n_features=elt, eps=0.1))
# rp_model <- rp_models_by_nc[[7]]
rca_dataset <- data.frame(class = train_encoded$class , rp_model$RP)


all_rp_models <- lapply(1:100 , function(index) RandomProjection(train, n_features=50, eps=0.1))


get_recon_error <- function(N,rp_model)
{
  # reconstruct matrix
  restr <- rp_model$RP[,1:N] %*% t(rp_model$R[,1:N])
  
  sse(train_encoded,restr)
  
}

sapply(seq(10,100,10) , function(x) get_recon_error(x,rp_models_by_nc[[8]]))


all_rp_errors <- sapply(all_rp_models , function(x) get_recon_error(50,x))
summary(all_rp_errors)









get_cluster_distribution <- function(model,dataset)
{
  
  dataset$kmeans_cluster <- model$cluster
  
  # Get distribution of data points by cluster
  cluster_sizes <- with(dataset , tapply(kmeans_cluster, kmeans_cluster, length))
  percent_pos <- with(dataset , tapply(class, kmeans_cluster, sum)) / cluster_sizes
  percent_neg <- 1 - percent_pos
  
  total_percent_pos <- with(dataset , tapply(class, kmeans_cluster, sum)) / sum(dataset$class)
  total_percent_neg = (with(dataset , tapply(class, kmeans_cluster, length)) - with(dataset , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(dataset , class == 0))
  
  return(environment())
  
}


start <- proc.time()
kmeans(rca_dataset[,-1] , 3)
end <- proc.time() - start

# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)

rca_kmeans_by_k <- lapply(k_values , function(k) kmeans(rca_dataset[,-1] , k))

sapply(rca_kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(kmeans_by_k , function(x) get_cluster_distribution(x,rca_dataset))
rca_kmeans_by_k_dist <- dist_by_k

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)


## Examine the contents of the clusters

train_income$cluster <- rca_kmeans_by_k[[1]]$cluster
train_encoded$cluster <- rca_kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[1]]$percent_neg,2),"%")
                     ,rca_kmeans_by_k_dist[[1]]$cluster_sizes
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[1]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

train_income$cluster <- rca_kmeans_by_k[[2]]$cluster
train_encoded$cluster <- rca_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[2]]$percent_neg,2),"%")
                     ,rca_kmeans_by_k_dist[[2]]$cluster_sizes
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * rca_kmeans_by_k_dist[[2]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)









# Determine number of clusters
wss <- (nrow(train_encoded[,-1])-1)*sum(apply(train_encoded[,-1],2,var))
for (i in 2:15) wss[i] <- sum(kmeans(train_encoded[,-1], 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")


###########################################################################################################################################################################################
######### EM Clustering on RCA-reduced data ##################################################################################################################################################
###########################################################################################################################################################################################

# # run EM clustering on dataset with first five PCA components
rca_dataset_5 <- rca_dataset[,1:6]
start <- proc.time()
exp_model <- Mclust(rca_dataset_5[,-1] , initialization=list(subset=sample(1:nrow(rca_dataset_5), size=16125)))
end <- proc.time() - start

rca_exp_model <- exp_model

# add cluster assignements based of which cluster has the highest probability amoung the choices
rca_dataset_5$cluster <- apply(rca_exp_model$z , 1 , which.max)
rca_dataset_5$class <- train_encoded$class

cluster_sizes <- with(rca_dataset_5 , tapply(cluster, cluster, length))
percent_pos <- with(rca_dataset_5 , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- (with(rca_dataset_5 , tapply(class, cluster, sum)) / sum(rca_dataset_5$class))
total_percent_neg = (with(rca_dataset_5 , tapply(class, cluster, length)) - with(rca_dataset_5 , tapply(class, cluster, sum)) ) / nrow(subset(rca_dataset_5 , class == 0))

train_encoded$cluster <- train_income$cluster <- rca_dataset_5$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)


train_encoded$cluster <- train_income$cluster <- NULL
rca_dataset_5$class <- NULL






# rbind(percent_pos , cluster_sizes)

x  <- rca_dataset_5
x$cluster_class <- ifelse(x$em_cluster %in% c(2) , 1, 0)

# get accuracy
nrow(subset(x,class == cluster_class)) / nrow(x)

# get recall
nrow(subset(x,class == cluster_class & class == 1)) / nrow(subset(x,class == 1))

# get sensitivity
nrow(subset(x,class == cluster_class & class == 0)) / nrow(subset(x,class == 0))



rca_dataset_5$em_cluster <- NULL




#******************************************************************************************************************************************************************************************
##########################################################################################################################################################################################
## Implement neural networks using components from RCA
##########################################################################################################################################################################################

train_encoded$em_cluster <- NULL
rca_data <- data.frame(rp_models_by_nc[[8]]$RP)

# Get feature names and create a formula object based off the features--to be used by the various algorithms
# use defined function to generate train and test sets using components from ICA
f <- function(num_components)
{
  
  train_rca <- data.frame(cbind(class = train_encoded$class[1:20000], rca_data[1:20000,1:num_components]))
  test_rca <- data.frame(cbind(class = train_encoded$class[-c(1:20000)], rca_data[-c(1:20000),1:num_components]))
  feature_names <- names(rca_data)
  model_formula <- as.formula(paste("class ~",paste(feature_names,collapse = "+")))
  
  return(environment())
  
}

# Implement neural network model using RCA components
net_start <- proc.time()
net_model <- neuralnet(formula(f(50)$train_rca) , data = f(50)$train_rca)
net_runtime <- proc.time() - net_start


# run neural network backpropagation algorithm for varying number of components
nc <- c(2,3,5,10,30,50,107)
nets_by_num_components <- lapply(nc , function(elt) neuralnet(formula(f(elt)$train_rca) , data = f(elt)$train_rca) )

# get computational speed of each neural net run varying by number of components used as input
net_times_by_num_components <- lapply(nc , function(elt) system.time(neuralnet(formula(f(elt)$train_rca) , data = f(elt)$train_rca) ))

net_components_results <- Map(function(model , N) try(get_net_model_info(model , train_set = f(N)$train_rca , test_set = f(N)$test_rca),silent = TRUE) , nets_by_num_components , nc)

net_components_results <- net_components_results[-6]

# get test and train performance
sapply(net_components_results , function(x) x$test_income_performance)
sapply(net_components_results , function(x) x$train_income_performance)


sapply(net_components_results , function(x) auc(roc(x$test_set$class , x$predicted_values)))

# get precision by number of components
sapply(net_components_results , get_precision)

# get specificity and sensitivity
sapply(net_components_results , function(x) get_metrics(x)$spec)
sapply(net_components_results , function(x) get_metrics(x)$sens)




#******************************************************************************************************************************************************************************************
##########################################################################################################################################################################################
## Implement Feature Selection via Information Gain
##########################################################################################################################################################################################

start <- proc.time()
info_gain <- information.gain(formula(train_encoded) , train_encoded)
end <- proc.time() - start


info_gain <- data.frame(var = rownames(info_gain) , info_gain)
rownames(info_gain) = NULL

info_gain <- arrange(info_gain , info_gain$attr_importance, decreasing = TRUE)

info_gain[1:10,]


train_copy <- train_income
train_copy$class <- NULL
train_copy <- data.frame(class=train_income$class , train_copy)

reduced_dataset_info_gain <- information.gain(formula(train_copy) , train_copy)
reduced_dataset_info_gain <- data.frame(var = rownames(reduced_dataset_info_gain) , reduced_dataset_info_gain)
rownames(reduced_dataset_info_gain) <- NULL

reduced_dataset_info_gain <- arrange(reduced_dataset_info_gain , reduced_dataset_info_gain$attr_importance, decreasing = TRUE)

pos_gain_data <- train_encoded[,as.character(subset(info_gain , info_gain$attr_importance > 0)$var)]



##########################################################################################################################################################################################
######## Kmeans clustering for information gain -- selected attributes  #######################################################################################################################
##########################################################################################################################################################################################


get_cluster_distribution <- function(model,dataset)
{
  
  dataset$kmeans_cluster <- model$cluster
  
  # Get distribution of data points by cluster
  cluster_sizes <- with(dataset , tapply(kmeans_cluster, kmeans_cluster, length))
  percent_pos <- with(dataset , tapply(class, kmeans_cluster, sum)) / cluster_sizes
  percent_neg <- 1 - percent_pos
  
  total_percent_pos <- with(dataset , tapply(class, kmeans_cluster, sum)) / sum(dataset$class)
  total_percent_neg = (with(dataset , tapply(class, kmeans_cluster, length)) - with(dataset , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(dataset , class == 0))
  
  return(environment())
  
}


# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)


info_gain_kmeans_by_k <- lapply(k_values , function(k) kmeans(pos_gain_data , k))
pos_gain_data$class <- train_encoded$class

sapply(kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(info_gain_kmeans_by_k, function(x) get_cluster_distribution(x,pos_gain_data))
info_gain_kmeans_by_k_dist <- dist_by_k


# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)




## Examine the contents of the clusters

train_income$cluster <- info_gain_kmeans_by_k[[1]]$cluster
train_encoded$cluster <- info_gain_kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[1]]$percent_neg,2),"%")
                     ,info_gain_kmeans_by_k_dist[[1]]$cluster_sizes
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[1]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

train_income$cluster <- info_gain_kmeans_by_k[[2]]$cluster
train_encoded$cluster <- info_gain_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(train_income$cluster)) 
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[2]]$percent_neg,2),"%")
                     ,info_gain_kmeans_by_k_dist[[2]]$cluster_sizes
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * info_gain_kmeans_by_k_dist[[2]]$total_percent_neg,2),"%")
                     ,round(tapply(train_income$age , train_income$cluster, mean))
                     
                     ,paste0(round(100 * tapply(train_encoded$marital_statusMarried.civ.spouse , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length),2),"%")
                     ,paste0(round(100 * tapply(train_encoded$relationshipHusband , train_income$cluster, sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
                     ,round(tapply(train_income$capital_gain , train_income$cluster, mean))
                     ,paste0(round(100 * tapply(train_encoded$sexMale,train_encoded$cluster,sum) / tapply(train_encoded$cluster,train_encoded$cluster,length) , 2),"%")
)

names(result) <- LETTERS[1:11]
write.table(result , "clipboard", sep="\t",row.names = FALSE)









pos_gain_data$class <- NULL
# Determine number of clusters
wss <- (nrow(pos_gain_data)-1)*sum(apply(pos_gain_data,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(pos_gain_data, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")



#########################################################################################################################################################################################
############### EM Clustering on Information Gain-Reduced Data ######################################################################################################################################################################
#########################################################################################################################################################################################

pos_gain_data$em_cluster <- NULL
pos_gain_data$class <- NULL

start <- proc.time()
# em_model <- Mclust(pos_gain_data)
exp_model <- Mclust(pos_gain_data , initialization=list(subset=sample(1:nrow(pos_gain_data), size=10000)))
end <- proc.time() - start

# add cluster assignements based of which cluster has the highest probability amoung the choices
pos_gain_data$em_cluster <- apply(exp_model$z , 1 , which.max)


cluster_sizes <- with(pos_gain_data , tapply(em_cluster, em_cluster, length))
percent_pos <- with(pos_gain_data , tapply(class, em_cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- with(pos_gain_data , tapply(class, em_cluster, sum)) / sum(pos_gain_data$class)
total_percent_neg = (with(pos_gain_data , tapply(class, em_cluster, length)) - with(pos_gain_data , tapply(class, em_cluster, sum)) ) / nrow(subset(pos_gain_data , class == 0))


rbind(percent_pos , cluster_sizes)



exp_model$BIC





#******************************************************************************************************************************************************************************************
##########################################################################################################################################################################################
## Implement neural networks Information-Gained Reduced Dataset
##########################################################################################################################################################################################


pos_gain_data$em_cluster <- NULL

pos_gain_data$class <- train_encoded$class # add class back in for training / testing
info_train <- pos_gain_data[1:20000,]
info_test <- pos_gain_data[-c(1:20000),]

# Implement neural network model using RCA components
net_start <- proc.time()
net_model <- neuralnet(formula(info_train) , data = info_train)
net_runtime <- proc.time() - net_start

net_model_3 <- neuralnet(formula(info_train) , data = info_train, hidden = 3)
get_net_model_info(net_model_3 , train_set = info_train , test_set = info_test)$test_income_performance


# get various metrics from result of neural net model
net_result <- get_net_model_info(net_model , train_set = info_train , test_set = info_test)
net_result$test_income_performance
net_result$train_income_performance
get_precision(net_result)
get_metrics(net_result)$spec
get_metrics(net_result)$sens

auc(roc(net_result$test_set$class , net_result$predicted_values))



net_result <- get_net_model_info(net_model_3 , train_set = info_train , test_set = info_test)
net_result$test_income_performance
net_result$train_income_performance
get_precision(net_result)
get_metrics(net_result)$spec
get_metrics(net_result)$sens

auc(roc(net_result$test_set$class , net_result$predicted_values))




# 
# # run neural network backpropagation algorithm for varying number of components
# nc <- c(2,3,5,10,30,50,107)
# nets_by_num_components <- lapply(nc , function(elt) neuralnet(formula(f(elt)$train_rca) , data = f(elt)$train_rca) )
# 
# # get computational speed of each neural net run varying by number of components used as input
# net_times_by_num_components <- lapply(nc , function(elt) system.time(neuralnet(formula(f(elt)$train_rca) , data = f(elt)$train_rca) ))
# 
# net_components_results <- Map(function(model , N) try(get_net_model_info(model , train_set = f(N)$train_rca , test_set = f(N)$test_rca),silent = TRUE) , nets_by_num_components , nc)
# 
# net_components_results <- net_components_results[-6]
# 
# # get test and train performance
# sapply(net_components_results , function(x) x$test_income_performance)
# sapply(net_components_results , function(x) x$train_income_performance)
# 
# 
# sapply(net_components_results , function(x) auc(roc(x$test_set$class , x$predicted_values)))
# 
# # get precision by number of components
# sapply(net_components_results , get_precision)
# 
# # get specificity and sensitivity
# sapply(net_components_results , function(x) get_metrics(x)$spec)
# sapply(net_components_results , function(x) get_metrics(x)$sens)






#########################################################################################################################################################################################
#########################################################################################################################################################################################
########## Train Neural Network Classifier using Clusters as Features ###################################################################################################################
#########################################################################################################################################################################################

## Define new dataset using the cluster indexes of the clusters generated on the dimensionality-reduced dataset
###################################################################################################################

rca_exp_model <- exp_model

new_data <- data.frame(pca_kmeans_by_k[[1]]$cluster , ica_kmeans_by_k[[1]]$cluster , rca_kmeans_by_k[[1]]$cluster , info_gain_kmeans_by_k[[1]]$cluster, 
                       apply(pca_exp_model$z , 1 , which.max) , apply(ica_exp_model$z , 1 , which.max) , apply(rca_exp_model$z , 1 , which.max))


new_data <- data.frame(pca_kmeans_by_k[[2]]$cluster , ica_kmeans_by_k[[2]]$cluster , rca_kmeans_by_k[[2]]$cluster , info_gain_kmeans_by_k[[2]]$cluster, 
                       apply(pca_exp_model$z , 1 , which.max) , apply(ica_exp_model$z , 1 , which.max) , apply(rca_exp_model$z , 1 , which.max))



new_data <- data.frame(pca_kmeans_by_k[[5]]$cluster , ica_kmeans_by_k[[5]]$cluster , rca_kmeans_by_k[[5]]$cluster , info_gain_kmeans_by_k[[5]]$cluster, 
                       apply(pca_exp_model$z , 1 , which.max) , apply(ica_exp_model$z , 1 , which.max) , apply(rca_exp_model$z , 1 , which.max))



# perform one hot encoding on the data
new_data <- data.frame(class = train_encoded$class , Reduce(cbind , lapply(new_data , class.ind)))

new_train <- new_data[1:20000,]
new_test <- new_data[-c(1:20000),]

# run neural net model on cluster-feature constructed data
nd_net_start <- proc.time()
nd_net_model <- neuralnet(formula(new_train) , data = new_train)
nd_net_runtime <- proc.time() - nd_net_start


# get various metrics from result of neural net model
net_result <- get_net_model_info(nd_net_model , train_set = new_train , test_set = new_test)
net_result$test_income_performance
net_result$train_income_performance
get_precision(net_result)
get_metrics(net_result)$spec
get_metrics(net_result)$sens

auc(roc(net_result$test_set$class , net_result$predicted_values))


################################################################################################


# ~ Now combine features from reduced datasets with cluster-based features #####################
################################################################################################

pca_dataset$class <- NULL
ica_dataset_5$class <- NULL
rca_dataset_5$class <- NULL
# try out different combinations of cluster-based features with dimension-reduced features

# combined <- data.frame(new_data , pos_gain_data)
# combined <- data.frame(new_data , pca_dataset)
# combined <- data.frame(new_data , ica_dataset_5)
combined <- data.frame(new_data , rca_dataset_5)

combined_train <- combined[1:20000,]
combined_test <- combined[-c(1:20000),]

# run neural net model on cluster-feature constructed data
combined_net_start <- proc.time()
combined_net_model <- neuralnet(formula(combined_train) , data = combined_train)
combined_net_runtime <- proc.time() - combined_net_start


# get various metrics from result of neural net model
net_result <- get_net_model_info(combined_net_model , train_set = combined_train , test_set = combined_test)
net_result$test_income_performance
net_result$train_income_performance
get_precision(net_result)
get_metrics(net_result)$spec
get_metrics(net_result)$sens

auc(roc(net_result$test_set$class , net_result$predicted_values))











### PCA Reconstruction
#################################################################################################################

train_encoded$cluster <- train_encoded$class <- NULL

get_recon_error <- function(N)
{

    # reconstruct matrix
    restr <- fit$x[,1:N] %*% t(fit$rotation[,1:N])
    
    # unscale and uncenter the data
    if(fit$scale != FALSE){
      restr <- scale(restr, center = FALSE , scale=1/fit$scale)
    }
    if(all(fit$center != FALSE)){
      restr <- scale(restr, center = -1 * fit$center, scale=FALSE)
    }
    
    train_encoded$cluster <- train_encoded$class <- NULL
 
    sse(train_encoded,restr)

}

sapply(seq(10,50,10) , get_recon_error)
get_recon_error(75)
get_recon_error(90)
get_recon_error(100)
