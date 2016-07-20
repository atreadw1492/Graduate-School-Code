## Script to implement a collection of unsupervised learning algorithms on exercise-related data
############################################################################################################################################################################################

# devtools::install_github("kassambara/factoextra")
# source("https://bioconductor.org/biocLite.R")
# biocLite("RDRToolbox")

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
train_norm <- data.frame(class = train_norm$class , train_norm[,-ncol(train_norm)])
train_norm$class <- ifelse(train_norm$class == "A" , 1,0)

# sapply(train_norm$class , function(elt) match(as.character(elt) , LETTERS))


test_norm <- as.data.frame(lapply(test , function(value) {if(is.factor(value)) value else (value - min(value)) / (max(value) - min(value))}))
test_norm <- data.frame(class = test_norm$class , test_norm[,-ncol(test_norm)])
test_norm$class <- ifelse(test_norm$class == "A" , 1,0)


dataSet <- rbind(train_norm,test_norm)


##########################################################################################################################################################################################
########## Implement K-means Clustering ##################################################################################################################################################
##########################################################################################################################################################################################

exercise_data <- dataSet[,-1]

start <- proc.time()
kmeans_model <- kmeans(exercise_data,5)
end <- proc.time() - start


# create field in train dataset for cluster labels
dataSet$kmeans_cluster <- kmeans_model$cluster

# Get distribution of data points by cluster
percent_pos <- with(dataSet , tapply(class, kmeans_cluster, sum)) / with(dataSet , tapply(kmeans_cluster, kmeans_cluster, length))
percent_neg <- 1 - percent_pos


ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}


get_cluster_dist <- function(model , x)
{
  
  x$kmeans_cluster <- model$cluster
  
  # Get distribution of data points by cluster
  cluster_sizes <- with(x , tapply(kmeans_cluster, kmeans_cluster, length))
  percent_pos <- with(x , tapply(class, kmeans_cluster, sum)) / cluster_sizes
  percent_neg <- 1 - percent_pos
  
  total_percent_pos <- with(x , tapply(class, kmeans_cluster, sum)) / sum(x$class)
  total_percent_neg = (with(x , tapply(class, kmeans_cluster, length)) - with(x , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(x , class == 0))
  
  return(environment())
  
}




# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)
dataSet$cluster <- NULL

kmeans_by_k <- lapply(k_values , function(k) kmeans(dataSet[,-1] , k))

# sapply(kmeans_by_k, function(x) ClusterPurity(x$cluster , dataSet$class ))

# get cluster distribution info
dist_by_k <- lapply(kmeans_by_k , function(x) get_cluster_dist(x,dataSet))

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)


## Examine the contents of the clusters

dataSet$cluster <- kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$percent_neg,2),"%")
                     ,dist_by_k[[1]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$percent_neg,2),"%")
                     ,dist_by_k[[2]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)


dataSet$cluster <- NULL

# 
# # test kmeans by different algorithms
# algorithm_list <- c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen")
# 
# kmeans_by_algoritm <- lapply(algorithm_list , function(x)  kmeans(dataSet[,-1] , 3 , algorithm = x))
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




# Determine number of clusters
wss <- (nrow(dataSet[,-1])-1)*sum(apply(dataSet[,-1],2,var))
for (i in 2:15) wss[i] <- sum(kmeans(dataSet[,-1], 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")



##################################################################################################################################################################################################################
##### Expectation Maximiation Clustering ##################################################################################################################################################
###########################################

# train <- train_encoded[,-1]

# start <- proc.time()
# em_model <- Mclust(train_encoded[,-1] , initialization=list(subset=sample(1:nrow(train_encoded[,-1]), size=16125)))
# end <- proc.time() - start


# train_dataset <- train_encoded[,intersect(names(train) , names(train_income))]
start <- proc.time()
starter_exp_model <- Mclust(exercise_data , initialization=list(subset=sample(1:nrow(exercise_data), size=16125)))
end <- proc.time() - start

# add cluster assignements based of which cluster has the highest probability amoung the choices
dataSet$cluster <- apply(starter_exp_model$z , 1 , which.max)

cluster_sizes <- with(dataSet , tapply(cluster, cluster, length))
percent_pos <- with(dataSet , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- with(dataSet , tapply(class, cluster, sum)) / sum(dataSet$class)
total_percent_neg = (with(dataSet , tapply(class, cluster, length)) - with(dataSet , tapply(class, cluster, sum)) ) / nrow(subset(dataSet , class == 0))


result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
                     
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)








##########################################################################################################################################################################################

##################################################################################################################################################################################################################
##### Principle Components Analysis ##################################################################################################################################################
###########################################


# Implement Pricipal Components Analysis
start <- proc.time()
fit <- prcomp(dataSet[,-1], cor=TRUE)
end <- proc.time() - start
summary(fit) # print variance accounted for 

# get eigenvalues of the components
eigenvalues <- fit$sdev**2
summary(eigenvalues)
sd(eigenvalues)

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




### PCA Reconstruction
#################################################################################################################

# define function to get sum of squared error between two matrices ~ use later for reconstruction error for PCA, RCA etc.
sse <- function(a,b) sum((a - (a + b) / 2)^2 + (b - (a + b) / 2)^2)

keep_class <- dataSet$class
dataSet$cluster <- dataSet$class <- NULL

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
  
  
  sse(dataSet,restr)
  
}

sapply(seq(10,50,10) , get_recon_error)
summary(fit)







## Implement k-means clustering on PCA components
#################################################

# get_cluster_dist <- function(model , train_set)
# {
#   
#   train_set$kmeans_cluster <- model$cluster
#   
#   # Get distribution of data points by cluster
#   cluster_sizes <- with(train_set , tapply(kmeans_cluster, kmeans_cluster, length))
#   percent_pos <- with(train_set , tapply(class, kmeans_cluster, sum)) / cluster_sizes
#   percent_neg <- 1 - percent_pos
#   
#   total_percent_pos <- with(train_set , tapply(class, kmeans_cluster, sum)) / sum(train_set$class)
#   total_percent_neg = (with(train_set , tapply(class, kmeans_cluster, length)) - with(train_set , tapply(class, kmeans_cluster, sum)) ) / nrow(subset(train_set , class == 0))
#   
#   return(environment())
#   
# }

pca_data <- data.frame(fit$x)

# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)

# run k-means on dataset with first 50 principle components
pca_kmeans_by_k <- lapply(k_values , function(k) kmeans(pca_data[,1:50] , k))

pca_data$class <- as.vector(dataSet$class) # add label in for distribution information
dist_by_k <- pca_kmeans_by_k_dist <- lapply(pca_kmeans_by_k,function(x) get_cluster_dist(x,pca_data))


# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(pca_kmeans_by_k_dist , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(pca_kmeans_by_k_dist , function(x) x$total_percent_pos)

lapply(pca_kmeans_by_k_dist , function(x) x$total_percent_neg)


## Examine the contents of the clusters

dataSet$cluster <- pca_kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$percent_neg,2),"%")
                     ,dist_by_k[[1]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- pca_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$percent_neg,2),"%")
                     ,dist_by_k[[2]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- NULL
#######################################################################################################

# Determine number of clusters
wss <- (nrow(pca_dataset[,-1])-1)*sum(apply(pca_dataset[,-1],2,var))
for (i in 2:15) wss[i] <- sum(kmeans(pca_dataset[,-1], 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")









## EM clustering on PCA-reduced data
####################################

pca_dataset <- data.frame(class = dataSet$class , fit$x)

# run EM clustering on dataset with first five PCA components
pca_dataset <- pca_dataset[,1:6]
start <- proc.time()
pca_exp_model <- Mclust(pca_dataset[,-1] , initialization=list(subset=sample(1:nrow(pca_dataset), size=10000)))
end <- proc.time() - start

# add cluster assignements based of which cluster has the highest probability amoung the choices
pca_dataset$cluster <- apply(pca_exp_model$z , 1 , which.max)


cluster_sizes <- with(pca_dataset , tapply(cluster, cluster, length))
percent_pos <- with(pca_dataset , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- with(pca_dataset , tapply(class, cluster, sum)) / sum(pca_dataset$class)
total_percent_neg = (with(pca_dataset , tapply(class, cluster, length)) - with(pca_dataset , tapply(class, cluster, sum)) ) / nrow(subset(pca_dataset , class == 0))


result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
                     
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)










# rbind(percent_pos , cluster_sizes)

pca_dataset$em_cluster <- NULL


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
ica_model <- icafast(exercise_data , 3)
end <- proc.time() - start

num_components <- c(3,5,10,15,20,30,50,52)
ica_by_num_components <- lapply(num_components , function(elt) icafast(exercise_data , elt))


lapply(ica_by_num_components , function(x) x$vafs)



ica_dataset <- data.frame(class = dataSet$class , ica_by_num_components[[6]]$S) # use this for clustering later ~ 30 components



x <- icafast(train , 20)

icafast(X,nc,center=TRUE,maxit=100,tol=1e-6,Rmat=diag(nc),
        alg=c("par","def"),fun=c("logcosh","exp","kur"),alpha=1)



# ica_all_vars <- icafast(train , ncol(train))
# names(ica_all_vars$vafs) <- 1:length(ica_all_vars$vafs)
# barplot(ica_all_vars$vafs[1:20] , col="darkblue", main="Variance Explained by Components in ICA")
# lines(ica_all_vars$vafs[1:20],col="red",type="o")


ica_by_algorithm <- lapply(c("logcosh","exp","kur"), function(alg) icafast(train , 40, alg = alg))


# look at kurtosis of independent components
all_kurtosis <- lapply(ica_by_num_components , function(x) kurtosis(x$S))
lapply(all_kurtosis,summary)

control_k <- sort(kurtosis(dataSet[,-1]))
summary(control_k)

# k <- sort(kurtosis(ica_all_vars$S))
# summary(k)

Reduce(rbind,lapply(all_kurtosis,summary))



######################################################################################################################################################################################################
######################################################################################################################################################################################################

##### Kmeans clustering on ICA-reduced dataset ########################################
#######################################################################################

# test different values of k
k_values <- c(3,5,7,10,15,20,25,30)

ica_kmeans_by_k <- lapply(k_values , function(k) kmeans(ica_dataset[,-1] , k))

# sapply(ica_kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(ica_kmeans_by_k , function(x) get_cluster_distribution(x,dataSet))

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)


## Examine the contents of the clusters

dataSet$cluster <- ica_kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$percent_neg,2),"%")
                     ,dist_by_k[[1]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- ica_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$percent_neg,2),"%")
                     ,dist_by_k[[2]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- NULL









## EM clustering on ICA-reduced data
####################################

# # run EM clustering on dataset with first five PCA components
ica_dataset_5 <- ica_dataset[,1:6]
ica_start <- proc.time()
ica_exp_model <- Mclust(ica_dataset_5[,-1] , initialization=list(subset=sample(1:nrow(ica_dataset_5), size=16125)))
ica_end <- proc.time() - start



# add cluster assignements based of which cluster has the highest probability amoung the choices
ica_dataset_5$cluster <- apply(ica_exp_model$z , 1 , which.max)


cluster_sizes <- with(ica_dataset_5 , tapply(cluster, cluster, length))
percent_pos <- with(ica_dataset_5 , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- (with(ica_dataset_5 , tapply(class, cluster, sum)) / sum(ica_dataset_5$class))
total_percent_neg = (with(ica_dataset_5 , tapply(class, cluster, length)) - with(ica_dataset_5 , tapply(class, cluster, sum)) ) / nrow(subset(ica_dataset_5 , class == 0))


result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
                     
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

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
# # rbind(percent_pos , cluster_sizes)
# 
# x  <- ica_dataset_5
# x$cluster_class <- ifelse(x$em_cluster %in% c(2) , 1, 0)
# 
# # get accuracy
# nrow(subset(x,class == cluster_class)) / nrow(x)
# 
# # get recall
# nrow(subset(x,class == cluster_class & class == 1)) / nrow(subset(x,class == 1))
# 
# # get sensitivity
# nrow(subset(x,class == cluster_class & class == 0)) / nrow(subset(x,class == 0))
# 
# 
# 
# ica_dataset_5$em_cluster <- NULL
# 
# 


##################################################################################################################################################################################################################
##### Random Projection ##################################################################################################################################################
###########################################

# https://github.com/chappers/randomProjection


start <- proc.time()
rp_model = RandomProjection(dataSet[,-1], n_features=50, eps=0.1)
end <- proc.time() - start

# rca_dataset <- data.frame(class = train_encoded$class , rp_model$RP)


# Implement random projection matrices by various number of components
rp_models_by_nc <- lapply(num_components , function(elt) RandomProjection(dataSet[,-1], n_features=elt, eps=0.1))
rp_model <- rp_models_by_nc[[6]]
rca_dataset <- data.frame(class = dataSet$class , rp_model$RP)


all_rp_models <- lapply(1:10 , function(index) RandomProjection(dataSet, n_features=50, eps=0.1))


get_recon_error <- function(N,rp_model)
{
  # reconstruct matrix
  restr <- rp_model$RP[,1:N] %*% t(rp_model$R[,1:N])
  
  sse(dataSet,restr)
  
}

recon_errors <- sapply(seq(10,50,10) , function(x) get_recon_error(x,rp_models_by_nc[[7]]))


all_rp_errors <- sapply(all_rp_models , function(x) get_recon_error(50,x))
summary(all_rp_errors)
sd(all_rp_errors)






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

rca_kmeans_by_k <- lapply(k_values , function(k) kmeans(rca_dataset[,-1] , k))

# sapply(rca_kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(rca_kmeans_by_k , function(x) get_cluster_distribution(x,rca_dataset))

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)


## Examine the contents of the clusters

## Examine the contents of the clusters

dataSet$cluster <- ica_kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$percent_neg,2),"%")
                     ,dist_by_k[[1]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- ica_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$percent_neg,2),"%")
                     ,dist_by_k[[2]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- NULL

# 
# # Determine number of clusters
# wss <- (nrow(train_encoded[,-1])-1)*sum(apply(train_encoded[,-1],2,var))
# for (i in 2:15) wss[i] <- sum(kmeans(train_encoded[,-1], 
#                                      centers=i)$withinss)
# plot(1:15, wss, type="b", xlab="Number of Clusters",
#      ylab="Within groups sum of squares")


##############################################################################################################################################################################################
######### EM Clustering on RCA-reduced data ##################################################################################################################################################
###########################################################################################################################################################################################


# # run EM clustering on dataset with first five PCA components
rca_dataset_5 <- rca_dataset[,1:6]
start <- proc.time()
rca_exp_model <- Mclust(rca_dataset_5[,-1] , initialization=list(subset=sample(1:nrow(rca_dataset_5), size=10000)))
end <- proc.time() - start


# add cluster assignements based of which cluster has the highest probability amoung the choices
rca_dataset_5$cluster <- apply(rca_exp_model$z , 1 , which.max)
rca_dataset_5$class <- dataSet$class

cluster_sizes <- with(rca_dataset_5 , tapply(cluster, cluster, length))
percent_pos <- with(rca_dataset_5 , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- (with(rca_dataset_5 , tapply(class, cluster, sum)) / sum(rca_dataset_5$class))
total_percent_neg = (with(rca_dataset_5 , tapply(class, cluster, length)) - with(rca_dataset_5 , tapply(class, cluster, sum)) ) / nrow(subset(rca_dataset_5 , class == 0))

dataSet$cluster <- rca_dataset_5$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
                     
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- NULL









#******************************************************************************************************************************************************************************************
##########################################################################################################################################################################################
## Implement Feature Selection via Information Gain
##########################################################################################################################################################################################

start <- proc.time()
info_gain <- information.gain(formula(dataSet) , dataSet)
end <- proc.time() - start


info_gain <- data.frame(var = rownames(info_gain) , info_gain)
rownames(info_gain) = NULL

info_gain <- arrange(info_gain , info_gain$attr_importance, decreasing = TRUE)

info_gain[1:10,]


summary(info_gain$attr_importance)

pos_gain_data <- dataSet[,as.character(info_gain$var[1:39])]


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


info_kmeans_by_k <- lapply(k_values , function(k) kmeans(pos_gain_data , k))
pos_gain_data$class <- dataSet$class

sapply(kmeans_by_k, function(x) ClusterPurity(x$cluster , train_encoded$class ))

# get cluster distribution info
dist_by_k <- lapply(info_kmeans_by_k , function(x) get_cluster_distribution(x,pos_gain_data))

# get proportion of postive labels by cluster and size of cluster for the different values of k
lapply(dist_by_k , function(x) rbind(x$percent_pos , x$cluster_sizes))


lapply(dist_by_k , function(x) x$total_percent_pos)

lapply(dist_by_k , function(x) x$total_percent_neg)



## Examine the contents of the clusters

dataSet$cluster <- info_kmeans_by_k[[1]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[1]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$percent_neg,2),"%")
                     ,dist_by_k[[1]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[1]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- info_kmeans_by_k[[2]]$cluster
result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100 * dist_by_k[[2]]$percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$percent_neg,2),"%")
                     ,dist_by_k[[2]]$cluster_sizes
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_pos,2),"%")
                     ,paste0(round(100 * dist_by_k[[2]]$total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)

dataSet$cluster <- NULL






pos_gain_data$class <- NULL
# Determine number of clusters
wss <- (nrow(pos_gain_data)-1)*sum(apply(pos_gain_data,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(pos_gain_data, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")


####################################################################################################################################################################################################
#### EM Clustering on Information Gain-Reduced Data
####################################################################################################################################################################################################


pos_gain_data$em_cluster <- NULL
pos_gain_data$class <- NULL

start <- proc.time()
# em_model <- Mclust(pos_gain_data)
info_exp_model <- Mclust(pos_gain_data , initialization=list(subset=sample(1:nrow(pos_gain_data), size=1000)))
info_end <- proc.time() - start


pos_gain_data$cluster <- apply(info_exp_model$z , 1 , which.max)
pos_gain_data$class <- dataSet$class

cluster_sizes <- with(pos_gain_data , tapply(cluster, cluster, length))
percent_pos <- with(pos_gain_data , tapply(class, cluster, sum)) / cluster_sizes
percent_neg <- 1 - percent_pos

total_percent_pos <- with(pos_gain_data , tapply(class, cluster, sum)) / sum(pos_gain_data$class)
total_percent_neg <- (with(pos_gain_data , tapply(class, cluster, length)) - with(pos_gain_data , tapply(class, cluster, sum)) ) / nrow(subset(pos_gain_data , class == 0))


result <- data.frame(cluster = 1:length(unique(dataSet$cluster)) 
                     ,paste0(round(100*percent_pos,2),"%")
                     ,paste0(round(100*percent_neg,2),"%")
                     ,cluster_sizes
                     ,paste0(round(100 * total_percent_pos,2),"%")
                     ,paste0(round(100 * total_percent_neg,2),"%")
                     
                     ,round(100 * tapply(dataSet$accel_belt_z , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_arm_y , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$accel_forearm_x , dataSet$cluster, mean),2)
                     ,round(100 * tapply(dataSet$yaw_forearm , dataSet$cluster, mean),2)
                     
)

names(result) <- LETTERS[1:10]
write.table(result , "clipboard", sep="\t",row.names = FALSE)






