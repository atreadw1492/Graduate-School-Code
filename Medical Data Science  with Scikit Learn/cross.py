import models
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean

import utils

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	
	# Get the train indices and test indices for each k-fold iteration
	kfold = KFold(n = len(Y) , n_folds = k)
	
	
	# Train logistic regression classifier on each fold train set
	lr_pred = [models.logistic_regression_pred(X[train_index] , Y[train_index], X[test_index]) for train_index,test_index in kfold]
	    
       	# Use the kfold object to get the true Y values for each fold test set
       	Y_true_values = [Y[test_index] for train_index , test_index in kfold]
       	
       	# Get the various classications metrics for the logistic regression models implemented on each fold
	metrics = [ models.classification_metrics(Y_pred,Y_true) for Y_pred,Y_true in zip(lr_pred , Y_true_values) ]
	
	
	mean_accuracy = mean([elt[0] for elt in metrics])
	
	mean_auc = mean([elt[1] for elt in metrics])
	
	return mean_accuracy,mean_auc


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	
	randomizedCV = ShuffleSplit(n = len(Y) , n_iter = iterNo, test_size = test_percent)
	
	# Train logistic regression classifier on each fold train set
	lr_pred = [models.logistic_regression_pred(X[train_index] , Y[train_index], X[test_index]) for train_index,test_index in randomizedCV]
	    
       	# Use the kfold object to get the true Y values for each fold test set
       	Y_true_values = [Y[test_index] for train_index , test_index in randomizedCV]
       	
       	# Get the various classications metrics for the logistic regression models implemented on each fold
	metrics = [ models.classification_metrics(Y_pred,Y_true) for Y_pred,Y_true in zip(lr_pred , Y_true_values) ]
	
	# Calculate mean accuracy across all iterations
	mean_accuracy = mean([elt[0] for elt in metrics])
	
	# Calculate mean AUC across all iterations
	mean_auc = mean([elt[1] for elt in metrics])
	
	return mean_accuracy,mean_auc


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

