import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn import metrics

#input: X_train, Y_train and X_test
#output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):
	#TODO: train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
	
	
	logistic_regression_model = LogisticRegression(random_state = 545510477)

	logistic_regression_fit = logistic_regression_model.fit(X_train , Y_train)
	
	Y_pred = logistic_regression_fit.predict(X_test)
	
	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_pred(X_train, Y_train, X_test):
	#TODO:train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
	
	linear_svc_model = LinearSVC(random_state = 545510477)
	
	linear_svc_fit = linear_svc_model.fit(X_train , Y_train)
	
	Y_pred = linear_svc_fit.predict(X_test)
	

	return Y_pred

#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.

	
        decision_tree_model = DecisionTreeClassifier(max_depth = 5 , random_state = 545510477)
        
        decision_tree_fit = decision_tree_model.fit(X_train , Y_train)
        
        Y_pred = decision_tree_fit.predict(X_test)	
	
	return Y_pred


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#TODO: Calculate the above mentioned metrics
	#NOTE: It is important to provide the output in the same order
	
	# Get accuracy
	accuracy = accuracy_score(Y_true , Y_pred)
	
	# Get auc
	fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
	auc = metrics.auc(fpr,tpr)
	
	# Get precision
	precision = precision_score(Y_true , Y_pred)
	
	# Get recall
	recall = recall_score(Y_true , Y_pred)
	
	# Get f1-score
	f1score = f1_score(Y_true , Y_pred)
	
	
	return accuracy,auc,precision,recall,f1score

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print "______________________________________________"
	print "Classifier: "+classifierName
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print "Accuracy: "+str(acc)
	print "AUC: "+str(auc_)
	print "Precision: "+str(precision)
	print "Recall: "+str(recall)
	print "F1-score: "+str(f1score)
	print "______________________________________________"
	print ""
	
def get_data_from_svmlight(svmlight_file):
    data_train = load_svmlight_file(svmlight_file,n_features=3190)
    X_train = data_train[0]
    Y_train = data_train[1]
    return X_train, Y_train

def main():
	#X_train, Y_train = utils.get_data_from_svmlight(os.getcwd() + "/deliverables/features_svmlight.train")
	#X_test, Y_test = utils.get_data_from_svmlight("../data/features_svmlight.validate")
	
	X_train, Y_train = get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = get_data_from_svmlight("../data/features_svmlight.validate")

	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train,X_test),Y_test)

if __name__ == "__main__":
	main()
	