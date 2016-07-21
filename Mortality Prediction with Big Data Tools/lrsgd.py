#!/usr/bin/env python
"""
Implement your own version of logistic regression with stochastic
gradient descent.

Author: Andrew Treadway
Email : atreadway6@gatech.edu
"""

#import collections
import math

#from sklearn.linear_model import SGDClassifier

class LogisticRegressionSGD:

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.mu = mu
        self.eta = eta
        self.weight = [0.0] * n_feature
        
        self.n_feature = n_feature

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """

        #patient_feature_values = [elt[1] for elt in X]
        # Get a list of all the feature values
        patient_feature_keys = [elt[0] for elt in X] # corresponds to which weight to update
        x_i_values = [elt[1] for elt in X]
        


        # initialize x values to all zeros
        all_x_values = [0.0] * self.n_feature
        #print "N_feature is: " + str(self.n_feature) + "\n\n\n\n"

        # Change the features with non-zero x values to actual values
        for index,feature_id in enumerate(patient_feature_keys):
            all_x_values[feature_id] = x_i_values[index]
        

        #print patient_feature_keys
        #print "\n\n\n"
        #
        #print all_x_values
        
        current_weights = self.weight
        # get dot product of current weights and x_i values for i_th patient
        dot_product = sum([elt1 * elt2 for elt1,elt2 in zip(current_weights,all_x_values)])
        p = 1 / ( 1 + math.exp(-dot_product) )     
                                    
        self.weight = [ self.update_weights(x_val , y , self.weight[j] , p) for j,x_val in enumerate(all_x_values) ]                                    
                                    

            
  
        pass
            #return updated_weight

    def predict(self, X):
        return 1 if self.predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
        
    def update_weights(self,x_j,y,w_j,p):
        
        updated_w_j = w_j + self.eta * ( (y - p) * x_j - (2 * self.mu * w_j) )
        
        return updated_w_j
