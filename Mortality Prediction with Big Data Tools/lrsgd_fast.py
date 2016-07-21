#!/usr/bin/env python


import math


class LogisticRegressionSGD:

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.mu = mu #change
        self.eta = eta
        self.weight = [0.0] * n_feature
        
        self.n_feature = n_feature

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """

        # Get a list of all the feature values
        patient_feature_keys = [elt[0] for elt in X] # corresponds to which weight to update
        x_i_values = [elt[1] for elt in X]
        
        #patient_feature_values = [elt[1] for elt in X]
        
        #print patient_feature_keys
        #print "\n\n\n"
        
        current_weights = self.weight        
        
        dot_product = sum([elt1 * elt2 for elt1,elt2 in zip(current_weights,x_i_values)])
        p = 1 / ( 1 + math.exp(-dot_product) )
                                    
        for index , j in enumerate(patient_feature_keys):  
            #print j
            self.weight[j] = self.update_weights( X[index][1] , y , self.weight[j], p)
            

        pass


    def predict(self, X):
        return 1 if self.predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
        
    def update_weights(self,x_j,y,w_j,p):
                
        updated_w_j = w_j + self.eta * ( (y - p) * x_j - (2 * self.mu * w_j) )
        
        return updated_w_j
