
# load packages
import numpy as np
import RTLearner as rt

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner , kwargs = {"leaf_size":1} , bags = 20, boost = False, verbose = False):
        self.learner = learner
        self.bags = bags
        self.boost = boost        
        self.kwargs = kwargs
        self.verbose = verbose
        
        #learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False
        
        


    def addEvidence(self,dataX,dataY):

        
#        evidence = self.build_tree(dataX,dataY)
        
        bags = self.bags
        learner = self.learner
        kwargs = self.kwargs
        verbose = self.verbose
        
        sample_size = int(np.round(0.6 * dataX.shape[0]))
        
        choices = list(range(0 , dataX.shape[0]))

        # get random sample indexes
        random_indexes = []
        for i in range(0,bags):
            random_indexes.append(np.random.choice(choices , size = sample_size , replace = True))

        # get random samples from the random indexes chosen previously
        random_samples = []
        random_Ys = []
        for elt in random_indexes:        
            random_samples.append(dataX[elt])
            random_Ys.append(dataY[elt])
        
        #learner = RTLearner


        trees = []
        for x , y in zip(random_samples , random_Ys):
            learner_object = learner(leaf_size = kwargs['leaf_size'] , verbose = verbose)
            learner_object.addEvidence(x,y)
            tree = learner_object
            trees.append(tree)
            
        
        
        self.evidence = trees
        
    def query(self,points):

        models = self.evidence        
        
        y_options = []
        for model in models:
            y_options.append(np.array(model.query(points)))
                    
        
        y_preds = np.mean(y_options,axis=0)
        
        
        return y_preds
        
        
#        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"