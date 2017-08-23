
import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size ,verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        
        
    def build_tree(self,dataX,dataY):

        leaf = 'Leaf'
        leaf_size = self.leaf_size
        
        if dataX.shape[0] == 1:
            return np.asarray([[-1 , dataY[0] , np.nan , np.nan]])
        if len(set(dataY)) == 1:
            return np.asarray([[-1 , dataY[0] , np.nan , np.nan]])
            
        # if number of data points is less than leaf size, average y values
        if dataX.shape[0] <= leaf_size:
            #print dataX.shape[0]
            avg_y = np.mean(dataY)
            return np.asarray([[-1 , avg_y , np.nan , np.nan]])
        
        # determine random feature to split on
        i = np.random.randint(0,dataX.shape[1])
        split_1 = np.random.randint(0,dataX.shape[0])
        split_2 =  np.random.randint(0,dataX.shape[0])
        
        while split_2 == split_1:
            split_2 =  np.random.randint(0,dataX.shape[0])
        
        split_value = np.mean((dataX[split_1 , i] , dataX[split_2 , i] ))

        x_left = dataX[dataX[:,i] <= split_value]
        left_indexes = np.where(dataX[:,i] <= split_value)        
        y_left = dataY[left_indexes]
        
        x_right = dataX[dataX[:,i] > split_value]
        right_indexes = np.where(dataX[:,i] > split_value)        
        y_right = dataY[right_indexes]
        
        while len(x_left) == 0 or len(x_right) == 0:
            # determine random feature to split on
            i = np.random.randint(0,dataX.shape[1])
            split_1 = np.random.randint(0,dataX.shape[0])
            split_2 =  np.random.randint(0,dataX.shape[0])
            
            while split_2 == split_1:
                split_2 =  np.random.randint(0,dataX.shape[0])
            
            split_value = np.mean((dataX[split_1 , i] , dataX[split_2 , i] ))            
            
            
            x_left = dataX[dataX[:,i] <= split_value]
            left_indexes = np.where(dataX[:,i] <= split_value)        
            y_left = dataY[left_indexes]
            
            x_right = dataX[dataX[:,i] > split_value]
            right_indexes = np.where(dataX[:,i] > split_value)        
            y_right = dataY[right_indexes]
        
        
        
        left_tree = self.build_tree(x_left , y_left)
        
        #print("right")
        right_tree = self.build_tree(x_right , y_right)      

        #print left_tree
        #print right_tree
        root = np.asarray([[i , split_value , 1 , left_tree.shape[0] + 1]])
        #print root
        
        #return np.asarray([root,left_tree,right_tree])        
        return np.vstack((root , left_tree , right_tree))


    def addEvidence(self,dataX,dataY):
        
        evidence = self.build_tree(dataX,dataY)
        
        self.evidence = evidence
        
        
    def query(self,points):

        model = self.evidence        
        
        #points = Xtest
        
        #point = points[0]
        

        y_preds = []   
        all_counts = []
        for point_index in range(0,len(points)):
            point = points[point_index]
            row_index = 0    
            counter = 1
            
            keep_while = True
            while(model[row_index , 0] != -1 ) and keep_while:
#==============================================================================
#                 if counter > 1000:
#                     y_pred = np.mean(y_preds)
#                     keep_while = False
#                 else:               
#==============================================================================
                skip = False
                #print 'test start'
                split_val = model[row_index , 1]
                feature_index = model[row_index , 0]
                feature_value = point[feature_index]        
                
                if feature_value <= split_val:
                    change = 1
                    row_index = row_index + 1
                    #print 'test first if'
                else:
                    if model[row_index , 3] == model[row_index , 3]:
                        change = int(model[row_index , 3])
                        row_index = row_index + int(model[row_index , 3])
                        #print 'test first else'
                
                if feature_value <= split_val and row_index >= model.shape[0]:
                    row_index = row_index - change
                    y_pred = model[row_index , 1]
                    skip = True
                elif feature_value > split_val and row_index >= model.shape[0]:
                    row_index = row_index - change
                    y_pred = model[row_index , 1] 
                    skip = True
                
                if not skip:                
                    y_pred = model[row_index , 1]
                    
                    if y_pred != y_pred and feature_value <= split_val:
                        row_index = row_index - 1 # go back up branch
                        if model[row_index , 3] == model[row_index , 3]:
                            row_index = row_index + int(model[row_index , 3]) # and down other side
                    elif y_pred != y_pred and feature_value > split_val:
                        if model[row_index , 3] == model[row_index , 3]:
                            row_index = row_index - int(model[row_index , 3])
                        row_index = row_index + 1
                
                y_pred = model[row_index , 1]
                counter = counter + 1
                                        
            all_counts.append(counter)
            y_preds.append(y_pred)
        
        
        return y_preds
        
        
#        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__=="__main__":
    print "..."