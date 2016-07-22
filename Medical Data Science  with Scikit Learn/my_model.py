import utils
import pandas as pd
from sklearn import ensemble
from etl import aggregate_events


def my_features():

	# Get train data from svmlight_file	
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	
	# Read in the test events and feature_map data
	events = pd.read_csv("../data/test/events.csv")
	feature_map = pd.read_csv("../data/test/event_feature_map.csv")
	
	# Aggregate test events data using the aggregate_events method from etl.py
	aggregated_events = aggregate_events(events , None, feature_map , "../data/test/test_aggregated_events.csv")

	# Create the test features
	patient_feautures = create_test_features(aggregated_events)

        # Generate the test features file	
	save_test_features(patient_feautures,"../deliverables/test_features.txt")
	
	# Get test data from svmlight_file created above
	X_test, patient_ids = utils.get_data_from_svmlight("../deliverables/test_features.txt")
	
	
	return X_train,Y_train,X_test



def my_classifier_predictions(X_train,Y_train,X_test):
	
	# Define a Gradient Boosting Classifier object

	model = ensemble.GradientBoostingClassifier(random_state = 545510477 , n_estimators =200, max_depth=6)

        
	# Fit Gradient Boosting model to training data
	fit = model.fit(X_train.toarray(),Y_train)


	# Use fit to get test probability predictions
	Y_pred = fit.predict_proba(X_test.toarray())
	
	Y_pred = [elt[1] for elt in Y_pred]
	
	
	return Y_pred


def create_test_features(aggregated_events):
    
          
	# Get list of all unique patient ids from filtered_events
	patient_ids = [elt for elt in set(aggregated_events.patient_id)]
	patient_ids = sorted(patient_ids)
	
	# Create dummy values for Y_test       
	#dummy_alive_or_dead = [0] * len(patient_ids)
	

	## Define function to map patient_id to list containing tuples of the form (feature_id,feature_value) 
	## for each feature associated with patient_id
	def get_feature_tuples(patient_id):
		
		single_patient_data = aggregated_events[aggregated_events.patient_id == patient_id]
		
		return list(single_patient_data.apply(lambda row: (row["feature_id"],row["feature_value"]) , axis = 1))
		
		
	# Loop over each patient_ids, getting the associated (feature_id,feature_value) tuples for each patient_id
	list_of_feature_tuples = map(get_feature_tuples , patient_ids)
	
	
	patient_features = dict(zip(patient_ids,list_of_feature_tuples))

	#mortality = dict(zip(patient_ids,dummy_alive_or_dead))

	return patient_features


def save_test_features(patient_features, test_features_deliverable):

    # Get sorted list of all patient_ids
    patient_ids = sorted(patient_features.keys())
    patient_ids = map(int , patient_ids)

     
    # Sort each feature list in patient_features by feature_id
    for patient_id in patient_ids:
        
        patient_features[patient_id] = sorted(patient_features[patient_id] , key = lambda x: x[0])          
               

    # Get SVMLight-formatted row for each patient in patient_ids
    all_feature_rows = map(lambda x: utils.bag_to_svmlight(patient_features[x]) + " " , patient_ids)
    
    
    # Combine patient id's with <feature_id , feature_value> data for the deliverable file
    deliverable_format_rows = [str(id) + " " + data for id,data in zip(patient_ids,all_feature_rows)]

    # Combine rows together into a single writable string; one row for each line in the file to be created
    deliverable_feature_data = reduce(lambda x, y: x + "\n" + y , deliverable_format_rows) + "\n"
    
    
    # Open write-connection to deliverable file
    deliverable = open(test_features_deliverable, 'wb')

    # Write deliverable to file
    deliverable.write(deliverable_feature_data);
    
    # Close connection to deliverable files
    deliverable.close()


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)

if __name__ == "__main__":
    main()

	