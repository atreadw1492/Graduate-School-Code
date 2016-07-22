import os
import pandas as pd
import datetime
from utils import bag_to_svmlight

def read_csv(filepath):
    
    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + "events.csv")
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + "mortality_events.csv")

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + "event_feature_map.csv")

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    

    # Get list of all unique patient ids
    patient_ids = [elt for elt in set(events.patient_id)]


    # Get patient ids of all alive patients
    alive_patient_ids = set(patient_ids).difference(mortality.patient_id)


    # Use alive patient ids to retrieve events data for alive patients from the "events" data frame
    alive_events_data = events[events.patient_id.isin(alive_patient_ids)]

    
    # Get the index date / last event date for each alive patient
    alive_last_event_date = alive_events_data.groupby(["patient_id"])["timestamp"].max()
    
    # Format the index date for each alive patient into a pandas data frame
    alive_indx_date = pd.DataFrame({"patient_id" : list(alive_last_event_date.keys()) , "indx_date" : list(alive_last_event_date) })
    alive_indx_date = alive_indx_date[["patient_id" , "indx_date"]]


    # Get the most recent visit date for each deceased patient
    death_datetime_date = [datetime.datetime.strptime(elt,"%Y-%m-%d") for elt in mortality.timestamp]
    
    # Get the date 30 days prior to the death date (the index date) for each dead patient
    dead_indx_date = [elt - datetime.timedelta(30) for elt in death_datetime_date]
        
    # Format the index dates for each dead patient back into a date string format: yyyy-mm-dd
    dead_indx_date = [datetime.datetime.strftime(elt,"%Y-%m-%d") for elt in dead_indx_date]

    # Format the index date for each dead patient into a pandas data frame
    dead_indx_date = pd.DataFrame({"patient_id" : mortality.patient_id , "indx_date" : dead_indx_date})
    dead_indx_date = dead_indx_date[["patient_id" , "indx_date"]]

    
    # Stack the alive_indx_date and dead_indx_date dataframes together
    indx_date = alive_indx_date.append(dead_indx_date)
        

    # Reset index of data frame
    indx_date = indx_date.reset_index()

    
    # Convert add to add in the H / M / S format
    indx_date["indx_date"] = [datetime.datetime.strftime(datetime.datetime.strptime(elt,"%Y-%m-%d") , "%Y-%m-%d %H:%M:%S") for elt in indx_date.indx_date]


    # Write indx_date to csv file
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    # Join events data with indx_date
    joined_data = pd.merge(left = events, right = indx_date, how = "left", on="patient_id")
    
    # check
    joined_data["timestamp"] = [datetime.datetime.strftime(datetime.datetime.strptime(elt,"%Y-%m-%d") , "%Y-%m-%d %H:%M:%S") for elt in joined_data.timestamp]

    # Get the number of days between each event timestamp and the index date for each respective patient
    window_lengths = [datetime.datetime.strptime(indx,"%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(ts,"%Y-%m-%d %H:%M:%S")  for indx,ts in zip(joined_data.indx_date , joined_data.timestamp)]
    joined_data["window_lengths"] = [elt.days for elt in window_lengths]
    
    
    # Get the subset of joined_data such that any event timestamp is within 2000 days less than or equal to the index date for each patient
    filtered_events = joined_data[(joined_data.window_lengths >= 0) & (joined_data.window_lengths <= 2000)]
    
    # Reset filtered_events index
    filtered_events = filtered_events.reset_index()
    
    
    # Get only the columns needed for filtered_events: patient id, event id, and value
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]
    
    # Write filtered_events to csv file
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    # Filter out events with N/A values
    filtered_events_df = filtered_events_df[filtered_events_df.value == filtered_events_df.value]
    
    # Join filtered_events_df with feature_map_df to get the idx (feature_id) equivalent of event_id
    filtered_events_df = pd.merge(left = filtered_events_df , right = feature_map_df , how = "left" ,  on = "event_id")
    
    filtered_events_df = filtered_events_df.rename(columns = {"idx" : "feature_id"})
    
    
    # Get subset of filtered_events_df where the the event is a lab event 
    lab_events = filtered_events_df[filtered_events_df.event_id.isin([elt for elt in filtered_events_df.event_id if elt[0:3] == "LAB"])]

    # Get subset of filtered_events_df where the event is not a lab event
    non_lab_events = filtered_events_df[filtered_events_df.event_id.isin([elt for elt in filtered_events_df.event_id if elt[0:3] != "LAB"])]
    
    # Aggregate lab events data by getting the count by patient_id / event_id
    lab_aggregated_events = lab_events.groupby(["patient_id" , "feature_id"])["value"].count().reset_index()
    #lab_aggregated_events = lab_events.groupby(["patient_id" , "event_id"])["value"].count().reset_index()
    
    # Aggregate non-lab (diagnostics and medication) events data by getting the average value by patient_id / event_id
    non_lab_aggregated_events = non_lab_events.groupby(["patient_id" , "feature_id"])["value"].sum().reset_index()
    
    
    # Stack the lab and non-lab aggregated data together
    aggregated_events = lab_aggregated_events.append(non_lab_aggregated_events)
    
    # Reset index for aggregated_events
    aggregated_events = aggregated_events.reset_index()

    # Rename 'value' column to 'feature_value'
    aggregated_events = aggregated_events.rename(columns = {"value" : "feature_value"})
    
    
    ## Normalize values in aggregated_events using min-max approach
    
    # First, get max value and min value by feature_id
    max_value_by_feature = aggregated_events.groupby(["feature_id"])["feature_value"].max().reset_index()
    max_value_by_feature = max_value_by_feature.rename(columns = {"feature_value" : "max_feature_value"})
      
    min_value_by_feature = aggregated_events.groupby(["feature_id"])["feature_value"].min().reset_index()    
    min_value_by_feature = min_value_by_feature.rename(columns = {"feature_value" : "min_feature_value"})
   
    # Merge max_value_by_feature and min_value_by_feature data frames
    feature_value_data = pd.merge(left = max_value_by_feature , right = min_value_by_feature , how = "inner" , on = "feature_id")
    
    # Merge aggregated_events with feature_value_data to get the min / max values for each feature_id
    aggregated_events = pd.merge(left = aggregated_events, right = feature_value_data , how = "left" , on = "feature_id")
        
    
    def normalize(row):   
    
        return row["feature_value"] / row["max_feature_value"]
    
    
    
    # Use normalize function to perform min-max normalization on the feature_value values in aggregated_events
    aggregated_events["feature_value"] = aggregated_events.apply(lambda row: normalize(row) , axis = 1)
    
    
    # Select only needed columns for aggregated_events: patient_id, feature_id, and feature_value
    aggregated_events = aggregated_events[["patient_id" , "feature_id" , "feature_value"]]
    
    # Write aggregated_events to csv file
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    
    
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    
    # Get list of all unique patient ids from filtered_events
    patient_ids = [elt for elt in set(filtered_events.patient_id)]
    
    # Get list of all unique alive patient ids
    alive_patient_ids = [elt for elt in set(patient_ids).difference(mortality.patient_id)]

    # Loop over patient_ids to determine which patients are alive (return 1) or dead (return 0)
    #alive_or_dead = map(lambda elt: 1 if elt in alive_patient_ids else 0 , patient_ids)
    
    alive_or_dead = map(lambda elt: 0 if elt in alive_patient_ids else 1 , patient_ids)
    

    ## Define function to map patient_id to list containing tuples of the form (feature_id,feature_value) 
    ## for each feature associated with patient_id
    def get_feature_tuples(patient_id):
        
        single_patient_data = aggregated_events[aggregated_events.patient_id == patient_id]
        
        return list(single_patient_data.apply(lambda row: (row["feature_id"],row["feature_value"]) , axis = 1))
        
        
    # Loop over each patient_ids, getting the associated (feature_id,feature_value) tuples for each patient_id
    list_of_feature_tuples = map(get_feature_tuples , patient_ids)
    
    
    patient_features = dict(zip(patient_ids,list_of_feature_tuples))

    mortality = dict(zip(patient_ids,alive_or_dead))

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    # Get sorted list of all patient_ids
    patient_ids = sorted(patient_features.keys())
    patient_ids = map(int , patient_ids)

     
    # Sort each feature list in patient_features by feature_id
    for patient_id in patient_ids:
        
        patient_features[patient_id] = sorted(patient_features[patient_id] , key = lambda x: x[0])          
               
                         
    all_feature_rows = map(lambda x: bag_to_svmlight(patient_features[x]) + " " , patient_ids)
    
    svmlight_feature_data = [str(int(mortality[x])) + " " + y for x,y in zip(patient_ids , all_feature_rows)]
    

    # Combine rows together into a single writable string; one row for each line in the file to be created
    svmlight_feature_data = reduce(lambda x, y: x + "\n" + y , svmlight_feature_data) + "\n"
    
    
    # Combine patient id's with <feature_id , feature_value> data for the deliverable file
    deliverable_format_rows = [str(id) + " " + str(float(mortality[id])) + " " + data for id,data in zip(patient_ids,all_feature_rows)]

    # Combine rows together into a single writable string; one row for each line in the file to be created
    deliverable_feature_data = reduce(lambda x, y: x + "\n" + y , deliverable_format_rows) + "\n"
    
    
    # Open write-connections to deliverable files
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    
    # Write deliverable1 to file    
    deliverable1.write(svmlight_feature_data)
    
    # Write deliverable2 to file
    deliverable2.write(deliverable_feature_data);
    
    # Close connection to deliverable files
    deliverable1.close()
    deliverable2.close()
    

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()




