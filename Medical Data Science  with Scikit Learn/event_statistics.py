import time
import pandas as pd
import numpy as np
import datetime

def read_csv(filepath):
    
    # Read "events" data into Pandas Data Frame
    events = pd.read_csv(filepath + "events.csv")

    # Read mortality data into Pandas Data Frame
    mortality = pd.read_csv(filepath + "mortality_events.csv")

    return events, mortality

def event_count_metrics(events, mortality):

    # Get patient ids of all alive patients
    alive_patient_ids = set(events.patient_id).difference(mortality.patient_id)

    # Use alive patient ids to retrieve events data for alive patients from the "events" data frame
    alive_events_data = events[events.patient_id.isin(alive_patient_ids)]
    
    # Use dead patient ids from "mortality" data frame to get events associated with deceased patients
    dead_events_data = events[events.patient_id.isin(mortality.patient_id)]
    
    
    avg_dead_event_count = np.mean(dead_events_data.groupby(["patient_id"])["event_id"].count())

    max_dead_event_count = np.max(dead_events_data.groupby(["patient_id"])["event_id"].count())

    min_dead_event_count = np.min(dead_events_data.groupby(["patient_id"])["event_id"].count())


    avg_alive_event_count = np.mean(alive_events_data.groupby(["patient_id"])["event_id"].count())

    max_alive_event_count = np.max(alive_events_data.groupby(["patient_id"])["event_id"].count())

    min_alive_event_count = np.min(alive_events_data.groupby(["patient_id"])["event_id"].count())



    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count


def encounter_count_metrics(events, mortality):

    # Get patient ids of all alive patients
    alive_patient_ids = set(events.patient_id).difference(mortality.patient_id)

    # Use alive patient ids to retrieve events data for alive patients from the "events" data frame
    alive_events_data = events[events.patient_id.isin(alive_patient_ids)]
    
    # Get the count of unique dates each alive patient visited the ICU; grouped by patient id
    alive_encounters_data = alive_events_data.groupby(["patient_id"])["timestamp"].apply(lambda x: len(x.unique()))
    
    
    # Use dead patient ids from "mortality" data frame to get events associated with deceased patients
    dead_events_data = events[events.patient_id.isin(mortality.patient_id)]
    
    # Get the count of unique dates each deceased patient visited the ICU; grouped by patient id
    dead_encounters_data = dead_events_data.groupby(["patient_id"])["timestamp"].apply(lambda x: len(x.unique()))
    
    
    avg_dead_encounter_count = np.mean(dead_encounters_data)

    max_dead_encounter_count = np.max(dead_encounters_data)

    min_dead_encounter_count = np.min(dead_encounters_data)


    avg_alive_encounter_count = np.mean(alive_encounters_data)



    max_alive_encounter_count = np.max(alive_encounters_data)

    min_alive_encounter_count = np.min(alive_encounters_data)

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):

    # Get patient ids of all alive patients
    alive_patient_ids = set(events.patient_id).difference(mortality.patient_id)

    # Use alive patient ids to retrieve events data for alive patients from the "events" data frame
    alive_events_data = events[events.patient_id.isin(alive_patient_ids)]
    
    # Get the last visit date for each alive patient
    alive_last_event_dates = [datetime.datetime.strptime(elt,"%Y-%m-%d") for elt in alive_events_data.groupby(["patient_id"])["timestamp"].max()]
    
    # Get the first visit date for each alive patient
    alive_first_event_dates = [datetime.datetime.strptime(elt,"%Y-%m-%d") for elt in alive_events_data.groupby(["patient_id"])["timestamp"].min()]
    
    # Get the length of time, in days, between the first visit and last visit for each alive patient
    alive_rec_len_data = [last - first for last,first in zip(alive_last_event_dates,alive_first_event_dates)]
    alive_rec_len_data = [delta.days for delta in alive_rec_len_data]
        
        
        
    # Use dead patient ids from "mortality" data frame to get events associated with deceased patients
    dead_events_data = events[events.patient_id.isin(mortality.patient_id)]
    
    # Get the last visit date for each deceased patient
    dead_last_event_dates = [datetime.datetime.strptime(elt,"%Y-%m-%d") for elt in dead_events_data.groupby(["patient_id"])["timestamp"].max()]
    
    # Get the first visit date for each deceased patient
    dead_first_event_dates = [datetime.datetime.strptime(elt,"%Y-%m-%d") for elt in dead_events_data.groupby(["patient_id"])["timestamp"].min()]
    
    # Get the length of time, in days, between the first visit and last visit for each deceased patient
    dead_rec_len_data = [last - first for last,first in zip(dead_last_event_dates,dead_first_event_dates)]
    dead_rec_len_data = [delta.days for delta in dead_rec_len_data]

    
    
    avg_dead_rec_len = np.mean(dead_rec_len_data)

    max_dead_rec_len = np.max(dead_rec_len_data)

    min_dead_rec_len = np.min(dead_rec_len_data)

    avg_alive_rec_len = np.mean(alive_rec_len_data)

    max_alive_rec_len = np.max(alive_rec_len_data)

    min_alive_rec_len = np.min(alive_rec_len_data)

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
 
    #Modify the filepath to point to the CSV files in train_data
    train_path = "../data/train/"
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()



