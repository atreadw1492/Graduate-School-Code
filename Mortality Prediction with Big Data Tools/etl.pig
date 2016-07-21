-- ***************************************************************************
-- TASK
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- Steps have been provided to guide you.
-- You can include as many intermediate steps as required to complete the calculations.
-- ***************************************************************************

-- ***************************************************************************
-- TESTS
-- To test, please change the LOAD path for events and mortality to ../../test/events.csv and ../../test/mortality.csv
-- 6 tests have been provided to test all the subparts in this exercise.
-- Manually compare the output of each test against the csv's in test/expected folder.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
events = LOAD '../../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
mortality = LOAD '../../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************


-- using left join will get all events
eventswithmort = JOIN events BY patientid LEFT OUTER , mortality BY patientid;  -- perform join of events and mortality by patientid;
eventswithmort = FOREACH eventswithmort GENERATE events::patientid AS patientid , eventid, etimestamp, value, mtimestamp, label;

-- filter out relations where mtimestamp is null since these refer to patients that are alive
deadevents = FILTER eventswithmort BY mtimestamp IS NOT NULL;

-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp
deadevents = FOREACH deadevents GENERATE patientid, eventid, value, etimestamp, label, SubtractDuration(mtimestamp,'P30D') AS index_date;
deadevents = FOREACH deadevents GENERATE patientid, eventid, value, label, DaysBetween(index_date , etimestamp) AS time_difference;



-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

-- get events data of all alive patients
aliveevents = FILTER eventswithmort BY mtimestamp IS NULL ;


-- r1 = foreach (group some_relation by some_id) generate group as some_new_id, MAX(some_relation.some_datetime_column) as some_datetime;

-- alive_temp = GROUP aliveevents ALL ;
-- alive_index_dates = FOREACH alive_temp GENERATE aliveevents.patientid AS patientid , MAX(aliveevents.etimestamp) AS index_date;

--************************************
-- temp = group aliveevents by patientid;
-- describe temp;
alive_index_dates = FOREACH ( GROUP aliveevents BY patientid ) GENERATE group AS patientid, MAX(aliveevents.etimestamp) AS index_date;

--alive_index_dates = FOREACH aliveevents GENERATE patientid , MAX(etimestamp) AS index_date;
-- describe alive_temp;
describe alive_index_dates;
describe aliveevents;

-- join events with alive_index_dates and filter so that only the alive patients' data is pulled in
aliveevents = JOIN aliveevents BY patientid LEFT OUTER , alive_index_dates BY patientid;
aliveevents = FOREACH aliveevents GENERATE  aliveevents::patientid AS patientid, eventid, value, 0 AS label, DaysBetween(index_date , etimestamp) AS time_difference;

describe aliveevents;




--*************************************************************************************************************************************************************************
--*************************************************************************************************************************************************************************
--*************************************************************************************************************************************************************************
--*************************************************************************************************************************************************************************


--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
STORE deadevents INTO 'deadevents' USING PigStorage(',');

-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************

-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)




filtered = UNION aliveevents, deadevents;
filtered = FILTER filtered BY (value IS NOT NULL) AND (time_difference <= 2000 AND time_difference >= 0);



--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
STORE filtered INTO 'filtered' USING PigStorage(',');





--***************************************************************************************************************************************************************************************
--***************************************************************************************************************************************************************************************
--***************************************************************************************************************************************************************************************
--***************************************************************************************************************************************************************************************




-- -- ***************************************************************************
-- -- Aggregate events to create features
-- -- ***************************************************************************

-- -- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)
describe filtered;
grouped_patient_events = GROUP filtered BY (patientid,eventid);
describe grouped_patient_events;

featureswithid =  FOREACH grouped_patient_events GENERATE FLATTEN(group) , COUNT(filtered) AS featurevalue;
--describe featureswithid;


-- --TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');





-- -- ***************************************************************************
-- -- Generate feature mapping
-- -- ***************************************************************************
-- compute the set of distinct eventids obtained from previous step, sort them by eventid and then rank these features by eventid to create (idx, eventid). Rank should start from 0.
all_features = FOREACH featureswithid GENERATE eventid;
all_features = DISTINCT all_features;
all_features = ORDER all_features BY eventid;

all_features = RANK all_features;
all_features = FOREACH all_features GENERATE ($0 - 1) AS idx , $1;


-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');

describe featureswithid;
describe all_features;

-- perform join of featureswithid and all_features by eventid and replace eventid with idx. It is of the form (patientid, idx, featurevalue)
features = JOIN featureswithid BY group::filtered::events::eventid LEFT OUTER , all_features BY events::eventid;
describe features;
features = FOREACH features GENERATE  featureswithid::group::filtered::patientid AS patientid , all_features::idx AS idx , featureswithid::featurevalue AS featurevalue;


-- -- -- --TEST-4
features = ORDER features BY patientid, idx;
STORE features INTO 'features_map' USING PigStorage(',');



-- -- ***************************************************************************
-- -- Normalize the values using min-max normalization
-- -- ***************************************************************************

-- group events by idx and compute the maximum feature value in each group. It is of the form (idx, maxvalues)
temp = FOREACH features GENERATE idx , featurevalue;
describe temp;

-- maxvalues = FOREACH ( GROUP temp BY idx ) GENERATE group AS idx, MAX(featurevalue) AS maxvalues;

maxvalues = GROUP temp BY idx;
maxvalues = FOREACH maxvalues GENERATE group AS idx , MAX($1.$1) AS maxvalues;

describe maxvalues;

-- join features and maxvalues by idx
normalized = JOIN features BY idx LEFT OUTER , maxvalues BY idx;
normalized = FOREACH normalized GENERATE features::patientid AS patientid , features::idx AS idx, features::featurevalue AS featurevalue, maxvalues::maxvalues AS maxvalues;


-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)
features = FOREACH normalized GENERATE patientid , idx , (double)featurevalue / (double)maxvalues AS normalizedfeaturevalue;



-- -- --TEST-5
features = ORDER features BY patientid, idx;
STORE features INTO 'features_normalized' USING PigStorage(',');

-- -- ***************************************************************************
-- -- Generate features in svmlight format
-- -- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- -- e.g.  1,1,1.0
-- --  	 1,3,0.8
-- --	 2,1,0.5
-- --       3,3,1.0
-- -- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- -- ***************************************************************************
-- -- Split into train and test set
-- -- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- -- e.g. 1,1
-- --	2,0
-- --      3,1
-- -- ***************************************************************************

-- create it of the form (patientid, label) for dead and alive patients
labels = FOREACH filtered GENERATE patientid , label;



-- --Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
STORE samples INTO 'samples' USING PigStorage(' ');

-- -- randomly split data for training and testing
samples = FOREACH samples GENERATE RANDOM() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');