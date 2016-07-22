

package edu.gatech.cse8803.phenotyping

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD

//import scala.tools.nsc.transform.patmat.Logic.PropositionalLogic.True

//import java.util.Date // adding the Date class in


object T2dmPhenotype {
  /**
    * Transform given data set to a RDD of patients and corresponding phenotype
    *
    * @param medication medication RDD
    * @param labResult lab result RDD
    * @param diagnostic diagnostic code RDD
    * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
    */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {

    val sc = medication.sparkContext

    /** Hard code the criteria */
    val type1_dm_dx = Set("code1", "250.01","250.03","250.11","250.13","250.21","250.23","250.31","250.33","250.41","250.43","250.51","250.53","250.61","250.63","250.71","250.73","250.81","250.83","250.91","250.93")
    val type1_dm_med = Set("med1", "lantus","insulin glargine","insulin aspart","insulin detemir","insulin lente","insulin nph","insulin reg","insulin,ultralente")

    val type2_dm_dx = Set("code2","250.3","250.32","250.2","250.22","250.9","250.92","250.8","250.82","250.7","250.72","250.6","250.62","250.5","250.52","250.4","250.42","250.00","250.02")
    val type2_dm_med = Set("med2","chlorpropamide","diabinese","diabanase","diabinase","glipizide","glucotrol","glucotrol XL","glucatrol ","glyburide","micronase","glynase","diabetamide","diabeta","glimepiride","amaryl","repaglinide","prandin","nateglinide","metformin","rosiglitazone","pioglitazone","acarbose","miglitol","sitagliptin","exenatide","tolazamide","acetohexamide","troglitazone","tolbutamide","avandia","actos","actos","glipizide")

    // Set set of all unique patient ids
    val all_patients = diagnostic.map{elt => elt.patientID}.distinct().collect().toSet

    // Get subset of diagnostic RDD containing patients with type 1 diagnosis
    val diag_with_type1 = diagnostic.filter{elt => type1_dm_dx.contains(elt.code)}

    // Get non-type1 patients set
    val non_type1_patients = all_patients diff diag_with_type1.map{elt => elt.patientID}.distinct().collect().toSet

    // Use above to get full subset of diagnostic containing data for patients without type 1 diagnosis
    val not_diag_type1 = diagnostic.filter{elt => non_type1_patients.contains(elt.patientID)}


    // continue filter; get patients with type 2 diagnosis
    val patients_with_type2 = diagnostic.filter{elt => type2_dm_dx.contains(elt.code)}.map{elt => elt.patientID}.distinct().collect().toSet
    val patients_after_2_filters = not_diag_type1.filter{elt => patients_with_type2.contains(elt.patientID)}


    // split patients into no / yes for type 1 medication
    val patients_with_med_type1 = medication.filter{elt => type1_dm_med.contains(elt.medicine)}.map{elt => elt.patientID}.distinct().collect().toSet
    val patients_no_med_type1 = all_patients diff patients_with_med_type1

    val patients_rdd_with_med_type1 = patients_after_2_filters.filter{elt => patients_with_med_type1.contains(elt.patientID)}
    val patients_rdd_no_med_type1 = patients_after_2_filters.filter{elt => patients_no_med_type1.contains(elt.patientID)} // map to case patients


    // Take patients with med type 1 and split on yes / no for medication type 2
    val patients_with_med_type2 = medication.filter{elt => type2_dm_med.contains(elt.medicine)}.map{elt => elt.patientID}.distinct().collect().toSet
    val patients_no_med_type2 = all_patients diff patients_with_med_type2

    val patients_rdd_with_med_type2 = patients_rdd_with_med_type1.filter(elt => patients_with_med_type2.contains(elt.patientID))
    val patients_rdd_no_med_type2 = patients_rdd_with_med_type1.filter(elt => patients_no_med_type2.contains(elt.patientID)) // map to case patients


    // Take patients with med type 2 and split on yes / no
    val patients_with_both_meds = patients_rdd_with_med_type2

    // get subsets of medication for type 1 med and type 2 meds only
    val type1_meds = medication.filter{elt => type1_dm_med.contains(elt.medicine)}
    val type2_meds = medication.filter{elt => type2_dm_med.contains(elt.medicine)}

    val patient_list_with_meds = patients_with_both_meds.map{elt => elt.patientID}.distinct().collect().toSet.toList

    type1_meds.cache()
    type2_meds.cache()


    //*****************************************************************************************************************
    // Use this function to filter where type 2 DM medication precedes type 1 DM medication
    def date_test_list(patient_id : String) : Boolean = {

      val a1 = type1_meds.filter(elt => elt.patientID == patient_id)
      val min_date1 = a1.map { elt => elt.date }.distinct().collect().min

      val a2 = type2_meds.filter(elt => elt.patientID == patient_id)
      val min_date2 = a2.map { elt => elt.date }.distinct().collect().min

      val date_test = min_date1.getTime - min_date2.getTime > 0

      date_test
    }


    val type2_precedes_type1_patient_list = patient_list_with_meds.filter{elt => date_test_list(elt)} // map to case patients
    val type2_precedes_type1_patient_rdd = medication.filter{elt => type2_precedes_type1_patient_list.contains(elt.patientID)} // map to case patients



    // Now, combine together all patients to be classified as case patients

    val no_med_type1 = patients_rdd_no_med_type1.map{elt => elt.patientID}.collect().toSet.toList
    val no_med_type2 = patients_rdd_no_med_type2.map{elt => elt.patientID}.collect().toSet.toList

    val case_patients_list = List.concat(no_med_type1,no_med_type2,type2_precedes_type1_patient_list)


    /** Find CASE Patients */

    val casePatients = sc.parallelize(case_patients_list.map{elt => (elt,1)})

    /*********** Get control patients ******************************/
    val patients_with_glucose = labResult.filter{elt => elt.testName.contains("glucose")}

    //val abnormal_1 = labResult.filter(elt => (elt.testName == "hba1c") )

    def test_abnormal(elt : LabResult) : Boolean = {

        if(elt.testName == "hba1c" && elt.value >= 6)
          {
              false
          }

        else if(elt.testName == "hemoglobin a1c" && elt.value >= 6)
          {
              false
          }
        else if(elt.testName == "fasting glucose" && elt.value >= 110)
          {
              false
          }

        else if(elt.testName == "fasting blood glucose" && elt.value >= 110)
          {
              false
          }
        else if (elt.testName == "fasting plasma glucose" && elt.value > 110)
          {
              false
          }

        else if(elt.testName == "glucose" && elt.value > 110)
          {
              false
          }

        else if(elt.testName == "glucose, serum" && elt.value > 110 )
          {
              false
          }

        else
          {
            true
          }
    }


    val patients_with_abnormal_lab_values = patients_with_glucose.filter{elt => test_abnormal(elt)}


    val dm_dx = Set("dm_dx_values","790.21","790.22","790.2","790.29","648.81","648.82","648.83","648.84","648","648","648.01","648.02","648.03","648.04","791.5","277.7","V77.1","256.4")
    val dm_dx_patients = diagnostic.filter{elt => (dm_dx.contains(elt.code)) || (elt.code.contains("250.0"))}.map{elt => elt.patientID}.collect().toSet

    val non_dm_dx_patients = all_patients diff dm_dx_patients


    val control_patients_data = patients_with_abnormal_lab_values.filter{elt => non_dm_dx_patients.contains(elt.patientID)}

    // get final list of all control patients
    val control_patients_list = control_patients_data.map{elt => elt.patientID}.distinct().collect().toList

    /** Find CONTROL Patients */
    val controlPatients = sc.parallelize(control_patients_list.map{elt => (elt,2)})

    // get other patients
    val other_patients_set = (all_patients diff case_patients_list.toSet) diff control_patients_list.toSet
    val other_patients_list = other_patients_set.toList


    /** Find OTHER Patients */
    val others = sc.parallelize(other_patients_list.map{elt => (elt,3)})

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }
}
