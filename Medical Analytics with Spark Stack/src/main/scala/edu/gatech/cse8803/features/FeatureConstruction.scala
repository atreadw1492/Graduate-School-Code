
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model.{LabResult, Medication, Diagnostic}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {


    diagnostic.cache()

    //val all_patients = diagnostic.map{elt => elt.patientID}
    val patients_with_diag = diagnostic.map{elt => (elt.patientID , elt.code)}
    //val diagnostic_counts = patients_with_diag.map(elt => elt).mapValues(_.size)
    val diagnostic_counts = patients_with_diag.countByValue

    val diagnostic_list = (diagnostic_counts.keys.toList zip diagnostic_counts.values.map{elt => elt.toDouble}.toList).toMap.toList

    diagnostic.sparkContext.parallelize(diagnostic_list)


  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {


    val patients_with_med = medication.map{elt => (elt.patientID , elt.medicine)}
    val med_counts = patients_with_med.countByValue

    val medication_list = (med_counts.keys.toList zip med_counts.values.map{elt => elt.toDouble}.toList).toMap.toList

    medication.sparkContext.parallelize(medication_list)


    //medication.sparkContext.parallelize(List((("patient", "med"), 1.0)))
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */

    val patients_with_test = labResult.map{elt => ((elt.patientID , elt.testName), elt.value)}
    val patients_with_test_list = patients_with_test.collect().toList

    def sumByKeys[A](tuples: List[(A, Double)]): List[(A, Double)] =
      tuples groupBy (_._1) map { case (k, v) => (k, v.map(_._2).sum) } toList


    val lab_sums = sumByKeys(patients_with_test_list).toMap
    val lab_counts = labResult.map{elt => (elt.patientID , elt.testName)}.countByValue

    val lab_avgs = lab_sums.keys.toList.map{elt => (elt , lab_sums(elt) / lab_counts(elt))}


    labResult.sparkContext.parallelize(lab_avgs)
    //labResult.sparkContext.parallelize(List((("patient", "lab"), 1.0)))
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   * @param diagnostic RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {

    diagnostic.cache()

    //val all_patients = diagnostic.map{elt => elt.patientID}
    val filtered_patients = diagnostic.filter{elt => candiateCode.contains(elt.code)}
    val patients_with_diag = filtered_patients.map{elt => (elt.patientID , elt.code)}

    val diagnostic_counts = patients_with_diag.countByValue

    val diagnostic_list = (diagnostic_counts.keys.toList zip diagnostic_counts.values.map{elt => elt.toDouble}.toList).toMap.toList

    diagnostic.sparkContext.parallelize(diagnostic_list)

    //diagnostic.sparkContext.parallelize(List((("patient", "diagnostics"), 1.0)))
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   * @param medication RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */

    val filtered_patients = medication.filter{elt => candidateMedication.contains(elt.medicine)}
    val patients_with_med = filtered_patients.map{elt => (elt.patientID , elt.medicine)}
    val med_counts = patients_with_med.countByValue

    val medication_list = (med_counts.keys.toList zip med_counts.values.map{elt => elt.toDouble}.toList).toMap.toList

    medication.sparkContext.parallelize(medication_list)


    //medication.sparkContext.parallelize(List((("patient", "med"), 1.0)))
  }


  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   * @param labResult RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */

    val filtered_patients = labResult.filter{elt => candidateLab.contains(elt.testName)}
    val patients_with_test = filtered_patients.map{elt => ((elt.patientID , elt.testName), elt.value)}
    val patients_with_test_list = patients_with_test.collect().toList

    def sumByKeys[A](tuples: List[(A, Double)]): List[(A, Double)] =
      tuples groupBy (_._1) map { case (k, v) => (k, v.map(_._2).sum) } toList


    val lab_sums = sumByKeys(patients_with_test_list).toMap
    val lab_counts = labResult.map{elt => (elt.patientID , elt.testName)}.countByValue

    val lab_avgs = lab_sums.keys.toList.map{elt => (elt , lab_sums(elt) / lab_counts(elt))}


    labResult.sparkContext.parallelize(lab_avgs)


    //labResult.sparkContext.parallelize(List((("patient", "lab"), 1.0)))
  }


  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   * @param sc SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map*/
    val feature_list = feature.collect().toList
    val featureMap = feature_list.map{elt => elt._1._2}.distinct.zipWithIndex.toMap

    val numFeature = featureMap.size



    val feature_with_id = feature.map{elt => (elt._1._1 , (featureMap(elt._1._2) , elt._2))}.collect.toList

    val feature_groups = feature_with_id.groupBy(_._1)

    val all_patients = feature_groups.keys.toList

    val feature_formatted = all_patients.map{p => (p , feature_groups(p).map{elt => elt._2})}

    val vector_list = feature_formatted.map{x => (x._1 , Vectors.sparse(numFeature , x._2))}

    val result = sc.parallelize(vector_list)

    result

  }
}


