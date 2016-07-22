
package edu.gatech.cse8803.graphconstruct

import edu.gatech.cse8803.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object GraphLoader {
  /** Generate Bipartite Graph using RDDs
    *
    * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
    * @return: Constructed Graph
    *
    * */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
           medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {


    val sc = patients.sparkContext

    val vertexPatient: RDD[(VertexId, VertexProperty)] = patients.map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))

    val p2v = vertexPatient.map{elt => elt._1}.collect.toList

    val patient2VertexId = p2v.zip(p2v).toMap
    //val patient2VertexId = vertexPatient.map{elt => elt._1}.collect.toList.zipWithIndex.toMap




    // get size of vertexPatient
    val startDiagIndex = patient2VertexId.values.max + 1

    /** Create diagnostic vertex */
    val diagnosticVertexIdRDD = diagnostics.
      map(_.icd9code).
      distinct.
      zipWithIndex.
      map{case(icd9code, zeroBasedIndex) =>
        (icd9code, zeroBasedIndex + startDiagIndex)} // make sure no conflict with patient vertex id

    val diagnostic2VertexId = diagnosticVertexIdRDD.collect.toMap

    val diagnosticVertex = diagnosticVertexIdRDD.
      map{case(icd9code, index) => (index, DiagnosticProperty(icd9code))}.
      asInstanceOf[RDD[(VertexId, VertexProperty)]]


    // add size of diagnostic vertex to patient vertex
    val startLabIndex = diagnostic2VertexId.values.max + 1
    /** Create lab result vertex */

    val labVertexIdRDD = labResults.
      map(_.labName).
      distinct.
      zipWithIndex.
      map{case(labName, zeroBasedIndex) =>
        (labName, zeroBasedIndex + startLabIndex)}.cache // make sure no conflict with patient vertex id

    val lab2VertexId = labVertexIdRDD.collect.toMap

    val labVertex = labVertexIdRDD.
      map{case(labName, index) => (index, LabResultProperty(labName))}.
      asInstanceOf[RDD[(VertexId, VertexProperty)]].cache


    /** Create medication vertex
      */

    val startMedIndex = lab2VertexId.values.max + 1

    val medicationVertexIdRDD = medications.
      map(_.medicine).
      distinct.
      zipWithIndex.
      map{case(medicine, zeroBasedIndex) =>
        (medicine, zeroBasedIndex + startMedIndex)}.cache // make sure no conflict with patient vertex id

    val medicine2VertexId = medicationVertexIdRDD.collect.toMap

    val medicineVertex = medicationVertexIdRDD.
      map{case(medicine, index) => (index, MedicationProperty(medicine))}.
      asInstanceOf[RDD[(VertexId, VertexProperty)]].cache


    // Create edges **************************************************************************************//

    // filter diagnostics
    val diag_max_date = diagnostics.map{elt => ((elt.patientID , elt.icd9code) , elt)}.reduceByKey((v1,v2) => { if( v1.date > v2.date ) v1 else v2 }).map{ case (key, elt) => elt }.cache


    val lab_max_date = labResults.map{elt => ((elt.patientID , elt.labName) , elt)}.reduceByKey((v1,v2) => { if( v1.date > v2.date ) v1 else v2 }).map{ case (key, elt) => elt }.cache

    val med_max_date = medications.map{elt => ((elt.patientID , elt.medicine) , elt)}.reduceByKey((v1,v2) => { if( v1.date > v2.date ) v1 else v2 }).map{ case (key, elt) => elt }.cache

    // create edges
    val patientDiagEdges = diag_max_date.
      map{elt => Edge(
        patient2VertexId(elt.patientID.toLong), // src id
        diagnostic2VertexId(elt.icd9code), // target id
        PatientDiagnosticEdgeProperty(Diagnostic(elt.patientID, elt.date, elt.icd9code, elt.sequence)).asInstanceOf[EdgeProperty] // edge property
      )}

    val patientMedEdges = med_max_date.
      map{elt => Edge(
        patient2VertexId(elt.patientID.toLong), // src id
        medicine2VertexId(elt.medicine), // target id
        PatientMedicationEdgeProperty(Medication(elt.patientID, elt.date, elt.medicine)).asInstanceOf[EdgeProperty] // edge property
      )}

    val patientLabEdges = lab_max_date.
      map{elt => Edge(
        patient2VertexId(elt.patientID.toLong), // src id
        lab2VertexId(elt.labName), // target id
        PatientLabEdgeProperty(LabResult(elt.patientID, elt.date, elt.labName, elt.value)).asInstanceOf[EdgeProperty] // edge property
      )}



    val diagPatientEdges = diag_max_date.
      map{elt => Edge(
        diagnostic2VertexId(elt.icd9code), // src id
        patient2VertexId(elt.patientID.toLong), // target id
        PatientDiagnosticEdgeProperty(Diagnostic(elt.patientID, elt.date, elt.icd9code, elt.sequence)).asInstanceOf[EdgeProperty] // edge property
      )}

    val medPatientEdges = med_max_date.
      map{elt => Edge(
        medicine2VertexId(elt.medicine), // src id
        patient2VertexId(elt.patientID.toLong), // target id
        PatientMedicationEdgeProperty(Medication(elt.patientID, elt.date, elt.medicine)).asInstanceOf[EdgeProperty] // edge property
      )}

    val labPatientEdges = lab_max_date.
      map{elt => Edge(
        lab2VertexId(elt.labName), // src id
        patient2VertexId(elt.patientID.toLong), // target id
        PatientLabEdgeProperty(LabResult(elt.patientID, elt.date, elt.labName, elt.value)).asInstanceOf[EdgeProperty] // edge property
      )}


      // combine vertices
      val vertices = sc.union(vertexPatient , labVertex, medicineVertex, diagnosticVertex)
      val edges = sc.union(patientDiagEdges , patientLabEdges, patientMedEdges, diagPatientEdges, labPatientEdges, medPatientEdges)


    // Making Graph
    val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertices, edges)
    //val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertexPatient, edgePatientPatient)

    graph
  }


  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Three Application", "local")



}
