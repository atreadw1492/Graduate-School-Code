
package edu.gatech.cse8803.jaccard

import edu.gatech.cse8803.model._
import edu.gatech.cse8803.model.{EdgeProperty, VertexProperty}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {


    // get all patient ids
    val patientsRDD = graph.vertices.filter{elt => elt._2.getClass.getName == "edu.gatech.cse8803.model.PatientProperty"}
    val temp_patients = patientsRDD.map{elt => elt._1}
    val patientsList = temp_patients.collect.toList

    val neighbors = graph.collectNeighborIds(EdgeDirection.Out).filter{elt => elt._1.toLong <= patientsList.max}
    neighbors.cache() //store neighbors in memory for efficiency

    //get subset of neighbors pertaining to input patientID

    val x = neighbors.filter{elt => elt._1 == patientID}.collect
    val starter = x.toList(0)._2.toSet

    val all_sets = patientsList.map{pID => neighbors.filter{elt => elt._1 == pID}.collect.toList(0)._2.toSet}

    val result = all_sets.map{elt =>
      if(elt.union(starter).size != 0){
      elt.intersect(starter).size.toFloat / elt.union(starter).size}
      else 0}


    // need to map results to patient ids
    val result2id = patientsList.zip(result)

    val closest = result2id.sortBy(-_._2).map{elt => elt._1}

    closest.filter{elt => elt != patientID}
    //neighbors.map{elt => e}

  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {

    val sc = graph.edges.sparkContext

    // get all patient ids
    val patientsRDD = graph.vertices.filter{elt => elt._2.getClass.getName == "edu.gatech.cse8803.model.PatientProperty"}
    val temp_patients = patientsRDD.map{elt => elt._1}
    val patientsList = temp_patients.collect.toList

    val neighbors = graph.collectNeighborIds(EdgeDirection.Either).filter{elt => elt._1.toLong <= patientsList.max}
    neighbors.cache() //store neighbors in memory for efficiency



    //val x = graph.edges.cartesian(graph.edges)
    val x = temp_patients.cartesian(temp_patients) //get all possible patient1-patient2 combinations

    val pairs = x.filter{elt => elt._1 < elt._2} // filter out duplicates
    pairs.cache() // store pairs in memory

    val neighborMap = neighbors.map{elt => (elt._1 , elt._2.toSet)}.collect.toMap

    val result = pairs.map{ elt => (elt ,
          if (neighborMap(elt._1).union(neighborMap(elt._2)).size != 0)
            {neighborMap(elt._1).intersect(neighborMap(elt._2)).size.toFloat / neighborMap(elt._1).union(neighborMap(elt._2)).size}
          else 0.0)}

    //result
    val finalResult = result.map{elt => (elt._1._1.toLong , elt._1._2.toLong , elt._2.toDouble)}.sortBy(-_._3)

    finalResult
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /** 
    Helper function

    Given two sets, compute its Jaccard similarity and return its result.
    If the union part is zero, then return 0.
    */

    val result = a.union(b).size / a.intersect(b).size

    result
  }
}
