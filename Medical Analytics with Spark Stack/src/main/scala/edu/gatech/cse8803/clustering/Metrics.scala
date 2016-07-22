

package edu.gatech.cse8803.clustering

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   *             \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return purity
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */

    val N = clusterAssignmentAndLabel.count.toDouble
    val K =  clusterAssignmentAndLabel.map{elt => elt._1}.distinct.collect.toList

    val all_classes = clusterAssignmentAndLabel.map{elt => elt._2}.distinct.collect()

    def get_max_of_classes(cluster_subset : RDD[(Int, Int)]) : Double = {
      all_classes.map{elt => cluster_subset.filter{y => y._2 == elt}.count}.max
    }

    //val all_clusters = K.map{k => clusterAssignmentAndLabel.filter{elt => elt._1 == k}}
    //val each_iteration = K.map{elt => get_max_of_classes()}

    val each_iteration = K.map{k => clusterAssignmentAndLabel.filter{elt => elt._1 == k}}.map{elt => get_max_of_classes(elt)}

    val result = each_iteration.sum / N

    result

  }

  def get_confusion_matrix(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {

    //val N = clusterAssignmentAndLabel.count.toDouble
    val K =  clusterAssignmentAndLabel.map{elt => elt._1}.distinct.collect.toList.sorted

    val all_classes = clusterAssignmentAndLabel.map{elt => elt._2}.distinct.collect().toList.sorted

    val cluster_labels = clusterAssignmentAndLabel.collect.toList

    //val class_to_cluster_map =  (all_classes , K).zipped.toMap

    val by_class = all_classes.map{elt => cluster_labels.filter{x => x._2 == elt}}

    val matrix1 = by_class(0)
    val matrix2 = by_class(1)
    val matrix3 = by_class(2)

    val confusion_list =  List(K.map{k => matrix1.count(_._1==k)},K.map{k => matrix2.count(_._1==k)},K.map{k => matrix3.count(_._1==k)})

    val col1 = List(0,1,2).map{elt => confusion_list(elt)(0)}
    val col2 = List(0,1,2).map{elt => confusion_list(elt)(1)}
    val col3 = List(0,1,2).map{elt => confusion_list(elt)(2)}



    println("Cluster 1")
    println(col1.map{elt => elt / col1.sum.toDouble})
    println("\n\nCluster 2")
    println(col2.map{elt => elt / col2.sum.toDouble})
    println("\n\nCluster 3")
    println(col3.map{elt => elt / col3.sum.toDouble})
    println("\n\n")


    val result = 0.0

    result

  }

}
