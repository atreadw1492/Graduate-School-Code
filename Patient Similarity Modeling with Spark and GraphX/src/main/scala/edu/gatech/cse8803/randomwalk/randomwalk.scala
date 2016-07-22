package edu.gatech.cse8803.randomwalk


import edu.gatech.cse8803.model.{PatientProperty, EdgeProperty, VertexProperty}
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {

    val rankedGraph = runWithOptions(graph , numIter, alpha, patientID)
    val rankedVertices = rankedGraph.vertices

    val patientsRDD = graph.vertices.filter{elt => elt._2.getClass.getName == "edu.gatech.cse8803.model.PatientProperty"}
    val temp_patients = patientsRDD.map{elt => elt._1}
    val patientsList = temp_patients.collect.toList

    val rankedMap = rankedVertices.collect.toList.toMap

    val result = patientsList.map{elt => (elt , rankedMap(elt))}.sortBy(-_._2).map{elt => elt._1}.filter{elt => elt != patientID}


    result
  }


  def runWithOptions(graph: Graph[VertexProperty, EdgeProperty], numIter: Int, resetProb: Double = 0.15,
                                                  src: Long): Graph[Double, Double] =
  {

    var rankGraph: Graph[Double, Double] = graph
      // Associate the degree with each vertex
      .outerJoinVertices(graph.outDegrees) { (vid, vdata, deg) => deg.getOrElse(0) }
      // Set the weight on the edges based on the degree
      .mapTriplets( e => 1.0 / e.srcAttr, TripletFields.Src )
      // Set the vertex attributes to the initial pagerank values
      .mapVertices { (id, attr) =>
      if (!(id != src)) resetProb else 0.0
    }

    def delta(u: VertexId, v: VertexId): Double = { if (u == v) 1.0 else 0.0 }

    var iteration = 0
    var prevRankGraph: Graph[Double, Double] = null
    while (iteration < numIter) {
      rankGraph.cache()

      val rankUpdates = rankGraph.aggregateMessages[Double](
        ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr), _ + _, TripletFields.Src)


      prevRankGraph = rankGraph
      val rPrb = if (1 > 0) {
        (src: VertexId, id: VertexId) => resetProb * delta(src, id)
      } else {
        (src: VertexId, id: VertexId) => resetProb
      }

      rankGraph = rankGraph.joinVertices(rankUpdates) {
        (id, oldRank, msgSum) => rPrb(src, id) + (1.0 - resetProb) * msgSum
      }.cache()

      rankGraph.edges.foreachPartition(x => {}) // also materializes rankGraph.vertices
      prevRankGraph.vertices.unpersist(false)
      prevRankGraph.edges.unpersist(false)

      iteration += 1
    }

    rankGraph
  }






}
