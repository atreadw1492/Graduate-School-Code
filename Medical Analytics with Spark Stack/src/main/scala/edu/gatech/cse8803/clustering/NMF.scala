package edu.gatech.cse8803.clustering



import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices


object NMF {

  /**
   * Run NMF clustering
    *
    * @param V The original non-negative matrix
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrices W and H in RowMatrix and DenseMatrix format respectively
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {


    // Initialize W and H
    val W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache)
    val H = BDM.rand[Double](k, V.numCols().toInt)

    // Calculate W * H
    val M = multiply(W,H)

    // Get initial error between V and M
    val M_V_diff = dotSubtract(M,V)
    val diff_squared = dotProd(M_V_diff,M_V_diff)

    val x = diff_squared.rows.map{elt => elt}.map{x => x.toArray.toList}

    val error = x.collect.toList.flatten.sum



    //******************************************************************************************************



    def update_H(W: RowMatrix ,V: RowMatrix, H: BDM[Double] ): BDM[Double] = {
      val WTV = computeWTV(W, V)

      val H_update_numerator = H :* WTV
      val WTW = computeWTV(W, W)
      val H_update_bottom = WTW * H

      val H_update = H_update_numerator :/ H_update_bottom
      H_update
    }

    //*************************************************************************************************


    def update_W(W: RowMatrix ,V: RowMatrix, H: BDM[Double] ): RowMatrix = {

      val VHT = multiply(V, H.t)
      val W_numerator = dotProd(W, VHT)
      val WH = multiply(W, H)
      val W_bottom = multiply(WH, H.t)

      val W_update = dotDiv(W_numerator, W_bottom)

      W_update
    }


    var index = 1
    var H_new = H
    var W_new = W

    while(index <= 100)
      {
        W_new = update_W(W_new,V,H_new)
        W_new.rows.cache.count
        H_new = update_H(W_new,V,H_new)

        index = index + 1
      }




    // Return W and H
    (W_new,H_new)

  }



  /** compute the mutiplication of a RowMatrix and a dense matrix */
  private def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {


    val d_flatten = (0 to d.cols - 1).toList.map{colVar => (0 to d.rows - 1).toList.map{rowVar => d(rowVar,colVar)}}.flatten.toArray
    val d_matrix = Matrices.dense(d.rows,d.cols,d_flatten)

    val result = X.multiply(d_matrix)

    result

  }

 /** get the dense matrix representation for a RowMatrix */
  private def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    null
  }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {

   val denseRows = W.rows.zip(V.rows).map{ case(w: Vector, v: Vector) => BDM.create(w.size,1,w.toArray) * BDM.create(1,v.size,v.toArray)}.reduce(_+_)

    denseRows
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  def dotSubtract(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :- toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }


}