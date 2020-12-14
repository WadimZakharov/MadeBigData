package org.apache.spark.ml

import org.apache.spark.ml.feature.{LSH, LSHModel, MinHashLSH, MinHashLSHModel}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util.{Identifiable, MLWriter, SchemaUtils}
import org.apache.spark.sql.types.StructType

import scala.util.Random


class CosLSH(override val uid: String) extends LSH[CosLSHModel] with HasSeed {

  
  override def setInputCol(value: String): this.type = super.setInputCol(value)

  
  override def setOutputCol(value: String): this.type = super.setOutputCol(value)

  
  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  
  def this() = {
    this(Identifiable.randomUID("cos_lsh"))
  }
  
  
  def setSeed(value: Long): this.type = set(seed, value)
  
  override protected[ml] def createRawLSHModel(inputDim: Int): CosLSHModel = {
    val rand = new Random(0)
    val randHyperPlanes: Array[Vector] = {
      Array.fill($(numHashTables)) {
        val randArray = Array.fill(inputDim)({if (rand.nextGaussian() > 0) 1.0 else -1.0})
        linalg.Vectors.fromBreeze(breeze.linalg.Vector(randArray))
      }
    }
    new CosLSHModel(uid, randHyperPlanes)
  }

  
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }

  
  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}


class CosLSHModel private[ml](
                                   override val uid: String,
                                   private[ml] val randHyperPl: Array[Vector])
  extends LSHModel[CosLSHModel] {

  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  private[ml] def this(randHyperPl: Array[Vector]) =
    this(Identifiable.randomUID("cos_lsh"), randHyperPl)

  override protected[ml] def hashFunction(elems: linalg.Vector): Array[linalg.Vector] = {
    val hash = randHyperPl.map {randHyperPl => if (elems.dot(randHyperPl) >= 0) 1 else -1}
    hash.map(Vectors.dense(_))
  }

  override protected[ml] def keyDistance(x: linalg.Vector, y: linalg.Vector): Double = {
    if (Vectors.norm(x, 2) == 0 || Vectors.norm(y, 2) == 0) {1.0}
    else {1.0 - x.dot(y) / (Vectors.norm(x, 2) * Vectors.norm(y, 2))}
  }

  override protected[ml] def hashDistance(x: Seq[linalg.Vector], y: Seq[linalg.Vector]): Double = {
    x.zip(y).map(value => if (value._1 != value._2) 0 else 1).sum.toDouble / x.size
  }

  override def copy(extra: ParamMap): CosLSHModel = {
    val copied = new CosLSHModel(uid, randHyperPl).setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = ???
}
