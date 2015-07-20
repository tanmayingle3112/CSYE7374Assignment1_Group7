import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

/**
 * Created by Tanmay on 6/19/2015.
 */
object LogRegression_LBFGS {
  def main(args: Array[String]) {

    val sc = new SparkContext("local[2]", "LogRegressionLBFGS")
    val rawData = sc.textFile("data\\WineClassification.csv")



    //converting data to LabeledPoint RDD
    val Data1 = rawData.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }.cache()


    val vectors = Data1.map(lp => lp.features)

    val scaler = new StandardScaler(withMean = true, withStd =
      true).fit(vectors)
    val parsedData = Data1.map(lp => LabeledPoint(lp.label,
      scaler.transform(lp.features)))


    //splitting into training and test dataset
    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training1 = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()
    val training = splits(0).cache()
    val test = splits(1)





    //Building the Model

    val numFeatures = Data1.take(1)(0).features.size



    val numCorrections = 10
    val convergenceTol = 1e-4
    val maxNumIterations = 20
    val regParam = 0.01
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))

    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
      training1,
      new LogisticGradient(),
      new SquaredL2Updater(),
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeightsWithIntercept)

    val lrModel = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))






    //Accuracy training
    val lrTotalCorrect = training.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracy = lrTotalCorrect / training.count







    //Accuracy test
    val lrTotalCorrect1 = test.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracy1 = lrTotalCorrect1/test.count




    //Area under ROC training
    lrModel.clearThreshold()

    val scoreAndLabels = training.map { point =>
      val score = lrModel.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()



    //Area under PR

    val auPR = metrics.areaUnderPR()








    //Area under ROC test


    val scoreAndLabels1 = test.map { point =>
      val score1 = lrModel.predict(point.features)
      (score1, point.label)
    }

    // Get evaluation metrics.
    val metrics1 = new BinaryClassificationMetrics(scoreAndLabels1)
    val auROC1 = metrics1.areaUnderROC()



    //Area under PR

    val auPR1 = metrics1.areaUnderPR()






    println("Area under ROC training= " + auROC)
    println("Area under ROC test= " + auROC1)
    println("Area under PR training= " + auPR)
    println("Area under PR test= " + auPR1)
    print(" Accuracy training= " + lrAccuracy  )
    print(" Accuracy test= " + lrAccuracy1  )






    sc.stop()
  }
}
