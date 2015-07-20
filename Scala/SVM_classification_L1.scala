import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by Tanmay on 6/19/2015.
 */
object SVM_classification_L1 {
  def main(args: Array[String]) {


  val sc = new SparkContext("local[2]", "SVM_L1")
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
  val training = splits(0).cache()
  val test = splits(1)


    //Building the Model

  val model = new SVMWithSGD()
  model.optimizer.setRegParam(0.001).setNumIterations(50).setStepSize(1).setUpdater(new L1Updater)

  val SVMModelSGD = model.run(training)


  //Accuracy training
  val lrTotalCorrect = training.map { point =>
    if (SVMModelSGD.predict(point.features) == point.label) 1 else 0
  }.sum
  val lrAccuracy = lrTotalCorrect / training.count


  //Accuracy test
  val lrTotalCorrect1 = test.map { point =>
    if (SVMModelSGD.predict(point.features) == point.label) 1 else 0
  }.sum
  val lrAccuracy1 = lrTotalCorrect1 / test.count


  //Area under ROC training
  SVMModelSGD.clearThreshold()

  val scoreAndLabels = training.map { point =>
    val score = SVMModelSGD.predict(point.features)
    (score, point.label)
  }

  // Get evaluation metrics.
  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  val auROC = metrics.areaUnderROC()


  //Area under PR

  val auPR = metrics.areaUnderPR()


  //Area under ROC test


  val scoreAndLabels1 = test.map { point =>
    val score1 = SVMModelSGD.predict(point.features)
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
  print(" Accuracy training= " + lrAccuracy)
  print(" Accuracy test= " + lrAccuracy1)



  sc.stop()


}
}
