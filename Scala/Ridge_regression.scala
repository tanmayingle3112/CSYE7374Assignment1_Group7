import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, LabeledPoint}

/**
 * Created by Tanmay on 6/20/2015.
 */
object Ridge_regression {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "Ridge")

    // Load and parse the data

    val data = sc.textFile("data\\WineRegression.csv")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()





    //splitting into training and test dataset
    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)




    // Building the model

    val model1 = new RidgeRegressionWithSGD()

    model1.optimizer.setStepSize(0.0005).setNumIterations(250).setRegParam(0)

    val model = model1.run(training)


    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }


    //testing the model

    val valuesAndPreds1 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }



    println(model.weights)


    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Root Mean Squared Error = " + math.sqrt(MSE))

    val MSE1 = valuesAndPreds1.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("test Root Mean Squared Error = " + math.sqrt(MSE1))



    sc.stop()

  }

}
