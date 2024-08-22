from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Sentiment Analysis with Spark MLlib") \
    .getOrCreate()

# Load and Prepare Data
data = spark.read.csv("sentiment_data.csv", header=True, inferSchema=True)
data.show(5)

# Preprocessing: Tokenization, Stop Words Removal, TF-IDF
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Define the Model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create a Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# Split the Data into Training and Test Sets
(trainingData, testData) = data.randomSplit([0.8, 0.2], seed=1234)

# Train the Model
model = pipeline.fit(trainingData)

# Evaluate the Model
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy = {accuracy:.2f}")

# Perform Sentiment Analysis on New Data
new_data = spark.createDataFrame([
    (0, "I love the new design of your website!"),
    (1, "The product quality has really dropped."),
], ["id", "text"])

new_predictions = model.transform(new_data)
new_predictions.select("id", "text", "prediction").show()

# Stop the Spark Session
spark.stop()
