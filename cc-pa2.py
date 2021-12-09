# Created by Christopher Lewis for CloudComputing project assignment 2
# With help by https://piotrszul.github.io/spark-tutorial/notebooks/3.1_ML-Introduction.html

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.functions import *
from pyspark.ml.regression import RandomForestRegressor

spark = SparkSession \
	.builder \
	.appName("Cloud Computing PA2") \
	.config("spark.some.config.option", "some-value") \
	.getOrCreate()

# configure CSV split between training dataframe and test dataframe
csvData = spark.read.csv('winequality-white.csv', header='true', inferSchema='true', sep=';')
(trainingDF, testDF) = csvData.randomSplit([0.8, 0.2])

features = [c for c in csvData.columns if c != 'quality']

transformAssembler = VectorAssembler(inputCols=features, outputCol="features")

# The actual call to linear regression function that has customized hyperparameters
linearRegression = LinearRegression(maxIter=101, regParam=0.3, elasticNetParam=0.3, featuresCol="features", labelCol="quality")

# the ml model pipeline. Runs transformAssembler in one stage and linearRegression in the other
pipeline = Pipeline(stages=[transformAssembler, linearRegression])

# train the pipeline on the training dataframe
lrPipelineModel = pipeline.fit(trainingDF)

# make predictions
testPredictionsDF = lrPipelineModel.transform(testDF)

# create a search grid with the cross-product of the parameter values (9 pairs)
search_grid = ParamGridBuilder() \
	.addGrid(linearRegression.regParam, [0.1, 0.3, 0.6]) \
	.addGrid(linearRegression.elasticNetParam, [0.4, 0.6, 0.8]).build()

# The evaluator we use for RMSE analysis
regressionEvaluator = RegressionEvaluator(labelCol='quality', predictionCol="prediction", metricName="rmse")

#validates our answers to improve accuracy
validator = CrossValidator(estimator = pipeline, estimatorParamMaps = search_grid, evaluator = regressionEvaluator, numFolds = 3)

#trains the validator and then makes predictions
cvModel = validator.fit(trainingDF)
cvTestPredictionsDF = cvModel.transform(testDF)

#Output from validator
print("RMSE on test data with CV = %g" % regressionEvaluator.evaluate(cvTestPredictionsDF))

# define the random forest function with hyperparameters
rf = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=70, maxBins=128, maxDepth=10, \
	minInstancesPerNode=10, seed=33)

#define the pipeline for transformAssembler into random forest as the second/last stage
rfPipeline = Pipeline(stages=[transformAssembler, rf])

# train the random forest model
rfPipelineModel = rfPipeline.fit(trainingDF)

# make predictions on test data
rfTestPredictions = rfPipelineModel.transform(testDF)
print("Random Forest RMSE on test data = %g" % regressionEvaluator.evaluate(rfTestPredictions))