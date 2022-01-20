# Databricks notebook source
import os
import csv

seattle_crimes_csv = "/FileStore/tables/SPD_Crime_Data__2008_Present.csv"

df = spark.read.format("csv") \
  .option("inferSchema", True) \
  .option("header", True) \
  .load(seattle_crimes_csv).cache()

display(df)

# COMMAND ----------

sample_df = df.cache()#.sample(False, 0.1, seed=0)
display(sample_df)

# COMMAND ----------

# Visualize data
sample_df.select("Offense Parent Group").distinct().show()
sample_df.select("Offense").distinct().show()


# COMMAND ----------



# COMMAND ----------

# Count by offense parent group
count_by_pgroup_df = sample_df.groupBy("Offense Parent Group").count().orderBy("count", ascending=False)
display(count_by_pgroup_df)


# COMMAND ----------

# report time - start time 
from pyspark.sql.functions import col
from pyspark.sql.functions import *
sample_df = sample_df.withColumn('Hours to Report', (to_timestamp(col("Report DateTime"), "MM/dd/yyyy hh:mm:ss a").cast("long")\
                                         - to_timestamp(col("Offense Start Datetime"), "MM/dd/yyyy hh:mm:ss a").cast("long")) / 3600)
# 
display(sample_df.groupBy("Offense Parent Group").avg("Hours to Report").orderBy("avg(Hours to Report)", ascending=False), truncate=False)

sample_df.cache()

# COMMAND ----------

import datetime
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql import functions as F

# Select relevant columns
samp_df = sample_df.select('Offense Start DateTime', 'Offense End DateTime', 'Report DateTime', 'Crime Against Category', 'Offense Parent Group',
                           'Offense', 'Offense Code', 'Precinct', 'Sector', 'Beat', 'MCPP', '100 Block Address', 'Longitude', 'Latitude', 'Hours to Report')

# drop nulls 
samp_df = samp_df.dropna()

# convert to timestamps, calculate
samp_df = samp_df.withColumn('hourOffense', dayofmonth(to_timestamp(col("Offense Start DateTime"), "MM/dd/yyyy hh:mm:ss a")))
samp_df = samp_df.withColumn('monthOffense',month(to_timestamp(col("Offense Start DateTime"), "MM/dd/yyyy hh:mm:ss a")))
samp_df = samp_df.withColumn('yearOffense', year(to_timestamp(col("Offense Start DateTime"), "MM/dd/yyyy hh:mm:ss a")))
samp_df = samp_df.select('hourOffense', 'monthOffense', 'yearOffense', 'Hours to Report', 'Crime Against Category', 'MCPP', 'Longitude', 'Latitude', 'Offense Parent Group')
                    
cols = samp_df.columns
# samp_df.printSchema()

# COMMAND ----------

display(samp_df)

# COMMAND ----------

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer
# from pyspark.ml.feature import OneHotEncoder

# samp_df = samp_df.withColumnRenamed("Offense Parent Group", "Label")


# # Leave ordinal numerical features as is
# # For nominal features (no relationship, e.g. cat dog house), use StringIndexer (if string) then OHE

# # String Indexer 
# c_si = StringIndexer(inputCol="Crime Against Category", outputCol="Crime Against Category Indexed")
# mcpp_si = StringIndexer(inputCol="MCPP", outputCol="MCPP Indexed")
# l_si = StringIndexer(inputCol="Label", outputCol="Label Indexed")

# # One-hot encoding
# mcpp_ohe = OneHotEncoder(inputCol="MCPP Indexed", outputCol="MCPP OHE")
# c_ohe = StringIndexer(inputCol="Crime Against Category Indexed", outputCol="Crime Against Category OHE")


# #Create pipeline and pass all stages
# pipeline = Pipeline(stages=[c_si, mcpp_si, l_si, mcpp_ohe, c_ohe])
# # Transform
# transformed_df = pipeline.fit(samp_df).transform(samp_df)

# COMMAND ----------


# String Indexer 
c_si = StringIndexer(inputCol="Crime Against Category", outputCol="Crime Against Category Indexed")
mcpp_si = StringIndexer(inputCol="MCPP", outputCol="MCPP Indexed")

# One-hot encoding
mcpp_ohe = OneHotEncoder(inputCol="MCPP Indexed", outputCol="MCPP OHE")
c_ohe = StringIndexer(inputCol="Crime Against Category Indexed", outputCol="Crime Against Category OHE")


#Create pipeline and pass all stages
pipeline = Pipeline(stages=[c_si, mcpp_si, mcpp_ohe, c_ohe])
# Transform
samp_df = pipeline.fit(samp_df).transform(samp_df)
# Remove intermediate cols
samp_df = samp_df.drop("Crime Against Category", "MCPP",  \
                  "Crime Against Category Indexed", "MCPP Indexed")
# Rename columns 
samp_df = samp_df.withColumnRenamed("MCPP OHE", "MCPP")
samp_df = samp_df.withColumnRenamed("Crime Against Category OHE", "Crime Against Category")

samp_df.show()
# transformed_df = transformed_df.
# transformed_df.show()

# COMMAND ----------


def resample(base_features,ratio,class_field,base_class):
    pos = base_features.filter(col(class_field)==base_class)
    neg = base_features.filter(col(class_field)!=base_class)
    total_pos = pos.count()
    total_neg = neg.count()
    sampled = pos.sample(False, ratio)
    return sampled.union(neg)
samp_df = resample(samp_df, 0.3, "Offense Parent Group", "LARCENY-THEFT")
display(samp_df)
  

# COMMAND ----------


cdf = samp_df.groupBy("Offense Parent Group").count().orderBy("count", ascending=False)
display(cdf)


# COMMAND ----------


# Vector assembler, input: cols, output: single "features" col 
# Vector assembler cannot take strings
from pyspark.ml.feature import VectorAssembler
samp_df = samp_df.withColumnRenamed("Offense Parent Group", "Label")

cols=samp_df.columns
cols.remove("Label")

assembler = VectorAssembler(inputCols=cols,outputCol="features")

# transform dataset 
transformed_df = assembler.transform(samp_df)
transformed_df.select("features").show(truncate=False)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="Label", outputCol="indexedLabel").fit(transformed_df)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(transformed_df)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = transformed_df.randomSplit([0.7, 0.3])

# Train a RandomForest model.
# rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
# Train a LogisticRegression model
mod = LogisticRegression(labelCol="indexedLabel", featuresCol="indexedFeatures",maxIter=15, family="multinomial")
# 
# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

# COMMAND ----------

# Remove intermediate cols
# transformed_df = transformed_df.drop("Crime Against Category", "MCPP", "Offense Parent Group", \
#                                      "Crime Against Category Indexed", "MCPP Indexed")
# transformed_df.show()

# COMMAND ----------

# Rename columns 
transformed_df = transformed_df.withColumnRenamed("MCPP OHE", "MCPP")
transformed_df = transformed_df.withColumnRenamed("Crime Against Category OHE", "Crime Against Category")
# transformed_df = transformed_df.withColumnRenamed("Offense Parent Group OHE", "Label")


# COMMAND ----------


# Vector assembler, input: cols, output: single "features" col 
# Vector assembler cannot take strings
from pyspark.ml.feature import VectorAssembler

cols=transformed_df.columns
cols.remove("Label")

assembler = VectorAssembler(inputCols=cols,outputCol="features")

# transform dataset 
transformed_df = assembler.transform(transformed_df)
transformed_df.select("features").show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# Scale/normalize features 
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
transformed_df=standardscaler.fit(transformed_df).transform(transformed_df)
transformed_df.select("features","Scaled_features").show(5)

# COMMAND ----------

train, test = transformed_df.randomSplit([0.8, 0.2], seed=12345)
print('Train size: ', train.count())
print('Test size: ', test.count())


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Train with logistic regression
lr = LogisticRegression(labelCol="Label", featuresCol="features",maxIter=15)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
predict_test.select("Label","prediction").show(10)

# COMMAND ----------


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# COMMAND ----------

predict_test.select("Label","prediction").show(200)

# COMMAND ----------

display(predict_test, truncate=False)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction")

# Make predicitons
predictionAndLabel = predict_test.select("Label", "prediction")

# Get metrics
acc = evaluatorMulti.evaluate(predictionAndLabel, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndLabel, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndLabel, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndLabel, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluatorMulti.evaluate(predictionAndLabel)

# COMMAND ----------

print('acc', acc)
print('f1', f1)
print('weightedPrecision', weightedPrecision)
print('weightedRecall', weightedRecall)
print('auc', auc)

# COMMAND ----------


