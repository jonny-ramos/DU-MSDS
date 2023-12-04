# Databricks notebook source
# Jonathan Ramos
# COMP4334 Assignment 2
# 11/15/2023

import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, TimestampType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


HPC_hf_schema = StructType(\
    [StructField('index', LongType(), True), \
     StructField('trial_n', StringType(), True), \
     StructField('HPC_beta_hi_e', DoubleType(), True), \
     StructField('HPC_beta_hi_l', DoubleType(), True), \
     StructField('HPC_gamma_lo_e', DoubleType(), True), \
     StructField('HPC_gamma_lo_l', DoubleType(), True), \
     StructField('HPC_gamma_mid_e', DoubleType(), True), \
     StructField('HPC_gamma_mid_l', DoubleType(), True), \
     StructField('HPC_gamma_hi_e', DoubleType(), True), \
     StructField('HPC_gamma_hi_l', DoubleType(), True), \
     ])

HPC_lf_schema = StructType(\
    [StructField('index', LongType(), True), \
     StructField('trial_n', StringType(), True), \
     StructField('HPC_delta_e', DoubleType(), True), \
     StructField('HPC_delta_l', DoubleType(), True), \
     StructField('HPC_theta_e', DoubleType(), True), \
     StructField('HPC_theta_l', DoubleType(), True), \
     StructField('HPC_alpha_e', DoubleType(), True), \
     StructField('HPC_alpha_l', DoubleType(), True), \
     StructField('HPC_beta_lo_e', DoubleType(), True), \
     StructField('HPC_beta_lo_l', DoubleType(), True), \
     ])

PFC_hf_schema = StructType(\
    [StructField('index', LongType(), True), \
     StructField('trial_n', StringType(), True), \
     StructField('PFC_beta_hi_e', DoubleType(), True), \
     StructField('PFC_beta_hi_l', DoubleType(), True), \
     StructField('PFC_gamma_lo_e', DoubleType(), True), \
     StructField('PFC_gamma_lo_l', DoubleType(), True), \
     StructField('PFC_gamma_mid_e', DoubleType(), True), \
     StructField('PFC_gamma_mid_l', DoubleType(), True), \
     StructField('PFC_gamma_hi_e', DoubleType(), True), \
     StructField('PFC_gamma_hi_l', DoubleType(), True), \
     ])

PFC_lf_schema = StructType(\
    [StructField('index', LongType(), True), \
     StructField('trial_n', StringType(), True), \
     StructField('PFC_delta_e', DoubleType(), True), \
     StructField('PFC_delta_l', DoubleType(), True), \
     StructField('PFC_theta_e', DoubleType(), True), \
     StructField('PFC_theta_l', DoubleType(), True), \
     StructField('PFC_alpha_e', DoubleType(), True), \
     StructField('PFC_alpha_l', DoubleType(), True), \
     StructField('PFC_beta_lo_e', DoubleType(), True), \
     StructField('PFC_beta_lo_l', DoubleType(), True), \
     ])

ispc_schema = StructType(\
    [StructField('index', LongType(), True), \
     StructField('trial_n', StringType(), True), \
     StructField('theta_ispc_e', DoubleType(), True), \
     StructField('theta_ispc_l', DoubleType(), True), \
     StructField('alpha_ispc_e', DoubleType(), True), \
     StructField('alpha_ispc_l', DoubleType(), True), \
     StructField('beta_lo_ispc_e', DoubleType(), True), \
     StructField('beta_lo_ispc_l', DoubleType(), True), \
     StructField('beta_hi_ispc_e', DoubleType(), True), \
     StructField('beta_hi_ispc_l', DoubleType(), True), \
     StructField('gamma_lo_ispc_e', DoubleType(), True), \
     StructField('gamma_lo_ispc_l', DoubleType(), True), \
     ])

pac_schema = StructType(\
    [StructField('index', LongType(), True), \
     StructField('trial_n', StringType(), True), \
     StructField('30Hz-6Hz pac', DoubleType(), True), \
     StructField('30Hz-8Hz pac', DoubleType(), True), \
     StructField('30Hz-10Hz pac', DoubleType(), True), \
     StructField('40Hz-6Hz pac', DoubleType(), True), \
     StructField('40Hz-8Hz pac', DoubleType(), True), \
     StructField('40Hz-10Hz pac', DoubleType(), True), \
     StructField('50Hz-6Hz pac', DoubleType(), True), \
     StructField('50Hz-8Hz pac', DoubleType(), True), \
     StructField('50Hz-10Hz pac', DoubleType(), True), \
     StructField('60Hz-6Hz pac', DoubleType(), True), \
     StructField('60Hz-8Hz pac', DoubleType(), True), \
     StructField('60Hz-10Hz pac', DoubleType(), True), \
     StructField('70Hz-6Hz pac', DoubleType(), True), \
     StructField('70Hz-8Hz pac', DoubleType(), True), \
     StructField('70Hz-10Hz pac', DoubleType(), True), \
     StructField('80Hz-6Hz pac', DoubleType(), True), \
     StructField('80Hz-8Hz pac', DoubleType(), True), \
     StructField('80Hz-10Hz pac', DoubleType(), True), \
     StructField('90Hz-6Hz pac', DoubleType(), True), \
     StructField('90Hz-8Hz pac', DoubleType(), True), \
     StructField('90Hz-10Hz pac', DoubleType(), True), \
     StructField('100Hz-6Hz pac', DoubleType(), True), \
     StructField('100Hz-8Hz pac', DoubleType(), True), \
     StructField('100Hz-10Hz pac', DoubleType(), True), \
     ])

# data is split across these 24 smaller csvs
# First getting all our Vehicle animals together
# coke cue
HPC_hf = spark.read.format('csv').option('header', True).schema(HPC_hf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_coke_HPC_high_freq.csv').drop('index')
HPC_lf = spark.read.format('csv').option('header', True).schema(HPC_lf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_coke_HPC_low_freq.csv').drop('index')
PFC_hf = spark.read.format('csv').option('header', True).schema(PFC_hf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_coke_PFC_high_freq.csv').drop('index')
PFC_lf = spark.read.format('csv').option('header', True).schema(PFC_lf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_coke_PFC_low_freq.csv').drop('index')
ispc = spark.read.format('csv').option('header', True).schema(ispc_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_coke_PFC_HPC_ispcs.csv').drop('index')
pac = spark.read.format('csv').option('header', True).schema(pac_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_coke_PFC_HPC_pac.csv').drop('index')

# building cocaine dataframe, with label col: 1 denotes coke cue presentation
coc = HPC_hf.join(HPC_lf, on='trial_n').join(PFC_hf, on='trial_n').join(PFC_lf, on='trial_n').join(ispc, on='trial_n').join(pac, on='trial_n')
coc = coc.withColumn('cue', f.lit('coke'))

# saline cue
HPC_hf = spark.read.format('csv').option('header', True).schema(HPC_hf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_saline_HPC_high_freq.csv').drop('index')
HPC_lf = spark.read.format('csv').option('header', True).schema(HPC_lf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_saline_HPC_low_freq.csv').drop('index')
PFC_hf = spark.read.format('csv').option('header', True).schema(PFC_hf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_saline_PFC_high_freq.csv').drop('index')
PFC_lf = spark.read.format('csv').option('header', True).schema(PFC_lf_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_saline_PFC_low_freq.csv').drop('index')
ispc = spark.read.format('csv').option('header', True).schema(ispc_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_saline_PFC_HPC_ispcs.csv').drop('index')
pac = spark.read.format('csv').option('header', True).schema(pac_schema).load('dbfs:/FileStore/tables/CUE1_Vehicle_saline_PFC_HPC_pac.csv').drop('index')

# building saline dataframe, with cue col
sal = HPC_hf.join(HPC_lf, on='trial_n').join(PFC_hf, on='trial_n').join(PFC_lf, on='trial_n').join(ispc, on='trial_n').join(pac, on='trial_n')
sal = sal.withColumn('cue', f.lit('saline'))

# merge the two sets
data = coc.union(sal)

# split into train and static test sets
train, test = data.randomSplit(weights=[0.7,0.3], seed=1234)

# check current number of partitions
print(test.rdd.getNumPartitions())

# repartition into 50 partitions
test = test.repartition(50)
print(test.rdd.getNumPartitions())

# write to directory as 50 separate smaller files to simulate stream
dbutils.fs.rm('FileStore/tables/cocaineCue/', True)
test.write.format('csv').option('header', True).save('FileStore/tables/cocaineCue/')

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import PCA
# cue indexer
cueIndexer = StringIndexer(inputCol='cue', outputCol='label')

# logistic regression, no tuning
lr = LogisticRegression()

# toss feature cols (all floats) into vector assembler
cols = [c for c in data.columns if not c in set(['index', 'trial_n', 'label', 'cue'])]
vecAssem = VectorAssembler(inputCols=cols, outputCol='features')

# pipeline
myStages = [cueIndexer, vecAssem, lr]

p = Pipeline(stages=myStages)

# fit pipeline on train data
pModel = p.fit(train)

# check predictions on training data
predTrain = pModel.transform(train)

# transform static data with fitted pipeline
predTest = pModel.transform(test)


# COMMAND ----------

# evaluate
train_predictions = predTrain.select(f.col('prediction'), f.col('label'))
test_predictions = predTest.select(f.col('prediction'), f.col('label'))

# confusion matrix
print('\nTrain Confusion Matrix')
TP = train_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 1)).count()
FP = train_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 0)).count()
TN = train_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 0)).count()
FN = train_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 1)).count()
print(f'   1\t 0\n1  {TP}\t {FP}\n0  {FN}\t {TN}')
print(f'train accuracy: {(TP+TN)/(TP+FP+FN+TN)}')

# confusion matrix
print('\nTest Confusion Matrix')
TP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 1)).count()
FP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 0)).count()
TN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 0)).count()
FN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 1)).count()
print(f'   1\t 0\n1  {TP}\t {FP}\n0  {FN}\t {TN}')
print(f'test accuracy: {(TP+TN)/(TP+FP+FN+TN)}')

evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction', metricName='areaUnderROC')
print('\nTrain area under ROC: ', evaluator.evaluate(predTrain))
print('Test area under ROC: ', evaluator.evaluate(predTest))


# COMMAND ----------

# new schema for compiled dataframe stream
cueSchema = StructType([\
    StructField('trial_n', StringType(), True), \
    StructField('HPC_beta_hi_e', DoubleType(), True), \
    StructField('HPC_beta_hi_l', DoubleType(), True), \
    StructField('HPC_gamma_lo_e', DoubleType(), True), \
    StructField('HPC_gamma_lo_l', DoubleType(), True), \
    StructField('HPC_gamma_mid_e', DoubleType(), True), \
    StructField('HPC_gamma_mid_l', DoubleType(), True), \
    StructField('HPC_gamma_hi_e', DoubleType(), True), \
    StructField('HPC_gamma_hi_l', DoubleType(), True), \
    StructField('HPC_delta_e', DoubleType(), True), \
    StructField('HPC_delta_l', DoubleType(), True), \
    StructField('HPC_theta_e', DoubleType(), True), \
    StructField('HPC_theta_l', DoubleType(), True), \
    StructField('HPC_alpha_e', DoubleType(), True), \
    StructField('HPC_alpha_l', DoubleType(), True), \
    StructField('HPC_beta_lo_e', DoubleType(), True), \
    StructField('HPC_beta_lo_l', DoubleType(), True), \
    StructField('PFC_beta_hi_e', DoubleType(), True), \
    StructField('PFC_beta_hi_l', DoubleType(), True), \
    StructField('PFC_gamma_lo_e', DoubleType(), True), \
    StructField('PFC_gamma_lo_l', DoubleType(), True), \
    StructField('PFC_gamma_mid_e', DoubleType(), True), \
    StructField('PFC_gamma_mid_l', DoubleType(), True), \
    StructField('PFC_gamma_hi_e', DoubleType(), True), \
    StructField('PFC_gamma_hi_l', DoubleType(), True), \
    StructField('PFC_delta_e', DoubleType(), True), \
    StructField('PFC_delta_l', DoubleType(), True), \
    StructField('PFC_theta_e', DoubleType(), True), \
    StructField('PFC_theta_l', DoubleType(), True), \
    StructField('PFC_alpha_e', DoubleType(), True), \
    StructField('PFC_alpha_l', DoubleType(), True), \
    StructField('PFC_beta_lo_e', DoubleType(), True), \
    StructField('PFC_beta_lo_l', DoubleType(), True), \
    StructField('theta_ispc_e', DoubleType(), True), \
    StructField('theta_ispc_l', DoubleType(), True), \
    StructField('alpha_ispc_e', DoubleType(), True), \
    StructField('alpha_ispc_l', DoubleType(), True), \
    StructField('beta_lo_ispc_e', DoubleType(), True), \
    StructField('beta_lo_ispc_l', DoubleType(), True), \
    StructField('beta_hi_ispc_e', DoubleType(), True), \
    StructField('beta_hi_ispc_l', DoubleType(), True), \
    StructField('gamma_lo_ispc_e', DoubleType(), True), \
    StructField('gamma_lo_ispc_l', DoubleType(), True), \
    StructField('30Hz-6Hz pac', DoubleType(), True), \
    StructField('30Hz-8Hz pac', DoubleType(), True), \
    StructField('30Hz-10Hz pac', DoubleType(), True), \
    StructField('40Hz-6Hz pac', DoubleType(), True), \
    StructField('40Hz-8Hz pac', DoubleType(), True), \
    StructField('40Hz-10Hz pac', DoubleType(), True), \
    StructField('50Hz-6Hz pac', DoubleType(), True), \
    StructField('50Hz-8Hz pac', DoubleType(), True), \
    StructField('50Hz-10Hz pac', DoubleType(), True), \
    StructField('60Hz-6Hz pac', DoubleType(), True), \
    StructField('60Hz-8Hz pac', DoubleType(), True), \
    StructField('60Hz-10Hz pac', DoubleType(), True), \
    StructField('70Hz-6Hz pac', DoubleType(), True), \
    StructField('70Hz-8Hz pac', DoubleType(), True), \
    StructField('70Hz-10Hz pac', DoubleType(), True), \
    StructField('80Hz-6Hz pac', DoubleType(), True), \
    StructField('80Hz-8Hz pac', DoubleType(), True), \
    StructField('80Hz-10Hz pac', DoubleType(), True), \
    StructField('90Hz-6Hz pac', DoubleType(), True), \
    StructField('90Hz-8Hz pac', DoubleType(), True), \
    StructField('90Hz-10Hz pac', DoubleType(), True), \
    StructField('100Hz-6Hz pac', DoubleType(), True), \
    StructField('100Hz-8Hz pac', DoubleType(), True), \
    StructField('100Hz-10Hz pac', DoubleType(), True), \
    StructField('cue', StringType(), True) \
])

# source
sourceStream = spark.readStream.format('csv') \
    .option('encoding', 'UTF-8') \
    .option('header', True) \
    .schema(cueSchema) \
    .option('maxFilesPerTrigger', 1) \
    .load('dbfs:/FileStore/tables/cocaineCue')

# fit model to stream
predTestStream = pModel.transform(sourceStream)

# sink, outputMode append, not complete since we are not aggregating
sinkStream = predTestStream.writeStream.outputMode('append') \
    .format('memory') \
    .queryName('testPreds') \
    .trigger(processingTime='10 seconds') \
    .start()

# COMMAND ----------

# evaluate early on in the stream
df_preds = spark.sql("select trial_n, prediction, label from testPreds")
df_preds.show()
# evaluate
test_predictions = df_preds.select(f.col('prediction'), f.col('label'))

# confusion matrix
print('\nTest Confusion Matrix')
TP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 1)).count()
FP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 0)).count()
TN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 0)).count()
FN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 1)).count()
print(f'   1\t 0\n1  {TP}\t {FP}\n0  {FN}\t {TN}')
print(f'Test accuracy: {(TP+TN)/(TP+FP+FN+TN)}')

evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction', metricName='areaUnderROC')
print('Test area under ROC: ', evaluator.evaluate(df_preds.select(f.col('prediction').cast('double'), f.col('label').cast('double'))))

# COMMAND ----------

# as we evaluate later in the stream, the results approach results of static window
df_preds = spark.sql("select trial_n, prediction, label from testPreds")
df_preds.show()
# evaluate
test_predictions = df_preds.select(f.col('prediction'), f.col('label'))

# confusion matrix
print('\nTest Confusion Matrix')
TP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 1)).count()
FP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 0)).count()
TN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 0)).count()
FN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 1)).count()
print(f'   1\t 0\n1  {TP}\t {FP}\n0  {FN}\t {TN}')
print(f'Test accuracy: {(TP+TN)/(TP+FP+FN+TN)}')

evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction', metricName='areaUnderROC')
print('Test area under ROC: ', evaluator.evaluate(df_preds.select(f.col('prediction').cast('double'), f.col('label').cast('double'))))

# COMMAND ----------

# as we evaluate later in the stream, the results approach results of static window
df_preds = spark.sql("select trial_n, prediction, label from testPreds")
df_preds.show()
# evaluate
test_predictions = df_preds.select(f.col('prediction'), f.col('label'))

# confusion matrix
print('\nTest Confusion Matrix')
TP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 1)).count()
FP = test_predictions.filter((f.col('prediction') == 1) & (f.col('label') == 0)).count()
TN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 0)).count()
FN = test_predictions.filter((f.col('prediction') == 0) & (f.col('label') == 1)).count()
print(f'   1\t 0\n1  {TP}\t {FP}\n0  {FN}\t {TN}')
print(f'Test accuracy: {(TP+TN)/(TP+FP+FN+TN)}')

evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction', metricName='areaUnderROC')
print('Test area under ROC: ', evaluator.evaluate(df_preds.select(f.col('prediction').cast('double'), f.col('label').cast('double'))))

# COMMAND ----------


