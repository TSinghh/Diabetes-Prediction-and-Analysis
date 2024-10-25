!pip install pyspark
import pyspark
from pyspark.sql import SparkSession

spark= SparkSession.builder.appName('PySpark').getOrCreate()
df_pyspark=spark.read.csv('/content/diabetes.csv')
df_pyspark.columns
df_pyspark

DataFrame[Pregnancies: string, Glucose: string, BloodPressure: string, SkinThickness: string, Insulin: string, BMI: string, DiabetesPedigreeFunction: string, Age: string, Outcome: string]

df_pyspark=spark.read.csv('/content/diabetes.csv',header=True,inferSchema=True)
df_pyspark

DataFrame[Pregnancies: int, Glucose: int, BloodPressure: int, SkinThickness: int, Insulin: int, BMI: double, DiabetesPedigreeFunction: double, Age: int, Outcome: int]
df_pyspark.printSchema()
type(df_pyspark)

df_pyspark.head(3)

df_pyspark.select(['DiabetesPedigreeFunction','Outcome']).show()

df_pyspark.drop('Pregnancies').show()

df_pyspark.withColumnsRenamed({'DiabetesPedigreeFunction':'Diabetes Percentage','Pregnancies':'No.of Pregnancies'}).show()

from pyspark.sql.functions import when

df_pyspark.withColumn('Results',
when(df_pyspark['Outcome'] == 1, "Yes").otherwise("No")).show()

df_pyspark.filter((df_pyspark['Outcome']==1)).select(['Age',"BloodPressure","Outcome"]).show()

df_pyspark.groupBy('Outcome').count().show()

df_pyspark.groupBy('BloodPressure').count().show()

df_pyspark.groupBy().min('DiabetesPedigreeFunction').show()

df_pyspark.groupBy().max('DiabetesPedigreeFunction').show()

df_pyspark.groupBy().sum('Pregnancies').show()

from pyspark.ml.feature import VectorAssembler

train= SparkSession.builder.appName('Vectorassembler').getOrCreate()

training_dt=train.read.csv('/content/diabetes.csv',header=True,inferSchema=True)

training_dt.show()

training_dt.columns

featureassembler=VectorAssembler(inputCols=['Insulin','Age'],outputCol='Independent Features')

output=featureassembler.transform(training_dt)

output.show()
 
final_data=output.select("Independent Features","Outcome")

final_data.show()

from pyspark.ml.regression import LinearRegression

train_data,test_data=final_data.randomSplit([0.45,0.55])

regressor=LinearRegression(featuresCol='Independent Features',labelCol='Outcome')
regressor=regressor.fit(train_data)


regressor.coefficients

regressor.intercept

pred_results=regressor.evaluate(test_data)
pred_results.predictions.show()

x = pred_results.predictions.select("Independent Features").rdd.flatMap(lambda x: x).collect()
y = pred_results.predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

x_values = [vector[0] for vector in x]

import matplotlib.pyplot as plt
plt.scatter(x_values, y)
plt.xlabel("Independent Features")
plt.ylabel("Predictions")
plt.show()
