import pickle
import findspark
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col, udf

spark = SparkSession.builder.appName("Sentiment").getOrCreate()
sc = spark.sparkContext

# Unpickle file
model_rdd_pkl = sc.binaryFiles("file:///home/dh2409/Documents/big_data/logistic.sav")
model_rdd_data = model_rdd_pkl.collect()

bow_vectorizer_rdd_pkl = sc.binaryFiles("file:///home/dh2409/Documents/big_data/bow_vectorizer")
bow_vectorizer_rdd_data = bow_vectorizer_rdd_pkl.collect()


# Load and broadcast python object over spark nodes
creditcardfrauddetection_model = pickle.loads(model_rdd_data[0][1])
broadcast_creditcardfrauddetection_model = sc.broadcast(creditcardfrauddetection_model)
print(broadcast_creditcardfrauddetection_model.value)

bow = pickle.loads(bow_vectorizer_rdd_data[0][1])
broadcast_bow = sc.broadcast(bow)
print(broadcast_bow.value)


#read data
dataset = pd.read_csv('file:///home/dh2409/Documents/big_data/data.csv', sep='|')
dataset.head()
X = dataset.drop(["Rating"], axis = 1)
feature_columns = X.columns.to_list()

# Create spark dataframe for prediction
df = spark.read.csv('file:///home/dh2409/Documents/big_data/data.csv', header=True, sep='|')


#user-define function
def predict(*cols):
    """input: text
       output: positive probability 
    """
    data = broadcast_bow.value.transform([cols[0]])
    prediction = broadcast_creditcardfrauddetection_model.value.predict_proba(data)
    return 1 if float(prediction[0,1]) >= 0.5 else 0

predict_udf = udf(predict, IntegerType())

if __name__ == '__main__':
    df = df.withColumn("score", predict_udf(*feature_columns))
    df.write.csv("file:///home/dh2409/Documents/big_data/data_prediction.csv")
    sc.stop()

