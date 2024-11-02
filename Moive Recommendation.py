# Databricks notebook source
# MAGIC %md 
# MAGIC ### Spark Moive Recommendation 
# MAGIC In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline

# COMMAND ----------

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part1: Data ETL and Data Exploration

# COMMAND ----------

# Please donwload your data from the website then upload to databricks at first https://grouplens.org/datasets/movielens/latest/
# 参考hw1，自己完善 :) 



# COMMAND ----------

# DBTITLE 1,Note: please download the data before running this cell
movies_df = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings_df = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links_df = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags_df = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)

# COMMAND ----------

movies_df.show(5)

# COMMAND ----------

ratings_df.show(10)

# COMMAND ----------

links_df.show(5)

# COMMAND ----------

tags_df.show(5)

# COMMAND ----------

tmp1 = ratings_df.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings_df.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

# COMMAND ----------

tmp1 = sum(ratings_df.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings_df.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Spark SQL and OLAP 

# COMMAND ----------

movies_df.registerTempTable("movies")
ratings_df.registerTempTable("ratings")
links_df.registerTempTable("links")
tags_df.registerTempTable("tags")

# COMMAND ----------

# MAGIC %md ### Q1: The number of Users

# COMMAND ----------

q1_result=spark.sql("Select Count(Distinct userId) as Number_of_Users from ratings")
q1_result.show()

# COMMAND ----------

# MAGIC %md ### Q2: The number of Movies

# COMMAND ----------

q3_result_1=spark.sql("Select Count(movieId) as Number_of_Rated_Moives From movies Where movieID in (Select movieId From ratings)")
q3_result_1.show()

# COMMAND ----------

# MAGIC %md ### Q3:  How many movies are rated by users? List movies not rated before

# COMMAND ----------

q3_result_2=spark.sql("Select movieId, title From movies Where movieID not in (Select movieId From ratings)")
q3_result_2.show()

# COMMAND ----------

# MAGIC %md ### Q4: List Movie Genres

# COMMAND ----------

q4_result=spark.sql("Select Distinct explode(split(genres,'[|]')) as genres From movies Order by 1")
q4_result.show()

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %md ### Q5: Movie for Each Category

# COMMAND ----------

q5_result_1=spark.sql("Select genres,Count(movieId) as Number_of_Moives From(Select explode(split(genres,'[|]')) as genres, movieId From movies) Group By 1 Order by 2 DESC")
q5_result_1.show()

# COMMAND ----------

# MAGIC %sql 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part2: Spark ALS based approach for training model
# MAGIC We will use an Spark ML to predict the ratings, so let's reload "ratings.csv" using ``sc.textFile`` and then convert it to the form of (user, item, rating) tuples.

# COMMAND ----------

ratings_df.show()

# COMMAND ----------

movie_ratings=ratings_df.drop('timestamp')

# COMMAND ----------

# Data type convert
from pyspark.sql.types import IntegerType, FloatType
movie_ratings = movie_ratings.withColumn("userId", movie_ratings["userId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("movieId", movie_ratings["movieId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("rating", movie_ratings["rating"].cast(FloatType()))

# COMMAND ----------

movie_ratings.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ALS Model Selection and Evaluation
# MAGIC
# MAGIC With the ALS model, we can use a grid search to find the optimal hyperparameters.

# COMMAND ----------

# import package
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# COMMAND ----------

#Create test and train set
(training, test) = movie_ratings.randomSplit([0.8, 0.2])

# COMMAND ----------

#Create ALS model
als = ALS(maxIter=5, rank=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

# COMMAND ----------

#Tune model using ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(als.regParam, [0.1, 0.3])  # Try fewer values, e.g., 0.1 and 0.3
             .addGrid(als.rank, [10])            # Test only rank=10 to start
             .addGrid(als.maxIter, [5])          # Fix maxIter to 5
             .build())


# COMMAND ----------

# Define evaluator as RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

# COMMAND ----------

# Build Cross validation 
cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)


# COMMAND ----------

# Fit the ALS model
cvModel = cv.fit(training)

# COMMAND ----------

#Extract best model from the tuning exercise using ParamGridBuilder
bestModel=cvModel.bestModel

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model testing
# MAGIC And finally, make a prediction and check the testing error.

# COMMAND ----------

#Generate predictions and evaluate using RMSE
predictions=bestModel.transform(test)
rmse = evaluator.evaluate(predictions)

# COMMAND ----------

#Print evaluation metrics and model parameters
print ("RMSE = "+str(rmse))
print ("**Best Model**")
print (" Rank:"), 
print (" MaxIter:"), 
print (" RegParam:"), 

# COMMAND ----------

predictions.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model apply and see the performance

# COMMAND ----------

alldata=bestModel.transform(movie_ratings)
rmse = evaluator.evaluate(alldata)
print ("RMSE = "+str(rmse))

# COMMAND ----------

alldata.registerTempTable("alldata")

# COMMAND ----------

# MAGIC %sql select * from alldata

# COMMAND ----------

# MAGIC %sql select * from movies join alldata on movies.movieId=alldata.movieId

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Recommend moive to users with id: 575, 232. 
# MAGIC you can choose some users to recommend the moives 

# COMMAND ----------

# Install and import Koalas if needed
!pip install koalas
import databricks.koalas as ks


# COMMAND ----------

userRecs = bestModel.recommendForAllUsers(10)

# Convert PySpark DataFrames to Koalas DataFrames
userRecs_ks = userRecs.to_koalas()
movies_ks = movies_df.to_koalas()

def movieRecommendation(inputId):
  recs_list=[]
  for recs in userRecs_ks.loc[str(inputId), 'recommendations']:
    recs_list.append(str(recs[0]))
  return (movies_ks[movies_ks['movieId'].isin(recs_list)])


# COMMAND ----------

# Get top movie recommendations for user with ID 148
recommended_movies_148 = movieRecommendation(85)
print("Top recommendations for user 148:")
print(recommended_movies_148)

# COMMAND ----------

print(userRecs_ks.index.unique())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the similar moives for moive with id: 463, 471
# MAGIC You can find the similar moives based on the ALS results

# COMMAND ----------

itemFactors=bestModel.itemFactors.to_koalas()

# COMMAND ----------

def similarMovies(inputId, matrix='cosine_similarity'):
  try:
    movieFeature=itemFactors.loc[itemFactors.id==str(inputId),'features'].to_numpy()[0]
  except:
    return 'There is no movie with id ' + str(inputId)
  
  if matrix=='cosine_similarity':
    similarMovie=pd.DataFrame(columns=('movieId','cosine_similarity'))
    for id,feature in itemFactors.to_numpy():
      cs=np.dot(movieFeature,feature)/(np.linalg.norm(movieFeature) * np.linalg.norm(feature))
      similarMovie=similarMovie.append({'movieId':str(id), 'cosine_similarity':cs}, ignore_index=True)
    similarMovie_cs=similarMovie.sort_values(by=['cosine_similarity'],ascending = False)[1:11]
    joint=similarMovie_cs.merge(movies_ks.to_pandas(), left_on='movieId', right_on = 'movieId', how = 'inner')
  if matrix=='euclidean_distance':
    similarMovie=pd.DataFrame(columns=('movieId','euclidean_distance'))
    for id,feature in itemFactors.to_numpy():
      ed=np.linalg.norm(np.array(movieFeature)-np.array(feature))
      similarMovie=similarMovie.append({'movieId':str(id), 'euclidean_distance':ed}, ignore_index=True)
    similarMovie_ed=similarMovie.sort_values(by=['euclidean_distance'])[1:11]
    joint=similarMovie_ed.merge(movies_ks.to_pandas(), left_on='movieId', right_on = 'movieId', how = 'inner')
  return joint[['movieId','title','genres']]

# COMMAND ----------

similarMovies(463)

# COMMAND ----------

print('Similar movies based on cosine similarity matrix are as follows.')
similarMovies(471, 'cosine_similarity')

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Write the report 
# MAGIC motivation: As artificial intelligence pervails in internet industry, more and more ecommerce platforms start to characterize their recommendation systems in order to provide better service. Collaborative filtering is one of the most popular recommendation algorithm which can be implemented with Alternating Least Squares (ALS) model in Spark ML. It would be a interesting and significant attempt to create a movie recommender for movie rating sites users.
# MAGIC
# MAGIC step1. Data ETL and Data Exploration
# MAGIC
# MAGIC I firstly loaded the rating data, established corresponding spark dataframes and checked out the basic information of the dataset.
# MAGIC
# MAGIC step2. Online Analytical Processing
# MAGIC
# MAGIC I performed analysis on the dataset from multi angle and gained some intuitive insights.
# MAGIC
# MAGIC step3. Model Selection
# MAGIC
# MAGIC I built up the ALS model and tuned the hyperparameter using 5-fold cross-validation, applying the optimal hyperparameters on the best final model.
# MAGIC
# MAGIC step4. Model Evaluation
# MAGIC
# MAGIC Finally, I evaluated the recommendation model by measuring the root-mean-square error of rating prediction on the testset.
# MAGIC
# MAGIC step5. Model Application: Recommend moive to users
# MAGIC
# MAGIC For given users, I wrote a function to dirctly recommend 10 movies which they may be interested in based on the model.
# MAGIC
# MAGIC step6. Model Application: Find the similar moives
# MAGIC
# MAGIC I also applid the ALS results on finding the similar moives for a given movie. I used two matrix to evaluate the similarity between movies: cosine similarity and euclidean distance, which can be used sperately depends on situations.
# MAGIC
# MAGIC Output and Conclusion
# MAGIC
# MAGIC In this project, I built a ALS model with Spark APIs based on MovieLens dataset, predicted the ratings for the movies and made specific recommendation to users accordingly. The RMSE of the best model is approximately 0.88.
