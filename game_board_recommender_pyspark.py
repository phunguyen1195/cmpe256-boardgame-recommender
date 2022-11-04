#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
from pyspark.sql.functions import col, explode
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# In[18]:


sc = SparkContext
spark = SparkSession.builder.appName("DataFrame").getOrCreate()


# In[19]:


movies = spark.read.csv("games.csv",header=True)
themes = spark.read.csv("subcategories.csv",header=True)
ratings = spark.read.csv("train.csv",header=True)


# In[24]:


ratings = ratings.\
    withColumn('UserID', col('UserID').cast('integer')).\
    withColumn('GameID', col('GameID').cast('integer')).\
    withColumn('Rating', col('Rating').cast('float'))


# In[26]:


# Count the total number of ratings in the dataset
numerator = ratings.select("Rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("UserID").distinct().count()
num_movies = ratings.select("GameID").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator *1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")


# In[27]:


userId_ratings = ratings.groupBy("UserID").count().orderBy('count', ascending=False)
userId_ratings.show()


# In[28]:


movieId_ratings = ratings.groupBy("GameID").count().orderBy('count', ascending=False)
movieId_ratings.show()


# In[31]:


(train, test) = ratings.randomSplit([0.8, 0.2], seed = 1234)

# Create ALS model
als = ALS(userCol="UserID", itemCol="GameID", ratingCol="Rating", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")

# Confirm that a model called "als" was created
type(als)


# In[35]:


# Import the requisite items
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()
            #             .addGrid(als.maxIter, [5, 50, 100, 200]) \

           
# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))



# In[36]:


# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Confirm cv was built
print(cv)


# In[ ]:


#Fit cross validator to the 'train' dataset
model = cv.fit(train)

#Extract best model from the cv model above
best_model = model.bestModel



# In[ ]:


# Print best_model
print(type(best_model))

# Complete the code below to extract the ALS model parameters
print("**Best Model**")

# # Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())

# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())

# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())


# In[ ]:


# View the predictions
test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)


# In[ ]:


with open('results_spark.txt', 'w') as writefile:
    writefile.write(str(best_model._java_obj.parent().getRank()))
    writefile.write('/n')
    writefile.write(str(best_model._java_obj.parent().getMaxIter()))
    writefile.write('/n')
    writefile.write(str(best_model._java_obj.parent().getRegParam()))
    writefile.write('/n')
    writefile.write(str(RMSE))
    


# In[ ]:




