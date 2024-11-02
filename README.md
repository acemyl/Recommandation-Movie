motivation: As artificial intelligence pervails in internet industry, more and more ecommerce platforms start to characterize their recommendation systems in order to provide better service. Collaborative filtering is one of the most popular recommendation algorithm which can be implemented with Alternating Least Squares (ALS) model in Spark ML. It would be a interesting and significant attempt to create a movie recommender for movie rating sites users.

step1. Data ETL and Data Exploration

I firstly loaded the rating data, established corresponding spark dataframes and checked out the basic information of the dataset.

step2. Online Analytical Processing

I performed analysis on the dataset from multi angle and gained some intuitive insights.

step3. Model Selection

I built up the ALS model and tuned the hyperparameter using 5-fold cross-validation, applying the optimal hyperparameters on the best final model.

step4. Model Evaluation

Finally, I evaluated the recommendation model by measuring the root-mean-square error of rating prediction on the testset.

step5. Model Application: Recommend moive to users

For given users, I wrote a function to dirctly recommend 10 movies which they may be interested in based on the model.

step6. Model Application: Find the similar moives

I also applid the ALS results on finding the similar moives for a given movie. I used two matrix to evaluate the similarity between movies: cosine similarity and euclidean distance, which can be used sperately depends on situations.

Output and Conclusion

In this project, I built a ALS model with Spark APIs based on MovieLens dataset, predicted the ratings for the movies and made specific recommendation to users accordingly. The RMSE of the best model is approximately 0.88.
