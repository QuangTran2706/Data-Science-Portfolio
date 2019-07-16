# Recommender Systems with IBM Watson

### Project Overview
In this project, I built several recommenders using the real data provided by IBM Watson Studio of the interactions that users have with articles on the IBM community platform.

Four recommendation systems are built in the Jupyter Notebook:
1. Rank-based: recommend articles based on popularity only. this is particularly good for cold start problem - provide recommendations for new users with no history in the system.
2. User-user collaborative filtering: find users similar to the target user with similar browsing behaviors, then recommend the articles that similar users have read. if a tie occurs, favor the users with more article interaction and articles viewed more times.
3. Content-based: apply NLP techniques to convert the article titles into feature vectors, recommend new articles with similar wording in titles as the ones the user has read. if a tie occurs, favor articles viewed more times.
4. Matrix Factorization: split the dataset into train and test set, decompose the training user-article matrix into 3 matrices with different latent factors, use the dot product to predict whether the user in the test will read the particular article. explore the impact of different latent factors on the prediction errors.
