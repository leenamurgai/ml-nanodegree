# ml-nanodegree
## [Machine Learning Udacity Nanodegree program](https://classroom.udacity.com/nanodegrees/nd009/syllabus/core-curriculum)

[Course Notes](https://github.com/leenamurgai/ml-nanodegree/blob/master/ml-nanodegree.pdf)

### 0. Foundations ###
1. Artificial Intelligence
2. Machine Learning
3. Data Science

### 1. Model Evaluation and Validation ###

**Project 1:** Predict Boston House Prices

* In this project we use Decision Tree Regression to predict house prices.
* We have 506 examples in our dataset and 13 features for each example.
* We explore the data and discuss the nature of the problem.
* We discuss the available performance metrics and choose an appropriate one.
* We split out dataset into a training and testing set and discuss grid search and cross validation.
* We look at learning graphs and complexity graphs to inform our choice of complexity for the model.
* Having chose a model we make a prediction for the price of a house given its features.
* We sanity check the predicted price comparing the output price with the statistics for our dataset and use KNN to compare it to the 10 nearest neighbours.

### 2. Supervised Learning ###

**Project 2:** Build a Student Student Intervention System

* In this project we are interested in building a system which detects when a student is at risk of failing and would benefit from early intervention.
* We have data for 395 students and 30 features for each example in addition to whether they passed or failed.
* We discuss the nature of the data we have.
* The data requires a fair amount of processing, transforming features to binary and removing redundant data.
* We discuss the performance metric F<sub>1</sub> score and how to make appropriate choices to ensure it is meaningful.
* We discuss how to split the data into training and testing sets to account for the imbalance.
* We calculate baseline performance metrics for our classifier.
* We choose 3 models (Support Vector Machine, K-Nearest Neighbours and Bernoulli Naive Bayes), discuss the pros and cons of each and choose our final model based on performance (Bernoulli Naive Bayes).
* We discuss how our chosen model works and fine tune it giving a final F<sub>1</sub> score and specifying the parameters which yielded the result.

### 3. Unsupervised Learning ###

**Project 3:** Creating Customer Segments

* In this project we are interested in segmenting the customers of a wholesale retailer so that when making changes, the retailer can assess the impact on the different types of customers independently
* The retailer has data for 440 customers, specifically their spending in 6 product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen)
* We look for relationships between the features
  1. We see if we can predict one feature given the others using a Decision Tree Regressor
  2. We look at the feature distributions and scatter plots for all pairs of features looking for correlation
* Noting all the distributions to be positively skewed we transform our data into log space
* We use Tukey's method for identifying outliers and from these select which data points to remove from our analysis
* We calculate a correlation matrix heat map
* We use principal component analysis to reduce the dimensionality of the problem from 6  to 2 (these 2 dimensions explain 73% of the variance in the data)
* We use K-means and Gaussian Mixture Model clustering algorithms in our reduced space to find customer segments

### 4. Reinforcement Learning ###

**Project 4:** Train a smart cab

* In this project we use reinforcement learning to teach a cab how to efficiently navigate an idealised grid-like city to get from A to B within a specified time limit
* The cab must decide given based in input information whether to turn right, left, go straight or stay put. Input information includes:
  * the direction it would ideally move in next to get to the destination as quickly as possible (left / right / straight)
  * the traffic light colour (red / green)
  * location of other vehicles at the intersection and the direction they intend to travel in (US right of way rules apply)
  * time remaining
* We implement an initial (not so smart) cab which navigates the city randomly
* We reduce the state space by considering which information is necessary in order to know if actions are legal
* We implement Q-learning and discuss the choices for the initial action
* We apply reinforcement learning and tweak parameters (discount factor, initial Q-values) to improve performance
* We find and discuss the optimal policy and analyse the penalty/reward structure

### 5. Capstone Project ###

**Project 5:** Digit Recognition in Natural Scene Images

* We design a system which, given a natural image of a digit, is capable of recognising the digit at the centre of it
* We use neural networks and more specifically TensorFlow to do this
* We train and test our system on the Street View House Numbers ([SVHN](http://ufldl.stanford.edu/housenumbers/)) dataset obtained from Google Street View images
* The dataset is large containing 630,420 labelled colour images, each one is a 32x32x3 array
* We download and extract the data
* We choose our performance metric and explore the data looking at the distribution of digits to find a lower bound for it we also note human performance on the dataset (98%)
* We convert the images to grey scale to reduce its size and normalise it and save our data format which makes it possible to access it as NumPy arrays
* We discuss the algorithms and techniques we use: Deep Neural Networks, Stochastic Gradient Descent, Xavier weight initialisation, L2 regularisation, Dropout
* Our general approach is to start with a simple model and gradually increase its complexity. The size of the dataset is such that once the model was complex enough, training the network became computationally too expensive and we were constrained by our hardware
* A three layer network is sufficient to get 93% accuracy
