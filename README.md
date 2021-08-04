# [Coursera] Machine Learning
For future reference, this repo will store the contents of this course. The `Lectures` directory contains the lecture slides and corresponding quizzes (with solutions) for each and every topic, all of which are organized according to the weekly structure of the course. Additionally, there is a separate directory for each (completed) programming assignment with instructions and source code. The programming in this course was done using Octave/MATLAB.

---

**Course Website:** https://www.coursera.org/learn/machine-learning/home/info  
> [![about.png](https://i.postimg.cc/rpsqD0Tv/about.png)](https://postimg.cc/LJcc7snT)
---

**Course Syllabus:**
## Week 1
**Introduction**: Introduces the core idea of teaching a computer to learn concepts using data—without being explicitly programmed.
- What is Machine Learning?
- Supervised Learning
- Unsupervised Learning

**Linear Regression with One Variable**: Linear regression predicts a real-valued output based on an input value. Discuss the application of linear regression to housing price prediction, present the notion of a cost function, and introduce the gradient descent method for learning.
- Model Representation
- Cost Function
- Gradient Descent
- Gradient Descent For Linear Regression

**Linear Algebra Review**: Provides a refresher on linear algebra concepts. Basic understanding of linear algebra is necessary for the rest of the course, especially as it begins to cover models with multiple variables.
- Matrices and Vectors
- Addition and Scalar Multiplication
- Matrix Vector Multiplication
- Matrix Matrix Multiplication
- Matrix Multiplication Properties
- Inverse and Transpose

## Week 2
**Linear Regression with Multiple Variables**: What if your input has more than one value? Show how linear regression can be extended to accommodate multiple input features. Also discusses best practices for implementing linear regression.
- Multiple Features
- Gradient Descent For Multiple Variables
- Gradient Descent in Practice
- Features and Polynomial Regression
- Normal Equation
- Normal Equation Noninvertibility

**Octave/MATLAB Tutorial**: This course includes programming assignments designed to help understand how to implement the learning algorithms in practice. Introduces Octave/MATLAB, which is needed to complete the programming assignments.
- Basic Operations
- Moving Data Around
- Computing on Data
- Plotting Data
- Control Statements
- Vectorization

`machine-learning-ex1` contains **Linear Regression** programming assignment

## Week 3
**Logistic Regression**: Logistic regression is a method for classifying data into discrete outcomes. Introduces the notion of classification, the cost function for logistic regression, and the application of logistic regression to multi-class classification.
- Classification
- Hypothesis Representation
- Decision Boundary
- Cost Function
- Simplified Cost Function and Gradient Descent
- Advanced Optimization
- Multiclass Classification: One-vs-all

**Regularization**: Machine learning models need to generalize well to new examples that the model has not seen in practice. Introduces regularization, which helps prevent models from overfitting the training data.
- The Problem of Overfitting
- Cost Function
- Regularized Linear Regression
- Regularized Logistic Regression

`machine-learning-ex2` contains **Logistic Regression** programming assignment

## Week 4
**Neural Networks Representation**: Neural networks is a model inspired by how the brain works. It is widely used today in many applications: when your phone interprets and understand your voice commands, it is likely that a neural network is helping to understand your speech; when you cash a check, the machines that automatically read the digits also use neural networks.
- Non-linear Hypotheses
- Neurons and the Brain
- Model Representation
- Examples and Intuitions
- Multiclass Classification

`machine-learning-ex3` contains **Multi-class Classification and Neural Networks** programming assignment

## Week 5
**Neural Networks Learning**: Introduces the backpropagation algorithm that is used to help learn parameters for a neural network.
- Cost Function
- Backpropagation Algorithm
- Backpropagation Intuition
- Unrolling Parameters
- Gradient Checking
- Random Initialization
- Autonomous Driving

`machine-learning-ex4` contains **Neural Network Learning** programming assignment

## Week 6
**Advice for Applying Machine Learning**: Applying machine learning in practice is not always straightforward. Shares best practices for applying machine learning in practice, and discusses the best ways to evaluate performance of the learned models.
- Deciding What to Try Next
- Evaluating a Hypothesis
- Model Selection and Train/Validation/Test Sets
- Diagnosing Bias vs. Variance
- Regularization and Bias/Variance
- Learning Curves

**Machine Learning System Design**: To optimize a machine learning algorithm, understanding where the biggest improvements can be made is needed. Discusses how to understand the performance of a machine learning system with multiple parts, and also how to deal with skewed data.
- Prioritizing What to Work On
- Error Analysis
- Error Metrics for Skewed Classes
- Trading Off Precision and Recall
- Data For Machine Learning

`machine-learning-ex5` contains **Regularized Linear Regression and Bias/Variance** programming assignment

## Week 7
**Support Vector Machines**: Support vector machines, or SVMs, is a machine learning algorithm for classification. Introduces the idea and intuitions behind SVMs and discuss how to use it in practice.
- Optimization Objective
- Large Margin Intuition
- Mathematics Behind Large Margin Classification
- Kernels
- Using an SVM

`machine-learning-ex6` contains **Support Vector Machines** programming assignment

## Week 8
**Unsupervised Learning**: Unsupervised learning is used to build models that help us understand our data better. Discusses the k-Means algorithm for clustering that enables us to learn groupings of unlabeled data points.
- Introduction to Unsupervied Learning
- K-Means Algorithm
- Optimization Objective
- Random Initialization
- Choosing the Number of Clusters

**Dimensionality Reduction**: Introduces Principal Components Analysis, and shows how it can be used for data compression to speed up learning algorithms as well as for visualizations of complex datasets.
- Motivaion: Data Compression
- Motivation: Visualization
- Principal Component Analysis Algorithm
- Reconstruction from Compressed Representation
- Choosing the Number of Principal Components
- Advice for Applying PCA

`machine-learning-ex7` contains **K-Means Clustering and PCA** programming assignment

## Week 9
**Anomaly Detection**: Given a large number of data points, sometimes it is necessary to figure out which ones vary significantly from the average. For example, in manufacturing, we may want to detect defects or anomalies. Shows how a dataset can be modeled using a Gaussian distribution, and how the model can be used for anomaly detection.
- Problem Motivation
- Gaussian Distribution
- Anomaly Detection Algorithm
- Developing and Evaluating an Anomaly Detection System
- Anomaly Detection vs. Supervised Learning
- Choosing What Features to Use
- Multivariate Gaussian Distribution
- Anomaly Detection using the Multivariate Gaussian Distribution

**Recommnder Systems**: When you buy a product online, most websites automatically recommend other products that you may like. Recommender systems look at patterns of activities between different users and different products to produce these recommendations. Introduces recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.
- Problem Formulation
- Content Based Recommendations
- Collaborative Filtering
- Collaborative Filtering Algorithm
- Vectorization: Low Rank Matrix Factorization
- Mean Normalization

`machine-learning-ex8` contains **Anomaly Detection and Recommender Systems** programming assignment

## Week 10
**Large Scale Machine Learning**: Machine learning works best when there is an abundance of data to leverage for training. Discusses how to apply the machine learning algorithms with large datasets.
- Learning With Large Datasets
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent
- Stochastic Gradient Descent Convergence
- Online Learning
- Map Reduce and Data Parallelism

## Week 11
**Application Example Photo OCR**: Identifying and recognizing objects, words, and digits in an image is a challenging task. Discusses how a pipeline can be built to tackle this problem and how to analyze and improve the performance of such a system.
- Problem Description and Pipeline
- Sliding Windows
- Getting Lots of Data and Artificial Data
- Ceiling Analysis: What Part of the Pipeline to Work on Next

---

Upon completion of this course, I earned a [certificate](https://www.coursera.org/account/accomplishments/certificate/WFVW6BY7SMFG). If you are also taking this course, feel free to use this repo; however, I encourage you to attempt the quizzes and programming assignments on your own, first.
