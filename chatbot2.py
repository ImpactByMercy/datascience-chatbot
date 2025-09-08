import re
import random
# chatbot.py

print("🤖 Data Science Chatbot started!")
print("Type 'exit' to quit.\n")

# Dictionary of 50 starter Data Science questions and answers
responses = {
    "what is data science": "Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge from data.",
    "what is machine learning": "Machine Learning is a subset of AI that allows systems to learn patterns from data and make predictions without being explicitly programmed.",
    "what is artificial intelligence": "AI is the simulation of human intelligence processes by machines, especially computer systems.",
    "what is deep learning": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data.",
    "what is supervised learning": "Supervised Learning is a type of ML where the model is trained on labeled data.",
    "what is unsupervised learning": "Unsupervised Learning is a type of ML where the model finds patterns in unlabeled data.",
    "what is reinforcement learning": "Reinforcement Learning is a type of ML where an agent learns by interacting with the environment and receiving rewards or penalties.",
    "what is regression": "Regression is a statistical method to model the relationship between a dependent variable and one or more independent variables.",
    "what is classification": "Classification is a type of supervised learning where the goal is to predict discrete labels.",
    "what is clustering": "Clustering is a type of unsupervised learning where data points are grouped based on similarity.",
    "what is data cleaning": "Data Cleaning is the process of fixing or removing incorrect, corrupted, or incomplete data.",
    "what is data preprocessing": "Data Preprocessing is preparing raw data for analysis by cleaning, normalizing, and transforming it.",
    "what is feature engineering": "Feature Engineering is creating new input features from existing data to improve model performance.",
    "what is feature selection": "Feature Selection is choosing the most relevant features to use in model training.",
    "what is overfitting": "Overfitting occurs when a model learns the training data too well and performs poorly on new data.",
    "what is underfitting": "Underfitting occurs when a model is too simple and fails to capture patterns in the data.",
    "what is cross validation": "Cross Validation is a technique to evaluate ML models by splitting data into training and validation sets multiple times.",
    "what is a confusion matrix": "A Confusion Matrix is a table used to evaluate classification models, showing true positives, false positives, true negatives, and false negatives.",
    "what is precision": "Precision measures the proportion of true positives among all positive predictions.",
    "what is recall": "Recall measures the proportion of true positives among all actual positive cases.",
    "what is f1 score": "F1 Score is the harmonic mean of precision and recall, balancing both metrics.",
    "what is bias in machine learning": "Bias is the error introduced by approximating a real-world problem too simply in a model.",
    "what is variance in machine learning": "Variance is the error introduced by a model being too sensitive to training data.",
    "what is standardization": "Standardization scales data to have zero mean and unit variance.",
    "what is normalization": "Normalization scales data to a fixed range, usually [0,1].",
    "what is PCA": "Principal Component Analysis (PCA) is a technique to reduce dimensionality while preserving variance.",
    "what is correlation": "Correlation measures the strength and direction of a linear relationship between two variables.",
    "what is covariance": "Covariance measures how two variables change together.",
    "what is outlier": "An outlier is a data point that differs significantly from other observations.",
    "what is missing data": "Missing data refers to absent or null values in a dataset.",
    "what is a dataset": "A dataset is a collection of data, often organized in rows (observations) and columns (features).",
    "what is a data frame": "A DataFrame is a 2D data structure in Python (pandas) for storing data in rows and columns.",
    "what is numpy": "NumPy is a Python library for numerical computing, supporting arrays and matrices.",
    "what is pandas": "Pandas is a Python library for data manipulation and analysis, providing DataFrames.",
    "what is matplotlib": "Matplotlib is a Python library for creating static, animated, and interactive visualizations.",
    "what is seaborn": "Seaborn is a Python library based on matplotlib for statistical data visualization.",
    "what is exploratory data analysis": "EDA is the process of summarizing main characteristics of data, often using visualizations.",
    "what is hypothesis testing": "Hypothesis Testing is a statistical method to determine if there is enough evidence to support a claim about a population.",
    "what is p-value": "P-value indicates the probability of obtaining results as extreme as observed under the null hypothesis.",
    "what is t-test": "T-test compares the means of two groups to see if they are statistically different.",
    "what is chi-square test": "Chi-Square Test checks if there is a significant association between categorical variables.",
    "what is random forest": "Random Forest is an ensemble learning method using multiple decision trees for classification or regression.",
    "what is decision tree": "Decision Tree is a model that splits data based on feature values to make predictions.",
    "what is logistic regression": "Logistic Regression is used to model binary outcomes using a logistic function.",
    "what is linear regression": "Linear Regression models the relationship between variables using a linear equation.",
    "what is k-means": "K-Means is an unsupervised algorithm that clusters data into K groups based on similarity.",
    "what is elbow method": "The Elbow Method helps choose the optimal number of clusters in K-Means by plotting SSE.",
    "what is gradient descent": "Gradient Descent is an optimization algorithm to minimize the cost function of a model.",
    "what is learning rate": "Learning Rate controls the step size during gradient descent optimization.",
    "what is epoch": "An Epoch is one complete pass of the training dataset through the learning algorithm.",
    "what is batch size": "Batch Size is the number of samples processed before the model is updated.",
    "what is activation function": "Activation Function introduces non-linearity in neural networks, e.g., ReLU, Sigmoid.",
    "what is confusion matrix": "A Confusion Matrix shows performance of classification models with TP, FP, TN, FN."
}

# Function to get chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    for key in responses:
        if key in user_input:
            return responses[key]
    return "Sorry, I don't know the answer to that yet."

# Main chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break
    response = chatbot_response(user_input)
    print("Bot:",response)