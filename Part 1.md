1. Classification Basics
a. Difference between classification and regression
Classification is about predicting categories or labels. The output is discrete (like “spam” or “not spam”).
Regression predicts numerical values. The output is continuous (like predicting house prices).

Examples:
Binary Classification (only 2 classes):
Spam vs. Not Spam email detection
Predicting if a customer will churn (Yes/No)
Multiclass Classification (more than 2 classes):
Handwritten digit recognition (0–9)
Classifying types of flowers (Iris-setosa, Iris-virginica, Iris-versicolor)

Evaluation Metrics
Accuracy – The percentage of correct predictions out of all predictions.
Accuracy = (Correct predictions / Total predictions)
Precision – Out of all items predicted as positive, how many were actually positive?
High precision means few false positives.
Recall – Out of all the actual positive items, how many did the model correctly detect?
High recall means few false negatives.
F1 Score – The harmonic mean of precision and recall. It balances both metrics, useful when classes are imbalanced.

Confusion Matrix – A table that shows:
True Positives
True Negatives
False Positives
False Negatives
It helps visualize where the model is making mistakes.

2. Logistic Regression
a. Why is logistic regression a classification algorithm?
Even though it has “regression” in its name, it predicts probabilities of classes (like 0 or 1). It outputs a value between 0 and 1 and then converts it into a class label, making it a classification model.

b. Role of the sigmoid function
The sigmoid function squeezes any real number into a 0–1 range, so it can be interpreted as a probability.
If the probability > 0.5, the model usually predicts “1”; otherwise “0”.

c. Advantages and disadvantages
Advantages:
Simple and easy to interpret
Works well when the classes are linearly separable
Disadvantages:
Struggles with complex, non-linear data
Not ideal for high-dimensional or large feature spaces without regularization

3. K-Nearest Neighbors (KNN)
a. Why KNN is non-parametric and lazy
Non-parametric means the algorithm does not assume any fixed form or equation for the data.
Lazy learning means it does not train a model in advance; instead, it stores the data and makes decisions only when a new point needs to be classified.

b. Steps of KNN classification
Choose a value of K (number of neighbors).
Calculate the distance between the new data point and all existing training points (e.g., Euclidean distance).
Pick the K closest neighbors.
Perform majority voting among those neighbors.
Assign the class that appears most often.

c. Effect of small K vs. large K
Small K (e.g., K = 1):
More sensitive to noise
Can lead to overfitting
Large K:
Smoother decision boundary
More stable predictions
Might underfit if K is too large

d. Importance of feature scaling
KNN relies on distance to compare data points.
If features are not scaled (e.g., age in years vs. income in pesos), the feature with larger values dominates the distance calculation.
