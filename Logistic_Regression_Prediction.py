import math

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Example linear model output
z = 1.72
prob = sigmoid(z)
decision = 1 if prob >= 0.5 else 0

print("Probability:", prob)
print("Class prediction:", decision)

import numpy as np

w = np.array([0.4, -0.2])
x = np.array([1.5, 0.7])
b = 0.1

z = np.dot(w, x) + b
prob = sigmoid(z)
decision = 1 if prob >= 0.5 else 0

print("z =", z)
print("Probability =", prob)
print("Class =", decision)
