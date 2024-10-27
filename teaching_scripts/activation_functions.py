""" 
Just a script to plot various activation functions.

author: Fabrizio Musacchio
date: Okt 27, 2024
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

# set global properties for all plots:
plt.rcParams.update({'font.size': 14})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% DEFINE PATHS
RESULTSPATH = '../results/teaching_material/'
# check whether the results path exists, if not, create it:
if not os.path.exists(RESULTSPATH):
    os.makedirs(RESULTSPATH)
# %% ACTIVATION FUNCTIONS
# define the activation functions:

# sigmoid: f(x) = 1 / (1 + exp(-x))
# Sigmoid is used in the output layer of a neural network to normalize the output values to a probability distribution.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# Tanh is similar to the sigmoid function but has a range of [-1, 1], which helps with training by centering the data.
def tanh(x):
    return np.tanh(x)

# ReLU: f(x) = max(0, x)
# ReLU is a simple and effective activation function that has been shown to outperform sigmoid and tanh in many cases.
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU: f(x) = max(alpha*x, x)
# Leaky ReLU is a variant of ReLU that allows a small gradient when x < 0, which can help with training stability.
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

# Exponential Linear Unit: f(x) = x if x > 0, alpha*(exp(x)-1) if x <= 0
# ELU is similar to ReLU but allows a small negative output when x < 0, which can help with training stability.
def elu(x, alpha=1):
    return np.where(x > 0, x, alpha*(np.exp(x)-1))

# Swish: f(x) = x / (1 + exp(-beta*x)) = x * sigmoid(beta*x)
# Swish is a self-gated activation function, where the gating mechanism is controlled by the input itself.
def swish(x, beta=1):
    return x * sigmoid(beta * x)  

# Mish: f(x) = x * tanh(log(1 + exp(x)))
# Mish is a smooth and non-monotonic activation function that has been shown to outperform ReLU and Swish.
def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

# Softmax: f(x) = exp(x) / sum(exp(x))
# Softmax is used in the output layer of a neural network to normalize the output values to a probability distribution.
def softmax(x):
    exps = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exps / np.sum(exps)
# %% PLOT ACTIVATION FUNCTIONS
# define the x range:
x = np.linspace(-5, 5, 100)

# create a loop plotting the differen activation functions in separate plots:
activation_functions = [sigmoid, tanh, relu, leaky_relu, elu, swish, mish]
activation_names = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Swish', 'Mish']
for i, activation in enumerate(activation_functions):
    plt.figure(figsize=(6, 4))
    plt.plot(x, activation(x))
    plt.title(activation_names[i])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTSPATH + f'activation_{activation_names[i]}.png', dpi=300)
    plt.show()
    
# plotting softmax separately as it is typically used over a range of values:
x_softmax = np.linspace(-2, 2, 11)  # define a small vector of values
plt.figure(figsize=(6, 4))
plt.plot(x_softmax, softmax(x_softmax), 'o-')  # plot softmax on vector
plt.title('Softmax')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTSPATH + 'activation_Softmax.png', dpi=300)
plt.show()
# %% END
    