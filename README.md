# Two-Layer Neural Network (from Scratch)

This project implements a **2-layer feedforward neural network** using only NumPy — no deep-learning frameworks.  
It was built to understand how **forward propagation**, **backpropagation**, and **gradient descent** work at the mathematical and code level.

---

## Overview

The network learns to classify handwritten digits (e.g., MNIST) from flattened 28×28 pixel images (784 features).

**Architecture:**
- **Input layer:** 784 neurons (one per pixel)
- **Hidden layer:** 10 neurons, ReLU activation  
- **Output layer:** 10 neurons, Softmax activation  

---

## Mathematical Formulation

### Forward Propagation

Given:
- $X \in \mathbb{R}^{784 \times m}$ : input matrix (each column = one example)  
- $W^{[1]} \in \mathbb{R}^{10 \times 784}$, $b^{[1]} \in \mathbb{R}^{10 \times 1}$  
- $W^{[2]} \in \mathbb{R}^{10 \times 10}$, $b^{[2]} \in \mathbb{R}^{10 \times 1}$

Steps:

$$
\begin{aligned}
Z^{[1]} &= W^{[1]} X + b^{[1]} \\
A^{[1]} &= \text{ReLU}(Z^{[1]}) = \max(0, Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= \text{softmax}(Z^{[2]})
\end{aligned}
$$

The **softmax** function converts logits into probabilities:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

### Backpropagation

We minimize the cross-entropy loss:

$$
L = -\frac{1}{m} \sum_{i=1}^m y_i \log(\hat{y}_i)
$$

Gradients are computed as:

$$
\begin{aligned}
dZ^{[2]} &= A^{[2]} - Y \\
dW^{[2]} &= \frac{1}{m} \, dZ^{[2]} (A^{[1]})^T \\
db^{[2]} &= \frac{1}{m} \sum dZ^{[2]} \\
dZ^{[1]} &= (W^{[2]})^T dZ^{[2]} \odot g'(Z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} \, dZ^{[1]} X^T \\
db^{[1]} &= \frac{1}{m} \sum dZ^{[1]}
\end{aligned}
$$

where  

$$
g'(Z^{[1]}) =
\begin{cases}
1, & \text{if } Z^{[1]} > 0 \\
0, & \text{otherwise}
\end{cases}
$$

---

### Parameter Update (Gradient Descent)

$$
\begin{aligned}
W^{[l]} &= W^{[l]} - \alpha \, dW^{[l]} \\
b^{[l]} &= b^{[l]} - \alpha \, db^{[l]}
\end{aligned}
$$

where $\alpha$ is the learning rate.

---

# Issues:
~~Currently this has a pretty low accuracy, and I hope to improve it with some time~~.  
~~As of right now it's stuck at 10% :(~~
It now gets about 95% on the test set :) 
The root cause was that the raw pixel values (0–255) created extremely large activations early in the network.
This pushed the softmax outputs into saturation, which collapsed the gradients and prevented learning entirely.
The solution I've figured out (or so forgot to do) was to normalize the dataset. 
Normalizing the input to [0,1] fixed the numerical instability:
- activations remained in a reasonable range
- softmax produced meaningful class probabilities
- gradients flowed correctly

