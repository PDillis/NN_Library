import numpy as np
import matplotlib.pyplot as plt


# ------------- Basic Functions --------------- #

def normalize_rows(x):
	# Normalize the rows of a numpy array x of any size
	x_norm = np.linalg.norm(x, axis=1, keepdims=True)
	return x/x_norm


# ------------ Activation Functions -------------#

def arctan(x):
	# Computes the arctan of a numpy array x of any size
	return np.arctan(x)


def binary(x):
	# Returns 1 if x>=0, else 0 for a real number x
	return 1 if x >= 0 else 0


def v_binary(binary, x):
	# Returns the binary step for a numpy array x of any size
	v_binary = np.vectorize(binary)
	return v_binary(x)


def elu(x, alpha):
	# Returns alpha * (e^x - 1) for a real number x
	return alpha * (np.exp(x) - 1) if x < 0 else x


def v_elu(elu, x, alpha):
	# Vectorization of ELU: Computes elu of a numpy array x of any size
	v_elu = np.vectorize(elu)
	return v_elu(x, alpha)


def relu(x):
	# Computes the max(0, x) of a numpy array x of any size
	return np.maximum(0, x)


def sigmoid(x):
	# Computes the sigmoid of a numpy array x of any size
	return 1/(1+np.exp(-x))


def softmax(x):
	# Compute the softmax of a numpy array x of any size
	x_exp = np.exp(x)
	x_sum = np.sum(x_exp, axis=1, keepdims=True)
	return x_exp/x_sum


def tanh(x):
	# Compute the tanh of a numpy array x of any size
	return np.tanh(x)


# ------ Gradients of Activation Functions ------- #

def d_arctan(x):
	# Returns the derivative of the arctan evaluated at a numpy array x of any size
	return 1/(1 + np.power(x, 2))


def d_binary(x):
	# Returns the derivative of the binary evaluated at a real number x
	# It is actually undefined for x=0, but we will avoid this for the time being
	return 0 if x != 0 else 1


def d_v_binary(d_binary, x):
	# Returns the derivative of the binary evaluated at a numpy array x of any size
	d_v_binary = np.vectorize(d_binary)
	return d_v_binary(x)


def d_elu(x, alpha):
	# Returns the derivative of the elu function evaluated at the real number x
	return elu(x, alpha) + alpha if x < 0 else 1


def d_v_elu(d_elu, x, alpha):
	# Returns the derivative of the elu function evaluated at a numpy array x of any size
	d_v_elu = np.vectorize(d_elu)
	return d_v_elu(x, alpha)


def d_relu(x):
	# Returns the derivative of the relu function evaluated at a real number x
	return 0 if x < 0 else 1


def d_v_relu(d_relu, x):
	# Returns the derivative of the relu function evaluated at a numpy array x of any size
	d_v_relu = np.vectorize(d_relu)
	return d_v_relu(x)


def d_sigmoid(x):
	# Returns the derivative of the sigmoid evaluated at a numpy array x of any size
	return sigmoid(x) * (1 - sigmoid(x))


def d_tanh(x):
	# Returns the derivative of the tanh evaluated at a numpy array x of any size
	return 1 - np.power(tanh(x), 2)


# -------------- Reshaping images --------------- #

def image2vector(image):
	# image is of shape (length, height, depth) (usually depth = 3), so we
	# obtain a flat 'vector' image of shape (length*height*depth, 1)
	return image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

# ------------------- Loss ---------------------- #

def compute_loss(a3, y):
	# Implement the loss function

	m = y.shape[1]
	logprobs = np.multiply(-np.log(a3), y) + np.multiply(-np.log(1-a3), 1-y)
	loss = 1./m * np.nansum(logprobs)

	return loss


# -------------- Loss Functions ----------------- #

def l1_loss(y_result, y_truth):
	# Calculate the L1 loss of the R^m vector y_result vs the 'truth' vector y_truth
	return np.sum(abs(y_result-y_truth))


def l2_loss(y_result, y_truth):
	# Calculate the L2 loss of the R^m vector y_result vs the 'truth' vector y_truth
	return np.sum(np.dot(y_truth-y_result, y_truth-y_result))


# --------- Parameter Initialization ------------ #

def initialize_parameters_zeros(layer_dims):
	# Returns W1, b1, ..., WL, bL, the weights matrices an biases of each layer in the NN. The
	# input will be the layer dimensions (an array).

	parameters = {}
	L = len(layer_dims)

	for l in range(L):
		parameters["W" + str(l+1)] = np.zeros((layer_dims[l + 1], layer_dims[l]))
		parameters["b" + str(l+1)] = np.zeros((layer_dims[l + 1], 1))

	return parameters


def initialize_parameters_random(layer_dims):
	# Returns W1, b1, ..., WL, bL, the weights matrices and biases of each layer in the NN. The
	# input will be the layer dimensions (an array).

	parameters = {}
	L = len(layer_dims)

	for l in range(L):
		parameters["W" + str(l + 1)] = np.random.randn(layer_dims[l + 1], layer_dims[l])
		parameters["b" + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))

	return parameters


def initialize_parameters_he(layer_dims):
	# Returns W1, b1, ..., WL, bL, the weights matrices and biases of each layer in the NN. The
	# input will be the layer dimensions (an array).

	parameters = {}
	L = len(layer_dims)

	for l in range(L):
		parameters["W" + str(l + 1)] = np.random.randn(layer_dims[l + 1], layer_dims[l]) * np.sqrt(2/layer_dims[l])
		parameters["b" + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))

	return parameters

# ------------- Parameter Update ---------------- #

def update_parameters(parameters, grads, learning_rate):
	# Update the parameters using gradient descent

	# The number of layers in the NN is:
	L = len(parameters) // 2

	# Update rule for each parameter:
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	return parameters


# ----------------- Plotting -------------------- #

def plot_costs(costs, learning_rate):
	# Plot the costs as the training progresses
	plt.plot(costs)
	plt.ylabel(str(costs).capitalize())
	plt.xlabel('Iterations')
	plt.title("Learning rate: " + str(learning_rate))
	plt.show()


def plot_decision_boundary(model, X, Y):
	x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

	# Generate a grid of step-size h
	h = 0.01

	xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=h),
	                     np.arange(start=y_min, stop=y_min, step=h))

	# Predict the function value for the whole grid
	z = model(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)

	# Plot the contour and training examples
	plt.contour(xx, yy, z, cmap=plt.cm.Spectral)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
	plt.show()


def plot_boundaries(title, xmin, xmax, ymin, ymax, parameters, x_train, y_train):
	plt.title(str(title))
	axes = plt.gca()
	axes.set_xlim([xmin, xmax])
	axes.set_ylim([ymin, ymax])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), x_train, y_train)


# ----------------- Predictions -------------------- #

def predict_decision(parameters, x):
	# Returns a vector of 1 or 0 if the prediction is true.
	aL, cache = forward_propagation(x, parameters)
	predictions = (aL > 0.5)

	return predictions


# -------------- Forward Propagation ----------------- #

def forward_propagation(x, parameters):
	# Implements forward propagation and computes the loss. Returns yhat and the cache.

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]

	# Our NN architecture is Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid
	z1 = np.dot(W1, x) + b1
	a1 = relu(z1)
	z2 = np.dot(W2, a1) + b2
	a2 = relu(z2)
	z3 = np.dot(W3, a2) + b3
	a3 = sigmoid(z3)

	cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

	return a3, cache

# --------------- Backpropagation ------------------- #

def backpropagation(x, y, cache):
	# Implement backpropagation. Returns the dictionary of gradients.

	m = x.shape[1]
	(z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

	dz3 = 1./m * (a3-y)
	dW3 = np.dot(dz3, a2.T)
	db3 = np.sum(dz3, axis=1, keepdims=True)

	da2 = np.dot(W3.T, dz3)
	dz2 = np.multiply(da2, np.int64(a2 > 0))
	dW2 = np.dot(dz2, a1.T)
	db2 = np.sum(dz2, axis=1, keepdims=True)

	da1 = np.dot(W2.T, dz2)
	dz1 = np.multiply(da1, np.int64(a1 > 0))
	dW1 = np.dot(dz1, x.T)
	db1 = np.sum(dz1, axis=1, keepdims=True)

	gradients = {"dz3": dz3, "dW3": dW3, "db3": db3, "da2": da2,
	             "dz2": dz2, "dW2": dW2, "db2": db2, "da1": da1,
	             "dz1": dz1, "dW1": dW1, "db1": db1}

	return gradients
