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


# -------------- Loss Functions ----------------- #

def l1_loss(y_result, y_truth):
	# Calculate the L1 loss of the R^m vector y_result vs the 'truth' vector y_truth
	return np.sum(abs(y_result-y_truth))


def l2_loss(y_result, y_truth):
	# Calculate the L2 loss of the R^m vector y_result vs the 'truth' vector y_truth
	return np.sum(np.dot(y_truth-y_result, y_truth-y_result))


# ----------------- Plotting -------------------- #

def plot_costs(costs, learning_rate):
	# Plot the costs as the training progresses
	plt.plot(costs)
	plt.ylabel(str(costs).capitalize())
	plt.xlabel('Iterations')
	plt.title("Learning rate: " + str(learning_rate))
	plt.show()
