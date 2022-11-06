import numpy as np


class LogisticRegression():

	def __init__(self, lrate = 0.01, epochs=1000):
		self.lrate = lrate
		self.epochs = epochs

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def cost_function(self, h, theta, y):
		m = len(y)
		cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
		return cost

	def gradient_descent(self, x, h, theta, y, m):
		gradient_value = np.dot(x.T, (h - y)) / m
		theta -= self.lrate * gradient_value
		return theta

	def fit(self, x, y):
		self.theta = []
		self.cost = []
		x = np.insert(x, 0, 1, axis=1)
		m = len(y)
		for i in np.unique(y):
			y_one_vs_all = np.where(y == i, 0, 1)
			theta = np.zeros(x.shape[1])
			cost = []
			for _ in range(self.epochs):
				z = x.dot(theta)
				h = self.sigmoid(z)
				theta = self.gradient_descent(x, h, theta, y_one_vs_all, m)
				cost.append(self.cost_function(h, theta, y_one_vs_all))
			self.theta.append((theta, i))
			self.cost.append((cost, i))
		np.save("values.npy", self.theta)
		return self

	def load_values(self, file):
		self.theta = np.load(file)

	def predict(self, x):
		x = np.insert(x, 0, 1, axis=1)
		x_predicted = [max((self.sigmoid(i.dot(theta)), c) for theta, c in self.theta)[1] for i in x]

	def score(self, x, y):
		''' Tests accuracy '''
		score = sum(self.predict(x) == y) / len(y)
		return score

	def plot_cost(self, costh):
		for cost, c in costh:
			plt.plot(range(len(cost)),cost,'r')
			plt.title("Convergence Graph of Cost Function of class " + str(c) +" vs All")
			plt.xlabel("Number of Iterations")
			plt.ylabel("Cost")
			plt.show()