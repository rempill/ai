import numpy as np

class MySGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, learningRate=0.001, noEpochs=1000, batch_mode=False):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            if batch_mode:
                # Batch Gradient Descent
                gradients = [0.0 for _ in range(len(x[0]) + 1)]
                for i in range(len(x)):
                    ycomputed = self.eval(x[i])
                    crtError = ycomputed - y[i]
                    for j in range(len(x[0])):
                        gradients[j] += crtError * x[i][j]
                    gradients[-1] += crtError * 1
                # Average and update
                for j in range(len(gradients)):
                    gradients[j] /= len(x)
                    self.coef_[j] -= learningRate * gradients[j]
            else:
                # Stochastic Gradient Descent
                for i in range(len(x)):
                    ycomputed = self.eval(x[i])
                    crtError = ycomputed - y[i]
                    for j in range(len(x[0])):
                        self.coef_[j] -= learningRate * crtError * x[i][j]
                    self.coef_[-1] -= learningRate * crtError * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        return [self.eval(xi) for xi in x]


class MySGDClassifier:
    def __init__(self, lr=0.01, epochs=1000, loss='log'):
        self.lr = lr
        self.epochs = epochs
        self.loss = loss  # 'log', 'hinge', 'perceptron'
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        # Add bias term
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        self.w = np.zeros(X_bias.shape[1])
        for epoch in range(self.epochs):
            for i in range(n_samples):
                xi = X_bias[i]
                yi = y[i]
                z = np.dot(xi, self.w)
                if self.loss == 'log':
                    # Logistic loss (sigmoid)
                    y_pred = self.sigmoid(z)
                    grad = (y_pred - yi) * xi
                elif self.loss == 'hinge':
                    # Hinge loss (SVM)
                    # y in {-1, 1} for hinge loss
                    y_hinge = 2 * yi - 1
                    margin = y_hinge * z
                    if margin < 1:
                        grad = -y_hinge * xi
                    else:
                        grad = np.zeros_like(xi)
                elif self.loss == 'perceptron':
                    # Perceptron loss
                    y_perc = 2 * yi - 1
                    if y_perc * z <= 0:
                        grad = -y_perc * xi
                    else:
                        grad = np.zeros_like(xi)
                else:
                    raise ValueError(f"Unknown loss: {self.loss}")
                self.w -= self.lr * grad

    def predict_proba(self, X):
        """
        Returns sigmoid(z) for all loss types. For 'log' loss, this is the true probability.
        For 'hinge' and 'perceptron', this is just a score in (0,1), not a calibrated probability.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        z = np.dot(X_bias, self.w)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
