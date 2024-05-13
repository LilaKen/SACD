import numpy as np
import torch.nn.functional as F


def softmax_np(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)



class BLS_MultiClass:
    def __init__(self, input_dim, num_enhance_nodes, num_classes=5):
        self.input_dim = input_dim
        self.num_enhance_nodes = num_enhance_nodes
        self.num_classes = num_classes
        self.weights = []
        self.bias = []
        self.beta = None

    def random_init(self):
        weights = np.random.randn(self.input_dim, self.num_enhance_nodes)
        bias = np.random.randn(self.num_enhance_nodes)
        return weights, bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def add_enhance_nodes(self, X):
        weights, bias = self.random_init()
        self.weights.append(weights)
        self.bias.append(bias)

        V = X @ weights + bias
        H = self.sigmoid(V)
        return H

    def fit(self, X, Y):
        # 初始的映射层
        H = X

        # 增加增强节点并获取它们的输出
        H_enhanced = self.add_enhance_nodes(X)
        H = np.hstack([H, H_enhanced])

        # 伪逆和线性回归
        if self.beta is None:
            self.beta = np.linalg.pinv(H) @ Y
        else:
            H_inv = np.linalg.pinv(H)
            self.beta = np.vstack([self.beta, H_inv @ Y - H_inv @ H @ self.beta])


    def predict(self, X):
        H = X
        for weights, bias in zip(self.weights, self.bias):
            V = X @ weights + bias
            H_enhanced = self.sigmoid(V)
            H = np.hstack([H, H_enhanced])

        logits = H @ self.beta
        probabilities = softmax_np(logits)
        return probabilities

    def evaluate(self, X, Y):
        predictions = np.argmax(self.predict(X), axis=1)
        correct = np.sum(predictions == np.argmax(Y, axis=1))
        return correct / len(Y)
