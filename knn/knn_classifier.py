import numpy as np


class KnnClassifier:
    def __init__(self,
                 k: int):
        self.k = k

        self.m: int
        self.n: int
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.m, self.n= self. x_train.shape

    def predict(self, x_test):
        m_test = x_test.shape[0]
        y_predict = np.zeros(m_test)
        for i in range(m_test):
            query = x_test[i]
            neighbors = np.zeroes(self.k)
            neighbors = self.find_neighbors(query)
            y_predict[i] = np.argmax(np.bincount(neighbors))

        return y_predict

    def find_neighbors(self, query):
        euclidean_distance = np.zeros(self.m)
        for i in range(self.m):
            euclidean_distance[i] = self.euclidean(query, self.x_train[i])
        sorted_inx = euclidean_distance.argsort()
        sorted_y_train = self.y_train[sorted_inx]
        return sorted_y_train[:self.k]

    @staticmethod
    def euclidean(vector_x, vector_y):
        return np.sqrt(np.sum(np.square(vector_y - vector_x)))
