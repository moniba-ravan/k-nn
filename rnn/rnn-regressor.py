import numpy as np


class RnnRegressor:
    def __init__(self,
                 r: int):
        self.r = r

        self.m = None
        self.n = None
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.m, self.n = self.x_train.shape

    def predict(self, x_test):
        m_test = x_test.shape[0]
        y_predict = np.zeros(m_test)
        for i in range(m_test):
            query = x_test[i]
            neighbors = np.zeros(self.m)
            neighbors = self.find_neighbors(query)
            y_predict[i] = np.mean(neighbors)

        return y_predict

    def find_neighbors(self, query):
        euclidean_distance = np.zeros(self.m)
        sorted_y_train = np.zeros(self.m)
        last_idx = 0
        for i in range(self.m):
            euclidean_distance[i] = self.euclidean(query, self.x_train[i])
            if euclidean_distance[i] <= self.r:
                sorted_y_train[last_idx] = self.y_train[i]

        return sorted_y_train[:last_idx]

    @staticmethod
    def euclidean(vector_x, vector_y):
        return np.sqrt(np.sum(np.square(vector_y - vector_x)))
