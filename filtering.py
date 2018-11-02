import copy
import numpy as np
from scipy.stats import norm

class CollaborativeFiltering:
    def __init__(self, k=5):
        self.usr_num = 0
        self.data = None
        self.item_num = 0
        self.k = k

    def fit(self, data):
        self.data = data
        self.usr_num = len(self.data)
        self.item_num = len(self.data[0])
        return

    def predict(self, predict_id):
        vect = self.data[predict_id[0]]
        vec_norm = np.linalg.norm(vect)
        
        candidates = [0] * self.k
        ratings = [0] * self.k
        
        for line in self.data:
            if line[predict_id[1]] != 0:
                sim = np.dot(np.array(line), np.array(vect)) / vec_norm / np.linalg.norm(line)
                if sim > min(candidates):
                    pos = candidates.index(min(candidates))
                    candidates[pos] = sim
                    ratings[pos] = line[predict_id[1]]
                    
        if not np.sum(np.array(ratings)):
            return 5
        else:
            return np.mean(np.nonzero(np.array(ratings)))

    def evaluation(self, data, test_x, test_y):
        # test_x list whose element is predict_id (row, col)
        # test_y list of actual ratings
        
        # train the model
        self.fit(data)

        # evaluation
        res = 0
        for i in range(len(test_x)):
            rating_hat = self.predict(test_x[i])
            res += (rating_hat - test_y[i])**2

        return res
