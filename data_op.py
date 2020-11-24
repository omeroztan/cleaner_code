"""This file is responsible to manage numpy arrays needed for the neural network.
Steps are; splitting the data to train and validation, normalizing and giving the data time-stepped manner"""
from getting_values import PathManager, PathToValues

class DataManager:
    """ I need to put reliable docstrings here"""
    def __init__(self, data_domain, index, step_size):

        sender = PathManager(data_domain, index)
        test_data_path, train_data_path = sender.send()

        converter = PathToValues(test_data_path, train_data_path)
        self.train_data = converter.convert_train()
        self.test_data = converter.convert_test()

        self.step_size = step_size

        self.train, self.validation = self.split(0.9)
        self.norm_train = self.normalize(self.train, self.train)
        self.norm_validation = self.normalize(self.validation, self.train)
        self.x_train, self.y_train = self.step(self.norm_train, steps=step_size)
        self.x_validation, self.y_validation = self.step(self.norm_validation, steps=step_size)

    def normalize(self, normalized, according):
        """normalize(self, data, train_data)"""
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0.001, 0.999))
        scaler.fit(according)
        return scaler.transform(normalized)

    def step(self, data, steps=None):
        if steps:
            steps = steps
        else:
            steps = self.step_size

        import numpy as np

        x, y = list(), list()
        for i in range(len(data)-steps):
            seq_x = data[i:i+steps]
            seq_y = data[i+steps]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

    def split(self, split_number):
        data = self.train_data

        return data[:int(len(data) * split_number)], \
               data[-int(len(data) * (1 - split_number)):]  # train, test

# I am not sure, whether we need another class here...
