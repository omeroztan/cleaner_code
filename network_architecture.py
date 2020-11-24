from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D

' how will we construct a good searching function and suitable for evoluiotn etc..'

class NeuralArch:
    def __init__(self, data_manager_obj):
        # todo give better names
        self.model = Sequential()
        self.data_obj = data_manager_obj
        self.first_layer_size = self.data_obj.x_train.shape[1]
        self.x = self.data_obj.x_train.shape[1:]

    def build_nn(self):
        # todo how to have a dynamic architecture

        self.model.add(Dense(self.first_layer_size, activation='relu'))

        self.model.add(LSTM(6, input_shape=self.x,  activation='relu', return_sequences=True))
        self.model.add(Dropout(0.5))

        self.model.add(LSTM(4, activation='relu'))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(1, activation='relu'))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        return self.model
