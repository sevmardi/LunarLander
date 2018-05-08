from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Activation


class Network:

    def __init__(self, space, output):
        super(Network, self).__init__()
        self.space = space
        self.output_space = output

    def model(self):
        model = Sequential()
        print("Input shape: " + str((1, ) + self.input_space))
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.input_shape))

        model.add(Dense(40))
        mode.add(Activation('relu'))

        model.add(Dense(40))
        mode.add(Activation('relu'))

        model.add(Dense(self.output_space))
        mode.add(Activation('linear'))

        return model
