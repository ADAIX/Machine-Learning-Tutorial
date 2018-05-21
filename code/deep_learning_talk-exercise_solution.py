"""
Solution for the exercise of slide 49

Architecture for the neural net depicted

Number of free parameters given by the `summary()` method

"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Conv2D(10, (5, 5), input_shape=(32, 32, 1)))
model.add(MaxPool2D())
model.add(Conv2D(25, (5, 5)))
model.add(MaxPool2D())
model.add(Conv2D(100, (4, 4)))
model.add(MaxPool2D())
# To transform a 3D tensor into a vector
# one needs to flatten it
model.add(Flatten())
model.add(Dense(10))

# Should give you the solution: 47,645 parameters
model.summary(50)

