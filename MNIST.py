# Import library
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Define a Sequential model
model = Sequential()
# Add Conv2D layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
# View the model summary
print(model.summary())
# Load the model
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Reshape to fit the input size of ConvNet
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
# Converting into categorical value
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# Standardising the pixel values
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, batch_size=64, epochs=5, validation_data=(test_images, test_labels))

