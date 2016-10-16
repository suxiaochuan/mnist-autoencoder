from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils


input_dim = 28 * 28
encoding_dim = [500, 300]
n_categories = 10

input_img = Input(shape=(input_dim, ))
encoded_img = Input(shape=(encoding_dim[0], ))

encoded = list()
encoded.append(Dense(encoding_dim[0], activation='relu',
                     activity_regularizer=regularizers.activity_l1(10e-5))(input_img))
encoded.append(Dense(encoding_dim[1], activation='relu',
                     activity_regularizer=regularizers.activity_l1(10e-5))(encoded_img))

decoded = list()
decoded.append(Dense(input_dim, activation='sigmoid')(encoded[0]))
decoded.append(Dense(encoding_dim[0], activation='sigmoid')(encoded[1]))

autoencoder = list()
autoencoder.append(Model(input=input_img, output=decoded[0]))
autoencoder.append(Model(input=encoded_img, output=decoded[1]))

encoder = list()
encoder.append(Model(input=input_img, output=encoded[0]))
encoder.append(Model(input=encoded_img, output=encoded[1]))

autoencoder[0].compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder[1].compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, y_train_category), (x_test, y_test_category) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

y_train = np_utils.to_categorical(y_train_category)
y_test = np_utils.to_categorical(y_test_category)

print x_train.shape
print x_test.shape

autoencoder[0].fit(x_train, x_train, nb_epoch=50, batch_size=512, shuffle=True, validation_data=(x_test, x_test))

encoded_train = encoder[0].predict(x_train)
encoded_test = encoder[0].predict(x_test)

print encoded_train.shape
print encoded_test.shape

autoencoder[1].fit(encoded_train, encoded_train, nb_epoch=50, batch_size=512, shuffle=True,
                   validation_data=(encoded_test, encoded_test))

encoded_twice_train = encoder[1].predict(encoded_train)
encoded_twice_test = encoder[1].predict(encoded_test)

print encoded_twice_train.shape
print encoded_twice_test.shape

softmax_input = Input(shape=(encoding_dim[1], ))
softmax_output = Dense(n_categories, activation='softmax')(softmax_input)
softmax = Model(input=softmax_input, output=softmax_output)
softmax.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
softmax.fit(encoded_twice_train, y_train, nb_epoch=50, batch_size=512, shuffle=True,
            validation_data=(encoded_twice_test, y_test))

score_train = softmax.evaluate(encoded_twice_train, y_train)
score_test = softmax.evaluate(encoded_twice_test, y_test)
print 'Train accuracy:', score_train[1]
print 'Test accuracy:', score_test[1]


