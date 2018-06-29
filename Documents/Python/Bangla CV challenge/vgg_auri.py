from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Activation,Dropout
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint

X_train_all= np.load("X_train_all.npy")
X_test_all = np.load("X_test_all.npy")
y_train_all = np.load("y_train_all.npy")

X_train_all = X_train_all.reshape(X_train_all.shape[0],64, 64,1).astype('float32')
X_test_all = X_test_all.reshape(X_test_all.shape[0],64, 64,1).astype('float32')

X_train_all = X_train_all / 255
X_test_all = X_test_all / 255

indices=list(range(len(X_train_all)))
np.random.seed(42)
np.random.shuffle(indices)

ind=int(len(indices)*0.80)
# train data
X_train=X_train_all[indices[:ind]]
y_train=y_train_all[indices[:ind]]
# validation data
X_val=X_train_all[indices[-(len(indices)-ind):]]
y_val=y_train_all[indices[-(len(indices)-ind):]]



datagen = ImageDataGenerator(
    #    featurewise_center=True,
   #     samplewise_center=True,
  #      featurewise_std_normalization=True,
 #       samplewise_std_normalization=True,
#        zca_whitening=True,
#        zca_epsilon=1e-06,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
#        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
 #       horizontal_flip=True,
    #    fill_mode='nearest'
        )


datagen.fit(X_train)
datagen.fit(X_val)




print(X_train_all.shape)
print(X_test_all.shape)
print(y_train_all.shape)

def VGG_16(weights_path=None):
    model = Sequential()
    #  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Conv2D(64, (5, 5), input_shape=(64, 64, 1), activation='relu', padding='same'))
    #  model.add(ZeroPadding2D((1,1),input_shape=(64,64,1)))
    #  model.add(Conv2D(64, (3, 3), activation='relu'))
    # output size = 64*64*64

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    #   model.add(ZeroPadding2D((1,1)))
    #  model.add(Conv2D(64, (3, 3), activation='relu'))
    # output size = 64*64*64
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # output size = 32*32*64

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #   model.add(ZeroPadding2D((1,1)))
    #  model.add(Conv2D(128, (3, 3), activation='relu'))
    # output size = 32*32*128

    #   model.add(ZeroPadding2D((1,1)))
    #  model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # output size = 32*32*128
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # output size = 16*16*128

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #   model.add(ZeroPadding2D((1,1)))
    #   model.add(Conv2D(256, (3, 3), activation='relu'))
    # output size = 16*16*256
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #   model.add(ZeroPadding2D((1,1)))
    #   model.add(Conv2D(256, (3, 3), activation='relu'))
    # output size = 16*16*256
    #   model.add(ZeroPadding2D((1,1)))
    #   model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # output size = 16*16*256
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # output size = 8*8*256

    #   model.add(ZeroPadding2D((1,1)))
    #   model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #  model.add(ZeroPadding2D((1,1)))
    #  model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #   model.add(ZeroPadding2D((1,1)))
    #   model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # 8*8*512
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # 4*4*512
    #
    #    model.add(ZeroPadding2D((1,1)))
    #    model.add(Conv2D(512, 3, 3, activation='relu'))
    #    model.add(ZeroPadding2D((1,1)))
    #    model.add(Conv2D(512, 3, 3, activation='relu'))
    #    model.add(ZeroPadding2D((1,1)))
    #    model.add(Conv2D(512, 3, 3, activation='relu'))
    #    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    if weights_path:
        model.load_weights(weights_path)

    return model


model = VGG_16()
model.summary()


path_model='model_filter.h5' # save model at this location after each epoch
K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model = VGG_16() # create the model
K.set_value(model.optimizer.lr,1e-3)

h=model.fit(x=X_train,
            y=y_train,
            batch_size=1,
            epochs=2,
            verbose=1,
            validation_data=(X_val,y_val),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )
