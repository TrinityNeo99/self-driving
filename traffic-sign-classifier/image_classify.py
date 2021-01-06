import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

training_file = "train.p"
validation_file = "train.p"
testing_file = "train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

signnames = pd.read_csv('signnames.csv')

# 可以随机展示9张图片
for i in range(9):
    index = random.randint(0, len(X_train))
    plt.subplot(3, 3, i + 1)
    signname = signnames["SignName"][int(y_train[index])]
    plt.title(signname)
    plt.imshow(X_train[index])
    plt.axis('off')

# plt.show()

# 对数据进行预处理
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def data_process(data):
    # this process include image normalization amd grayscale
    # data's shape is (num_image, 32, 32, 3)
    new_data=[]
    for img in data:
        gray_img = rgb2gray(img)
        norm_img=(gray_img-128)/128
        new_data.append(norm_img)
    return np.array(new_data).reshape((len(new_data), 32, 32, 1))

X_train=data_process(X_train)
X_valid=data_process(X_valid)
X_test=data_process(X_test)
print("数据预处理完成")

# 数据预处理后的结果展示
for i in range(9):
    index=random.randint(0,len(X_valid))
    plt.subplot(3,3,i+1)
    signname=signnames["SignName"][int(y_valid[index])]
    plt.title(signname)
    plt.imshow(X_valid[index][:,:,0])
    plt.axis('off')
print("数据预处理展示")
# plt.show()

# 进行训练
from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_valid=to_categorical(y_valid)
y_test=to_categorical(y_test)
#
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from keras.layers import Conv2D, MaxPooling2D
inputs = Input(shape=(32,32,1))
x=Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
x=Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x=MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
x=Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x=Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x=MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
x=Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x=Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x=MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
x=Flatten()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.7)(x)
outputs=Dense(42, activation='softmax')(x)

model = Model(inputs=inputs,outputs=outputs)

from keras.callbacks import TensorBoard

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_valid,y_valid), batch_size=32, epochs=3, callbacks=[TensorBoard(log_dir='./log')])
model.save('model.h5')
print("训练完成")

from keras.models import load_model
model= load_model("model.h5")
print("************************")
model.evaluate(X_test, y_test)
pred=model.predict(X_test[0:10])
for i in range(9):
    plt.subplot(3,3,i+1)
    signname=signnames["SignName"][np.argmax(pred[i])]
    plt.title(signname)
    plt.imshow(X_test[i][:,:,0])
    plt.axis('off')
print("评估完成")
plt.show()