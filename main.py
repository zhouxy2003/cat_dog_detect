import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Part 1 - Data Preprocessing

## 1.1对图像进行变换 增加泛化
train_datagen = ImageDataGenerator(
                    rescale=1./255,          #像素归一化
                    shear_range=0.2,         #改变图像形状
                    zoom_range=0.2,          #进行缩放
                    horizontal_flip=True,    #水平翻转
                )

test_datagen = ImageDataGenerator(
                    rescale=1. / 255,  # 像素归一化
                )

## 1.2 创建训练集生成器
train_set = train_datagen.flow_from_directory(
                    directory='./dataset/training_set',   #数据集路径
                    target_size=(64,64),                  #尺寸大小
                    batch_size=32,                        #每个批次大小
                    class_mode='binary'                   #分类模式：二分类
                )

test_set = test_datagen.flow_from_directory(
                    directory='./dataset/test_set',   #数据集路径
                    target_size=(64,64),                  #尺寸大小
                    batch_size=32,                        #每个批次大小
                    class_mode='binary'                   #分类模式：二分类
                )


#Part 2 - Building the CNN
cnn = tf.keras.models.Sequential() #是一个顺序模型 可以往里面加东西

# 2.1 first convolution layer
cnn.add(tf.keras.layers.Conv2D(
                    filters=32,               # 卷积层数量
                    kernel_size=(3,3),        # 卷积核大小 3x3矩阵
                    activation='relu',        # max(0,x) 引入非线性因素
                    input_shape=(64,64,3)     # 对应之前的图像大小,3是指图像为彩色rgb,黑白则为1，只指定一次
                )
)

# 2.2 pooling layer
cnn.add(tf.keras.layers.MaxPool2D(
                    pool_size=(2,2),         # 2x2大小矩阵
                    strides=(2,2)            # 步数为2
                )
)

# 2.3 second convolution layer
cnn.add(tf.keras.layers.Conv2D(
                    filters=32,               # 卷积层数量
                    kernel_size=(3,3),        # 卷积核大小 3x3矩阵
                    activation='relu',        # max(0,x) 引入非线性因素
                )
)
cnn.add(tf.keras.layers.MaxPool2D(
                    pool_size=(2,2),         # 2x2大小矩阵
                    strides=(2,2)            # 步数为2
                )
)

# 2.4 flattening layer
cnn.add(tf.keras.layers.Flatten())

# 2.5 full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))   # 图像大小为64x64 神经元为128

# 2.6 output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print(cnn.summary())
import visualkeras
visualkeras.layered_view(cnn, to_file='cat_dog_cnn.png', legend=True)

# Part 3 Training the CNN

# 3.1 compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3.2 training the CNN and evaluating the test_set
cnn.fit(x=train_set, validation_data=test_set,epochs=30)

cnn.save_weights('cat_dog_detect.weights.h5')

cnn.load_weights('cat_dog_detect.weights.h5')

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('./dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64,3))
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(train_set.class_indices)

print(result)





















