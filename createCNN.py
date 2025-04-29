import tensorflow as tf

def create_cnn_model():

    # Part 2 - Building the CNN
    cnn = tf.keras.models.Sequential()  # 是一个顺序模型 可以往里面加东西

    # 2.1 first convolution layer
    cnn.add(tf.keras.layers.Conv2D(
        filters=32,  # 卷积层数量
        kernel_size=(3, 3),  # 卷积核大小 3x3矩阵
        activation='relu',  # max(0,x) 引入非线性因素
        input_shape=(64, 64, 3)  # 对应之前的图像大小,3是指图像为彩色rgb,黑白则为1，只指定一次
    )
    )

    # 2.2 pooling layer
    cnn.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),  # 2x2大小矩阵
        strides=(2, 2)  # 步数为2
    )
    )

    # 2.3 second convolution layer
    cnn.add(tf.keras.layers.Conv2D(
        filters=32,  # 卷积层数量
        kernel_size=(3, 3),  # 卷积核大小 3x3矩阵
        activation='relu',  # max(0,x) 引入非线性因素
    )
    )
    cnn.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),  # 2x2大小矩阵
        strides=(2, 2)  # 步数为2
    )
    )

    # 2.4 flattening layer
    cnn.add(tf.keras.layers.Flatten())

    # 2.5 full connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))  # 图像大小为64x64 神经元为128

    # 2.6 output layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    return cnn