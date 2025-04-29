import numpy as np
from keras.preprocessing import image
from createCNN import create_cnn_model

cnn = create_cnn_model()
cnn.load_weights('./cat_dog_detect.weights.h5')

test_image = image.load_img('./dataset/single_prediction/cod4.jpg', target_size=(64,64,3))
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
#training_set.class_indices

if result[0][0] == 1:
    print('This is a dog🐕')
else:
    print('This is a cat🐱')

