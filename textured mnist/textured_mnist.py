from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import keras.backend as K
import pickle

def load_mnist():
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test


path = '../data/normalized_brodatz/'

images = []

for i in range(1, 112):
    filename = path + "D" + str(i) + ".tif"
    im = Image.open(filename)
    images.append(np.array(im))

images = np.array(images)

num_images = 5
fig = plt.figure(figsize=(7., 7.))
grid = ImageGrid(fig, 111, nrows_ncols=(num_images, num_images), axes_pad=0.1)
for ax, im in zip(grid, images[0:num_images*num_images]):
    ax.imshow(im)
plt.show()

# now we have loaded textures we want to be able to get a
# 28x28 slice from them
width = 28
height = 28

dimx = images[0].shape[0]
dimy = images[0].shape[1]


x = np.random.randint(0, dimx-28)
y = np.random.randint(0, dimy-28)

small_images = []
for i in range(num_images**2):
    img_small = images[i][x:x+width, y:y+height]
    small_images.append(img_small)
num_images = 5
fig = plt.figure(figsize=(7., 7.))
grid = ImageGrid(fig, 111, nrows_ncols=(num_images, num_images), axes_pad=0.1)
for ax, im in zip(grid, small_images[0:num_images*num_images]):
    ax.imshow(im)
plt.show()

# mnist mask
x_train, x_test, y_train, y_test = load_mnist()

x_train_modified = []
x_test_modified = []

# recommended textures
idx = [3, 4, 6, 14, 16, 17, 46, 52, 80, 104]

for i in range(len(x_train)):
    mask = x_train[i]
    label = y_train[i]
    threshold = 0.1

    # create mask
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row][col] > threshold:
                mask[row][col] = 1
            else:
                mask[row][col] = 0

    img_small = images[idx[label]][x:x + width, y:y + height]
    x_train_modified.append(img_small * mask.reshape(width, height))

for i in range(len(x_test)):
    mask = x_test[i]
    label = y_test[i]
    threshold = 0.1

    # create mask
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row][col] > threshold:
                mask[row][col] = 1
            else:
                mask[row][col] = 0

    img_small = images[idx[label]][x:x + width, y:y + height]
    x_test_modified.append(img_small * mask.reshape(width, height))

num_images = 10
fig = plt.figure(figsize=(7., 7.))
grid = ImageGrid(fig, 111, nrows_ncols=(num_images, num_images), axes_pad=0.1)
for ax, im in zip(grid, x_train_modified[0:num_images**2]):
    ax.imshow(im)
plt.show()

# save the dataset

save = True

if save:
    print("Saving data")
    pickle.dump(np.array(x_train_modified), open('x_train_textured.p', 'wb'))
    pickle.dump(np.array(x_test_modified), open('x_test_textured.p', 'wb'))


print("Finished")
