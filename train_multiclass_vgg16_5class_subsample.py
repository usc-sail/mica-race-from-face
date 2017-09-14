'''
@Krishna Somandepalli - July 2017

Train a simple deep VGG-style CNN to predict race from face.
The race databases were constructed from here: https://docs.google.com/spreadsheets/d/16XkCRkipjKMGVZ1GQXG3ZOgUgtcYewbfIYgezlmM9Gc/edit#gid=0
5 race classes: caucasian, african, eastasian, asianindian, latino (nativeamerican/pacificis ignored due to lack of data)
split data into train and test manually - make sure test data has unique identities not seen in training. 
Test has balanced class data; Train has highly imbalanced class data

NOTE on DATA
The data cannot be released since some of the image databases required signing a data release document
You can recreate the database from the above google document
The race labels acquired from movie characters has been updated here by identity in case if one wants to use it for CASIA or such databases
The preprocessing scripts have been updated
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import metrics
from keras.callbacks import TensorBoard
import random
import json
import numpy as np
from itertools import groupby, islice, cycle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from PIL import Image
from keras.callbacks import CSVLogger
import tensorflow as tf


#function to read a list of image files and return an array for training/testing
ImLoad = lambda f: \
            np.asarray( [np.asarray(Image.open(i))*(1./255.0) for i in f] )[..., np.newaxis]

#tensorflow image format - standard VGG-16 with modifications for grayscale images
def generate_vgg16_conf1(num_classes, in_shape = (100, 100, 1)):
    """ modified  - smaller version of original VGG16  """
    # Block 1
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', \
                                              name='block1_conv1', input_shape=in_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    return model


def generate_vgg16(num_classes, in_shape = (100, 100, 1)):
    """ modified  - smaller version of original VGG16 with BatchNorm and Dropout """
    # Block 1
    with tf.device('/cpu:0'):
	    model = Sequential()
	    model.add(Conv2D(32, (3, 3),  padding='same', \
						name='block1_conv1', input_shape=in_shape))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))

	    model.add(Conv2D(32, (3, 3),  padding='same', name='block1_conv2'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

	    # Block 2
	    model.add(Conv2D(64, (3, 3),  padding='same', name='block2_conv1'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    
	    model.add(Conv2D(64, (3, 3), padding='same', name='block2_conv2'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

	    # Block 3
	    model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv1'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    
	    model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv2'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    
	    model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv3'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

	    # Block 4
	    model.add(Conv2D(256, (3, 3), padding='same', name='block4_conv1'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    
	    model.add(Conv2D(256, (3, 3), padding='same', name='block4_conv2'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    
	    model.add(Conv2D(256, (3, 3), padding='same', name='block4_conv3'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

	    # Classification block
	    model.add(Flatten(name='flatten'))
	    model.add(Dense(1024, name='fc1'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    
	    model.add(Dense(1024, name='fc2'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(Dropout(0.2))
	    model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    return model



classes = ['african', 'asianindian', 'caucasian', 'eastasian', 'latino']#, 'nativeamerican']
num_classes = len(classes) #5

#load race labels from a dictionary of following format:
#{"latino":[im1.jpg, im2.jpg,....], "caucasian":[/path/to/image1.jpg, /path/to/imag2/jpg]}
all_images = json.load(open('SSD_ALL_im_race_dict.json', 'r'))

# label encoder for one-hot encoding
labeler = LabelEncoder()
labeler.fit(classes)

# reading and preparing test images - test images selected to keep identities unseen from training
all_test_images = [i.strip() for i in open('all_test_images.txt', 'r').readlines()]
all_test_labels = [i.strip() for i in open('all_test_labels.txt', 'r').readlines()]
all_test_ = zip(all_test_images, all_test_labels)
all_test = [i for i in all_test_ if i[1] in classes]

[random.shuffle(all_images[k]) for k in classes]


test_labels_int = labeler.transform([i[1] for i in all_test])
test_labels = np_utils.to_categorical(test_labels_int)

test_images = ImLoad([i[0] for i in all_test])

# num test images per class
N_test = 100

num_images_per_class = [len(all_images[k]) for k in classes]

# batch generator helper
def rcycle(iterable):
        #http://davidaventimiglia.com/python_generators.html
        # this is itertools.cycle but shuffle from the second cycle onwards
    saved = []                 # In-memory cache
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)  # Shuffle every batch
        for element in saved:
            yield element

#classes_rcycle = [rcycle( random.sample( sorted(all_images[k])[::-1][N_test:], \
#				len(all_images[k][N_test:]) ) ) for k in classes]

batch_size_per_class = 10
# number of images per class to subsample
min_class_size = min(num_images_per_class) - N_test
#num_batches_per_ep = (min_class_size - N_test)/batch_size_per_class
#num_epochs = (max(num_images_per_class) - N_test)/(min_class_size - N_test)

# randomly sample from all classes to the min_class_size + shuffle
ALL_IMAGES = [ random.sample([i for i in all_images[k] if i not in all_test_images], min_class_size) for k in classes]
[random.shuffle(i) for i in ALL_IMAGES]


#classes_rcycle = [rcycle( random.sample( [i for i in all_images[k] if i not in all_test_images], min_class_size ) +  ) \
#				for k in classes]
print('preparing the image batch generator ----')
classes_rcycle = [rcycle(i) for i in ALL_IMAGES]
print("DONE LOADING IMAGES - - -- - - - - - - - - - - - - - - ")

# The efforts taken here to write a batch generator are to ensure class balance in each batch!
#im_labels are fixed for each batch, so we neednot redo this in the tarinig loop
im_labels_ = []
for cl in classes:
    im_labels_ += [cl for _ in range(batch_size_per_class)]
#encoded labels
labels_encoded = labeler.transform(im_labels_)
#one hot encoded
im_labels = np_utils.to_categorical(labels_encoded)

def image_batch_generator(classes=classes, im_labels_all = im_labels):

    while True:
        im_list = []
        im_labels_ = []
        if im_labels_ is None: im_labels_ = []
        for cl_i, cl in enumerate(classes):
            im_list += [i for i in islice( classes_rcycle[cl_i], batch_size_per_class )]
            if im_labels_all is None: 
                im_labels_ += [cl for _ in range(batch_size_per_class)]
        if im_labels_all is None: 
            labels_encoded = labeler.transform(im_labels_)
                #one hot encoded
            im_labels = np_utils.to_categorical(labels_encoded)
        else: 
            im_labels = im_labels_all
        im_array = ImLoad(im_list)
        yield (im_array, im_labels)#, im_list)


# fn. to get class-wise performance
def get_val_accuracy(all_class_posteriors, all_true_labels=test_labels):
    all_class_labels = np.argmax(all_class_posteriors,1).tolist()
    all_true_labels = np.argmax(all_true_labels,1).tolist()
    cl_acc_list = []
    for cl_i, cl in enumerate(classes):
        # print(cl)
        cl_ix = np.where(np.array(all_true_labels)==cl_i)[0].tolist()
        cl_acc = (np.sum(np.array(all_true_labels)[cl_ix]==np.array(all_class_labels)[cl_ix] ) *100.0)/len(cl_ix)
        cl_acc_list.append(cl_acc)
    return cl_acc_list, np.mean(cl_acc_list)


# RUN THE CNN MODEL
# with tf.device('/cpu:0'):
if True:
	# model load arch
	model = generate_vgg16_conf1(num_classes=num_classes)

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr = 0.00001, decay = 1e-6)
	#opt = keras.optimizers.Adam()
	# compile the model
	model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])


	train_generator = image_batch_generator()

	num_epochs = 30
	
	csv_logger = CSVLogger('log_multiclass_conf1_5class_09_12_2017.csv', append=True, separator=';')
	model_checkpoint = keras.callbacks.ModelCheckpoint("multiclass_conf1_5class.{epoch:02d}.hdf5", \
				monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	
	model.fit_generator(
		train_generator,
		steps_per_epoch = 3500,
		epochs=num_epochs,
		validation_data= (test_images, test_labels),
		validation_steps = 10,
		callbacks = [csv_logger, model_checkpoint])

	model.save('multiclass_conf1_5class_%dep_09_12_2017.h5' % (num_epochs))
	# num_batches_per_ep = 1500
	# ### TRAINING LOOP
	# for e in range(num_epochs):
	#     print('Epoch', e)
	#     for b in range(num_batches_per_ep):
	#         im_array, im_labels = train_generator.next()
	#         if not b%100: print(b)
	#         # resume training
	#         # just before ending training for this epoch show accuracies, etc
	#         if b == num_batches_per_ep-1:
	#             model.fit(im_array, im_labels, batch_size=250, epochs=1, verbose=2,\
	#                 shuffle=True, callbacks = [ csv_logger ])
	
	#         else:
	#             model.fit(im_array, im_labels, batch_size=250, epochs=1, verbose=0, \
	#                 shuffle=True)
	
	#         # predict ans save model for the last batch for this epoch - until then train
	#         ## TESTING SUBLOOP
	#         if b == num_batches_per_ep-1:
	#             # pred_labels = model.predict(test_images, batch_size=250, verbose=1)
	#             pred_prob = model.predict_proba(test_images, batch_size=250, verbose=1)
	#             print( 'val acc = ', get_val_accuracy(pred_prob) )
	#             model.save('multilabel_subsample_racenet_all_ims_ep%d.h5' % (e))
	            # np.savez('pred_info_4_ep%d' % (e), \
				# true_labels = test_labels_, pred_prob = pred_prob, \
				# im_list = test_image_list)
	
