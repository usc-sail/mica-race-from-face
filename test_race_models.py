'''
@ Krishna Somandepalli July 2017 - performance evaluation scripts
usage:
edit the variable dir_path_glob and use the glob path style for class-specific directory of images
./test_race_model.py pretrained_model_in_keras_or_tf.hdf5 
OR if you have a bunch of predicted labels
./test_race_model.py predicted_posteriors.npz
format for predicted_posteriors.npz: x['pred_prob']: predicted posteriors, x['true_labels'],...

Outputs plots with class-specific ROC curves and micro/macro ROCs for performance eval
'''


from __future__ import print_function
from keras.models import load_model
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
import sys
TRAIN_MODEL = False
if not TRAIN_MODEL: K.set_learning_phase(0)
from pylab import *
import cv2
import glob, os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


labeler = LabelEncoder()

# where can I find class-specific images for testing!
dir_path_glob = "../all_images/test/%s/*.jpg"

# dimensions of the generated pictures for each filter.
img_width = 100
img_height = 100
n_channels = 1
classes = ['african', 'asianindian', 'caucasian', 'eastasian', 'latino']#, 'nativeam']
num_im_per_class = 50
true_labels = []
for i in classes:
	true_labels += [i for _ in range(num_im_per_class)]

model_fname = sys.argv[1]


if model_fname.endswith('.h5'):
	model = load_model(model_fname)
	# model = vgg16.VGG16(weights='imagenet', include_top=False)
	print('Model loaded.')

	#model.summary()

	all_class_posteriors = []
	all_class_labels = []
	all_true_labels = []

	for class_ix, class_ in enumerate(classes):
		print(class_),
		class_posteriors = []
		class_labels = []
		true_labels = [class_ix for i in range(1000)]
		for im_file in glob.glob(dir_path_glob % (class_)):
			im = (cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)*(1./255)).reshape((1,100,100,1))
			class_posteriors.append(model.predict(im, verbose=0)[0])
			class_labels.append(model.predict_classes(im, verbose=0)[0])
		print('accuracy = ', (sum(array(class_labels)==array(true_labels))*100.0)/len(class_labels), '%')
		all_class_posteriors.append(array(class_posteriors))
		all_class_labels += class_labels
		all_true_labels += true_labels
	all_class_posteriors = np.vstack(all_class_posteriors)


elif model_fname.endswith('npz'):
	pred_output = np.load(sys.argv[1])
	all_true_labels = labeler.fit_transform(true_labels).tolist()
	all_class_posteriors = pred_output['pred_prob']
	all_class_labels = np.argmax(all_class_posteriors,1).tolist()

	for cl_i, cl in enumerate(classes):
		print(cl)
		cl_ix = np.where(array(all_true_labels)==cl_i)[0].tolist()
		cl_acc = (sum(array(all_true_labels)[cl_ix]==array(all_class_labels)[cl_ix] ) *100.0)/len(cl_ix)
		print("accuracy = ", cl_acc, '%')


overall_acc = sum( array(all_true_labels) == array(all_class_labels) )/float(len(all_class_labels))
print('AVERAGE ACCURACY'),
print(overall_acc*100)
ion()
CM = confusion_matrix(all_true_labels, all_class_labels)
df_cm = pd.DataFrame(CM, index = classes,columns = classes)
figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='d')
# setp(s.axes.get_yticklabels(), rotation=0)
# sn.heatmap(df_cm, annot=True, fmt='%d')
# show()
all_true_labels_bin = label_binarize(all_true_labels, classes=[0,1,2,3,4,5])
n_classes = len(classes)
lw=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_true_labels_bin[:, i], all_class_posteriors[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(all_true_labels_bin.ravel(), all_class_posteriors.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
figure()
plot(fpr["micro"], tpr["micro"],
         label='binary-predicition-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plot(fpr["macro"], tpr["macro"],
         label='class-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plot(fpr[i], tpr[i], color=color, lw=lw,
             label='Class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plot([0, 1], [0, 1], 'k--', lw=lw)
xlim([0.0, 1.0])
ylim([0.0, 1.05])
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROCs for race from face prediction')
legend(loc="lower right")
# show()1

