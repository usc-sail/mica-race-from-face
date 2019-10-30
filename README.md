Predicting race from faces for movie data  
========
**NOTE:
Race is a social construct. We understand that our objective here of attempting to classify race, regardless of the definitional distinction between race and ethnicity, is an extremely culturally sensitive issue. Any approach we use to classify race will be done with as much respect as this problem deserves to be treated with and as delicately as possible -- in the ultimate interest of understanding diversity of representation of people on screen. This is a work in progress and has plenty of scope for improvement.**


Please see the details of the definitions of race, taxonomy of race recognition, database details, neural network models and performance evaluation details in the [wiki page here](https://github.com/usc-sail/mica-race-from-face/wiki)  


## Contents:  

### race labels  
actor_race.txt: labels for IMDB faces obtained as described in [Ramakrishna et. al., 2017, ACL](http://sail.usc.edu/publications/files/linguistic-analysis-differences_camera-ready.pdf)  
lfw_race_dict.json: mapping of race for identities in the [LFW dataset](http://vis-www.cs.umass.edu/lfw/)  

### aligning faces  
average_face.ppm: average face used for in-plane face alignment  
average_face.json: landmark information for the average face for alignment  
faceswap.py: utilities for face alignment  

average_faces: directory with average faces for the different classes in our data  

### Convolution neural networks (CNN) training scripts and pretrained models  
pretrained_models: Directory containing 5-class pretrained models  
train_multiclass_vgg16_5class_subsample.py: Keras training script and CNN architecture details  
test_race_models.py: performance evaluation scripts (ROC, accuracy, confusion matrix, etc.,)  
utils.py: miscellaneous handy utility functions  
