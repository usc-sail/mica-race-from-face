#@Krishna Somandepalli - you can never have enough util functions
# a collection of handy util functions (sources: stackoverflow, self,..)
# mostly scripts to clean up image directories before running CNN, etc
# 1. generate avg image from a directory fulll of images
# 2. find missing sequence of numbers
# 3. find invalid image files 


from pylab import *
import os
import json
import sys
import numpy as np
from PIL import Image
import glob
from skimage.io import imsave
'''
Create an average image with a directory full of images
'''
def generate_avg_image(dir_='.'):
	# change the glob structure to regex only certain images
	if type(dir_)==list: imlist = dir_
	elif type(dir_)==str: imlist = glob.glob(dir_+'/*')
	w,h = Image.open(imlist[0]).size
	N = len(imlist)
	arr = np.zeros((h,w),np.float)
	for im in imlist:
	    imarr = np.array(Image.open(im),dtype=np.float)
	    arr = arr+imarr/N
	arr = np.array(np.round(arr),dtype=np.uint8)
	#imshow(arr)
	#show()
	return arr

'''
Find missing elements in a sequence of numbers
Used for debugguing missing files in softlinked ranges
'''
def missing_elements(L):
	L = sorted(L)
	start, end = L[0], L[-1]
	return sorted(set(range(start, end + 1)).difference(L))

'''
Finds invalid image format files in a dir
'''
def find_bad_apples(dir_ = '.'):
    L = glob.glob(dir_+'/*')  
    Lnot = []
    for i in L:
        try: Image.open(i)
        except IOError: Lnot.append(i); print(i)
    return Lnot
