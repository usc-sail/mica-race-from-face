#!/usr/bin/python

"""
@Krishna Somandepalli
Part of the following functions are taken from:
http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
Courtesy: Matthew Earl - 2015

"""

import sys
import cv2
import dlib
import numpy
from pylab import *

PREDICTOR_PATH = "/data/workspace/mica/race/scripts/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

# google sdk lmarks
# FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = [2, 13, 14, 15, 16]  # list(range(48, 68))
RIGHT_BROW_POINTS = [3, 4]  # list(range(17, 22))
LEFT_BROW_POINTS = [5, 6]  # list(range(22, 27))
RIGHT_EYE_POINTS = [0]  # list(range(36, 42))
LEFT_EYE_POINTS = [1]  # list(range(42, 48))
NOSE_POINTS = [7, 8, 9, 10, 11, 12]  # list(range(27, 35))
# JAW_POINTS = #list(range(0, 17))
# DEFINE SOMETHING CALLED "STABLE POINTS" - eye corners and nose region
# STABLE_POINTS = [36,39,42,45,28,29,30]

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)
    print len(rects)
    if len(rects) == 0:
        raise NoFaces
    all_lmarks = []
    all_bbox = []
    for d in rects:
        # d = rects[0]
        bbox = [d.left(), d.top(), d.right(), d.bottom()]  # for d in rects])
        lmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
        all_lmarks.append(lmarks.tolist())
        all_bbox.append(bbox)
    return all_lmarks, all_bbox


def get_landmarks_one_face(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    d = rects[0]
    bbox = [d.left(), d.top(), d.right(), d.bottom()]  # for d in rects])
    lmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    return lmarks, bbox


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname)  # , cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s, b = get_landmarks(im)

    return im, s, b


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))

# im1, l1,b1 = read_im_and_landmarks(sys.argv[1])
# im2, l2,b2 = read_im_and_landmarks(sys.argv[2])
##
# M = transformation_from_points(l1[ALIGN_POINTS],
#                               l2[ALIGN_POINTS])
##
##mask = get_face_mask(im2, landmarks2)
##warped_mask = warp_im(mask, M, im1.shape)
##combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
##                          axis=0)
##
# warped_im2 = warp_im(im2, M, im1.shape)
# print warped_im2.shape
# l_w, b_w = get_landmarks(warped_im2)
##warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
##
##output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
# x1,y1,x2,y2 = b1
# cv2.imshow('input1',im1[y1:y2, x1:x2,:])
# x1,y1,x2,y2 = b2
# cv2.imshow('input2',im2[y1:y2, x1:x2,:])
# x1,y1,x2,y2 = b1
# cv2.imshow('output',warped_im2[y1:y2, x1:x2,:])
# cv2.imwrite('output.png',warped_im2)
# cv2.waitKey(0)
#
