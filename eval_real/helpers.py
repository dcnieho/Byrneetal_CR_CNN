# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:19:16 2019

@author: Marcus
"""

import numpy as np
import cv2
import numpy.linalg as alg


kernel_pup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
kernel_cr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))


# %%
def detect_pupil(img, intensity_threshold, size_limits, window_name=None):
    ''' Identifies pupil blob
    Args:
        img - grayscale eye image
        intensity_threshold - threshold used to find pupil area
        size_limite - [min_pupil_size, max_pupil_size]
        window_name - plots detected blobs is window
                        name is given

    Returns:
        (cx, cy) - center of gravity binary pupil blob
        area -  area of pupil blob
        countour_points - contour points of pupil
        ellipse - parameters of ellipse fit to pupil blob
            (x_centre,y_centre),(minor_axis,major_axis),angle, area

    '''

    # img = cv2.GaussianBlur(img,(55,55),0)

    # Threshold image to get binary image
    ret,thresh1 = cv2.threshold(img, intensity_threshold,
                               255,cv2.THRESH_BINARY)

    # Compute center location of image
    im_height, im_width = img.shape
    center_x, center_y = im_width/2, im_height/2

    # Close  holes in the pupil, e.g., created by the CR
    blobs = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel_pup)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel_pup)

    # Visualized blobs if windown name given
    if window_name:
        cv2.imshow(window_name, blobs)

    # Find countours of the detected blobs
    blob_contours, hierarchy  = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    # Find pupil but checking one blob at the time. Pupils are round, so
    # add checks for 'roundness' criteria
    # If serveral blobs are found, select the one
    # closest to the center
    '''
    For a blob to be a pupil candidate
    1. blob must have the right area
    2. must be circular
    '''

    pupil_detected = False
    old_distance_image_center = np.inf
    for i, cnt in enumerate(blob_contours):

        # Take convex hull of countour points to alleviate holes
        cnt = cv2.convexHull(cnt)

        # Only contours with enouth points are of interest
        if len(cnt) < 10:
            continue

        # Compute area and bounding rect around blob
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)

        # Check area and roundness criteria
        area_condition = (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.5)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= 0.2)

        # If these criteria are fulfilled, a pupil is probably detected
        if area_condition and symmetry_condition and fill_condition:
            # Compute moments of blob
            moments = cv2.moments(cnt)

            # Compute blob center of gravity from the moments
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # Compute distance blob - image center
            distance_image_center = np.sqrt((cx - center_x)**2 +
                                            (cy - center_y)**2)
            # Check if the current blob-center is closer
            # to the image center than the previous one
            if distance_image_center < old_distance_image_center:
                pupil_detected = True

                # Store pupil variables
                contour_points = cnt
                area = temp_area

                cx_best = cx
                cy_best = cy

                old_distance_image_center = distance_image_center

    # If no potential pupil is found, due to e.g., blinks,
    # return nans
    if not pupil_detected:
        cx_best = np.nan
        cy_best = np.nan
        area = np.nan
        contour_points = np.nan

    pupil_features = {'cog':(cx_best, cy_best), 'area':area, 'contour_points': contour_points}
    return pupil_features

#%%
def detect_cr(img, intensity_threshold, size_limits,
              pupil_cr_distance_max, pup_center, no_cr=2, cr_img_size = (20,20),
              window_name=None):
    ''' Identifies cr blob (must be located below pupil center)
    Args:
        img - grayscale eye image
        intensity_threshold - threshold used to find cr area(s)
        size_limite - [min_cr_size, max_cr_size]
        pupil_cr_distance_max - maximum allowed distance between
                                pupil and CR
        no_cr - number of cr's to be found

    Returns:
        cr - cr featuers
        cr_img -  image patch around CR

    '''
    cr = np.repeat([[np.nan, np.nan, np.nan]]
                   , no_cr, axis=0)
    contours=[]

    if np.isnan(pup_center[0]):
        return cr, None, None, None

    # Threshold image to get binary image
    ret,thresh1 = cv2.threshold(img, intensity_threshold,
                               255,cv2.THRESH_BINARY)

    # Close  holes in the cr, if any
    blobs = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel_cr)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel_cr)

    if window_name:
        cv2.imshow(window_name, blobs)

    # Find countours of the detected blobs
    blob_contours, hierarchy = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    cr = []
    cr_img = []
    patch_off = []
    for i, cnt in enumerate(blob_contours):
        # Only contours with enouth points are of interest
        if len(cnt) < 4:
            continue

        # Compute area and bounding rect around blob
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)

        # Check area and roundness criteria
        area_condition = (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.7)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= 0.7)

        # If these criteria are fulfilled, a pupil is probably detected
        if area_condition and symmetry_condition and fill_condition:
            # Compute moments of blob
            moments = cv2.moments(cnt)

            # Compute blob center of gravity from the moments
            # Coordinate system (0, 0) upper left
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # Check distance between pupil and cr
            d = np.sqrt((cx - pup_center[0])**2 + (cy - pup_center[1])**2)
            if d > pupil_cr_distance_max:
                continue

            if cy < (pup_center[1] - 0):
                continue

            # Cut out image patch around cr location
            cr_radie = cr_img_size[0]/2

            # Cut out regions around threshold CR
            x_range = [int(cx - cr_radie), int(cx + cr_radie)]
            y_range = [int(cy - cr_radie), int(cy + cr_radie)]
            # make sure we don't run off the image
            if x_range[0]<0:
                x_range = [x-x_range[0] for x in x_range]
            if y_range[0]<0:
                y_range = [y-y_range[0] for y in y_range]
            if x_range[1]>img.shape[1]:
                x_range = [x-(x_range[1]-img.shape[0]) for x in x_range]
            if y_range[1]>img.shape[0]:
                y_range = [y-(y_range[1]-img.shape[0]) for y in y_range]

            cr_imgt = [img  [y_range[0] : y_range[1],
                             x_range[0] : x_range[1]],
                       blobs[y_range[0] : y_range[1],
                             x_range[0] : x_range[1]]]

            cr.append([cx, cy, temp_area])
            patch_off.append([x_range[0], y_range[0]])
            cr_img.append(cr_imgt)
            cnt[:,0,0] -= x_range[0]
            cnt[:,0,1] -= y_range[0]
            contours.append(cnt)

    # if more crs than expected are found, then take the closest to the center
    if len(cr) > no_cr:
        dist = []
        for c in cr:
            dist.append(np.sqrt((c[0] - pup_center[0])**2 + \
                                (c[1] - pup_center[1])**2))

        # sort and select the closest distances
        idx = np.argsort(dist)
        cr = [cr[i] for i in idx[:no_cr]]
        patch_off = [patch_off[i] for i in idx[:no_cr]]
        cr_img = [cr_img[i] for i in idx[:no_cr]]
        contours = [contours[i] for i in idx[:no_cr]]

    # If the correct number of cr's are detected,
    # distinguish between them using x-position, i.e.,
    # give them an identity, cr1, cr2, cr2, etc.
    if len(cr) == no_cr:
        x_pos = []
        for c in cr:
            x_pos.append(c[0])

        # sort
        idx = np.argsort(x_pos)
        cr = [cr[i] for i in idx]
        patch_off = [patch_off[i] for i in idx]
        cr_img = [cr_img[i] for i in idx]
        contours = [contours[i] for i in idx]

    return cr, None if not cr_img else cr_img[0], patch_off, contours

def biqubic_calibration_with_cross_term(x, y, Y, data_length=None):

    '''
    def func(data, a, b, c, d, e, f):
        return a * data[0, :] + b * data[1, :] + c * data[2, :] + d * data[3, :] + e * data[4, :] + f * data[5, :]
    if not data_length:
        data_length = len(x)
    X = np.ones((6, data_length))
    X[1, :] = x
    X[2, :] = y
    X[3, :] = np.square(x)
    X[4, :] = np.square(y)
    X[5, :] = x * y
    popt, pcov = curve_fit(func, X, Y, method="lm")
    # '''
    # """
    X = np.zeros((len(x), 6))
    X[:, 0] = 1
    X[:, 1] = x
    X[:, 2] = y
    X[:, 3] = x ** 2
    X[:, 4] = y ** 2
    X[:, 5] = x * y

    x_1 = np.dot(X.T, X)
    x_2 = np.dot(alg.inv(x_1), X.T)
    coeff = np.dot(x_2, Y)
    # """
    pcov = ""
    return coeff, pcov

def biqubic_estimation_with_cross_term(x, y, coeff):
    X = np.zeros((len(x), 6))
    X[:, 0] = 1
    X[:, 1] = x
    X[:, 2] = y
    X[:, 3] = x ** 2
    X[:, 4] = y ** 2
    X[:, 5] = x * y

    return np.dot(X, coeff)