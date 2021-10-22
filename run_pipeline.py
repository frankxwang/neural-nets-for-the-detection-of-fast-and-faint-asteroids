import urllib.request
import os
import parmap
import glob
from tqdm import tqdm
import numpy as np
import astroscrappy
import subprocess
import matplotlib.pyplot as plt
import scipy.special
import scipy
import sep
import scipy.ndimage as ndi
from skimage.morphology import convex_hull_image
import skimage.measure
from astropy.io import fits
from astropy import wcs
from PIL import Image
import pickle as pkl
from numba import jit, cuda
import fitsio

import tensorflow as tf
from keras.models import Model, load_model
import efficientnet.keras as efn
from keras.layers import *
import cupy as cp

# file containing the model weights
weights_dir = "/home/fwang/oldhome/fwang/wandb/run-20201006_062622-3613h28f/model-best_0.00020242914979757084_0.9700428766983811_0.994513WEIGHTS.h5"

# directory for aligning the images
base_directory = "/media/etdisk16/ztf_neos/pipeline/resampled/"

# directory containing science and difference images
sci_dir = "/media/etdisk16/ztf_neos/full_days/20190605/"

# directory containing reference images
ref_dir = "/media/etdisk16/ztf_neos/reference/"

# output file containing all the positive detections
out_file = "/home/fwang/20190605run.pkl"

# swarp configuration file
swarp_dir = "/home/fwang/Pipeline/swarp.conf"

# size of images
size = 80

# threshold for whether or not an image is deemed to be a positive detection
model_threshold = 0.85

# create the modified EfficientNetB1 model
def make_model():
    images = Input((size, size, 2))

    x = efn.EfficientNetB1(input_shape=(size, size, 2), weights=None, include_top=False)(images)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(images, output)

    return model

model = make_model()

# load the pretrained weights
model.load_weights(weights_dir)

def sci_to_ref(sci_file):
    tokens = sci_file.split("_")
    return "_".join([tokens[0], tokens[2], tokens[3], tokens[4], tokens[6], "refimg.fits"])

# load in the telescope FITS data
sci_files = glob.glob(os.path.join(sci_dir, "*_sciimg.fits"))
print(len(sci_files))
diff_files = [sci.replace("sciimg.fits", "scimrefdiffimg.fits.fz") for sci in sci_files]
ref_files = [os.path.join(ref_dir, sci_to_ref(sci.split("/")[-1])) for sci in sci_files]

# image segmentation to preprocess images
def extract_object_slices(diff, factor=1.3, area_thresh=15, upper_area_thresh=40, solidity_thresh=0.5):
    
    # extract background and noise
    bkg = sep.Background(diff)
    rms = np.abs(bkg.rms())
    
    # sigma threshold by factor
    thresh = rms * factor + bkg.back()

    mask = np.logical_and(diff > thresh, diff !=0)

    # uses floodfill to label each grid point
    labels = skimage.measure.label(mask, connectivity=2)
    
    # extract the objects from the labelled grid
    # this function is numba accelerated
    areas, slices, binary_imgs = find_objects(labels, area_thresh)
    
    # filter objects by area, solidity
    positions = []
    slices_list = []
    binary_img_list = []
    for area, bound, binary_img in zip(areas, slices, binary_imgs):
        # if the area is super big, no need to worry about solidity
        if area >= upper_area_thresh:
            pos = ((bound[3] + bound[2]) // 2, (bound[1] + bound[0]) // 2)
            positions.append(pos)
            slices_list.append(bound)
            binary_img_list.append(binary_img)
        else:
            # for smaller objects, we want to check solidity (area / convex hull area)
            solidity = area / np.sum(convex_hull_image(binary_img))
            # make sure object doesn't have major holes in it
            # but if the solidity is 1 (perfectly filled object, it's probably an artifact)
            if solidity > solidity_thresh and solidity < 1:
                pos = ((bound[3] + bound[2]) // 2, (bound[1] + bound[0]) // 2)
                positions.append(pos)
                slices_list.append(bound)
                binary_img_list.append(binary_img)
    return positions, slices_list, binary_img_list

# given a labelled grid, extract all the objects, also filter by the area threshold
@jit(nopython=True)
def find_objects(labels, area_thresh):
    # preallocate bounds for numba compatibility
    bounds = np.zeros((np.max(labels), 4), dtype=np.uintc)
    bounds[:, 0] = 9999999
    bounds[:, 2] = 9999999
    
    # calculate the bounds 
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label = labels[i, j] - 1
            # unsegmented parts of the grid
            if label == -1:
                continue
            bounds[label][0] = min(i, bounds[label][0])
            bounds[label][1] = max(i + 1, bounds[label][1])
            bounds[label][2] = min(j, bounds[label][2])
            bounds[label][3] = max(j + 1, bounds[label][3])
    
    # get the metadata for each object, and also filter by area
    areas = []
    slices = []
    binary_imgs = []
    
    for i in range(len(bounds)):
        bound = bounds[i]
        if bound[0] == 9999999:
            continue
        y = bound[1] - bound[0]
        x = bound[3] - bound[2]
        
        # if the rectangular bounding box already is this small, the segmented object itself is probably under area_thresh
        # this speeds up the program signifcantly since we don't need to call np.sum so many times
        if y * x <= area_thresh + 4:
            continue
        
        # extract the pixel mask and find the area
        binary_img = labels[bound[0]:bound[1], bound[2]:bound[3]] == (i + 1)
        area = np.sum(binary_img)
        
        # only include object if it is larger than area_thresh
        if area >= area_thresh:
            areas.append(area)
            slices.append(bound)
            binary_imgs.append(binary_img)
                
    return areas, slices, binary_imgs

# normalize image through sigma clipping
def normalize(image, back, rms, bkg=None, sigma1=5, sigma2=5):
    image = np.array(image, copy=True)
    
    # if we are using a sep background, we can more efficiently subtract the background w/o needing to allocate another array
    if bkg:
        bkg.subfrom(image)
    else:
        image -= back
        
    image /= rms

    return image.clip(-sigma2, sigma1, out=image)

# crop images while padding out of bounds regions with 0
def crop_with_padding(arr, x1, x2, y1, y2):
    crop = arr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        return np.zeros((y2 - y1, x2 - x1))
    return np.pad(crop, ((max(-y1, 0), max(y2 - arr.shape[0], 0)), (max(-x1, 0), max(x2 - arr.shape[1], 0))), mode="constant")
    
# remove CCD crosstalk & ghosting artifacts
def clean_saturation_artifacts(sci_img, sci_file, ref_file, bounds):

    # get the mask of artifacts
    mask = get_saturation_mask(sci_file, ref_file, bounds)
    mask = mask.astype(bool)
    
    # mask out the image, sci_img should be normalized so the background level is 0 
    sci_img = np.array(sci_img)
    sci_img[mask] = 0
    
    return sci_img, mask

# get mask to remove artifacts
def get_saturation_mask(sci_file, ref_file, bounds):
    
    mask = np.zeros((size, size))
    
    # get the positions of the object on the original science image (not aligned to reference)
    ref_wcs = wcs.WCS(ref_file)
    ra_dec_bounds = ref_wcs.all_pix2world(bounds, 0, ra_dec_order=True)
    sci_wcs = wcs.WCS(sci_file)
    sci_bounds = sci_wcs.all_world2pix(ra_dec_bounds, 0, ra_dec_order=True)
    sci_bounds = np.rint(sci_bounds)
    sci_bounds = sci_bounds.astype(int)
    
    # cross-talk detection:
    before, after = sci_file.split("_q")
    
    # quadrant 1 bleeds into 2, 2 into 1, 3 into 4, and 4 into 3
    if int(after[0]) == 1:
        i = 2
    if int(after[0]) == 2:
        i = 1
    if int(after[0]) == 3:
        i = 4
    if int(after[0]) == 4:
        i = 3
    
    # get the image that might potentially bleed into this one
    filename_c = before + "_q" + str(i + 1) + "_sciimg.fits"
    
    # load that image
    img, header = fits.getdata(filename_c, header=True)
    
    # flip it across the x-axis since that's how crosstalk operates
    img_x_flip = img[:, ::-1]
    # crop the section we are looking at
    crop = crop_with_padding(img_x_flip, sci_bounds[1][0], sci_bounds[0][0], sci_bounds[0][1], sci_bounds[1][1])
    
    # crop the image and match the size
    crop_image = Image.fromarray(crop)
    crop_image = crop_image.resize((size, size), Image.NEAREST)
    crop = np.array(crop_image)[:, ::-1]
    
    # create mask of saturated images
    mask += crop > 10000
    
    # ghost detection:
    if int(after[0]) == 1 or int(after[0]) == 4:
        # quadrant 1 & 4 have left readout direction
        read_dir = -1
    else:
        # quadrant 2 & 3 have right readout
        read_dir = 1
    
    img, header = fits.getdata(sci_file, header=True)
    
    # mask out ghosts "moving" backward
    bounds_backwards = np.copy(sci_bounds)
    # ghosts appear at a set offset
    bounds_backwards[:, 0] -= read_dir * 2280
    
    x1 = bounds_backwards[1][0] - 100
    x2 = bounds_backwards[0][0] + 100
    y1 = bounds_backwards[0][1]
    y2 = bounds_backwards[1][1]
    
    # get a mask of saturated pixels, convolve them with a 1 by 100 kernel of just 1's to "spread out" the mask horizontally
    img_sat_mask = img > header["SATURATE"]
    if img_sat_mask[y1:y2, x1:x2].size != 0:
        img_sat_mask[y1:y2, x1:x2] = scipy.signal.convolve2d(img_sat_mask[y1:y2, x1:x2], np.ones((1, 100)), mode="same")
    
    # crop the ghost mask
    crop = crop_with_padding(img_sat_mask, bounds_backwards[1][0], bounds_backwards[0][0], bounds_backwards[0][1], bounds_backwards[1][1])
    crop_image = Image.fromarray(crop)
    crop_image = crop_image.resize((size, size), Image.NEAREST)
    crop = np.array(crop_image)[:, ::-1]
    
    mask += crop
    
    # forward, same as before
    bounds_forward = np.copy(sci_bounds)
    bounds_forward[:, 0] += read_dir * 870
    
    x1 = bounds_forward[1][0] - 100
    x2 = bounds_forward[0][0] + 100
    y1 = bounds_forward[0][1]
    y2 = bounds_forward[1][1]
    
    img_sat_mask = img > header["SATURATE"]
    if img_sat_mask[y1:y2, x1:x2].size != 0:
        img_sat_mask[y1:y2, x1:x2] = scipy.signal.convolve2d(img_sat_mask[y1:y2, x1:x2], np.ones((1, 100)), mode="same")
    
    crop = crop_with_padding(img_sat_mask, bounds_forward[1][0], bounds_forward[0][0], bounds_forward[0][1], bounds_forward[1][1])
    crop_image = Image.fromarray(crop)
    crop_image = crop_image.resize((size, size), Image.NEAREST)
    crop = np.array(crop_image)[:, ::-1]
    
    mask += crop
    
    # add some additional "spreading out" of the mask
    mask = scipy.signal.convolve2d(mask, np.ones((3, 3)), mode="same")
        
    return mask

# project original image onto the target using CuPy GPU acceleration
# use sampling grid of every res_pixel points to increase speed
# higher res_pixel => faster but less precision
def gpu_align(h_target, h_orig, target_image, orig_image, res_pixel=5):
    
    t_shape = target_image.shape
    
    # create sampling grid of points on the target image
    coords = np.mgrid[:t_shape[0]+res_pixel:res_pixel, :t_shape[1]+res_pixel:res_pixel].T
    shrink_size = coords.shape
    coords = coords.reshape(-1, 2)
    
    # extract the coordinate grid information for each image
    target_wcs = wcs.WCS(h_target)
    orig_wcs = wcs.WCS(h_orig)
    
    # project the sampling grid from the target to the original image
    target_ref_pix = cp.array(orig_wcs.all_world2pix(target_wcs.all_pix2world(coords, 0), 0).reshape(*shrink_size))
    # create a grid of sampling points using the scaled down version of the image
    all_coords = cp.mgrid[:t_shape[0]/res_pixel:1/res_pixel, :t_shape[1]/res_pixel:1/res_pixel].reshape(2, -1)
    
    # use interpolation to find where each point on all_coords falls on the original image
    x_vals = map_coordinates(target_ref_pix[..., 0], all_coords)
    y_vals = map_coordinates(target_ref_pix[..., 1], all_coords)
    
    # use interpolation to convert the original image coordinates to pixel values
    resamp = map_coordinates(cp.array(orig_image), cp.array([y_vals, x_vals])).reshape(t_shape).get()
    
    return resamp

# OLD CPU SWarp BASED METHOD: align the reference file to the science, returns science and reference
def swarp_align(sci_file, ref_file, use_cached=True):
    ref, header = fits.getdata(ref_file, header=True)

    filename_sci = sci_file.split("/")[-1]
    directory = base_directory + filename_sci[:-len(".fits")]
    
    failed = False
    
    if os.path.exists(directory) and use_cached:
        try:
            return [fits.getdata(directory + "/coadd.fits"), ref]
        except:
            # if this fails it means that this was one of the files that got truncated
            failed = True
    if not os.path.exists(directory) or not use_cached or failed:
        
        os.makedirs(directory, exist_ok=True)
        with open(directory + "/coadd.head", "w") as f:
            f.write(repr(header) + "\nEND     ")

        subprocess.run(["swarp", sci_file, "-c", swarp_dir, "-VERBOSE_TYPE", "QUIET"],
                         cwd=directory, stderr=subprocess.DEVNULL)
                       
    return [fits.getdata(directory + "/coadd.fits"), ref]

# check image for number of bad pixels and correct shape
def check_image(arr, bad_pix_max=80):
    return arr.shape == (size, size) \
           and np.count_nonzero(~np.isfinite(arr)) < bad_pix_max

# run the full pipeline on a science, reference, and difference image
# threshold is for the ML model, align=False if inputs are prealigned images, clean_sat for saturation artifact removal, clean for artifact + cosmic ray removal, use_swarp=False for GPU alignment
def process_images(sci_file, ref_file, diff_file, threshold=0.85, batch_size=2048, align=True, clean_sat=True, clean=True, use_swarp=False):
    
    # load images and align them
    if align:
        
        if use_swarp:
            
            sci, ref = swarp_align(sci_file, ref_file)
            sci = sci.astype(np.float32)
            ref = ref.astype(np.float32)
            
        else:
            
            sci, sci_head = fits.getdata(sci_file, header=True)
            ref, ref_head = fits.getdata(ref_file, header=True)
            sci = sci.astype(np.float32)
            ref = ref.astype(np.float32)
            
            sci = gpu_align(ref_head, sci_head, ref, sci)
        
        # for the compressed images fitsio is much faster
        diff = fitsio.read(diff_file)
    else:
        sci = sci_file
        ref = ref_file
        diff = diff_file
    
    # use image segmentation to get a list of "interesting" transient objects
    positions, slices_list, binary_img_list = extract_object_slices(diff)
    
    # project the positions of each object onto the reference frame
    wcs_ref = wcs.WCS(ref_file)
    wcs_sci = wcs.WCS(sci_file)
    positions = wcs_ref.all_world2pix(wcs_sci.all_pix2world(positions, 0, ra_dec_order=True), 0, ra_dec_order=True)
    
    # process and normalize the sci & reference images
    sci[sci == 0] = np.nan
    sci_bkg = sep.Background(sci)
    sci_bkg_back = sci_bkg.back()
    sci_bkg_rms = sci_bkg.rms()
    ref_bkg = sep.Background(ref)
    sci_norm = normalize(sci, sci_bkg_back, sci_bkg_rms)
    ref_norm = normalize(ref, ref_bkg.back(), ref_bkg.rms())
    
    # extract crops for every single position returned by image segmentation
    images = []
    orig_images = []
    bounds = []
    slices_list_filtered = []
    binary_img_list_filtered = []

    for i, pos in enumerate(positions):
        x, y = pos
        x = int(x)
        y = int(y)
        
        sci_norm_crop = sci_norm[y - size//2:y + size//2, x - size//2:x + size//2]
        ref_norm_crop = ref_norm[y - size//2:y + size//2, x - size//2:x + size//2]
        
        if not check_image(sci_norm_crop) or not check_image(ref_norm_crop):
            continue
        
        sci_crop = sci[y - size//2:y + size//2, x - size//2:x + size//2]

        sci_norm_crop[~np.isfinite(sci_norm_crop)] = 0
        ref_norm_crop[~np.isfinite(ref_norm_crop)] = 5
        
        images.append((sci_norm_crop, ref_norm_crop))
        orig_images.append((sci_crop, ref_norm_crop))
        
        bounds.append([(x - size//2, y - size//2), (x + size//2, y + size//2)])
        
        slices_list_filtered.append(slices_list[i])
        binary_img_list_filtered.append(binary_img_list[i])

    
    # send all the images through the model
    images = np.array(images)
    orig_images = np.array(orig_images)
    norm_images_transpose = np.transpose(images, [0, 2, 3, 1])
    preds = model.predict(norm_images_transpose, batch_size=batch_size)[:, 0]

    # threshold preds
    indexes = preds > threshold
    top_images = orig_images[indexes]
    top_bounds = np.array(bounds)[indexes]
    
    top_slices_list = [slices_list_filtered[i] for i in range(len(slices_list_filtered)) if indexes[i]]
    top_binary_img_list = [binary_img_list_filtered[i] for i in range(len(binary_img_list_filtered)) if indexes[i]]
    
    # apply artifact/cosmic ray cleaning if we have positive detetcions
    if len(top_images) > 0:
        
        if not clean:
            return top_images, images[indexes], preds, top_bounds
    
        clean_images = []

        for (sci, ref_norm), bounds in zip(top_images, top_bounds):
            # apply cosmic ray removal
            cleaned_cosmics_sci = astroscrappy.detect_cosmics(sci)[1]
            cleaned_cosmics_sci = normalize(cleaned_cosmics_sci, sci_bkg_back[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]], sci_bkg_rms[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]])
            if not clean_sat:
                clean_images.append([cleaned_cosmics_sci, ref_norm])
                continue
            try:
                # removal of CCD artifacts
                cleaned, mask = clean_saturation_artifacts(cleaned_cosmics_sci, sci_file, ref_file, bounds)
                clean_images.append([cleaned, ref_norm])
            except FileNotFoundError:
                clean_images.append([cleaned_cosmics_sci, ref])
                print(sci_file, ref_file, bounds)
        
        norm_images = np.array(clean_images)

        norm_images_transpose = np.transpose(norm_images, [0, 2, 3, 1])
        
        # rerun the cleaned images through the model
        preds = model.predict(norm_images_transpose, batch_size=batch_size)[:, 0]

        indexes = preds > threshold
        
        top_slices_list = [top_slices_list[i] for i in range(len(top_slices_list)) if indexes[i]]
        top_binary_img_list = [top_binary_img_list[i] for i in range(len(top_binary_img_list)) if indexes[i]]
        
        return top_images[indexes], norm_images[indexes], preds[indexes], top_bounds[indexes], top_slices_list, top_binary_img_list

    return [], [], [], [], [], []

# open the output file for storing all of the results
f_data = open(out_file, "wb")

total = 0

i = 0

# loop through each image
for sci, ref, diff in tqdm(list(zip(sci_files, ref_files, diff_files))):
    if not os.path.exists(sci) or not os.path.exists(ref) or not os.path.exists(diff):
        continue

    try:
        # process each image with the ML model
        top_imgs, norm_imgs, preds, top_bnds, slices, binary_imgs = process_images(sci, ref, diff, batch_size=2048)
        pkl.dump([top_imgs, norm_imgs, preds, top_bnds, slices, binary_imgs, i, (sci, ref, diff)], f_data)

        total += len(preds)
        print(len(top_imgs), total, i)
    except Exception as e:
        # record any images with possible errors
        print(e)
        pkl.dump([e, i, (sci, ref, diff)], f_data)
    
    i += 1
