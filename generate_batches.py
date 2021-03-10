import glob
import numpy as np
from astropy.io import fits
from astropy import wcs
import skimage.transform
import random
import scipy.stats as stats
import pickle as pkl
from functools import partial
from multiprocessing import Pool
import subprocess
from tqdm import tqdm
import tables
import os.path
import astroscrappy
import math
import scipy.special
import sep
import parmap
import scipy.ndimage as ndi
from skimage.morphology import convex_hull_image

# parameters for the streak distributions
params_gamma_width = [4.79235037, 11.40523025, 1.64361626]
params_gamma_amplitude = [1.09717868, 1.99999993, 3.7169828]
params_gamma_length = [4.16396709, 8.0180058, 2.68969172]

gain = 6.2

size = 80
# min distance from center of streak to the edge of the crop
min_offset = 35

file_batch_size_train = 4500
file_batch_size_val = 1900
# number of negative/positive images to extract per frame
# negative number is larger to reduce false positive rate
num_negative_images = 60
# hard negative images are image where there is a streak in the sci and ref images
num_hard_neg_images = 18
num_positive_images = 60

bad_pix_max = 80

# directory for aligned images
resample_dir = "/media/rd1/ztf_neos/pipeline/resampled/"

# directory for science images
science_dir = "/mnt/etdisk9/ztf_neos/training_examples/science/"

# directory for reference images
reference_dir = "/mnt/etdisk9/ztf_neos/training_examples/reference/"

# directory for difference images
difference_dir = "/mnt/etdisk9/ztf_neos/training_examples/difference/"

# output file, excluding extension
out_dir = "/mnt/etdisk9/ztf_neos/training_examples/batches/filtered_short_streaks_no_overlapV2"

# swarp configuration file
swarp_dir = "/home/fwang/Pipeline/swarp.conf"


def check_image(arr):
    return arr.shape == (size, size) \
           and np.count_nonzero(arr == 0) < bad_pix_max \
           and np.count_nonzero(~np.isfinite(arr)) < bad_pix_max


def extract_object_slices(diff, factor=1.3, area_thresh=15, upper_area_thresh=40, solidity_thresh=0.5):
    bkg = sep.Background(diff)
    rms = bkg.rms()
    thresh = rms * factor + bkg.back()

    mask = diff > thresh

    labels = skimage.measure.label(mask, connectivity=2)

    objects = ndi.find_objects(labels)

    filtered = []

    for i, sl in enumerate(objects):

        y = sl[0].stop - sl[0].start
        x = sl[1].stop - sl[1].start

        if y * x < area_thresh:
            continue

        label = i + 1

        img_mask = labels[sl] == label
        area = np.sum(img_mask)

        if area >= area_thresh:
            if area >= upper_area_thresh:
                filtered.append(sl)
            else:
                solidity = area / np.sum(convex_hull_image(img_mask))
                if solidity > solidity_thresh:
                    filtered.append(sl)

    return filtered


def extract_random_object_cutouts(sci, ref, diff, ref_bkg=None):

    if ref_bkg is None:
        ref_bkg = sep.Background(ref)

    mask = ref > (ref_bkg.back() + ref_bkg.rms() * 10)
    diff_bkg = sep.Background(diff)
    diff[mask] = diff_bkg.back()[mask]

    filtered = extract_object_slices(diff)

    crops = []

    for sl in filtered:
        ymin, ymax = sl[0].start, sl[0].stop
        xmin, xmax = sl[1].start, sl[1].stop
        xavg = (xmin + xmax) // 2
        yavg = (ymin + ymax) // 2

        offset = size // 2 - min_offset

        x = random.randint(xavg - offset - size // 2, xavg + offset - size // 2)

        y = random.randint(yavg - offset - size // 2, yavg + offset - size // 2)

        sci_crop = sci[y:y+size, x:x+size]
        ref_crop = ref[y:y+size, x:x+size]

        if check_image(sci_crop) and check_image(ref_crop):
            crops.append(((slice(y, y+size), slice(x, x+size)), (slice(ymin, ymax), slice(xmin, xmax))))

    return crops


def generate_streak(x_size, y_size, x_center, y_center, angle, amplitude, length, psf_width):
    x = np.arange(0, x_size)
    y = np.expand_dims(np.arange(0, y_size), -1)

    hx = (x - x_center) * math.cos(angle) - (y - y_center) * math.sin(angle)
    hy = (x - x_center) * math.sin(angle) + (y - y_center) * math.cos(angle)

    arr = np.exp(-hy ** 2 / (2 * psf_width ** 2)) * \
          (scipy.special.erf((hx + length / 2) / (psf_width * np.sqrt(2))) -
           scipy.special.erf((hx - length / 2) / (psf_width * np.sqrt(2))))

    arr *= amplitude / np.max(arr)

    return arr, arr > amplitude / 200


def implant_random_streak(image, bkg_rms, mask_pix, return_info=False, predetermined_vals=False, streak=None, mask=None,
                          factor=None, bkg=None):
    image = np.array(image, dtype=np.float32)

    while True:
        std = stats.gamma.rvs(*params_gamma_width) * 0.05

        if np.random.random() < 0.5:
            # based on the collected distribution (not below 10 however)
            length = stats.gamma.rvs(*params_gamma_length)
        elif np.random.random() < 0.2:
            # long streaks
            length = np.random.random() * 20 + 30
        else:
            # short streaks
            length = np.random.random() * 5 + 7

        if std > 0.5 and length / std > 1.2:
            break

    if not predetermined_vals:
        for _ in range(100):

            x = np.random.randint(min_offset, size - min_offset)
            y = np.random.randint(min_offset, size - min_offset)

            rotation = np.random.random() * np.pi * 2

            streak, mask = generate_streak(size, size, x, y, rotation, 1, length, std)
            if np.sum(mask_pix[mask]) / np.sum(mask) > 0.05:
                continue

            amp_max = np.average(bkg_rms) * 10

            # filter out top 10% of pixels, but at most till amp_max
            thresh = min(amp_max, np.percentile(image[mask], 90))

            pixels = image[mask]
            pixels = pixels[pixels < thresh]

            amp_min = np.std(pixels) * 3.25

            if amp_min < amp_max:
                break
        else:
            return None

        if np.random.random() < 0.8:
            factor = 1 + np.abs(np.random.randn()) * 0.5
        else:
            factor = np.random.random() * 1.75 + 1.75

        factor = min(max(1, amp_max / amp_min), factor)
    else:
        amp_min = np.std(image[mask]) * 3.25
        amp_min *= 1.3 * (1 + (np.abs(np.random.randn()) * 0.5))

    if std > 2.2 or length < 12 or std < 0.75:
        amp_min *= 3.9 / 3

    streak_scaled = streak * factor * amp_min
    streak_scaled = np.random.poisson(streak_scaled * gain) / gain

    if return_info:
        return image + streak_scaled, (x, y, rotation, length, std, streak, mask, factor, amp_min)

    return image + streak_scaled


def select_random_section(sci, ref):
    y_len, x_len = sci.shape
    x = np.random.randint(0, x_len - size)
    y = np.random.randint(0, y_len - size)

    return sci[y:y+size, x:x+size], ref[y:y+size, x:x+size], (slice(y, y + size), slice(x, x + size))


# align the reference file to the science, returns science and reference
# if realign is True then align the images again even if the directory already exists (has been aligned already)
def swarp_align(sci_file, ref_file, realign=False):
    ref, header = fits.getdata(ref_file, header=True)

    filename_sci = sci_file.split("/")[-1]
    directory = os.path.join(resample_dir, filename_sci[:-len(".fits")])

    try:
        os.makedirs(directory, exist_ok=realign)

    except FileExistsError:
        pass

    else:
        with open(directory + "/coadd.head", "w") as f:
            f.write(repr(header) + "\nEND     ")

        subprocess.run(["swarp", sci_file, "-c", swarp_dir, "-VERBOSE_TYPE", "QUIET"],
                       cwd=directory, stderr=subprocess.DEVNULL)

    return [fits.getdata(directory + "/coadd.fits"), ref]


def normalize(image, back, rms, sigma1=5, sigma2=5):
    image = np.array(image)

    image -= back

    image /= rms

    return np.clip(image, -sigma2, sigma1)


def convert_sci_to_ref(file):
    info = file.split("/")[-1].split("_")[1:-1]
    return os.path.join(reference_dir, "ztf_" + "_".join(info[1:4] + info[5:6]) + "_refimg.fits")


def convert_sci_to_diff(file):
    filename = file.split("/")[-1]
    return os.path.join(difference_dir, filename[:-len("sciimg.fits")] + "scimrefdiffimg.fits")


import time
def implant_random_streak_file(sci_file, ref_file, diff_file):

    sci, ref = swarp_align(sci_file, ref_file)
    diff, ref = swarp_align(diff_file, ref_file)

    sci = sci.astype(np.float32)
    sci[sci == 0] = np.nan

    sci_bkg = sep.Background(sci)
    sci_bkg_rms = sci_bkg.rms()
    sci_bkg_back = sci_bkg.back()

    ref = ref.astype(np.float32)
    ref_bkg = sep.Background(ref)
    ref_bkg_rms = ref_bkg.rms()
    ref_bkg_back = ref_bkg.back()

    diff = diff.astype(np.float32)
    diff[diff == 0] = np.nan

    # result[0] is the negative images, result[1] is the positive images
    result = []
    streak_metadata = []

    sections = extract_random_object_cutouts(sci, ref, diff)
    np.random.shuffle(sections)
    sections = sections[:400]

    sci_norm = normalize(sci, sci_bkg_back, sci_bkg_rms)
    sci_norm[~np.isfinite(sci_norm)] = 0
    ref_norm = normalize(ref, ref_bkg_back, ref_bkg_rms)
    ref_norm[~np.isfinite(ref_norm)] = 5

    mask_pix = np.logical_or(sci_norm >= 5, ref_norm >= 5)

    labels = []

    for slices, object_slice in sections:

        sci_crop = sci[slices]
        sci_crop = astroscrappy.detect_cosmics(sci_crop)[1]

        sci_crop = normalize(sci_crop, sci_bkg_back[slices], sci_bkg_rms[slices])

        ref_crop = ref_norm[slices]

        result.append([sci_crop, ref_crop])
        streak_metadata.append({
            "sci_file": sci_file,
            "ref_file": ref_file,
            "slice": slices,
            "real_slice": object_slice
        })
        labels.append(0)

    for _ in range(len(sections) * num_hard_neg_images // num_negative_images):
        while True:
            sci_crop, ref_crop, slices = select_random_section(sci, ref)

            if check_image(sci_crop) and check_image(ref_crop):
                break

        sci_crop = astroscrappy.detect_cosmics(sci_crop)[1]

        sci_rms = sci_bkg_rms[slices]
        sci_back = sci_bkg_back[slices]
        ref_rms = ref_bkg_rms[slices]
        ref_back = ref_bkg_back[slices]

        streak_result = implant_random_streak(sci_crop, sci_rms, mask_pix[slices], return_info=True)
        if streak_result is None:
            continue
        sci_crop_streak, (x_streak, y_streak, rotation, length, std, streak, mask, factor, amp_min) = streak_result
        ref_crop_streak = implant_random_streak(ref_crop, ref_rms, mask_pix[slices], predetermined_vals=True, streak=streak, mask=mask, factor=factor)

        sci_crop_streak = normalize(sci_crop_streak, sci_back, sci_rms)
        ref_crop_streak = normalize(ref_crop_streak, ref_back, ref_rms)

        streak_metadata.append({
            "x": x_streak,
            "y": y_streak,
            "amp_min": amp_min,
            "factor": factor,
            "length": length,
            "rotation": rotation,
            "width": std,
            "hard_neg": True,
            "sci_file": sci_file,
            "ref_file": ref_file,
            "slice": slices
        })

        result.append([sci_crop_streak, ref_crop_streak])
        labels.append(0)
        
    for _ in range(len(sections) * num_positive_images // num_negative_images):
        while True:
            sci_crop, ref_crop, slices = select_random_section(sci, ref)

            if check_image(sci_crop) and check_image(ref_crop):
                break

        sci_crop = astroscrappy.detect_cosmics(sci_crop)[1]

        sci_rms = sci_bkg_rms[slices]
        sci_back = sci_bkg_back[slices]
        ref_crop = ref_norm[slices]

        streak_result = implant_random_streak(sci_crop, sci_rms, mask_pix[slices], return_info=True)

        if streak_result is None:
            continue
        sci_crop_streak, (x_streak, y_streak, rotation, length, std, streak, mask, factor, amp_min) = streak_result

        sci_crop_streak = normalize(sci_crop_streak, sci_back, sci_rms)

        result.append([sci_crop_streak, ref_crop])

        streak_metadata.append({
            "x": x_streak,
            "y": y_streak,
            "amp_min": amp_min,
            "factor": factor,
            "length": length,
            "rotation": rotation,
            "width": std,
            "hard_neg": False,
            "sci_file": sci_file,
            "ref_file": ref_file,
            "slice": slices
        })

        labels.append(1)

    result = np.array(result).transpose((0, 2, 3, 1))

    return result, streak_metadata, labels


filenames = glob.glob(os.path.join(science_dir, "*.fits"))
filename_pairs = [(file, convert_sci_to_ref(file), convert_sci_to_diff(file)) for file in filenames]
filename_pairs = [pair for pair in filename_pairs if os.path.isfile(pair[1]) and os.path.isfile(pair[2])]
print("Number of Science Reference Difference Pairs: " + str(len(filename_pairs)))
assert file_batch_size_val + file_batch_size_train <= len(filename_pairs)
random.shuffle(filename_pairs)

files_train = filename_pairs[:file_batch_size_train]
files_val = filename_pairs[-file_batch_size_val:]

fileh = tables.open_file(out_dir + ".h5", mode='w')
atom = tables.Float32Atom()

print("Create Train Dataset")

train_images = fileh.create_earray(fileh.root, 'train_images', atom, (0, size, size, 2),  "Train Images",
                                   expectedrows=file_batch_size_train * (num_positive_images + num_negative_images + num_hard_neg_images))
train_labels = fileh.create_earray(fileh.root, 'train_labels', atom, (0,),  "Train Labels")
train_streak_meta = []

for file_pair in tqdm(files_train):

    sci_file, ref_file, diff_file = file_pair

    try:

        data, streak_metadata, labels = implant_random_streak_file(sci_file, ref_file, diff_file)

    except Exception as e:

        print(e, file_pair)
        continue

    train_images.append(data)
    train_labels.append(labels)
    train_streak_meta.extend(streak_metadata)

with open(out_dir + ".train.metadata.pkl", "wb") as f:
    pkl.dump(train_streak_meta, f)

print("Train Images Shape: " + str(train_images.shape))
print("Train Labels Shape: " + str(train_labels.shape))

print("Create Validation Dataset")

val_images = fileh.create_earray(fileh.root, 'val_images', atom, (0, size, size, 2),  "Val Images",
                                 expectedrows=file_batch_size_val * (num_positive_images + num_negative_images + num_hard_neg_images))
val_labels = fileh.create_earray(fileh.root, 'val_labels', atom, (0,),  "Val Labels")
val_streak_meta = []

labels = [0] * (num_negative_images + num_hard_neg_images) + [1] * num_positive_images

for file_pair in tqdm(files_val):

    sci_file, ref_file, diff_file = file_pair

    try:

        data, streak_metadata, labels = implant_random_streak_file(sci_file, ref_file, diff_file)

    except Exception as e:

        print(e, file_pair)
        continue

    val_images.append(data)
    val_labels.append(labels)
    val_streak_meta.extend(streak_metadata)

with open(out_dir + ".val.metadata.pkl", "wb") as f:
    pkl.dump(val_streak_meta, f)

print("Val Images Shape: " + str(val_images.shape))
print("Val Labels Shape: " + str(val_labels.shape))

fileh.close()


