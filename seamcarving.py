import sys
import numpy as np
import cv2 as cv
from scipy.ndimage.filters import convolve
from tqdm import trange
from numba import jit
from moviepy.editor import ImageSequenceClip

# Get Brightness
def get_brightness_image(img):
    return np.mean(img, axis=-1).astype("uint8")

# Find edges of image by using Sobel Filter
def get_edge_detection_image(img):

    brightness_img = get_brightness_image(img)

    kernel_dx = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

    kernel_dy = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])

    brightness_img = brightness_img.astype('float32')
    return np.absolute(convolve(brightness_img, kernel_dx)) + np.absolute(convolve(brightness_img, kernel_dy))
@jit
def get_least_energy_seams(img):
    energy_map = get_edge_detection_image(img)
    h, w, _ = img.shape

    energy_path_map = energy_map.copy()
    direction_map = np.zeros_like(energy_path_map, dtype='int16')

    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(direction_map[i - 1, j:j + 2])
                direction_map[i, j] = idx + j
                min_energy = energy_path_map[i - 1, idx + j]
            else:
                idx = np.argmin(energy_path_map[i - 1, j - 1:j + 2])
                direction_map[i, j] = idx + j - 1
                min_energy = energy_path_map[i - 1, idx + j - 1]

            energy_path_map[i, j] += min_energy

    return energy_path_map, direction_map
@jit
def draw_every_seams(img):
    _, direction_map = get_least_energy_seams(img)
    h, w, _ = img.shape
    img_seams = img.copy()
    color = [255, 0, 0] 
    img_seams[-1, :] = color

    for i in range(w):
        k = i
        for j in range(h-1, 0, -1):
            k = direction_map[j, k]
            img_seams[j-1, k] = color

    return img_seams

@jit
def remove_seam(img):
    energy_path_map, direction_map = get_least_energy_seams(img)
    h, w, _ = img.shape
    mask = np.ones((h, w), dtype='bool')
    k = np.argmin(energy_path_map[-1])
    color = [255, 0, 0] 
    img_seam = img.copy()
    img_removed_seam = img.copy()

    for i in range(h-1, -1, -1):
        mask[i, k] = False
        img_seam[i, k] = color
        k = direction_map[i, k]
    
    mask = np.stack([mask] * 3, axis=2)
    img_removed_seam = img_removed_seam[mask].reshape((h, w-1, 3))

    return img_seam, img_removed_seam


@jit
def remove_seam_compact_mode(img):
    energy_path_map, direction_map = get_least_energy_seams(img)
    h, w, _ = img.shape
    mask = np.ones((h, w), dtype='bool')
    k = np.argmin(energy_path_map[-1])
    img_removed_seam = img.copy()

    for i in range(h-1, -1, -1):
        mask[i, k] = False
        k = direction_map[i, k]
    
    mask = np.stack([mask] * 3, axis=2)
    img_removed_seam = img_removed_seam[mask].reshape((h, w-1, 3))

    return img_removed_seam

def seamcarving_img(img, scale):
    h, w, _ = img.shape
    new_w = int(scale * w)

    for i in trange(w - new_w): # use range if you don't want to use tqdm
        img = remove_seam_compact_mode(img)

    return img

def create_gif_seamcarving(img, scale, filename='test.gif'):
    images = []
    w = img.shape[1]
    new_w = int(scale * w)
    img_removed_seam = img
    for _ in trange(w - new_w): # use range if you don't want to use tqdm
        img_seam, img_removed_seam = remove_seam(img_removed_seam)
        img_seam_pad = np.zeros_like(img, dtype='uint8')
        img_seam_pad[:, 0:img_seam.shape[1], :] = img_seam      
        img_pad = np.zeros_like(img, dtype='uint8')
        img_pad[:, 0:img_removed_seam.shape[1], :] = img_removed_seam
        images.append(img_seam_pad[...,::-1])
        images.append(img_pad[...,::-1])
    
    clip = ImageSequenceClip(images, fps=20)
    clip.write_gif(filename, fps=20)

img = cv.imread("sample3.jpg")
cv.imshow("Brightness", get_brightness_image(img))
cv.imshow("Energy", get_edge_detection_image(img).astype("uint8"))
cv.imshow("seams", draw_every_seams(img))
img_seam, img_removed_seam = remove_seam(img)
cv.imshow("removed_seam", img_removed_seam)
cv.imshow("1 seam", img_seam)
cv.imshow("seam carving", seamcarving_img(img, 0.5))
# create_gif_seamcarving(img, 0.5)

# cv.waitKey(0)


