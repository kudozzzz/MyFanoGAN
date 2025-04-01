"""
Saving (array of) images to file.

Copyright (c) 2018 Thomas Schlegl ... get_img_width(), save_images_as_row()
Copyright (c) 2017 Ishaan Gulrajani ... save_images() - Image grid saver, based on color_grid_vis from github.com/Newmu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import imageio  # Replaces deprecated scipy.misc.imsave


def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if np.issubdtype(X.dtype, np.floating):
        X = (255.99 * X).astype(np.uint8)

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows  # Integer division

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3), dtype=np.uint8)
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw), dtype=np.uint8)

    for n, x in enumerate(X):
        j = n // nw  # Integer division
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    imageio.imwrite(save_path, img)


def get_img_width(img_w, border_sz, n_samples):
    return int((img_w + border_sz) * n_samples - border_sz)


def save_images_as_row(X, save_path, border_sz=2):
    # [0, 1] -> [0,255]
    if np.issubdtype(X.dtype, np.floating):
        X = (255.99 * X).astype(np.uint8)

    n_samples = X.shape[0]

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img_width = get_img_width(w, border_sz, n_samples)
        img = np.zeros((h, img_width, 3), dtype=np.uint8)
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img_width = get_img_width(w, border_sz, n_samples)
        img = np.zeros((h, img_width), dtype=np.uint8)

    b = 0
    for n, x in enumerate(X):
        img[:, n * w + b:n * w + w + b] = x
        b += border_sz

    imageio.imwrite(save_path, img)
