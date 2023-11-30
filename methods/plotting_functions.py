# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:24:38 2021

@author: ianni
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import re

def round_repeating_decimals(num, precision=8):
  
    start = precision - 1
    # Convert the number to a string
    num_str = str(num)[start:]
    
    # Use regex to find repeating decimals
    match = re.search(r'(\d+?)\1{5,}', num_str)
    
    # If repeating decimals are found, round the number and calculate the index
    if match:
        index = match.start() + start
        num = round(num, max(index, precision))
    else:
        index = None
        num = round(num, precision)
    
    return num

# round_repeating_decimals(2.999999926666667, precision=8)

def VisualizeImageGrayscale(image_3d):
  r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
  """
  vmin = torch.min(image_3d)
  image_2d = image_3d - vmin
  vmax = torch.max(image_2d)
  return (image_2d / vmax)

def VisualizeNumpyImageGrayscale(image_3d):
  r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
  """
  vmin = np.min(image_3d)
  image_2d = image_3d - vmin
  vmax = np.max(image_2d)
  return (image_2d / vmax)

def format_img(img_):
    img_ = img_     # unnormalize
    np_img = img_.numpy()
    tp_img = np.transpose(np_img, (1, 2, 0))
    return tp_img

def imshow(img, title="imshowfig", cmap='gray'):
    # img = img     # unnormalize
    npimg = img.detach().numpy()
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg, cmap=cmap)
    plt.savefig(str(title + '.png'))
    # plt.show()