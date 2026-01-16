import argparse
from PIL import Image
import numpy as np

from clustering.helpers import cluster_plot, kmeans_colours, rgb_to_xyz, xyz_to_lab

# Use argparse to get the image file, repulsion, and output format
parser = argparse.ArgumentParser(description='Generate a colourscheme based on an image')
parser.add_argument('path', help='path to the image that the colourscheme will be generated from', type=str)
parser.add_argument('-r', '--repulsion', help='force that means repel each other with', default=None, type=float)
parser.add_argument('--xresources', dest='xresources', help='output in xresources format', default=False, action='store_true')
parser.add_argument('--seed', help='seed for initial kmeans centroids, set to 0 for random', default=0, type=int)
parser.add_argument('-s', '--separation', help='force that means repel each other with', default=None, type=float)

args = parser.parse_args()
img_path = args.path
repulsion = args.repulsion
use_xresources = args.xresources
separation = args.separation
seed = args.seed


if seed != 0:
    np.random.seed(seed)

# Shrink the image to nxn
img = Image.open(img_path)
img.thumbnail((600, 600))

# Get colour values
counts_list = []
colours_list = []
width, height = img.size
for count, colour in img.getcolors(width*height):
    counts_list.append(count)
    colours_list.append(colour)
colours = np.array(colours_list)
counts = np.array(counts_list)

# Convert colours to CIELAB
r = colours[:,0]
g = colours[:,1]
b = colours[:,2]
L, a, b = xyz_to_lab(*rgb_to_xyz(r, g, b))

# Split data into categories based on luminosity percentiles
mask_background = L < np.percentile(L, 10)
mask_background2 = np.logical_and(L < np.percentile(L, 50), L > np.percentile(L, 40))
mask_foreground = L > np.percentile(L, 90)
mask_accent = np.logical_and(L < np.percentile(L, 90), L > np.percentile(L, 60))


# Combine the channels
x = np.stack([L, a, b], axis=1)

# Get the 3 foreground colours
x_f = x[mask_foreground]
counts_f = counts[mask_foreground]
means_f = kmeans_colours(x_f, counts_f, 3, separation=separation, repulsion=repulsion)

# Get the 10 accent colours
x_a = x[mask_accent]
counts_a = counts[mask_accent]
means_a = kmeans_colours(x_a, counts_a, 10, separation=separation, repulsion=repulsion)

# Get the 3 background colours
x_b = x[mask_background]
counts_b = counts[mask_background]
means_b = kmeans_colours(x_b, counts_b, 1, separation=separation, repulsion=repulsion)

x_b2 = x[mask_background2]
counts_b2 = counts[mask_background2]
means_b2 = kmeans_colours(x_b2, counts_b2, 2, separation=separation, repulsion=repulsion)

# Print the colours in a form for xrdb
def output_colours(bg, bg2, a, fg, xresources=False):
    if xresources:
        num_str = '*.color%d: #%02x%02x%02x'
        bg_str = '*.background: #%02x%02x%02x'
        fg_str = '*.foreground: #%02x%02x%02x'
        cursor_str = '*.cursor: #ffffff'
    else:
        num_str = '#define COLOUR%d #%02x%02x%02x'
        bg_str = '#define BACKGROUND #%02x%02x%02x'
        fg_str = '#define FOREGROUND #%02x%02x%02x'
        cursor_str = '#define CURSOR #ffffff'
    
    colours = np.concatenate([bg, bg2, a, fg], axis=0)
    for i in range(colours.shape[0]):
        colour = colours[i]
        print(num_str%(i, int(colour[0]), int(colour[1]), int(colour[2])))

    print(bg_str%(int(bg[0,0]), int(bg[0,1]), int(bg[0,2])))
    print(fg_str%(int(fg[-1,0]), int(fg[-1,1]), int(fg[-1,2])))
    print(cursor_str)

output_colours(means_b, means_b2,  means_a, means_f, use_xresources)
