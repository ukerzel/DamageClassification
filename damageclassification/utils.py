"""
Collection of various utils 
"""

import numpy as np

import imageio.v3 as iio
from PIL import Image
# we may have very large images (e.g. panoramic SEM images), allow to read them w/o warnings
Image.MAX_IMAGE_PIXELS = 933120000

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


import math


###
### load SEM images
### 
def load_image(filename : str) -> np.ndarray :
    """Load an SEM image 

    Args:
        filename (str): full path and name of the image file to be loaded

    Returns:
        np.ndarray: file as numpy ndarray
    """
    image =  iio.imread(filename,mode='F')

    return image



###
### show SEM image with boxes in various colours around each damage site
###
def show_boxes(image : np.ndarray, damage_sites : dict, box_size = [250,250],
               save_image = False, image_path : str = None) :
    """_summary_

    Args:
        image (np.ndarray): SEM image to be shown
        damage_sites (dict): python dictionary using the coordinates as key (x,y), and the label as value
        box_size (list, optional): size of the rectangle drawn around each centroid. Defaults to [250,250].
        save_image (bool, optional): save the image with the boxes or not. Defaults to False.
        image_path (str, optional) : Full path and name of the output file to be saved
    """

    _, ax = plt.subplots(1)
    fig = plt.imshow(image,cmap='gray')
    # do not show axis ticks (indicating pixels)
    plt.xticks([])
    plt.yticks([]) 

    for key, label in damage_sites.items():
        
        position = list([key[0],key[1]])

        # define colours of the rectangles overlaid on the image per damage type
        match label:
            case 'Inclusion':
                edgecolor = 'b'
            case 'Interface' :
                edgecolor = 'g'
            case 'Martensite' :
                edgecolor = 'r'
            case 'Notch':
                edgecolor = 'y'
            case 'Shadowing' :
                edgecolor = 'm'
            case _:
                edgecolor = 'k'
        
            
        rectangle = patches.Rectangle((position[1]-box_size[1]/2., position[0]-box_size[0]/2),
                                       box_size[0],box_size[1],
                                       linewidth=1,edgecolor=edgecolor,facecolor='none')
        ax.add_patch(rectangle)


    legend_elements = [Line2D([0], [0], color='b', lw=4, label='Inclusion'),
                       Line2D([0], [0], color='g', lw=4, label='Interface'),
                       Line2D([0], [0], color='r', lw=4, label='Martensite'),
                       Line2D([0], [0], color='y', lw=4, label='Notch'),
                       Line2D([0], [0], color='m', lw=4, label='Shadow'),
                       Line2D([0], [0], color='k', lw=4, label='Not Classified')
        ]

    ax.legend(handles=legend_elements,bbox_to_anchor=(1.04, 1), loc="upper left")

    if save_image:
        plt.savefig(image_path,dpi=1200,bbox_inches='tight' )
    plt.show()

    return fig, image_path


###
### cut out small images from panorama, append colour information
###
def prepare_classifier_input(panorama : np.ndarray, centroids : list, window_size = [250,250]) -> list :
    """Create a list of smaller images from the SEM panoramic image. 
       The neural networks expect images of a given size that are centered around a single damage site candiates.
       For each centroid (from the clustering step before), we cut out a smaller image from the panorama of the size
       expected by the classfier network.
       Since the networks expect colour images, we repeat the gray-scale image 3 times for a given candiate site.

    Args:
        panorama (np.ndarray): SEM input image
        centroids (list): list of centroids for the damage site candidates
        window_size (list, optional): Size of the image expected by the neural network later. Defaults to [250,250].

    Returns:
        list: List of "colour" images cut out from the SEM panorama, one per damage site candidate
    """

    panorama_shape = panorama.shape

    # list of the small images cut out from the panorama,
    # each of these is then fed into the classfier model
    images = []

    for i in range(len(centroids)):
        x1 = int(math.floor(centroids[i][0] - window_size[0]/2))
        y1 = int(math.floor(centroids[i][1] - window_size[1]/2))
        x2 = int(math.floor(centroids[i][0] + window_size[0]/2))
        y2 = int(math.floor(centroids[i][1] + window_size[1]/2))
    

        ##
        ## Catch the cases in which the extract would go
        ## over the boundaries of the original image
        ##
        if x1<0:
            x1 = 0
            x2 = window_size[0]
        if x2>= panorama_shape[0]:
            x1 = panorama_shape[0] - window_size[0]
            x2 = panorama_shape[0]
        if y1<0:
            y1 = 0
            y2 = window_size[1]
        if y2>= panorama_shape[1]:
            y1 = panorama_shape[1] - window_size[1]
            y2 = panorama_shape[1]

        # we now need to create the image path from the panoramic image that corresponds to the 
        # centroid, with the size determined by the window_size. 
        # First, we create an empty container with np.zeros()
        tmp_img = np.zeros((window_size[1],  window_size[0],1), dtype=float)

        # Then we copy over the patch of the panomaric image.
        # The later classfier expects colour images, i.e. 3 colour channels for RGB
        # Since we use gray-scale images, we only have one colour information, so we add the image to the first colour channel
        tmp_img[:,:,0] = panorama[x1:x2,y1:y2]

        # rescale the colour values
        tmp_img = tmp_img*2./255. - 1.

        # The classifier expects colour images, i.e. 3 colour channels.
        # We "fake" this by repeating the same gray-scale information 3 times, once per colour channel
        tmp_img_colour = np.repeat(tmp_img,3, axis=2) #3

        images.append(tmp_img_colour)
    

    return images







        
    

