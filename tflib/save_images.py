"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def unblockshaped(arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                   .swapaxes(1,2)
                   .reshape(h, w))

def save_images(X, save_path):

    # 1. PREPROCESSING
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')  # [0, 1] -> [0,255]

    # 2. GET WIDTH AND HEIGHT (to convert a 3D image -> 2D image)
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = int(rows), int(n_samples/rows)

    # CONVERT 3D ARRAY TO TILED 2D IMAGE (using some numpy magic)
    h, w = X[0].shape[:2]
    img = np.array(unblockshaped(X, h*nh, w*nw))
    
    # if X.ndim == 2:
    #     X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    # if X.ndim == 4:
    #     # BCHW -> BHWC
    #     X = X.transpose(0,2,3,1)
    #     h, w = X[0].shape[:2]
    #     img = np.zeros(( int(h*nh), int(w*nw), 3))
    
    # elif X.ndim == 3:
    #     h, w = X[0].shape[:2]
    #     img = np.zeros(( int(h*nh), int(w*nw) ))
    
    # print ('\nX:', X.shape, ' img:', img.shape)
    # print ('nh:', nh, 'nw:', nw, ' h:', h, ' w:', w)

    # for n, x in enumerate(X):
    #     j = n / nw
    #     i = n % nw
    #     print ('jh:', int(j*h), ':',int(j*h)+h, ' iw:', int(i*w),':', int(i*w)+w)
    #     img[ int(j*h) : int(j*h)+h, int(i*w) : int(i*w)+w ] = x

    # for img_id, img in enumerate(X):
    #     block_w = int(img_id / nw) #
    #     block_h = int(img_id % nw) #
    #     print (block_w, block_h)
    #     # print (int(block_w*w), int(block_w*w)+w, ',', int(block_h*h), int(block_h*h)+h, img[ int(block_w*w) : int(block_w*w)+w, int(block_h*h) : int(block_h*h)+h ].shape)
        # img[ int(block_w*w) : int(block_w*w)+w, int(block_h*h) : int(block_h*h)+h ] = img

    
    imsave(save_path, img)

# PROOF OF THE unblockshaped trick
def convert_3D_image_to_2D_image(url_image, nx = 28, ny = 28, img_copies = 128):
    def unblockshaped(arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                   .swapaxes(1,2)
                   .reshape(h, w))

    import numpy as np
    from scipy.misc import imread
    from scipy.misc import imresize
    import matplotlib.pyplot as plt
    # %matplotlib inline

    # 1. READ AND RESIZE IMAGE
    f, axarr = plt.subplots(1,2, figsize = (10,10))
    a = imresize(imread(url_image,flatten=True),(nx,ny)) 
    print ('a:',a.shape)
    axarr[0].imshow(a)

    # 2. DUPLICATE IMAGE AND CREATE A 3D ARRAY
    X = []
    for i in range(img_copies):
        X.append(a)
    X = np.array(X)
    print ('X:',X.shape)
    
    tileimg_rows = int(np.sqrt(img_copies))
    while img_copies % tileimg_rows != 0:
        tileimg_rows -= 1
    tileimg_cols = int(img_copies / tileimg_rows)
    
    print ('Tile Image: (', tileimg_rows * nx, tileimg_cols * ny,')')
    
    # 3. NUMPY MAGIC TO CONVERT 3D ARRAY INTO 2D IMAGE (TILE-FORMAT)
    tmp = unblockshaped(X, tileimg_rows*ny, tileimg_cols*nx)
    axarr[1].imshow(tmp)

# convert_3D_image_to_2D_image('sample_images.jpeg')