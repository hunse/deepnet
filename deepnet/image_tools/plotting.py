
import os

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

def display_available():
    return ('DISPLAY' in os.environ)

# figsize = (13,8.5)

# def figure(fignum=None, figsize=(8, 6)):
#     plt.ion()
#     if fignum is None:
#         f = plt.figure(figsize=figsize)
#     else:
#         f = plt.figure(fignum, figsize=figsize)
#         plt.clf()

#     return f

def show(image, ax=None, vlims=None, invert=False):
    kwargs = dict(interpolation='none')
    if vlims is not None:
        assert type(vlims) == tuple and len(vlims) == 2
        kwargs['vmin'], kwargs['vmax'] = vlims
    if image.ndim < 3:
        kwargs['cmap'] = 'gray' if not invert else 'gist_yarg'

    if ax is None: ax = plt.gca()
    ax.imshow(image, **kwargs)
    return ax

def tile(images, ax=None, rows=16, cols=24, random=False,
         grid=False, gridwidth=1, gridcolor='r', **show_params):
    """
    Plot tiled images to the current axis

    :images Each row is one flattened image
    """

    n_images = images.shape[0]
    imshape = images.shape[1:3]
    m, n = imshape
    n_channels = images.shape[3] if images.ndim > 3 else 1

    inds = np.arange(n_images)
    if random: npr.shuffle(inds)

    if n_channels == 1:
        img = np.zeros((m*rows, n*cols), dtype=images.dtype)
    else:
        img = np.zeros((m*rows, n*cols, n_channels), dtype=images.dtype)

    for ind in xrange(min(rows*cols, n_images)):
        i,j = (ind / cols, ind % cols)
        img[i*m:(i+1)*m, j*n:(j+1)*n] = images[inds[ind]]


    ax = show(img, ax=ax, **show_params)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if grid:
        for i in xrange(1,rows):
            ax.plot([-0.5, img.shape[1]-0.5], [i*m-0.5, i*m-0.5], '-',
                    color=gridcolor, linewidth=gridwidth)
        for j in xrange(1,cols):
            ax.plot([j*n-0.5, j*n-0.5], [-0.5, img.shape[0]-0.5], '-',
                    color=gridcolor, linewidth=gridwidth)

        ax.set_xlim([-0.5, img.shape[1]-0.5])
        ax.set_ylim([-0.5, img.shape[0]-0.5])
        ax.invert_yaxis()


def compare(imagesetlist,
            ax=None, rows=5, cols=20, vlims=None, grid=True, random=False):
    d = len(imagesetlist)

    nimages = imagesetlist[0].shape[0]
    imshape = imagesetlist[0].shape[1:]
    m, n = imshape
    img = np.zeros((d*m*rows, n*cols))

    inds = np.arange(nimages)
    if random:
        npr.shuffle(inds)

    for ind in range(min(rows*cols, nimages)):
        i,j = (ind / cols, ind % cols)
        for k in xrange(d):
            img[(d*i+k)*m:(d*i+k+1)*m, j*n:(j+1)*n] = \
                imagesetlist[k][inds[ind],:].reshape(imshape)

    ax = show(img, ax=ax, vlims=vlims)

    if grid:
        for i in xrange(1,rows):
            ax.plot( [-0.5, img.shape[1]-0.5], (d*i*m-0.5)*np.ones(2), 'r-' )
        for j in xrange(1,cols):
            ax.plot( [j*n-0.5, j*n-0.5], [-0.5, img.shape[0]-0.5], 'r-')

        ax.set_xlim([-0.5, img.shape[1]-0.5])
        ax.set_ylim([-0.5, img.shape[0]-0.5])
        ax.invert_yaxis()


def activations(acts, func, ax=None):
    if ax is None:
        ax = plt.gca()

    N = acts.size
    nbins = max(np.sqrt(N), 10)

    # minact = np.min(acts)
    # maxact = np.max(acts)
    minact, maxact = (-2, 2)
    ax.hist(acts.ravel(), bins=nbins, range=(minact,maxact), normed=True)

    # barwidth = (maxact - minact) / float(nbins)
    # leftout = np.sum(acts < minact) / float(N)
    # rightout = np.sum(acts > maxact) / float(N)
    # ax.bar([minact-barwidth, maxact], [leftout, rightout], width=barwidth)

    x = np.linspace(minact, maxact, 101)
    ax.plot(x, func(x))

    ax.set_xlim([minact, maxact])
    # ax.set_xlim([minact-barwidth, maxact+barwidth])


def filters(filters, ax=None, **kwargs):
    std = filters.std()
    tile(filters, ax=ax, vlims=(-2*std, 2*std), grid=True, **kwargs)


# class Imageset(object):
#     """
#     A container for a set of images, that facilitates common functions
#     including batches and visualization.
#     """

#     # batchlen = 100
#     figsize = (12,6)

#     def __init__(self, images, shape, batchlen=100, vlims=(0,1)):
#         """
#         :images[nparray]   Images, where each row is one (flattened) image
#         :shape[tuple]      Shape of an image (for unflattening)
#         :vlims[tuple]      Limits of image values (for display)
#         """

#         self.images = images
#         self.shape = tuple(shape)
#         self.batchlen = batchlen
#         self.vlims = vlims
#         self.num_examples = images.shape[0]
#         self.npixels = images.shape[1]
#         if self.npixels != shape[0]*shape[1]:
#             raise Exception('Shape must match number of pixels in images')

#     @property
#     def nbatches(self):
#         if self.num_examples % self.batchlen == 0:
#             return self.num_examples/self.batchlen
#         else:
#             return None

#     def todict(self):
#         return {'images': self.images, 'shape': self.shape, 'vlims': self.vlims}

#     @staticmethod
#     def fromdict(d):
#         return Imageset(**d)

#     def tofile(self, filename):
#         np.savez(filename, self.todict())

#     @staticmethod
#     def fromfile(filename):
#         d = np.load(filename)['arr_0'].item()
#         return Imageset.fromdict(d)

#     def imageset_like(self, images, shape=None):
#         return type(self)(images,
#                           shape=self.shape if shape is None else shape,
#                           vlims=self.vlims)

#     def image(self, i):
#         if i < 0 or i >= self.num_examples:
#             raise Exception('Invalid image index')
#         else:
#             return self.images[i,:]

#     def subset(self, imin, imax=None):
#         if imax is None:
#             return Imageset(self.images[0:imin], self.shape, vlims=self.vlims)
#         else:
#             return Imageset(self.images[imin:imax], self.shape, vlims=self.vlims)

#     @property
#     def batchshape(self):
#         if self.nbatches is None:
#             raise Exception('Examples cannot be evenly divided into batches')
#         else:
#             return (self.batchlen, self.npixels)

#     def batch(self, i):
#         if self.nbatches is None:
#             raise Exception('Examples cannot be evenly divided into batches')
#         elif i < 0 or i >= self.nbatches:
#             raise Exception('Invalid batch index')
#         else:
#             return self.images[i*self.batchlen:(i+1)*self.batchlen,:]

#     def show(self, ind, fignum=None):
#         figure(fignum=fignum, figsize=self.figsize)
#         show(plt.gca(), self.image(ind).reshape(self.shape), vlims=self.vlims)
#         plt.tight_layout()
#         plt.draw()

#     def tile(self, fignum=None, rows=16, cols=24, grid=False, random=False):
#         figure(fignum=fignum, figsize=self.figsize)
#         tile(plt.gca(), self.images, imshape=self.shape, vlims=self.vlims,
#              rows=rows, cols=cols, grid=grid, random=random)
#         plt.tight_layout()
#         plt.draw()

#     def compare(self, compim, fignum=None, **kwargs):
#         figure(fignum=fignum, figsize=self.figsize)
#         compare(plt.gca(), [self.images, compim.images], imshape=self.shape,
#                 vlims=self.vlims, **kwargs)
#         plt.tight_layout()
#         plt.draw()

#     def rmse(self, comim):
#         return np.mean(np.sqrt(np.mean((self.images - comim.images)**2, axis=1)))

