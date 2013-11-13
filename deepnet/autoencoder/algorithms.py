
import collections
import time

import numpy as np
import numpy.random
import scipy as sp
import scipy.optimize

import theano
import theano.tensor as T

from .. import image_tools as imtools

# def sgd_minibatch_fn(trainer, rate, clip=None):
#     x = T.matrix('x', dtype=trainer.dtype)
#     cost, ups = trainer.get_cost_updates(x)

#     grads = trainer.grads(cost)

#     updates = collections.OrderedDict(ups)

#     rate = T.cast(rate, dtype=trainer.dtype)
#     for param, grad in zip(trainer.params, grads):
#         if clip is not None:
#             grad = grad.clip(*clip)
#         updates[param] = param - rate*grad

#     return theano.function([x], cost, updates=updates,
#                            allow_input_downcast=True)

def sgd_minibatch_fn(trainer, rate, clip=None):
    x = T.matrix('x', dtype=trainer.dtype)
    cost, grads, updates = trainer.get_cost_grads_updates(x)

    rate = T.cast(rate, dtype=trainer.dtype)
    for param in trainer.params:
        grad = grads[param]
        if clip is not None:
            grad = grad.clip(*clip)
        updates[param] = param - rate*grad

    return theano.function([x], cost, updates=updates.items(),
                           allow_input_downcast=True)

def sgd(trainer, images, timages=None, test_fn=None,
        n_epochs=30, rate=0.05, clip=(-1,1),
        show=imtools.display_available(), vlims=None,
        save_fn=None):
    """
    Unsupervised training using Stochasitc Gradient Descent (SGD)
    """

    if timages is None:
        timages = images[:500]

    print "Performing SGD on a %d x %d autoencoder for %d epochs"\
        % (trainer.network.nvis, trainer.network.nhid, n_epochs)
    print "SGD params: %s" % dict(n_epochs=n_epochs, rate=rate, clip=clip)
    print "Trainer params: %s" % trainer.train_hypers

    ### create minibatch learning function
    train = sgd_minibatch_fn(trainer, rate=rate, clip=clip)

    exshape = images.shape[:1]
    imshape = images.shape[1:]
    # if len(exshape) == 1:
    #     ### split into batches
    #     batchlen = 100
    #     batches = images.reshape((-1, batchlen, np.prod(imshape)))
    # elif len(exshape) == 2:
    #     ### already in batches, so just collapse the shape
    #     batches = images.reshape(exshape + (np.prod(imshape),))
    # else:
    #     raise ValueError("Invalid input image shape %s" % images.shape)

    ### split into batches
    batchlen = 100
    batches = images.reshape((-1, batchlen, np.prod(imshape)))

    stats = dict(algorithm='sgd', n_epochs=0, cost=[],
                 hypers=dict(trainer.train_hypers))
    trainer.network.train_stats.append(stats)

    for epoch in xrange(n_epochs):
        # rate = rates[epoch] if epoch < len(rates) else rates[-1]
        cost = 0

        t = time.time()
        for batch in batches:
            cost += train(batch)
        t = time.time() - t

        test_stats = test(trainer, timages,
                          test_fn=test_fn, show=show, fignum=101, vlims=vlims)

        stats['n_epochs'] += 1
        stats['cost'].append(cost)
        for k, v in test_stats.items():
            if k not in stats: stats[k] = []
            stats[k].append(v)

        print "Epoch %d finished, t = %0.2f s, cost = %0.3e, %s" \
            % (epoch, t, cost, str(["%s = %0.2e" % (k,v) for k,v in test_stats.items()]))

        if save_fn is not None:
            save_fn()
            # trainer.network.to_file(save_name)

    return stats


def lbfgs(trainer, images, timages=None, test_fn=None,
          n_evals=10, clip=None,
          show=imtools.display_available(), vlims=None,
          save_fn=None):
    """
    Unsupervised training using limited-memory BFGS (L-BFGS)
    """

    batchlen = 5000

    if timages is None:
        timages = images[:500]

    print "Performing L-BFGS on a %d x %d autoencoder for %d function evals"\
        % (trainer.network.nvis, trainer.network.nhid, n_evals)
    print "L-BFGS params: %s" % dict(n_evals=n_evals, clip=clip)
    print "Trainer params: %s" % trainer.train_hypers

    test_stats = test(trainer, timages, test_fn=test_fn,
                      show=show, fignum=101, vlims=vlims)
    print ["%s = %0.3e" % (k,v) for k,v in test_stats.items()]

    ### Make functions to put parameters into one vector, and get them back.
    params = [p.get_value(borrow=False) for p in trainer.params]

    def split_params(params_vect):
        params_list = []
        i = 0
        for p in params:
            params_list.append(params_vect[i : i + p.size].reshape(p.shape))
            i += p.size
        return params_list

    def concat_params(params_list):
        return np.hstack([p.flatten() for p in params_list])

    p0 = concat_params(params).astype('float64')

    ### make Theano function
    s_p = T.vector(dtype=trainer.dtype)
    s_params = split_params(s_p)

    s_x = T.matrix('x', dtype=trainer.dtype)
    cost, grads, updates = trainer.get_cost_grads_updates(s_x)
    grads = [grads[p] for p in trainer.params]

    f_df = theano.function([s_x, s_p], [cost] + grads, updates=updates.items(),
                           givens=zip(trainer.params, s_params),
                           allow_input_downcast=True)

    ### make optimization function
    stats = dict(algorithm='lbfgs', n_evals=0, cost=[],
                 hypers=dict(trainer.train_hypers))
    trainer.network.train_stats.append(stats)

    # flatten images, and get indices
    images = images.reshape((images.shape[0], np.prod(images.shape[1:])))
    images_i = np.arange(len(images))

    def f_df_cast(p):
        t = time.time()
        i = np.random.choice(images_i, size=batchlen)
        x = images[i]
        outs = f_df(x, p)
        cost, grads = outs[0], outs[1:]
        grad = concat_params(grads)
        if clip is not None:
            grad = grad.clip(*clip)
        t = time.time() - t

        ### test
        if 1:
            for param, value in zip(trainer.params, split_params(p)):
                param.set_value(value.astype(param.dtype), borrow=False)

            test_stats = test(trainer, timages, test_fn=test_fn,
                              show=show, fignum=101, vlims=vlims)

            stats['n_evals'] += 1
            stats['cost'].append(cost)
            for k, v in test_stats.items():
                if k not in stats: stats[k] = []
                stats[k].append(v)

            print "Eval %d finished, t = %0.2f s, cost = %0.3e, %s" \
                % (stats['n_evals'], t, cost,
                   str(["%s = %0.2e" % (k,v) for k,v in test_stats.items()]))

        return cost.astype('float64'), grad.astype('float64')

    ### perform optimization
    t = time.time()
    p_opt, mincost, info_dct = sp.optimize.lbfgsb.fmin_l_bfgs_b(
        f_df_cast, p0, maxfun=n_evals, iprint=1)
    t = time.time() - t

    for param, opt_value in zip(trainer.params, split_params(p_opt)):
        param.set_value(opt_value.astype(param.dtype), borrow=False)

    print "Done. t = %0.3f s, cost = %0.2e" % (t, mincost)

    test_stats = test(trainer, timages, test_fn=test_fn,
                      show=show, fignum=101, vlims=vlims)
    print ["%s = %0.3e" % (k,v) for k,v in test_stats.items()]


def test(trainer, timages, test_fn=None,
         show=imtools.display_available(), fignum=None, vlims=None):
    # from ..image_tools import *
    # from ..image_tools import compare, activations, filters
    from .. import image_tools as imtools

    if test_fn is None:
        test_fn = trainer.network.compVHVraw

    imshape = timages.shape[1:]
    ims_shape = (-1,) + imshape
    x = timages.reshape((len(timages), -1))
    ha, h, ya, y = test_fn(x)

    rmse = imtools.rmse(x, y).mean()
    act = h.mean(axis=0)
    # test_stats = {'rmse': rmse, 'hidact': act}
    test_stats = collections.OrderedDict(
        rmse=rmse, hid_mean=act.mean(), hid_min=act.min(), hid_max=act.max())

    ### Show current results
    if show:
        import matplotlib.pyplot as plt

        # image_tools.figure(fignum=fignum, figsize=(12,12))
        plt.figure(fignum)
        plt.clf()
        rows, cols = 3, 2

        ax = plt.subplot(rows, 1, 1)
        imtools.compare([x.reshape(ims_shape), y.reshape(ims_shape)], vlims=vlims)

        ax = plt.subplot(rows, cols, cols+1)
        imtools.activations(ha, trainer.network.f.eval)

        ax = plt.subplot(rows, cols, cols+2)
        imtools.activations(ya, trainer.network.g.eval)

        ax = plt.subplot(rows, 1, 3)
        imtools.filters(trainer.network.filters, rows=8, cols=16)

        plt.tight_layout()
        plt.draw()

    return test_stats
