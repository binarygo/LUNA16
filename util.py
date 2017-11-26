import numpy as np
import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def find_first_neq(a, val):
    for i, x in enumerate(a):
        if np.any(np.abs(x - val) > 1.0e-12):
            return i


def find_bbox(a, margin, bg):
    margin = np.round(margin).astype(np.int)
    slices = []
    for axis in range(len(a.shape)):
        view = np.rollaxis(a, axis)
        dim = a.shape[axis]
        m = margin[axis]
        front = find_first_neq(view, bg)
        front = min(dim, max(0, front - m))
        back = find_first_neq(view[::-1], bg)
        back = min(dim, max(0, back - m))
        slices.append(slice(front, dim - back))
    return slices
