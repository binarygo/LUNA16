import hashlib
import os
import sys
import tensorflow as tf
from glob import glob

import util
import lung_scan


def to_example(pscan):
    std_image, nodules = pscan.tighten()
    size_z, size_y, size_x = std_image.shape
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'size_x': util.int64_feature([size_x]),
            'size_y': util.int64_feature([size_y]),
            'size_z': util.int64_feature([size_z]),
            'std_image': util.bytes_feature([std_image.tostring()]),
            'num_nodules': util.int64_feature([len(nodules)]),
            'nodule_x': util.float_feature([nod[0] for nod in nodules]),
            'nodule_y': util.float_feature([nod[1] for nod in nodules]),
            'nodule_z': util.float_feature([nod[2] for nod in nodules]),
            'nodule_d': util.float_feature([nod[3] for nod in nodules]),
        }))
    return example


def _hash_suid(suid):
    PREC_N = 1000000000
    return int(hashlib.md5(suid).hexdigest(), 16) % PREC_N / float(PREC_N)


def _write_example(example, file_path):
    writer = tf.python_io.TFRecordWriter(file_path)
    writer.write(example.SerializeToString())
    writer.close()
    

if __name__ == '__main__':
    file_path_list = sorted(glob('data/all_scans/*.npz'))
    for i, file_path in enumerate(file_path_list):
        print('Process %s: %d of %d'%(file_path, i, len(file_path_list)))
        sys.stdout.flush()
        pscan = lung_scan.ProcessedScan()
        pscan.load(file_path)

        p = _hash_suid(pscan.suid)
        output_dir = os.path.join('data/examples')
        output_split = None
        if p < 0.8:
            output_split = 'train'
        elif p < 0.9:
            output_split = 'valid'
        else:
            output_split = 'test'
        output_path = os.path.join(
            output_dir, output_split, pscan.suid + '.rio')
        _write_example(to_example(pscan), output_path)
