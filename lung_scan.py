import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

import util
import lung_segm


BACKGROUND_CT = -1000
MAX_CT = 400


def read_annot_df(annot_file_path):
    annot_df = pd.read_csv(annot_file_path)
    annot_df.set_index('seriesuid', inplace=True)
    return annot_df


class Scan(object):
    
    def __init__(self, file_path, annot_df):
        self.file_path = file_path
        self.suid = os.path.basename(file_path)[0:-4]

        itk = sitk.ReadImage(file_path)
        # x, y, z
        self.origin = np.asarray(itk.GetOrigin(), dtype=np.float)
        # x, y, z
        self.spacing = np.asarray(itk.GetSpacing(), dtype=np.float)
        # z, y, x
        self.image = sitk.GetArrayFromImage(itk)
        # [x, y, z, diameter]
        self.nodules = np.asarray(
            annot_df.loc[self.suid:self.suid], dtype=np.float)


def _apply_mask(image, mask):
    image = image.copy()
    image[mask==False] = BACKGROUND_CT
    return image


def _normalize(image):
    image = (image - BACKGROUND_CT) / float(MAX_CT - BACKGROUND_CT)
    image[image>1] = 1.
    image[image<0] = 0.
    return image.astype(np.float32)


class ProcessedScan(object):
    
    def __init__(self):
        return

    def init(self, scan, spacing=1.0):
        self.suid = scan.suid
        self.spacing = float(spacing)

        # Resample
        # x, y, z
        resize_factor = scan.spacing / np.asarray([self.spacing] * 3)
        self.image = ndimage.interpolation.zoom(
            scan.image, zoom=resize_factor[::-1], mode='nearest')
            
        # Lung mask
        self.mask = lung_segm.segm_3d(self.image, self.spacing)

        # Standardize image
        self.std_image = _normalize(_apply_mask(self.image, self.mask))
        
        # Nodules
        # x, y, z
        scan_image_size = (
            scan.spacing * np.asarray(scan.image.shape[::-1]))
        nodules = []
        for x, y, z, d in scan.nodules:
            nod = np.asarray([x, y, z])
            nod = (nod - scan.origin) / scan_image_size
            x, y, z = nod * np.asarray(self.image.shape[::-1])
            nodules.append([x, y, z, d / self.spacing])
        self.nodules = np.asarray(nodules, dtype=np.float)

    def save(self, file_path):
        np.savez_compressed(
            file_path,
            suid=self.suid,
            spacing=self.spacing,
            image=self.image,
            mask=self.mask,
            std_image=self.std_image,
            nodules=self.nodules)
        
    def load(self, file_path):
        ans = np.load(file_path)
        self.suid = ans['suid'].item()
        self.spacing = ans['spacing']
        self.image = ans['image']
        self.mask = ans['mask']
        self.std_image = ans['std_image']
        self.nodules = ans['nodules']
