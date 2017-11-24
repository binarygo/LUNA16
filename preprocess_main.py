import os
import sys
from glob import glob

import lung_scan


if __name__ == '__main__':
    file_path_list = sorted(glob('data/subset*/*.mhd'))
    annot_df = lung_scan.read_annot_df('data/CSVFILES/annotations.csv')
    for i, file_path in enumerate(file_path_list):
        print('Process %s: %d of %d'%(file_path, i, len(file_path_list)))
        sys.stdout.flush()
        scan = lung_scan.Scan(file_path, annot_df)
        pscan = lung_scan.ProcessedScan()
        pscan.init(scan)
        pscan.save(os.path.join('data/all_scans/', scan.suid + '.npz'))
