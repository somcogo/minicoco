from glob import glob
import os
import time

import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
from pycocotools.coco import COCO

coco = COCO('data/annotations/instances_val2017.json')
resize = Resize((64, 64))

trn_mean = np.asarray([0.46987239, 0.44626274, 0.40666163])
trn_std = np.asarray([0.43082439, 0.43424141, 0.42904485])

val_mean = np.asarray([0.46943393, 0.44602461, 0.40660645])
val_std = np.asarray([0.43071977, 0.43432737, 0.42935181])

val_path_list = glob('data/images/val2017/*')
val_path_list.sort()
trn_path_list = glob('data/images/train2017/*')
trn_path_list.sort()

trn_img_ids = [int(os.path.split(path)[-1][:12]) for path in trn_path_list]
trn_ann_ids = [coco.getAnnIds(img_id) for img_id in trn_img_ids]
val_img_ids = [int(os.path.split(path)[-1][:12]) for path in val_path_list]
val_ann_ids = [coco.getAnnIds(img_id) for img_id in val_img_ids]

f_trn = h5py.File('data/minicoco_trn.hdf5', 'w')
f_trn.create_dataset('data', shape=(25000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_trn.create_dataset('mask', shape=(25000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_trn.create_dataset('coco_img_ids', data=trn_img_ids)
f_trn.create_dataset('coco_ann_ids', data=trn_ann_ids)

f_val = h5py.File('data/minicoco_val.hdf5', 'w')
f_val.create_dataset('data', shape=(5000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_val.create_dataset('mask', shape=(5000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_trn.create_dataset('coco_img_ids', data=val_img_ids)
f_trn.create_dataset('coco_ann_ids', data=val_ann_ids)
f_val.close()

for i, path in enumerate(trn_path_list):
    image = Image.open(path)
    image = resize(image)
    np_data = np.asarray(image)
    if len(np_data.shape) == 2:
        np_data = np.stack([np_data, np_data, np_data], axis=2)
    np_data = np.divide(np_data-trn_mean, trn_std)
    f_trn['data'][i] = np_data

    img_id = trn_img_ids[i]
    img = coco.loadImgs(img_id)
    ann_ids = coco.getAnnIds(img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((img[0]['height'], img[0]['width']))
    for ann in anns:
        mask = np.maximum(coco.annToMask(ann) * ann['category_id'], mask)
    mask = np.asarray(resize(mask)).astype(int)
    f_trn['mask'][i] = mask




    img = coco.getImgIds

f_trn.close()