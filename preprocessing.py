from glob import glob
import os
import time

import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import torch
from pycocotools.coco import COCO

t1 = time.time()
coco = COCO('data/annotations/instances_train2017.json')
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

f_trn = h5py.File('data/minicoco_trn_v2.hdf5', 'w')
f_trn.create_dataset('data', shape=(25000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_trn.create_dataset('mask', shape=(25000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_trn.create_dataset('coco_img_ids', data=trn_img_ids)

t2 = time.time()
t3 = time.time()
print('setup:', t2 - t1)
for i, path in enumerate(trn_path_list):
    image = Image.open(path)
    image = resize(image)
    np_data = np.asarray(image) / 255
    if len(np_data.shape) == 2:
        np_data = np.stack([np_data, np_data, np_data], axis=2)
    np_data = np.divide(np_data-trn_mean, trn_std)
    f_trn['data'][i] = np_data

    img_id = trn_img_ids[i]
    img = coco.loadImgs(img_id)
    ann_ids = coco.getAnnIds(img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((1, 64, 64))
    for ann in anns:
        ann_mask = coco.annToMask(ann)
        ann_mask = np.expand_dims(ann_mask, axis=0)
        ann_mask = torch.Tensor(ann_mask)
        ann_mask = np.asarray(resize(ann_mask)).astype(int) * ann['category_id']
        mask = np.maximum(ann_mask, mask)
    mask = mask.transpose(1, 2, 0)
    f_trn['mask'][i] = mask
    f_trn.flush()
    if i % 1000 == 0:
        t4 = time.time()
        print(i, t4 - t3)
        t3 = time.time()

f_trn.close()

coco = COCO('data/annotations/instances_val2017.json')

val_img_ids = [int(os.path.split(path)[-1][:12]) for path in val_path_list]
val_ann_ids = [coco.getAnnIds(img_id) for img_id in val_img_ids]

f_val = h5py.File('data/minicoco_val_v2.hdf5', 'w')
f_val.create_dataset('data', shape=(5000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_val.create_dataset('mask', shape=(5000, 64, 64, 3), chunks=(1, 64, 64, 3), dtype='float32')
f_val.create_dataset('coco_img_ids', data=val_img_ids)

t3 = time.time()
for i, path in enumerate(val_path_list):
    image = Image.open(path)
    image = resize(image)
    np_data = np.asarray(image) / 255
    if len(np_data.shape) == 2:
        np_data = np.stack([np_data, np_data, np_data], axis=2)
    np_data = np.divide(np_data-val_mean, val_std)
    f_val['data'][i] = np_data

    img_id = val_img_ids[i]
    img = coco.loadImgs(img_id)
    ann_ids = coco.getAnnIds(img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((1, 64, 64))
    for ann in anns:
        ann_mask = coco.annToMask(ann)
        ann_mask = np.expand_dims(ann_mask, axis=0)
        ann_mask = torch.Tensor(ann_mask)
        ann_mask = np.asarray(resize(ann_mask)).astype(int) * ann['category_id']
        mask = np.maximum(ann_mask, mask)
    mask = mask.transpose(1, 2, 0)
    f_val['mask'][i] = mask
    f_val.flush()
    if i % 1000 == 0:
        t4 = time.time()
        print(i, t4 - t3)
        t3 = time.time()

f_val.close()