import os
import cv2
import yaml
import numpy as np
import shutil
import random
from glob import glob
from tqdm import tqdm
import subprocess

# --------- SETUP ---------
IMG_DIR = 'Training/images'     # input imgs
MASK_DIR = 'Training/masks'     # corr masks
OUT_DIR = 'YOLO_dataset'        # where final YOLO struct will go

CLASSES = ['tumor']             # class list (edit if needed)
IMG_EXT = '.jpg'
MASK_EXT = '.png'

SPLIT = 0.8                     # train-val split
SHOW_DEBUG = False              # draw contours (for checking)
LAUNCH_TRAINING = True
MODEL_TYPE = 'yolov8n-seg.pt'
EPOCHS = 100
IMG_SIZE = 512

# --------- DIR SETUP ---------
def make_dirs():
    for s in ['train', 'val']:
        os.makedirs(f'{OUT_DIR}/images/{s}', exist_ok=True)
        os.makedirs(f'{OUT_DIR}/labels/{s}', exist_ok=True)

# --------- CONVERT MASK → POLY ---------
def get_polygons(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    out = []

    for c in cnts:
        if cv2.contourArea(c) < 10:
            continue
        norm = c.reshape(-1, 2) / [w, h]
        flat = norm.flatten()
        if len(flat) >= 6:
            out.append('0 ' + ' '.join(f'{p:.6f}' for p in flat))

    return out

# --------- COPY IMG + LABEL ---------
def save_pair(img_path, split):
    name = os.path.basename(img_path)
    mask_path = os.path.join(MASK_DIR, name.replace(IMG_EXT, MASK_EXT))

    if not os.path.exists(mask_path):
        return

    mask = cv2.imread(mask_path, 0)
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    polys = get_polygons(bin_mask)
    if not polys:
        return

    shutil.copy(img_path, f'{OUT_DIR}/images/{split}/{name}')
    with open(f'{OUT_DIR}/labels/{split}/{name.replace(IMG_EXT, ".txt")}', 'w') as f:
        f.write('\n'.join(polys))

    if SHOW_DEBUG:
        vis = cv2.imread(img_path)
        for c in cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
        cv2.imshow("view", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --------- YAML FILE ---------
def write_yaml():
    stuff = {
        'train': os.path.abspath(f'{OUT_DIR}/images/train'),
        'val': os.path.abspath(f'{OUT_DIR}/images/val'),
        'nc': len(CLASSES),
        'names': CLASSES
    }
    with open(f'{OUT_DIR}/data.yaml', 'w') as f:
        yaml.dump(stuff, f)

# --------- START TRAINING ---------
def run_training():
    print("\n==> training...\n")
    cmd = [
        'yolo',
        'task=segment',
        'mode=train',
        f'model={MODEL_TYPE}',
        f'data={OUT_DIR}/data.yaml',
        f'epochs={EPOCHS}',
        f'imgsz={IMG_SIZE}'
    ]
    subprocess.run(' '.join(cmd), shell=True)

# --------- MAIN ---------
def main():
    make_dirs()
    imgs = sorted(glob(f'{IMG_DIR}/*{IMG_EXT}'))
    random.shuffle(imgs)

    cut = int(len(imgs) * SPLIT)
    train, val = imgs[:cut], imgs[cut:]

    for i in tqdm(train, desc='Train set'):
        save_pair(i, 'train')
    for i in tqdm(val, desc='Val set'):
        save_pair(i, 'val')

    write_yaml()
    print("✅ dataset ready!")

    if LAUNCH_TRAINING:
        run_training()

if __name__ == '__main__':
    main()
