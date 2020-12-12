'''
    detect_faces.py

    Daniel Jeong
    danielje@cs.cmu.edu

    Finds bounding boxes on faces using a pre-trained MTCNN model.

'''

from mtcnn import MTCNN
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys
sys.path.append('../')
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# CONSTANTS

NPZ_PATH = '../FER_plus.npz'

# Given a path with images, detects the faces in them
def detect_faces(model, dataset, show_flag=False):
    if not os.path.exists(NPZ_PATH):
        raise RuntimeError('[Error] Invalid path provided!')

    # Read images from directory
    #img_names = [name for name in os.listdir(path) if '.jpg' in name]
    #print(f'{len(img_names)} images found.')

    # Read images via npz file
    data = np.load(NPZ_PATH, allow_pickle=True)

    if dataset == 'train':
        imgs = data['xtrain']

    elif dataset == 'validation':
        imgs = data['xvalid']

    elif dataset == 'test':
        imgs = data['xtest']

    else:
        raise RuntimeError(f'[Error] Invalid dataset type provided: {dataset}')

    rgb_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in imgs])
    del imgs # Remove from memory, garbage collection will follow

    no_face = 0
    face = 0
    multiple_faces = 0
    show_idx = 0

    face_imgs = {}

    for idx in tqdm(range(rgb_imgs.shape[0])):
        #img = cv2.imread(os.path.join(path, img_name), 0)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = rgb_imgs[idx]

        pred = model.detect_faces(img)

        if len(pred) < 1:
            print(f'No faces detected in "image {idx}."')
            no_face += 1

        elif len(pred) > 1:
            print(f'Multiple faces detected in "image {idx}."')
            multiple_faces += 1

        else:
            # x, y, width, height
            face_imgs[idx] = np.array(pred[0]['box'])

            face += 1

        # Just for verification
        if show_flag and (show_idx < 5) and (len(pred) == 1):
            plt.figure(figsize=(8,8))
            plt.imshow(img, cmap='gray')
            ax = plt.gca()
            x, y, width, height = pred[0]['box']
            rect = Rectangle((x,y), width, height, fill=False, color='red')
            ax.add_patch(rect)
            plt.show()
            plt.close()

            show_idx += 1

    assert(rgb_imgs.shape[0] == (no_face + face + multiple_faces))
    print(f'\nNo face: {no_face}\nFace: {face}\nMultiple Faces: {multiple_faces}\n')

    with open(f'../FER_plus_{dataset}_bb.pkl', 'wb') as fh:
        pickle.dump(face_imgs, fh)
        print('Pickle file of bounding box labels saved.')

def main(**kwargs):
    dataset = kwargs['dataset']
    show_flag = kwargs['show']

    model = MTCNN()
    detect_faces(model, dataset, show_flag=show_flag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    main(**vars(args))