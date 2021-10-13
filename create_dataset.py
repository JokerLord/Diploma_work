import numpy as np
import glob
import argparse
import os

from tensorflow.keras.utils import to_categorical
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', help='Folder with folders to be transformed into npz')
    parser.add_argument('output_folder')
    args = parser.parse_args()

    for folder in os.listdir(args.input_folder):
        data = []
        labels = []
        gt_vectors = []
        for filename in glob.glob(args.input_folder + '\\' + folder + '\\' + "*"):
            img = np.array(Image.open(filename))
            data.append(img)
            start = filename.rfind('\\')
            end = filename.find('_')
            class_id = int(filename[start+1:end])
            labels.append(class_id)
            gt_vectors.append(to_categorical(class_id, num_classes=20))
        data = np.array(data)
        labels = np.array(labels)
        gt_vectors = np.array(gt_vectors)
        np.savez(args.output_folder + '\\' + folder, data=data, labels=labels, gt_vectors=gt_vectors)