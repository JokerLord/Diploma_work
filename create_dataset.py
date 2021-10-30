import multiprocessing
import os
import numpy as np
import argparse
import queue
from PIL import Image
from random import randint
from pathlib import Path
from multiprocessing import Process, Queue
from tensorflow.keras.utils import to_categorical
from time import time

PATCH_WIDTH: int = 224
PATCH_HEIGHT: int = 224
NUM_OF_PATCHES: int = 20

PIC_WIDTH: int = 10000
PIC_HEIGHT: int = 10000

FOLDER: str = 'D:\Learning\MSU\Year 4th\Diploma Files'


def scale_range(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


def create_dataset(scale: float, height: int, width: int, num_classes: int, num_imgs: int, input_folder_path: Path,
                   common_save_folder_path: Path):
    data = []
    labels = []
    gt_vectors = []
    size = (int(scale * height), int(scale * width))
    for class_num in range(num_classes):
        for img_num in range(num_imgs):
            img = np.array(Image.open(input_folder_path / Path(str(class_num) + '_' + str(img_num) + '.jpg')),
                           copy=False)
            for k in range(NUM_OF_PATCHES):
                x = randint(0, PIC_HEIGHT - size[0])
                y = randint(0, PIC_WIDTH - size[1])
                patch = img[x: x + size[0], y: y + size[1]]
                patch = np.array(Image.fromarray(patch).resize((height, width)))
                data.append(patch)
                labels.append(class_num)
                gt_vectors.append(to_categorical(class_num, num_classes=20))
    data = np.array(data)
    labels = np.array(labels)
    gt_vectors = np.array(gt_vectors)
    np.savez(common_save_folder_path / Path(format(scale, f'.1f')), data=data, labels=labels, gt_vectors=gt_vectors)


def do_job(tasks_to_do):
    while True:
        try:
            task = tasks_to_do.get_nowait()
            create_dataset(*task)
        except queue.Empty:
            break
        else:
            print(f'done with scale {task[0]}', flush=True)
    return True


def create_tests_parallel(args):
    common_save_folder_path = Path(FOLDER) / Path(args.save_folder)
    input_folder_path = Path(FOLDER) / Path(args.input)
    if not os.path.isdir(common_save_folder_path):
        os.mkdir(common_save_folder_path)

    input_params = [
        (scale, PATCH_HEIGHT, PATCH_WIDTH, args.num_of_classes, args.num_of_imgs, input_folder_path, common_save_folder_path) for
        scale in scale_range(0.5, 10.0, 0.1)]

    number_of_processes = multiprocessing.cpu_count() - 1
    tasks_to_do = Queue()
    processes = []

    for i in input_params:
        tasks_to_do.put(i)

    for _ in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_do,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def create_train(args):
    common_save_folder_path = Path(FOLDER) / Path(args.save_folder)
    input_folder_path = Path(FOLDER) / Path(args.input)
    if not os.path.isdir(common_save_folder_path):
        os.mkdir(common_save_folder_path)

    create_dataset(args.scale, PATCH_HEIGHT, PATCH_WIDTH, args.num_of_classes, args.num_of_imgs, input_folder_path,
                   common_save_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_test = subparsers.add_parser('test', help='Generate test datasets for various scales')
    parser_test.add_argument('input', help='Name of input folder')
    parser_test.add_argument('num_of_classes', type=int)
    parser_test.add_argument('num_of_imgs', type=int, help='Number of images in one class')
    parser_test.add_argument('save_folder', help='Name of save folder')
    parser_test.set_defaults(func=create_tests_parallel)

    parser_train = subparsers.add_parser('train', help='Generate train dataset for particular scale')
    parser_train.add_argument('scale', type=float)
    parser_train.add_argument('input', help='Name of input folder')
    parser_train.add_argument('num_of_classes', type=int)
    parser_train.add_argument('num_of_imgs', type=int, help='Number of images in one class')
    parser_train.add_argument('save_folder', help='Name of save folder')
    parser_train.set_defaults(func=create_train)

    args = parser.parse_args()
    args.func(args)
