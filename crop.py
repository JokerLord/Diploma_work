import multiprocessing
import os
import numpy as np
import argparse
import queue
from PIL import Image
from random import randint
from multiprocessing import Process, Queue
from pathlib import Path
from multiprocessing import Process, Queue

WIDTH = 224
HEIGTH = 224
NUM_OF_PATCHES = 20


def scale_range(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

def crop(scale: float, height: int, width: int, class_num: int, num_of_imgs: int, folder: Path, save_folder: Path):
    for img_num in range(num_of_imgs):
        input_file = folder / Path(str(class_num) + '_' + str(img_num) + '.png')
        img = np.array(Image.open(input_file))
        size = (int(scale * height), int(scale * width))
        for k in range(NUM_OF_PATCHES):
            x = randint(0, img.shape[0] - size[0])
            y = randint(0, img.shape[1] - size[1])
            patch = img[x: x + size[0], y: y + size[1]]
            Image.fromarray(patch).resize((height, width),
                                        resample=Image.BILINEAR).save(
                                        save_folder / Path(str(class_num) + '_' + str(img_num) + ' ' + str(k) + '.jpg'))


def do_job(tasks_to_do):
    while True:
        try:
            task = tasks_to_do.get_nowait()
            crop(*task)
        except queue.Empty:
            break
        else:
            print(f'done with class {task[3]}', flush=True)
    return True

def crop_parallel(scale: float, height: int, width: int, num_of_classes: int, num_of_imgs: int, folder: Path, save_folder: Path):
    input_params = [(scale, height, width, i, num_of_imgs, folder, save_folder) for i in range(num_of_classes)]

    number_of_processes = multiprocessing.cpu_count()
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

def make_tests(args):
    os.mkdir(Path(args.save_folder))
    for scale in scale_range(4.1, 10.0, 0.1):
        save_folder = Path(args.save_folder) /  Path('test' + format(scale, f'.1f'))
        os.mkdir(save_folder)
        crop_parallel(scale, HEIGTH, WIDTH, args.num_of_classes, args.num_of_imgs, args.input, save_folder)

def make_train(args):
    os.mkdir(Path(args.save_folder))
    save_folder = Path(args.save_folder) / Path('train' + format(args.scale, f'.1f'))
    os.mkdir(save_folder)
    crop_parallel(args.scale, HEIGTH, WIDTH, args.num_of_classes, args.num_of_imgs, Path(args.input), save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_test = subparsers.add_parser('test', help='Generate test datasets for various scales')
    parser_test.add_argument('input', help='Input folder')
    parser_test.add_argument('num_of_classes', type=int)
    parser_test.add_argument('num_of_imgs', type=int, help='Number of images in one class')
    parser_test.add_argument('save_folder')
    parser_test.set_defaults(func=make_tests)

    parser_train = subparsers.add_parser('train', help='Generate train dataset for particular scale')
    parser_train.add_argument('scale', type=float)
    parser_train.add_argument('input', help='Input folder')
    parser_train.add_argument('num_of_classes', type=int)
    parser_train.add_argument('num_of_imgs', type=int, help='Number of images in one class')
    parser_train.add_argument('save_folder')
    parser_train.set_defaults(func=make_train)

    args = parser.parse_args()
    args.func(args)
