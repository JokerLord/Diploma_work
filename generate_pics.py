import multiprocessing
from multiprocessing import process
from pathlib import Path
import queue
import numpy as np
from crop import CLASS_NUM
import cv2
import time
import os
import argparse
from PIL import Image, ImageDraw
from random import randint
from numpy.random import normal
from multiprocessing import Process, Queue


PIC_WIDTH = 10000
PIC_HEIGHT = 10000

COLORS = ['red', 'green', 'blue', 'yellow', 'purple']
FIGURES = ['rectangle', 'ellipse', 'cross', 'triangle']

FIGURE_WIDTH = 40
FIGURE_HEIGHT = 40

NOISE_LEVEL = 1
NOISE_SIGMA = 10.0

DENSITY_CONST = 4


def generate_pics(folder: Path, class_num: int, num_of_imgs: int):
    density = int(DENSITY_CONST * PIC_HEIGHT)

    fig = class_num % 4
    color = class_num // 4
    if color > 4:
        raise Exception("Class number can't be bigger than 20!")
    elif color == 0:
        r, g, b = (255, 0, 0)
    elif color == 1:
        r, g, b = (0, 255, 0)
    elif color == 2:
        r, g, b = (0, 0, 255)
    elif color == 3:
        r, g, b = (255, 255, 0)
    else:
        r, g, b = (128, 0, 128)

    for img_num in range(num_of_imgs):
        image = Image.new(mode="RGB", size=(PIC_WIDTH, PIC_HEIGHT),
                          color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        for k in range(density):
            dsize = randint(-5, 5)

            x = randint(0, PIC_WIDTH - FIGURE_WIDTH - dsize)
            y = randint(0, PIC_HEIGHT - FIGURE_HEIGHT - dsize)

            dcolor = randint(-20, 20)
            if FIGURES[fig] == 'rectangle':
                draw.rectangle(xy=[x, y, x + FIGURE_WIDTH + dsize, y + FIGURE_HEIGHT + dsize],
                               fill=(r + dcolor, g + dcolor, b + dcolor))
            elif FIGURES[fig] == 'ellipse':
                draw.ellipse(xy=[x, y, x + FIGURE_WIDTH + dsize, y + FIGURE_HEIGHT + dsize],
                             fill=(r + dcolor, g + dcolor, b + dcolor))
            elif FIGURES[fig] == 'cross':
                draw.line(xy=[(x, y), (x + FIGURE_WIDTH + dsize, y + FIGURE_HEIGHT + dsize)],
                          fill=(r + dcolor, g + dcolor, b + dcolor), width=7)
                draw.line(xy=[(x, y + FIGURE_HEIGHT + dsize), (x + FIGURE_WIDTH + dsize, y)],
                          fill=(r + dcolor, g + dcolor, b + dcolor), width=7)
            else:
                draw.polygon(xy=[(x, y + FIGURE_HEIGHT + dsize),
                                 (x + (FIGURE_WIDTH + dsize) // 2, y),
                                 (x + FIGURE_WIDTH + dsize, y + FIGURE_HEIGHT + dsize)],
                             fill=(r + dcolor, g + dcolor, b + dcolor))

        noise = normal(0.0, NOISE_SIGMA, (PIC_WIDTH // 2, PIC_HEIGHT // 2))
        noise = np.array(Image.fromarray(noise).resize((PIC_WIDTH, PIC_HEIGHT), resample=Image.BILINEAR))
        img = np.array(image, dtype=np.float64)
        for channel in range(img.shape[2]):
            img[:, :, channel] += NOISE_LEVEL * noise
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        Image.fromarray(img.astype(np.uint8)).save(folder / Path(str(class_num) + '_' + str(img_num) + '.png'))


def do_job(tasks_to_do):
    while True:
        try:
            task = tasks_to_do.get_nowait()
            generate_pics(*task)
        except queue.Empty:
            break
        else:
            print(f'done with class {task[1]}', flush=True)
    return True


def generate_pics_parallel(save_folder: Path, num_of_classes: int, num_of_imgs: int):
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    input_params = [(save_folder, i, num_of_imgs) for i in range(num_of_classes))]
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', help='Save folder')
    parser.add_argument('num_of_classes', type=int, help='Number of classes')
    parser.add_argument('num_of_imgs', type=int, help='Number of images in one class to generate')
    args = parser.parse_args()

    generate_pics_parallel(Path(args.save_folder), args.num_of_classes, args.num_of_imgs)
