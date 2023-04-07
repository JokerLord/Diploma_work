from pathlib import Path
import queue
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw
from multiprocessing import Process, Queue, cpu_count
from numpy.random import randint, normal

PIC_WIDTH = 10000
PIC_HEIGHT = 10000

COLORS = ["red", "green", "blue", "yellow", "purple"]
FIGURES = ["rectangle", "ellipse", "cross", "triangle"]

FIGURE_WIDTH = 40
FIGURE_HEIGHT = 40

NOISE_MEAN = 0.0

DENSITY_CONST = 4

FOLDER = "D:\\Learning\\MSU\\Year 4th\\Diploma Files"


# Функция генерирует num_of_imgs картинок для класса class_num
def generate_pics(save_folder_path: Path, class_num: int, num_of_imgs: int, is_noised: bool, noise_sigma: float):
    density = int(DENSITY_CONST * PIC_HEIGHT)

    fig, color = class_num % 4, class_num // 4
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

    dsizes = randint(-5, 5, size=(num_of_imgs, density))
    dcolors = randint(-20, 20, size=(num_of_imgs, density))

    for img_num in range(num_of_imgs):
        image = Image.new(mode="RGB", size=(PIC_WIDTH, PIC_HEIGHT),
                          color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        for k in range(density):
            dsize = dsizes[img_num, k]
            x = randint(0, PIC_WIDTH - FIGURE_WIDTH - dsize)
            y = randint(0, PIC_HEIGHT - FIGURE_HEIGHT - dsize)

            x1 = x + FIGURE_WIDTH + dsize
            y1 = y + FIGURE_HEIGHT + dsize

            dcolor = dcolors[img_num, k]
            r1, g1, b1 = r + dcolor, g + dcolor, b + dcolor

            if fig == 0:
                draw.rectangle(xy=(x, y, x1, y1), fill=(r1, g1, b1))
            elif fig == 1:
                draw.ellipse(xy=[x, y, x1, y1], fill=(r1, g1, b1))
            elif fig == 2:
                draw.line(xy=[(x, y), (x1, y1)], fill=(r1, g1, b1), width=7)
                draw.line(xy=[(x, y1), (x1, y)], fill=(r1, g1, b1), width=7)
            else:
                draw.polygon(xy=[(x, y1), (x + (FIGURE_WIDTH + dsizes[img_num, k]) // 2, y), (x1, y1)],
                             fill=(r1, g1, b1))
        # Добавление шума
        if is_noised:
            img = np.array(image, dtype=np.int16)
            noise = normal(NOISE_MEAN, noise_sigma, size=(img.shape[0] // 2, img.shape[1] // 2, 3)) + 128
            noise = np.array(Image.fromarray(noise.astype(np.uint8)).resize(img.shape[:2]))
            # noised = np.clip(img + noise - 128, 0, 255)
            noised = img + noise - 128
            noised[noised > 255] = 255
            noised[noised < 0] = 0
            image1 = Image.fromarray(noised.astype(np.uint8)).save(save_folder_path / Path(str(class_num) + '_' + str(img_num) + '.jpg'))
        else:
            image.save(save_folder_path / Path(f'{class_num}_{img_num}.jpg'))
        print(f"image {class_num}_{img_num} created!")


def do_job(tasks_to_do):
    while 1:
        try:
            # Вытаскиваем из очереди входные данные и передаем их функции generate_pics
            task = tasks_to_do.get_nowait()
            generate_pics(*task)
        # Если очередь пуста, выходим из цикла
        except queue.Empty:
            break
        else:
            print(f'done with class {task[1]}', flush=True)
    return True


# Функция генерирует num_of_imgs картинок для каждого класса параллельно
def generate_pics_parallel(save_folder_path: Path, num_of_classes: int, num_of_imgs: int, is_noised: bool,
                           noise_sigma: float):
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)

    # Входные параметры функции генерации картинок для каждого класса
    input_params = [(save_folder_path, i, num_of_imgs, is_noised, noise_sigma) for i in range(num_of_classes)]

    number_of_processes = cpu_count() - 1
    tasks_to_do = Queue()
    processes = []

    # Кладем входные параметры для каждого класса в очередь для параллельного выполнения
    for i in input_params:
        tasks_to_do.put(i)

    # Запускаем number_of_processes потоков
    for _ in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_do,))
        processes.append(p)
        p.start()

    # Ждем окончания выполнения всех запущенных потоков
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', help='Name of save folder')
    parser.add_argument('num_of_classes', type=int, help='Number of classes')
    parser.add_argument('num_of_imgs', type=int, help='Number of images in one class to generate')
    parser.add_argument('is_noised', type=int)
    parser.add_argument('noise_sigma', type=float)
    args = parser.parse_args()

    generate_pics_parallel(Path(FOLDER) / Path(args.save_folder), args.num_of_classes, args.num_of_imgs, args.is_noised,
                           args.noise_sigma)
