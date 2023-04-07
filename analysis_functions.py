import time
from typing import Tuple
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from create_dataset import scale_range
from PIL import Image
from pathlib import Path
from model import MySimpleClassifier
from tensorflow.keras import backend as K, models
from scipy.ndimage import gaussian_filter1d

from test_crops import TARGET_SIZE

FOLDER_PATH = "D:\\Learning\\MSU\\4th\\diploma_files"

NUM_OF_IMAGES: int = 5
NUM_OF_ROWS: int = 2

# NUM_OF_COLUMNS * NUM_OF_ROWS == length of avg_pool layer
NUM_OF_COLUMNS: int = 8

MIN_SCALE: float = 0.5
MAX_SCALE: float = 10.0
STEP: float = 0.1

X_POINTS = np.arange(MIN_SCALE, MAX_SCALE, STEP)

BATCH_SIZE: int = 61
NUM_OF_CLASSES: int = 20
TRUE_SCALE: float = 5.0

IMAGE_SIZE = 5000

PREDICTION_METHODS = ["all", "one", "global"]

EPS = 1e-10


def save(folder: str, scale_patches: list) -> None:
    path = Path(folder)
    if not path.exists():
        path.mkdir()
    ind = 0
    for patch in scale_patches:
        ind += 1
        patch.save(path / Path(f"{ind}.jpg"))


# Function returns {batch_size} number of scale pyramids of {class_num} class
# from test images (full_test) with # {noise_level} noise.
def get_scale_pyramid_batch(
    noise_level: int, class_num: int, batch_size: int, target_size: int = 224
) -> list:
    scales = [scale for scale in scale_range(MIN_SCALE, MAX_SCALE, STEP)]
    image_sizes = [
        int(IMAGE_SIZE / scale) + int(IMAGE_SIZE / scale) % 2
        for scale in scale_range(MIN_SCALE, MAX_SCALE, STEP)
    ]
    patch_sizes = [
        int(scale * TARGET_SIZE) + int(scale * TARGET_SIZE) % 2
        for scale in scale_range(MIN_SCALE, MAX_SCALE, STEP)
    ]
    image_nums = np.random.randint(NUM_OF_IMAGES, size=(batch_size,))

    diff = TARGET_SIZE // 2

    image_scales_path = Path(FOLDER_PATH) / Path(
        f"full_test_noise{noise_level}_scaled"
    )

    np_obj = np.load(image_scales_path / Path(f"class-{class_num}.npz"))
    scaled_images = []
    for i in range(len(image_sizes)):
        scaled_images.append(np_obj[f"{image_sizes[i]}"])

    batch = []
    for i in range(batch_size):
        scale_patches = []
        x, y = np.random.randint(
            patch_sizes[-1] // 2,
            IMAGE_SIZE - (patch_sizes[-1] // 2) - 1,
            size=2,
        )
        for j in range(len(image_sizes)):
            image_scaled = scaled_images[j][image_nums[i]]
            new_x, new_y = int(x / scales[j]), int(y / scales[j])
            patch = image_scaled[
                (new_x - diff): (new_x + diff),
                (new_y - diff): (new_y + diff),
            ]
            scale_patches.append(patch)
        batch.append(np.stack(scale_patches, axis=0))
    return batch


# Function returns feature map on avg_pool layer
def get_feature_maps_from_avg_pool_layer(
    model: MySimpleClassifier, scale_pyramid: np.ndarray
):
    layers = [layer for layer in model.model.layers]
    extract_out = K.function([model.model.input], [layers[-2].output])
    return extract_out([scale_pyramid])[0]


# Function returns gradient from the loss function for the {class_num}
# class with respect to avg_pool layer
def gradient_wrt_avg_pool_layer(
    model: MySimpleClassifier, scale_pyramid: np.ndarray, class_num: int
):
    layers = [layer for layer in model.model.layers]
    # строим нашу градиентную модель
    # build our gradient model
    grad_model = models.Model(
        inputs=[model.model.inputs],
        outputs=[layers[-2].output, model.model.output],
    )
    with tf.GradientTape() as tape:
        input_data = tf.cast(scale_pyramid, tf.float32)
        (conv_outputs, predictions) = grad_model(input_data)
        # get loss related to the corresponding class num
        loss = predictions[:, class_num]
    # use automatic differentiations for computing gradient
    return tape.gradient(loss, conv_outputs)


def derivative(function: np.ndarray):
    return np.gradient(function, STEP)


def smooth_function(y_points: np.ndarray, gauss_sigma: float):
    return gaussian_filter1d(y_points, gauss_sigma)  # smooth the curve


def smooth_function_derivative(y_points: np.ndarray, gauss_sigma: float):
    return derivative(smooth_function(y_points, gauss_sigma))


def identity_function(y_points: np.ndarray, gauss_sigma: float):
    return y_points


def show_graphs(
    feature_map: np.ndarray,
    weights: np.ndarray,
    best_weights_set: set,
    show_all: bool,
    gauss_sigma: float,
    function,
):
    fig = plt.figure(figsize=(25, 10))
    gs = fig.add_gridspec(NUM_OF_ROWS, NUM_OF_COLUMNS, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for i in range(NUM_OF_ROWS):
        for j in range(NUM_OF_COLUMNS):
            curr_weight = weights[i * NUM_OF_COLUMNS + j]
            if show_all:
                y_points = function(
                    feature_map[:, i * NUM_OF_COLUMNS + j], gauss_sigma
                )
            else:
                if curr_weight in best_weights_set:
                    y_points = function(
                        feature_map[:, i * NUM_OF_COLUMNS + j], gauss_sigma
                    )
                else:
                    y_points = np.zeros(shape=X_POINTS.shape)
            # plt.subplot(
            #     NUM_OF_ROWS, NUM_OF_COLUMNS, i * NUM_OF_COLUMNS + j + 1
            # )
            axs[i, j].plot(X_POINTS, y_points)  # type: ignore
    for ax in axs.flat:
        ax.set(xlabel='scale', ylabel='mean')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def show_predictions(
    batch_predictions: list, class_num: int, save_folder: str
) -> None:
    for i in range(len(PREDICTION_METHODS)):
        plt.plot(batch_predictions[i], label=PREDICTION_METHODS[i])
    plt.legend(loc="best")
    path = Path("graphics") / Path(save_folder)
    if not path.exists():
        path.mkdir()
    plt.savefig(path / Path(f"{class_num}.png"))
    plt.show()


def find_function_zeros(y_points: np.ndarray):
    zero_indexes = []
    for i in range(X_POINTS.shape[0] - 1):
        if y_points[i] == 0.0:
            zero_indexes.append(i)
        elif y_points[i] * y_points[i + 1] < 0.0:
            if abs(y_points[i]) < abs(y_points[i + 1]):
                zero_indexes.append(i)
            else:
                zero_indexes.append(i + 1)
    return zero_indexes


def find_function_extremes(y_points: np.ndarray):
    function_derivative = derivative(y_points)
    return find_function_zeros(function_derivative)


def is_close(a: float, b: float):
    return abs(a - b) <= EPS


def predict_scale(
    feature_map: np.ndarray,
    weights: np.ndarray,
    best_weights_set: set,
    gauss_sigma: float,
    prediction_method: str,
):
    scales = []
    for i in range(weights.shape[0]):
        function = np.asarray(smooth_function(feature_map[:, i], gauss_sigma))
        function_derivative = smooth_function_derivative(
            feature_map[:, i], gauss_sigma
        )

        curr_weight = weights[i]
        if curr_weight in best_weights_set:
            extreme_indexes = find_function_extremes(function)
            deriv_extreme_indexes = find_function_extremes(function_derivative)

            if prediction_method == "all":
                scales += [X_POINTS[i] for i in extreme_indexes]
                scales += [X_POINTS[i] for i in deriv_extreme_indexes]

            elif prediction_method == "one":
                if len(extreme_indexes) == 1:
                    scales.append(X_POINTS[extreme_indexes[0]])
                else:
                    if len(deriv_extreme_indexes) == 1:
                        scales.append(X_POINTS[deriv_extreme_indexes[0]])
            else:
                absolute_max = np.max(np.abs(function))
                for j in range(len(extreme_indexes)):
                    if is_close(
                        abs(function[extreme_indexes[j]]), absolute_max
                    ):
                        scales.append(X_POINTS[extreme_indexes[j]])

                derivative_absolute_max = np.max(np.abs(function_derivative))
                for j in range(len(deriv_extreme_indexes)):
                    if is_close(
                        abs(function_derivative[deriv_extreme_indexes[j]]),
                        derivative_absolute_max,
                    ):
                        scales.append(X_POINTS[deriv_extreme_indexes[j]])

    if len(scales) > 0:
        return np.median(scales)
    return None


def print_weights(weights: np.ndarray):
    for i in range(NUM_OF_ROWS):
        for j in range(NUM_OF_COLUMNS):
            print(f"{weights[i * NUM_OF_COLUMNS + j]:.5f}", end=" ")
        print()


def analyze_scale_pyramid(
    model: MySimpleClassifier,
    scale_pyramid: np.ndarray,
    class_num: int,
    best_weight_percentage: int,
    gauss_sigma: float,
    prediction_method: str,
):
    feature_map = get_feature_maps_from_avg_pool_layer(model, scale_pyramid)
    # compute gradient
    gradient = gradient_wrt_avg_pool_layer(model, scale_pyramid, class_num)
    # average gradient over all scales, take absolute value and get weight for
    # every channel
    weights = np.abs(np.mean(gradient, axis=0))  # weights has size (16, )
    # count number of channels we will take for scale calculating
    cnt = round(len(weights) * best_weight_percentage / 100)
    best_weights_set = set(sorted(weights)[::-1][:cnt])
    # show_graphs(feature_map, weights, best_weights_set, True, gauss_sigma,
    #             identity_function)
    # show_graphs(feature_map, weights, best_weights_set, True, gauss_sigma,
    #             smooth_function)
    # show_graphs(feature_map, weights, best_weights_set, False, gauss_sigma,
    #             smooth_function)
    # show_graphs(feature_map, weights, best_weights_set, False, gauss_sigma,
    #             smooth_function_derivative)
    # print_weights(weights)
    prediction = predict_scale(
        feature_map, weights, best_weights_set, gauss_sigma, prediction_method
    )
    # print(prediction)
    return prediction


def predict(
    model: MySimpleClassifier,
    class_data: list,
    class_num: int,
    best_percentage: int,
    gauss_sigma: float,
    prediction_method: str,
):
    predictions = []
    for scale_pyramid in class_data:
        prediction = analyze_scale_pyramid(
            model,
            scale_pyramid,
            class_num,
            best_percentage,
            gauss_sigma,
            prediction_method,
        )
        if prediction is not None:
            predictions.append(prediction)
    class_prediction = np.median(predictions)
    # return [class_prediction, predictions]
    return class_prediction


def write_to_csv(class_predictions: list, save_name: str) -> None:
    with open(save_name, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=" ", quotechar="|")
        csvwriter.writerow(PREDICTION_METHODS)
        for each_class_predictions in class_predictions:
            csvwriter.writerow(each_class_predictions)


def analyze_predictions(
    model: MySimpleClassifier,
    best_percentage: int,
    gauss_sigma: float,
    save_folder: str,
    csv_save_name: str,
) -> None:
    class_predictions = []
    for class_num in range(NUM_OF_CLASSES):
        scale_pyramid_batch = get_scale_pyramid_batch(0, class_num, BATCH_SIZE)
        batch_predictions = []
        each_class_predictions = []
        for prediction_method in PREDICTION_METHODS:
            class_prediction, predictions = predict(
                model,
                scale_pyramid_batch,
                class_num,
                best_percentage,
                gauss_sigma,
                prediction_method,
            )
            batch_predictions.append(sorted(predictions))
            each_class_predictions.append(class_prediction)
        class_predictions.append(each_class_predictions)
        show_predictions(batch_predictions, class_num, save_folder)
    write_to_csv(class_predictions, csv_save_name)


def predict_for_all_classes(
    model: MySimpleClassifier, best_percentage: int, gauss_sigma: float
) -> list:
    predictions = []
    for class_num in range(NUM_OF_CLASSES):
        scale_pyramid_batch = get_scale_pyramid_batch(0, class_num, BATCH_SIZE)
        prediction = predict(
                    model,
                    scale_pyramid_batch,
                    class_num,
                    best_percentage,
                    gauss_sigma,
                    PREDICTION_METHODS[0],
                )
        predictions.append(prediction)
    return predictions


def compute_metrics(class_predictions: list) -> Tuple[int, int, int]:
    mse, acc1, acc2 = 0, 0, 0
    for i in range(NUM_OF_CLASSES):
        mse += (class_predictions[i] - TRUE_SCALE) ** 2
        if 4.5 < class_predictions[i] < 5.5:
            acc1 += 1
        if 4 < class_predictions[i] < 6:
            acc2 += 1
    return mse, acc1, acc2
