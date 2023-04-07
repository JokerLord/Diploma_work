import numpy as np
from model import MySimpleClassifier
from pathlib import Path
from PIL import Image
from analysis_functions import (
    TRUE_SCALE,
    get_scale_pyramid_batch,
    predict,
    predict_for_all_classes,
    compute_metrics,
    analyze_scale_pyramid,
    analyze_predictions,
    PREDICTION_METHODS,
)

MIN_SCALE: float = 0.5
MAX_SCALE: float = 10.0
STEP: float = 0.1

X_POINTS = np.arange(MIN_SCALE, MAX_SCALE, STEP)
TRUE_SCALE: float = 5


def get_scale_index(scale: float) -> int:
    rounded_scale = round(scale, 1)
    return int((rounded_scale - MIN_SCALE) * 10)


def result_examples(num_of_samples: int, best_weight_percentage: int,
                    gauss_sigma: float) -> None:
    path = Path("tmp")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    classes = np.random.randint(16, 19, size=num_of_samples)
    for i in range(num_of_samples):
        scale_pyramid = get_scale_pyramid_batch(0, classes[i], 1)[0]
        scale_prediciton = analyze_scale_pyramid(
            loaded_light_model, scale_pyramid, classes[i],
            best_weight_percentage, gauss_sigma, PREDICTION_METHODS[0]
        )
        true_index = get_scale_index(5)
        prediction_index = get_scale_index(scale_prediciton)
        Image.fromarray(scale_pyramid[true_index]).save(path / f"{i}_{X_POINTS[true_index]:.1f}.jpg")
        Image.fromarray(scale_pyramid[prediction_index]).save(path / f"{i}_{X_POINTS[prediction_index]:.1f}.jpg")


if __name__ == "__main__":
    # upload light model, trained on scale 5.0
    loaded_light_model = MySimpleClassifier((224, 224, 3), 20, 4)
    loaded_light_model.load("saved_models/light_model_trained_on_scale_5.0")

    # with open("method_all_results_1", "w") as output_file:
    #     for best_percentage in range(40, 70, 10):
    #         for gauss_sigma in range(7, 17):
    #             predictions = predict_for_all_classes(
    #                 loaded_light_model, best_percentage, gauss_sigma
    #             )
    #             metrics = compute_metrics(predictions)
    #             output_file.write(
    #                 f"Accuracy with BP = "
    #                 f"{best_percentage} and sigma = "
    #                 f"{gauss_sigma}: {metrics[0]:.3f}, "
    #                 f"{metrics[1]}, {metrics[2]}\n"
    #             )
    #             for prediction in predictions:
    #                 output_file.write(f"{prediction:.3f} ")
    #             output_file.write("\n")
    #             print("CUNT")
    result_examples(3, 50, 8)
