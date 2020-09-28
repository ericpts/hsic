from typing import Dict
import tensorflow as tf
import tensorflow_datasets as tfds
import lib_problem
import gin
from lib_image import normalize_image
from lib_models import make_resnet50_model, IMAGE_SIZE


def filter_dataset(D: tf.data.Dataset, attribute_name: str, target_value):
    def f_filter(example: Dict) -> bool:
        return tf.equal(example["attributes"][attribute_name], target_value)

    return D.filter(f_filter)


def filter_for_id(example: Dict) -> bool:
    male = example["attributes"]["Male"]
    blond = example["attributes"]["Blond_Hair"]
    female = tf.logical_not(male)
    brunette = tf.logical_not(blond)

    return tf.logical_or(tf.logical_and(blond, female), tf.logical_and(brunette, male))


def filter_for_ood(example: Dict) -> bool:
    male = example["attributes"]["Male"]
    blond = example["attributes"]["Blond_Hair"]
    female = tf.logical_not(male)
    brunette = tf.logical_not(blond)

    return tf.logical_or(tf.logical_and(blond, male), tf.logical_and(brunette, female))


@gin.configurable
def extract_image_and_label(example, label: str = "gender"):
    assert label in ["hair", "gender"]

    X = example["image"]

    hair = tf.cast(example["attributes"]["Blond_Hair"], tf.int32)
    gender = tf.cast(example["attributes"]["Male"], tf.int32)

    if label == "hair":
        y = hair
    else:
        y = gender

    X = tf.image.convert_image_dtype(X, dtype=tf.float32, saturate=False)
    return (X, y)


def transform_image(X, y):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    X = tf.keras.layers.experimental.preprocessing.CenterCrop(
        orig_min_dim, orig_min_dim
    )(X)
    X = tf.keras.layers.experimental.preprocessing.Resizing(*IMAGE_SIZE)(X)
    X = normalize_image(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return X, y


def dataset_extract_images(D: tf.data.Dataset) -> tf.data.Dataset:
    D = D.map(extract_image_and_label, num_parallel_calls=4)
    D = D.batch(1).map(transform_image, num_parallel_calls=4)
    D = D.unbatch()
    return D


@gin.configurable
class CelebAProblem(lib_problem.Problem):
    def __init__(self):
        super().__init__("celeb_a_problem", make_base_model=make_resnet50_model)
        print(f"CelebA: Using picture size of {IMAGE_SIZE}")

    def generate_training_data(self) -> tf.data.Dataset:
        return dataset_extract_images(
            tfds.load("celeb_a", split="train").filter(filter_for_id)
        ).cache()

    def generate_id_testing_data(self) -> tf.data.Dataset:
        return dataset_extract_images(
            tfds.load("celeb_a", split="test").filter(filter_for_id)
        ).cache()

    def generate_ood_testing_data(self) -> tf.data.Dataset:
        return dataset_extract_images(
            tfds.load("celeb_a", split="test").filter(filter_for_ood)
        ).cache()
