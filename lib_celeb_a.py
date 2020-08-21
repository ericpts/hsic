from typing import Dict
import plotly.express as px
import torchvision.transforms as transforms
import tensorflow as tf
import tensorflow_datasets as tfds
import lib_problem


def make_resnet50_model():
    base_resnet = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", pooling="avg"
    )
    inputs = tf.keras.layers.Input((224, 224, 3))
    X = inputs
    X = base_resnet(X)
    features = tf.keras.layers.Dense(
        100, kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(X)

    X = features
    X = tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(X)
    return tf.keras.Model(inputs, outputs=[features, X])


def filter_dataset(D: tf.data.Dataset, attribute_name: str, target_value):
    def f_filter(example: Dict) -> bool:
        return tf.equal(example["attributes"][attribute_name], target_value)

    return D.filter(f_filter)


def filter_for_biased_hair_and_gender(example: Dict) -> bool:
    blonde_female = tf.logical_and(
        example["attributes"]["Blond_Hair"],
        tf.logical_not(example["attributes"]["Male"]),
    )
    brunette_male = tf.logical_and(
        example["attributes"]["Black_Hair"], example["attributes"]["Male"],
    )
    return tf.logical_or(blonde_female, brunette_male)


def extract_image_and_label(example):
    X = example["image"]
    y = example["attributes"]["Male"]
    y_biased = example["attributes"]["Blond_Hair"]

    X = tf.image.convert_image_dtype(X, dtype=tf.float32, saturate=False)
    return (X, y)


def normalize_image(image: tf.Tensor, mean, stddev):
    per_channel = []
    for i, (m, s) in enumerate(zip(mean, stddev)):
        per_channel.append((image[:, :, :, i] - m) / s)
    return tf.stack(per_channel, axis=-1)


def transform_image(X, y):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    X = tf.keras.layers.experimental.preprocessing.CenterCrop(
        orig_min_dim, orig_min_dim
    )(X)
    X = tf.keras.layers.experimental.preprocessing.Resizing(*target_resolution)(X)
    X = normalize_image(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return X, y


def end_to_end_dataset_transform(D: tf.data.Dataset) -> tf.data.Dataset:
    D = D.filter(filter_for_biased_hair_and_gender)
    D = D.map(extract_image_and_label)
    D = D.batch(1).map(transform_image)
    D = D.unbatch()
    D = D.shuffle(1_000)
    return D


@gin.configurable
class CelebAProblem(lib_problem.Problem):
    def __init__(self):
        super().__init__("celeb_a_problem", make_base_model=make_resnet50_model)

    def generate_training_data(self) -> tf.data.Dataset:
        return end_to_end_dataset_transform(
            tfds.load("celeb_a", split="train")
        ).shuffle(1_000)

    def generate_testing_data(self) -> tf.data.Dataset:
        return end_to_end_dataset_transform(tfds.load("celeb_a", split="test"))

