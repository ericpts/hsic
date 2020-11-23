import tensorflow as tf
import gin
from typing import Callable, Tuple
import lib_resnet


@gin.configurable
def get_image_size(image_size: Tuple[int, int] = (224, 224)):
    return image_size


@gin.configurable
def get_weight_regularizer(strength: float = 0.0001):
    return tf.keras.regularizers.l2(strength)


def make_vgg16_model(num_classes: int, image_size: Tuple[int, int, int]):
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", pooling="avg"
    )
    inputs = tf.keras.layers.Input(image_size, name="picture")
    X = inputs
    X = base(X)

    features = X

    X = tf.keras.layers.Dense(num_classes, kernel_regularizer=get_weight_regularizer())(
        features
    )
    return tf.keras.Model(inputs, outputs=[features, X])


def _make_resnet_common(base, num_classes: int, image_size: Tuple[int, int, int]):
    inputs = tf.keras.layers.Input(image_size, name="picture")
    X = inputs
    X = base(X)

    features = X

    X = tf.keras.layers.Dense(num_classes, kernel_regularizer=get_weight_regularizer())(
        features
    )
    return tf.keras.Model(inputs, outputs=[features, X])


def make_resnet18_model(num_classes: int, image_size: Tuple[int, int, int]):
    return _make_resnet_common(lib_resnet.resnet_18(), num_classes, image_size)


def make_resnet50_model(num_classes: int, image_size: Tuple[int, int, int]):
    return _make_resnet_common(lib_resnet.resnet_50(), num_classes, image_size)


def make_mlp_relu_model(
    num_classes: int, image_size: Tuple[int, int, int]
) -> tf.keras.Model:
    assert image_size[0:2] == (28, 28)
    inputs = tf.keras.layers.Input(image_size)
    X = inputs
    X = tf.keras.layers.Flatten()(X)
    reg = get_weight_regularizer()
    X = tf.keras.layers.Dense(20, kernel_regularizer=reg, activation="relu")(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(num_classes, kernel_regularizer=reg)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


def make_mlp_model(
    num_classes: int, image_size: Tuple[int, int, int]
) -> tf.keras.Model:
    assert image_size[0:2] == (28, 28)
    inputs = tf.keras.layers.Input(image_size)
    X = inputs
    X = tf.keras.layers.Flatten()(X)
    reg = get_weight_regularizer()
    X = tf.keras.layers.Dense(20, kernel_regularizer=reg)(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(num_classes, kernel_regularizer=reg)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


def get_make_function(
    model_name: str,
) -> Callable[[int, Tuple[int, int, int]], tf.keras.Model]:
    return {
        "vgg16": make_vgg16_model,
        "resnet18": make_resnet18_model,
        "resnet50": make_resnet50_model,
        "mlp": make_mlp_model,
        "mlp_relu": make_mlp_relu_model,
    }[model_name]
