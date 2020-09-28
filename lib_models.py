import tensorflow as tf
import gin
from typing import Callable, Tuple
import lib_resnet


@gin.configurable
def get_image_size(image_size: Tuple[int, int] = (224, 224)):
    return image_size


@gin.configurable
def get_weight_regularizer(strength: float = 0.01):
    return tf.keras.regularizers.l2(strength)


@gin.configurable
def get_n_classes(n_classes: int):
    return n_classes


def make_vgg16_model():
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", pooling="avg"
    )
    inputs = tf.keras.layers.Input((*get_image_size(), 3), name="picture")
    X = inputs
    X = base(X)

    features = X

    X = tf.keras.layers.Dense(
        get_n_classes(), kernel_regularizer=get_weight_regularizer()
    )(features)
    return tf.keras.Model(inputs, outputs=[features, X])


def _make_resnet_common(base):
    inputs = tf.keras.layers.Input((*get_image_size(), 3), name="picture")
    X = inputs
    X = base(X)

    features = X

    X = tf.keras.layers.Dense(
        get_n_classes(), kernel_regularizer=get_weight_regularizer()
    )(features)
    return tf.keras.Model(inputs, outputs=[features, X])


def make_resnet18_model():
    return _make_resnet_common(lib_resnet.resnet_18())


def make_resnet50_model():
    return _make_resnet_common(lib_resnet.resnet_50())


def make_mlp_model() -> tf.keras.Model:
    assert get_image_size() == (28, 28)
    inputs = tf.keras.layers.Input((28, 28, 3))
    X = inputs
    X = tf.keras.layers.Flatten()(X)
    reg = get_weight_regularizer()
    X = tf.keras.layers.Dense(20, kernel_regularizer=reg)(X)
    feature_extractor = X
    X = tf.keras.layers.Dense(get_n_classes(), kernel_regularizer=reg)(X)
    return tf.keras.Model(inputs, outputs=[feature_extractor, X])


def get_make_function(model_name: str) -> Callable[[], tf.keras.Model]:
    return {
        "vgg16": make_vgg16_model,
        "resnet18": make_resnet18_model,
        "resnet50": make_resnet50_model,
        "mlp": make_mlp_model,
    }[model_name]

