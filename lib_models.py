import tensorflow as tf
import gin
from typing import Callable

IMAGE_SIZE = (224, 224)


@gin.configurable
def get_weight_regularizer(strength: float = 0.01):
    return tf.keras.regularizers.l2(strength)


def get_make_function(model_name: str) -> Callable[[], tf.keras.Model]:
    return {"vgg16": make_vgg16_model, "resnet50": make_resnet50_model}[model_name]


def make_vgg16_model():
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", pooling="avg"
    )
    inputs = tf.keras.layers.Input((*IMAGE_SIZE, 3), name="picture")
    X = inputs
    X = base(X)

    features = X

    X = tf.keras.layers.Dense(2, kernel_regularizer=get_weight_regularizer())(features)
    return tf.keras.Model(inputs, outputs=[features, X])


def make_resnet50_model():
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", pooling="avg"
    )
    inputs = tf.keras.layers.Input((*IMAGE_SIZE, 3), name="picture")
    X = inputs
    X = base(X)

    features = X

    X = tf.keras.layers.Dense(2, kernel_regularizer=get_weight_regularizer())(features)
    return tf.keras.Model(inputs, outputs=[features, X])
