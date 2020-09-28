import tensorflow as tf


def normalize_image(image: tf.Tensor, mean, stddev):
    per_channel = []
    for i, (m, s) in enumerate(zip(mean, stddev)):
        per_channel.append((image[:, :, :, i] - m) / s)
    return tf.stack(per_channel, axis=-1)

