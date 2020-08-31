import sklearn
import sklearn.metrics
import tensorflow as tf


def compute_ensemble_distribution(y_hat):
    return tf.math.reduce_mean(y_hat, axis=1)


def pmax(y_hat_ensemble, _y_hat):
    return tf.math.reduce_max(y_hat_ensemble, axis=1)


def entropy(y_hat_ensemble, _y_hat):
    return -tf.math.reduce_sum(
        y_hat_ensemble * tf.math.log(y_hat_ensemble + tf.keras.backend.epsilon()),
        axis=1,
    )


def max_diff(y_hat_ensemble, y_hat):
    n_models = y_hat.shape[1]
    per_model = []
    for im in range(n_models):
        per_model.append(
            tf.math.reduce_max(tf.math.abs(y_hat_ensemble - y_hat[:, im, :]), axis=1)
        )
    ret = tf.math.reduce_max(per_model, axis=0)
    return ret


def average_diff(y_hat_ensemble, y_hat):
    n_models = y_hat.shape[1]
    per_model = []
    for im in range(n_models):
        per_model.append(
            tf.math.reduce_mean(tf.math.abs(y_hat_ensemble - y_hat[:, im, :]), axis=1)
        )
    ret = tf.math.reduce_mean(per_model, axis=0)
    return ret


def compute_auroc(in_y_hat, oo_y_hat, f_ensemble_score, higher_score_is_ood=True):
    id_score = f_ensemble_score(compute_ensemble_distribution(in_y_hat), in_y_hat)
    oo_score = f_ensemble_score(compute_ensemble_distribution(oo_y_hat), oo_y_hat)

    if higher_score_is_ood:
        id_label, oo_label = 0, 1
    else:
        id_label, oo_label = 1, 0

    y_true = tf.convert_to_tensor(
        ([id_label] * id_score.shape[0]) + ([oo_label] * oo_score.shape[0])
    )
    y_score = tf.concat([id_score, oo_score], axis=0)
    return sklearn.metrics.roc_auc_score(y_true, y_score)


def compute_metrics(in_y_hat, oo_y_hat):
    return {
        "auroc_pmax": compute_auroc(
            in_y_hat, oo_y_hat, pmax, higher_score_is_ood=False
        ),
        "auroc_entropy": compute_auroc(in_y_hat, oo_y_hat, entropy),
        "auroc_max_diff": compute_auroc(in_y_hat, oo_y_hat, max_diff),
        "auroc_average_diff": compute_auroc(in_y_hat, oo_y_hat, average_diff),
    }
