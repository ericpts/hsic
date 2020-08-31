import tensorflow as tf
import tensorflow_probability as tfp
import gin
from typing import List


def center_matrix(M):
    M = M - tf.reduce_mean(M, axis=0)
    return M


def dhsic(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    N = xs[0].shape[0]
    d = len(xs)

    centered_gram_matrices = [k.matrix(f, f) for f in xs]

    t1 = 1.0
    t2 = 1.0
    t3 = 2.0 / N

    for i in range(d):
        t1 = t1 * centered_gram_matrices[i]
        t2 = 1 / (N ** 2) * t2 * tf.reduce_sum(centered_gram_matrices[i])
        t3 = 1 / N * t3 * tf.reduce_sum(centered_gram_matrices[i], axis=1)

    # The original HSIC paper divides by 1 / (N - 1)^2, whereas the d-HSIC
    # formulation divides by 1 / (N^2).
    # In order to be consistent with previous work, we modify this d-HSIC to
    # also normalize by 1 / (N-1)^2
    biased_original = 1 / (N ** 2) * tf.reduce_sum(t1) + t2 - tf.reduce_sum(t3)
    biased_like_hsic = biased_original * (N / (N - 1)) ** 2
    return biased_like_hsic


def hsic(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    N = xs[0].shape[0]
    n_variables = len(xs)

    centered_gram_matrices = [center_matrix(k.matrix(f, f)) for f in xs]

    sums = []
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            sums.append(
                tf.linalg.trace(centered_gram_matrices[i] @ centered_gram_matrices[j])
            )
    return tf.reduce_mean(sums) / ((N - 1) ** 2)


def dcka(xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel):
    up = dhsic(xs, k)
    down = 1.0

    n_variables = len(xs)
    for i in range(n_variables):
        down *= tf.math.pow(1 / dhsic([xs[i], xs[i]], k), 1 / n_variables)

    return up * down


def cka(
    xs: List[tf.Tensor], k: tfp.math.psd_kernels.PositiveSemidefiniteKernel
) -> tf.Tensor:
    return hsic(xs, k) / tf.math.sqrt(hsic([xs[0], xs[0]], k) * hsic([xs[1], xs[1]], k))


def label_kernel(l_0, l_1):
    l_0 = tf.one_hot(l_0, 1)
    l_1 = tf.one_hot(l_1, 1)
    return tfp.math.psd_kernels.ExponentiatedQuadratic().apply(l_0, l_1)


def conditional_hsic(
    xs: List[tf.Tensor],
    labels: tf.Tensor,
    k: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
) -> tf.Tensor:
    n = xs[0].shape[0]
    n_variables = len(xs)
    tf.eye(n) - 1 / n

    def center(M):
        return M - tf.reduce_mean(M, axis=1)

    K = [k.matrix(f, f) for f in xs]
    one_hot_labels = tf.one_hot(labels, 1)
    K_z = k.matrix(one_hot_labels, one_hot_labels)

    K_xx = [center(x * K_z) for x in K]
    K = [center(x) for x in K]
    K_z = center(K_z)

    eps = 0.001
    R = eps * tf.linalg.inv(K_z + eps * tf.eye(n))

    pair_losses = []
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            X = R @ K_xx[i] @ R
            Y = R @ K[j] @ R
            pair_losses.append(tf.linalg.trace(X @ Y))
    ret = tf.reduce_mean(pair_losses) / n

    return ret


def conditional_cka(
    xs: List[tf.Tensor],
    labels: tf.Tensor,
    k: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
) -> tf.Tensor:
    return conditional_hsic(xs, labels, k) / tf.math.sqrt(
        conditional_hsic([xs[0], xs[0]], labels, k)
        * conditional_hsic([xs[1], xs[1]], labels, k)
    )


def manual_chsic(c, s, y, k):
    unique_labels, _ = tf.unique(y)
    unique_labels = sorted(list(unique_labels.numpy()))

    n = c.shape[0]
    C = {z: [i for i in range(n) if y[i] == z] for z in unique_labels}
    s1 = 0.0
    for i in range(n):
        for j in range(n):
            s1 += k.apply(c[i], c[j]) * k.apply(s[i], s[j]) * label_kernel(y[i], y[j])
    s1 /= n

    s2 = 0.0
    for z in unique_labels:
        for zz in unique_labels:
            a1 = 0.0
            a2 = 0.0
            for i in C[z]:
                for ii in C[zz]:
                    a1 += k.apply(c[i], c[ii])
                    a2 += k.apply(s[i], s[ii])
            a = label_kernel(z, zz) * a1 * a2
            a /= len(C[z]) * len(C[zz])
            s2 += a
    s2 /= n

    s3 = 0.0
    for z in unique_labels:
        for i in range(n):
            t_0 = 0.0
            for ii in C[z]:
                t_0 += k.apply(c[i], c[ii])
            t_1 = 0.0
            for jj in C[z]:
                t_1 += k.apply(s[i], s[jj])
            t = t_0 * t_1
            t *= label_kernel(y[i], z)
            t /= len(C[z])
            s3 += t
    s3 /= n

    return s1 + s2 - 2 * s3


@gin.configurable
def diversity_loss(
    features: List[tf.Tensor], y: tf.Tensor, independence_measure: str, kernel: str,
) -> tf.Tensor:
    if kernel == "linear":
        k = tfp.math.psd_kernels.Linear()
    elif kernel == "rbf":
        k = tfp.math.psd_kernels.ExponentiatedQuadratic()
    else:
        raise ValueError(f"Unknown kernel: {kernel}; should be one of linear, rbf")

    if independence_measure == "cka":
        return dcka(features, k)
    elif independence_measure == "conditional_cka":
        return conditional_cka(features, y, k)
    elif independence_measure == "hsic":
        return dhsic(features, k)
    elif independence_measure == "conditional_hsic":
        return conditional_hsic(features, y, k)
    else:
        raise ValueError(
            f"Unknown independence_measure: {independence_measure}; expected one of cka or hsic"
        )
