from libc.math cimport log

cdef float EPSILON = 1e-15


def kl(float observed_mean,
       float mean) -> float:
    p = min(max(observed_mean, EPSILON), 1 - EPSILON)
    q = min(max(mean, EPSILON), 1 - EPSILON)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


def compute_bound(int num_samples,
                  float observed_mean,
                  float divergence_bound,
                  float value,
                  float target_bound,
                  float precision = 1e-3) -> float:

    while value - target_bound > precision:
        mean = (value + target_bound) / 2.0
        if num_samples * kl(observed_mean, mean) > divergence_bound:
            target_bound = mean
        else:
            value = mean

    return (value + target_bound) / 2.0
