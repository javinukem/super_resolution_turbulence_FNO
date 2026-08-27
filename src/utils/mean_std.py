import numpy


def compute_mean_std_dataset(dataset, indices):
    """
    Takes a dataset and computes the mean/std for the chosen samples.
    dataset: HDF5 dataset (e.g. hr_states)
    indices: list of indices to include
    """
    x = dataset[indices[0]]
    C = x.shape[0]

    mean = numpy.zeros(C, dtype=numpy.float64)
    var = numpy.zeros(C, dtype=numpy.float64)
    n_total = 0

    for idx in indices:
        x = dataset[idx]
        x = x.reshape(C, -1)
        n_pixels = x.shape[1]

        mean += x.mean(axis=1) * n_pixels
        var += x.var(axis=1) * n_pixels
        n_total += n_pixels

    mean /= n_total
    var /= n_total
    std = numpy.sqrt(var)

    return mean, std
