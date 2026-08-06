import numpy as np
import utils_data

PATH_TO_DATA_label = "/home/hkerino/Documents/nextoso-dataset/data/10m/labels/"

density_all_class = utils_data.density_repartition_all_classes(
    PATH_TO_DATA_label, plot=True
)

thresholds, filtered_probas = utils_data.plot_filtered_probas(density_all_class, 20)

np.save("data_functions/thresholds.npy", thresholds)
