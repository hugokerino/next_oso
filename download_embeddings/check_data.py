import os
import json
import numpy as np
import re

PATH_npy = "/home/hkerino/Documents/PASTIS/alpha-earth_data"

# npy data
ids = []
id_inf = []
id_nan = []
id_zeros = []
for count, data in enumerate(sorted(os.listdir(PATH_npy))):
    d = np.load(os.path.join(PATH_npy, data))
    m = re.search(r"alpha-earth_(\d+)", data)
    id = int(m.group(1))
    if id in ids:
        print(f"Issue with {id}, doublon")
    else:
        ids.append(id)
    if np.isnan(d).any():
        id_nan.append(id)
    if np.isinf(d).any():
        id_inf.append(id)
    if d[d == 0].size / (128 * 128 * 128) >= 0.02:  # more than 2% cover by zeros
        id_zeros.append(f"{id:04d}")

print(f"Total len = {len(ids)}")
print(f"Number of patch with NaN issue : {len(id_nan)} : {id_nan}")
print(f"Number of patch with inf issue : {len(id_inf)} : {id_inf}")
print(f"Number of patch with zeros issue : {len(id_zeros)} : {id_zeros}")

print(f"Total patches : {count} ")
print(f"{((len(id_zeros) + len(id_inf) + len(id_nan)) / count) * 100:2f}% corrupted")

# Sauvegarder
with open("/home/hkerino/Documents/PASTIS/alpha-earth_data/id_issue.json", "w") as f:
    json.dump(id_zeros, f)
