import torch
import pandas as pd
from pathlib import Path
from itertools import islice
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


from data_functions import dataset
from supervised_models import lightning_module
from mains.main_utils import build_model, load_config
from data_functions.utils_data import to_RGB

"""
Sript to visualize attention from a trained LTAE model with S2 time series and S2 validity mask
"""

# Load configuration and build the model
config= load_config(train_cfg_path="./configs/supervised_models/config_local.yaml",
            model_cfg_path="./configs/supervised_models/LTAE.yaml")
fold = 0
config["fold_index"] = fold
config["selected_fold"] = config["folds"][fold]
config["encoder"]["return_att"] = True

#Load data loaders
_, _, dl_test = dataset.get_loaders(
    path_data=config["path_data"],
    use_mnt_data=config["use_mnt_data"],
    folds_train=config["selected_fold"][0],
    folds_val=config["selected_fold"][1],
    folds_test=config["selected_fold"][2],
    batch_size=config["batch_size"],
    num_worker=config["num_worker"],
    label_type=config["label_type"],
    label_type_test=config["label_type_test"],
    mask_needed=config["mask_need"],
    norm=True,
    shuffle=False,
    pass_id=True,
)
# Load the model checkpoint
encoder, decoder, batch_size = build_model(config)
log_dir = "/home/hkerino/Documents/next_oso/logs/to_share/LTAE_2026-10-09"
checkpoint_dir = Path(log_dir) / f"fold_{config['fold_index'] + 1}" / "Checkpoints"
checkpoint = next(checkpoint_dir.glob("*.ckpt"))
print(f"Loading checkpoint: {checkpoint}")


ltae_classifier = lightning_module.Classifier.load_from_checkpoint(
    checkpoint,
    weights_only=True,
    encoder=encoder,
    decoder=decoder,
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=0),
    batch_size=batch_size,
    num_class=config["num_class"],
    names_class=dl_test.dataset.name_labels,
)
ltae_classifier.eval()


BATCH_INDEX=8 # Chose batch index to compute/plot here
batch = next(islice(dl_test, BATCH_INDEX, BATCH_INDEX+1))
inputs, labels, ids = batch

#Forward pass
out, att = ltae_classifier.encoder(x=inputs[0], batch_positions=inputs[1], pad_mask=None)
print(f"ID : {ids[0]}")
print(f"Attention shape {att.shape}")
print(f"Out shape {out.shape}")

#Load corresponding data to plot
d = dataset.next_oso_Dataset(mask_needed=True, date_type="raw", norm=False)
input, label = d.__getitem__(ids[0])
s2, date, mask, geo = input

print(f"s2 shape : {s2.shape}")
print(f"date shape : {date.shape}")
print(f"mask shape : {mask.shape}")
print(f"geo shape : {geo.shape}")

c, t, h, w = s2.shape
print(f"Contains {t} dates")

head = 15 # Chose which head to plot
att_head = att[head].squeeze(0)  # (t,h,w)


t_start, t_end = 17,24 # Select time range

fig, ax = plt.subplots(3, t_end-t_start, figsize=(20, 12))


for i in range(t_start, t_end):
    ax[0, i - t_start].imshow(to_RGB(s2[:, i, :, :]))
    ax[0, i - t_start].set_title(
        f"{pd.Timestamp(date[i]).strftime('%d/%m/%Y')}", fontsize=15
    )
    ax[0, i-t_start].axis("off")

    ax[1, i - t_start].imshow(mask[i], vmin=0, vmax=1, cmap="grey_r")
    ax[1, i - t_start].add_patch(
        Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor="black", linewidth=1.5)
    )
    ax[1, i - t_start].axis("off")

    ax[2, i-t_start].imshow(att_head[i, :, :].detach().numpy())
    ax[2, i - t_start].axis("off")


ax[0, 0].text(
    -0.15,
    0.5,
    "Sentinel-2 RGB",
    transform=ax[0, 0].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)
ax[1, 0].text(
    -0.15,
    0.5,
    "Sentinel-2 mask",
    transform=ax[1, 0].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)
ax[2, 0].text(
    -0.15,
    0.5,
    f"Attention (head={head})",
    transform=ax[2, 0].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
