import os
import numpy as np
import geopandas as gpd
import ee
from argparse import ArgumentParser


def main(args):
    # bounding box of next oso dataset
    aois = gpd.read_file(args.aois_path)
    year = args.year

    # Initialize earthengine session and retrieve alpha-earth_embeddings
    ee.Initialize(project=args.earth_engine_project)

    # Collect embedding year
    embedding_collection = ee.ImageCollection(
        "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    ).filterDate(f"{year}-01-01", f"{year}-12-31")

    # Split AOIs across SLURM array tasks
    n_arrays = args.n_arrays
    array_index = args.array_index
    if array_index < 1 or array_index > n_arrays:
        raise ValueError(
            f"array_index must be between 1 and n_arrays, got {array_index} / {n_arrays}"
        )

    num_polygons = len(aois)
    if num_polygons == 0:
        print("Aucun AOI trouvé dans le fichier fourni.")
        return

    blocks = np.array_split(np.arange(num_polygons), n_arrays)
    selected_indices = blocks[array_index - 1]
    aois = aois.iloc[selected_indices].reset_index(drop=True)

    print(
        f"Task {array_index}/{n_arrays}: traitement de {len(aois)} polygones sur {num_polygons}"
    )

    bbox_done = os.listdir(args.save_npy_path)

    # Download embeddings per polygon
    for index, row in aois.iterrows():
        id, polygon = row["ID_PATCH"], row["geometry"]

        if f"alpha_earth_{f'{id:04d}'}.npy" not in bbox_done:
            bbox_done.append(f"alpha_earth_{f'{id:04d}'}.npy")
            aoi_ee = ee.Geometry(
                polygon.__geo_interface__, proj="EPSG:2154", geodesic=False
            )
            embeddings = embedding_collection.filterBounds(aoi_ee).first().clip(aoi_ee)

            # Lire toutes les bandes de A00 à A63
            band_names = [f"A{i:02d}" for i in range(64)]

            # Extraire les données pour toutes les bandes et créer un tableau NumPy
            all_data = []
            for band_name in band_names:
                try:
                    band = embeddings.select(band_name)
                    data = band.reduceRegion(
                        reducer=ee.Reducer.toList(),
                        geometry=aoi_ee,
                        scale=10,
                    ).getInfo()
                    band_data = np.flipud(
                        np.array(data.get(band_name)).reshape((128, 128))
                    )
                    all_data.append(band_data)
                except Exception as exc:
                    print(f"Erreur pour la bande {band_name}: {exc}")

            # Tableau NumPy contenant toutes les bandes
            dst = np.stack(all_data, axis=0)

            assert dst.shape == (64, 128, 128)

            # Save the reprojected image
            np.save(
                os.path.join(args.save_npy_path, f"alpha_earth_{id:04d}.npy"),
                dst,
            )

            print(
                f"Saved alpha-earth embedding for polygon {id:04d}\n {len(bbox_done)}/{len(aois)}"
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--aois_path",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/aoi/nextoso_extent.gpkg",
    )
    parser.add_argument(
        "--save_npy_path",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/alpha-earth_data",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
    )
    parser.add_argument(
        "--earth_engine_project", type=str, required=True, help="Ex:ee-hugobzhk"
    )
    parser.add_argument(
        "--array_index",
        type=int,
        default=1,
        help="ID de la tâche SLURM array (1-based)",
    )
    parser.add_argument(
        "--n_arrays", type=int, default=1, help="Nombre total de tâches SLURM array"
    )
    args = parser.parse_args()
    main(args)
