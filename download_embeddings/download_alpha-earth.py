import os
import numpy as np
import geopandas as gpd
import ee
import geemap
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.warp import Resampling
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

            geemap.ee_export_image(
                embeddings,
                filename=os.path.join(args.save_tif_path, f"alpha_earth_{id:04d}.tif"),
                region=aoi_ee,
                scale=10,
                crs="EPSG:2154",
            )

            # Reproject to 128x128
            src = rasterio.open(
                os.path.join(args.save_tif_path, f"alpha_earth_{id:04d}.tif")
            )
            dst = np.empty(
                (src.count, 128, 128),
                dtype=np.float32,
            )

            target_transform = from_bounds(
                *polygon.bounds,
                width=128,
                height=128,
            )
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )

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
        "--save_tif_path",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/alpha-earth_data_tif",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
    )
    parser.add_argument(
        "--earth_engine_project", type=str, required=True, help="Ex:ee-hugobzhk"
    )
    args = parser.parse_args()
    main(args)
