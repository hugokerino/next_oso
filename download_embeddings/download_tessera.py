from geotessera import GeoTessera
import geopandas as gpd
import numpy as np
from rasterio.io import MemoryFile
from shapely.geometry import box, Point
from pyproj import Transformer
from rasterio.transform import array_bounds
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform

import os
from rasterio.transform import from_bounds
from time import time
from argparse import ArgumentParser


def create_geom(embedding, crs, transform):
    tile_geom = box(
        *array_bounds(
            embedding.shape[0],
            embedding.shape[1],
            transform,
        )
    )

    to_tile_crs = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    return tile_geom, to_tile_crs


def reproject_embedding(
    embedding,
    transform,
    src_crs,
    dst_crs,
):
    H, W, C = embedding.shape

    left, bottom, right, top = array_bounds(
        H,
        W,
        transform,
    )

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        W,
        H,
        left,
        bottom,
        right,
        top,
    )

    dst = np.empty(
        (C, dst_height, dst_width),
        dtype=np.float32,
    )

    reproject(
        source=embedding.transpose(2, 0, 1),
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    return (
        dst.transpose(1, 2, 0),
        dst_transform,
        dst_crs,
    )


def merge_embeddings(
    embedding1,
    transform1,
    crs1,
    embedding2,
    transform2,
    crs2,
):
    """
    Fusionne deux embeddings GeoTessera.
    Retourne :
        mosaic, crs, transform
    """

    if crs1 != crs2:
        embedding2, transform2, crs2 = reproject_embedding(
            embedding=embedding2, transform=transform2, src_crs=crs2, dst_crs=crs1
        )
        assert crs1 == crs2, f"CRS mismatch: {crs1} != {crs2}"

    datasets = []
    memfiles = []

    for embedding, transform in [
        (embedding1, transform1),
        (embedding2, transform2),
    ]:
        H, W, C = embedding.shape

        memfile = MemoryFile()

        ds = memfile.open(
            driver="GTiff",
            height=H,
            width=W,
            count=C,
            dtype=embedding.dtype,
            crs=crs1,
            transform=transform,
        )

        ds.write(embedding.transpose(2, 0, 1))

        datasets.append(ds)
        memfiles.append(memfile)

    mosaic, out_transform = merge(datasets)

    for ds in datasets:
        ds.close()

    for memfile in memfiles:
        memfile.close()

    # retour format H,W,C
    mosaic = mosaic.transpose(1, 2, 0)

    return mosaic, crs1, out_transform


def main(args):
    start = time()
    gt = GeoTessera()
    from_2154_to_4326 = Transformer.from_crs("EPSG:2154", "EPSG:4326")

    save_path = args.save_path
    aois = gpd.read_file(args.aois_path)

    for count, aoi in aois.sort_values("new_id").iterrows():
        id = f"{aoi['new_id']:04d}"

        if os.path.exists(os.path.join(save_path, f"tessera_{id}.npy")):
            continue

        print(f"\n{id} is being processed")
        aoi = aoi["geometry"]

        pl = from_2154_to_4326.transform_bounds(
            aoi.bounds[0],
            aoi.bounds[1],
            aoi.bounds[2],
            aoi.bounds[3],
        )

        mean_lat = (pl[0] + pl[2]) / 2
        mean_lon = (pl[1] + pl[3]) / 2

        corners = {
            "high_left": (pl[2], pl[1]),
            "low_left": (pl[0], pl[1]),
            "low_right": (pl[0], pl[3]),
            "high_right": (pl[2], pl[3]),
        }

        embedding, crs, transform = gt.fetch_embedding(
            lon=mean_lon,
            lat=mean_lat,
            year=2020,
        )

        tile_geom, transform_to_crs_tile = create_geom(embedding, crs, transform)

        for name, (lat, lon) in corners.items():
            x, y = transform_to_crs_tile.transform(lon, lat)

            if tile_geom.contains(Point(x, y)):
                # print(f"Corner {name} already in the tile")
                continue

            # print(f"Need another tile for Corner {name}")
            embedding_corner, crs_corner, transform_corner = gt.fetch_embedding(
                lon=lon, lat=lat, year=2020
            )

            # Function to merge to tile from means and this corner
            embedding, crs, transform = merge_embeddings(
                embedding,
                transform,
                crs,
                embedding_corner,
                transform_corner,
                crs_corner,
            )  # TO DO, deal with different crs propely

            # Modify tile_geom variable from this merging
            tile_geom, transform_to_crs_tile = create_geom(
                embedding,
                crs,
                transform,
            )

        # print(f"Embedding shape = {embedding.shape}")

        # Reproject in aoi
        target_transform = from_bounds(
            *aoi.bounds,
            width=128,
            height=128,
        )

        dst = np.empty(
            (128, 128, 128),
            dtype=np.float32,
        )

        reproject(
            source=embedding.transpose(2, 0, 1),
            destination=dst,
            src_transform=transform,
            src_crs=crs,
            dst_transform=target_transform,
            dst_crs="EPSG:2154",
            resampling=Resampling.nearest,
        )

        np.save(os.path.join(save_path, f"tessera_{id}.npy"), dst)

    end = time()
    print(f"\nElpased time {(end - start) / 3600:.2f} hours")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--aois_path",
        type=str,
        default="/home/hkerino/Documents/nextoso-dataset/data/aoi/nextoso_extent.gpkg",
    )
    parser.add_argument(
        "--save_path", type=str, default="/home/hkerino/Documents/tessera/tessera_npy"
    )
    args = parser.parse_args()
    main(args)
