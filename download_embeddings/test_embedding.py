import ee
import geopandas as gpd

ee.Initialize(project="ee-hugobzhk")

aois = gpd.read_file("/home/hkerino/Documents/PASTIS/metadata.geojson")
id, polygon = aois.iloc[0]["ID_PATCH"], aois.iloc[0]["geometry"]
aoi_ee = ee.Geometry(polygon.__geo_interface__, proj="EPSG:2154", geodesic=False)

pastis_dates = ["2018-09-01", "2019-11-01"]
embedding_pastis = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(pastis_dates[0], pastis_dates[1])
    .filterBounds(aoi_ee)
)
print(f"Number of images in pastis collection: {embedding_pastis.size().getInfo()}")

# Collect embedding year
embedding_2018 = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate("2018-01-01", "2018-12-31")
    .filterBounds(aoi_ee)
)
print(f"Number of images in 2018 collection: {embedding_2018.size().getInfo()}")

embedding_2019 = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate("2019-01-01", "2019-12-31")
    .filterBounds(aoi_ee)
)
print(f"Number of images in 2019 collection: {embedding_2019.size().getInfo()}")


img_pastis = embedding_pastis.first()
print("PASTIS")
print(img_pastis.propertyNames().getInfo())
print(img_pastis.getInfo()["properties"]["system:time_start"])
print(img_pastis.getInfo()["properties"]["system:time_end"])

img_2018 = embedding_2018.first()
print("2018")
print(img_2018.propertyNames().getInfo())
print(img_2018.getInfo()["properties"]["system:time_start"])
print(img_2018.getInfo()["properties"]["system:time_end"])

img_2019 = embedding_2019.first()
print("2019")
print(img_2019.propertyNames().getInfo())
print(img_2019.getInfo()["properties"]["system:time_start"])
print(img_2019.getInfo()["properties"]["system:time_end"])
