# Geo Map Harmonizer (WIP)

This package includes an algorithm specialized in harmonizing geotiff maps.
Its main function is to generate a harmonized legend based on the spatial 
distribution of classes in two maps, aiming for maximum accuracy in comparison.

Additionally, the package provides data analysis functions for detailed
reports. Specific strategies for handling Big Data are implemented, such as
dividing maps into smaller blocks and consolidating results.

## Installation

1 - Install [GDAL](https://github.com/OSGeo/gdal) on your system.
2 - Install [Python GDAL Library](https://pypi.org/project/GDAL/)
3 - Install Geo Map Harmonizer with `pip install geo-map-harmonizer`.


## Utilization
The application uses a class to store the paths to the maps in .tif format,
and uses the map_legend_harmonizer method
to generate the legend harmonization for the two maps.

```python
from geomapharmonizer import GeoMapHamonizer

geomap = GeoMapHamonizer(path_map1="path/to/map1.tif", path_map2="path/to/map2.tif")

df_legends = geomap.map_legend_harmonizer()
```
