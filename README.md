# Geo Map Harmonizer

This package houses an algorithm specialized in harmonizing legends of GeoTIFF maps. Its primary function is the generation of a harmonized legend, based on the spatial distribution of classes in two maps, aiming to achieve maximum concordance between them. The algorithm can handle a variety of map types, especially Land Use and Land Cover (LULC) maps.

In addition to this core functionality, the package offers a set of data analysis functions for generating detailed reports. These analyses aim to provide insights into the characteristics of the processed maps, allowing a deeper understanding of the spatial dynamics of classes.

The package has been designed with specific strategies to handle Big Data. The implementation of techniques such as map division into smaller blocks and efficient result consolidation contributes to the scalability of the package, making it suitable for large-scale analyses.

Feel free to explore and integrate this package into your projects, leveraging its map harmonization and data analysis functionalities for your geospatial studies. If you have any questions or suggestions, please feel free to contact us.


## Installation

1 - Install [GDAL](https://github.com/OSGeo/gdal) on your system;

2 - Install [Python GDAL Library](https://pypi.org/project/GDAL/);

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
