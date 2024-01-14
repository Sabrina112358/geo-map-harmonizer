import logging
from geomapharmonizer import GeoMapHamonizer

geomap = GeoMapHamonizer(path_map1="mapb_col6_crop.tif", path_map2="terra_class.tif", logging_level=logging.DEBUG)

geomap.map_legend_harmonizer()