import os
import json
import logging
import numpy
from datetime import datetime, date, timedelta


def get_map_name(path_map: str):
    """
    Separa e retorna o nome do arquivo a partir do parth do arquivo
    """

    map_name = path_map.split(sep="/")[-1].split(sep=".")[0]
    return map_name


# Function to calculate the size in bytes of a raster file
def get_raster_size(path_map1: str):
    file_size = os.path.getsize(path_map1)
    logging.debug(file_size)
    return file_size

# Function to calculate the number of blocks based on the desired maximum size


def calculate_num_blocks(path_map1: str, max_block_size: int):
    raster_size = get_raster_size(path_map1=path_map1)
    num_blocks = int(numpy.ceil(raster_size / max_block_size))
    return num_blocks


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.bool_):
            return bool(obj)
        if isinstance(obj, (numpy.floating, numpy.complexfloating)):
            return float(obj)
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.string_):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        return super(NpEncoder, self).default(obj)
