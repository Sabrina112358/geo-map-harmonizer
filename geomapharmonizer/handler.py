from osgeo import gdal, osr
import logging
import os
import numpy
import json
from datetime import date, datetime, timedelta
from sklearn.metrics import confusion_matrix
import glob
import pandas
import sys

from geomapharmonizer.src.utils import (
    get_map_name,
    calculate_num_blocks,
    NpEncoder
)


class GeoMapHamonizer:

    def __init__(
            self,
            path_map1: str,
            path_map2: str,
            logging_level: int = logging.INFO,
            logging_format: str = '[%(asctime)s][%(levelname)s]  %(message)s',
            max_block_size = 100 * 1024 * 1024
    ) -> None:
        self.path_map1 = path_map1
        self.path_map2 = path_map2
        logging.basicConfig(format=logging_format, level=logging_level)
        self.max_block_size = max_block_size

    def check_maps_dependencies(self) -> bool:
        """
        Checa as dependencias dos mapas
        """

        # Carrega os arquivos
        dataset1 = gdal.Open(self.path_map1)
        dataset2 = gdal.Open(self.path_map2)

        # verifica se as dimensões dos arquivos são iguais
        if dataset1.RasterXSize != dataset2.RasterXSize or dataset1.RasterYSize != dataset2.RasterYSize:
            logging.error("As dimensões dos arquivos são diferentes.")
            return False

        # verifica se as projeções dos arquivos são iguais
        if osr.SpatialReference(dataset1.GetProjection()).GetAttrValue('AUTHORITY', 1) != osr.SpatialReference(dataset2.GetProjection()).GetAttrValue('AUTHORITY', 1):
            logging.error("As projeções dos arquivos são diferentes.")
            return False

        # verifica se as resoluções dos arquivos são iguais
        if round(dataset1.GetGeoTransform()[1], 4) != round(dataset1.GetGeoTransform()[1], 4):
            logging.error("As resoluções dos arquivos são diferentes.")
            return False

        # se tudo estiver correto, exibe uma mensagem de confirmação
        logging.info(
            "Os arquivos têm as mesmas dimensões, projeção e resolução.")

        return True
    
    def get_cross_tables(self):

        # Read files
        dataset1 = gdal.Open(self.path_map1)
        dataset2 = gdal.Open(self.path_map2)

        # Get the file dimensions
        cols = dataset1.RasterXSize
        rows = dataset1.RasterYSize

        logging.debug(
            f"Number of rows: {cols}, number of columns: {cols}, map size: {cols * rows} pixels.")

        # Calculate the number of blocks based on the desired maximum size
        num_blocks = calculate_num_blocks(
            path_map1=self.path_map1, max_block_size=self.max_block_size)

        num_blocks = int(numpy.ceil(num_blocks/2))

        logging.debug(f"Number of 100mb blocks: {num_blocks*num_blocks}.")

        # Calculate the block width and height
        step_col = int(cols / num_blocks)
        step_row = int(rows / num_blocks)

        logging.debug(f"Block height: {step_row}, block width: {step_col}.")

        # Create an empty dictonary that saves the class order of each block
        class_list = {}

        # Create an empty folder where the cross tables will be save
        path = "./temp_cross_tables"

        # check whether directory already exists
        if not os.path.exists(path):
            os.mkdir(path)

        # create a block count
        index = 0

        for i in range(0, rows, step_row):
            for j in range(0, cols, step_col):

                # Calculate the block size for this part
                width = min(step_col, cols - j)
                height = min(step_row, rows - i)

                # Read the blocks with the determined sizes
                block1 = dataset1.ReadAsArray(j, i, xsize=width, ysize=height)
                block1 = block1.reshape(-1)
                logging.debug(f"Bloco 1: {sys.getsizeof(block1)}.")

                block2 = dataset2.ReadAsArray(j, i, xsize=width, ysize=height)
                block2 = block2.reshape(-1)
                logging.debug(f"Bloco 2: {sys.getsizeof(block2)}.")

                # Calculate the unique pixel values for each map
                classes1 = numpy.unique(block1)
                classes2 = numpy.unique(block2)

                # Concatenate the values and generate the list of classes
                classes = numpy.unique(numpy.concatenate((classes1, classes2)))

                class_list[f"{i}_{j}"] = list(classes)

                # Calculate the confusion matrix
                block_confusion_matrix = confusion_matrix(
                    block1, block2, labels=classes)

                # Get the main name of the map files
                name_map1 = get_map_name(path_map=self.path_map1)
                name_map2 = get_map_name(path_map=self.path_map2)

                # Save the confusion matrix to a CSV file
                numpy.savetxt(
                    f'./temp_cross_tables/{name_map1}_{name_map2}_{i}_{j}.csv', block_confusion_matrix, delimiter=',')

                index += 1
                logging.debug(
                    f"Process in {round(index/(num_blocks * num_blocks)  * 100, 2)}%.")

        with open('./temp_cross_tables/class_list.txt', 'w') as fout:
            fout.write(json.dumps(class_list, cls=NpEncoder, indent=2))

    def map_legend_harmonizer(self):
        if not self.check_maps_dependencies():
            raise Exception("As dependencias dos mapas não batem.")
        
        self.get_cross_tables()

        return None
