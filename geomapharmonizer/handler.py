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


class GeoMapHarmonizer:
    """"
    This package includes an algorithm specialized in harmonizing geotiff maps.
    Its main function is to generate a harmonized legend based on the spatial
    distribution of classes in two maps, aiming for maximum accuracy in
    comparison.

    Additionally, the package provides data analysis functions for detailed
    reports. Specific strategies for handling Big Data are implemented, such as
    dividing maps into smaller blocks and consolidating results.

    In summary, the package provides an efficient and comprehensive solution
    for harmonizing geotiff maps, integrating an advanced algorithm, analysis
    tools, and strategies for dealing with large datasets.

    Attributes
    ---
    path_map1: str
        System path to the first map file in .tif format.
    path_map2: str
        System path to the second map file in .tif format.
    logging_level: int = 20
        Level of logs used in execution.
    logging_format: str = '[%(asctime)s][%(levelname)s] %(message)s'
        Format of logging output.
    max_block_size: int = 100 * 1024 * 1024
        Value of max legth of sub-block of each map
    """

    def __init__(
            self,
            path_map1: str,
            path_map2: str,
            logging_level: int = logging.INFO,
            logging_format: str = '[%(asctime)s][%(levelname)s]  %(message)s',
            max_block_size=100 * 1024 * 1024,
            null_value=0.0
    ) -> None:
        """
        Parameters
        ---
        path_map1: str
            System path to the first map file in .tif format.
        path_map2: str
            System path to the second map file in .tif format.
        logging_level: int = 20
            Level of logs used in execution.
        logging_format: str = '[%(asctime)s][%(levelname)s] %(message)s'
            Format of logging output.
        max_block_size: int = 100 * 1024 * 1024
            Value of max legth of sub-block of each map
        """
        self.path_map1 = path_map1
        self.path_map2 = path_map2
        logging.basicConfig(format=logging_format, level=logging_level)
        self.max_block_size = max_block_size
        self.null_value = null_value

    def check_maps_dependencies(self) -> bool:
        """
        Function to check map dependencies, such as dimension, projection and resolution.

        Returns: bool
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

    def get_cross_tables(self) -> None:
        """Method to generate cross tables files
        """

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

    def unify_cross_tables(
        self,
        path_map1: str,
        path_map2: str,
        null_value: float = 0.0
    ) -> pandas.DataFrame:
        """Method to unify all cross tables files in a pandas.DataFrame

        Args:
            path_map1 (str): System path to the first map file in .tif format.
            path_map2 (str): System path to the second map file in .tif format.
            null_value (float, optional): Value of null value in the maps files. Defaults to 0.0.

        Returns:
            pandas.DataFrame: DataFrame with the unify cross tables
        """

        with open('./temp_cross_tables/class_list.txt', 'r') as fout:
            class_list = fout.read()

        class_list = json.loads(class_list)

        class_items = []
        for key, value in class_list.items():
            class_items = numpy.unique(numpy.concatenate((class_items, value)))

        cross_table = pandas.DataFrame(
            0, index=class_items, columns=class_items)

        files = glob.glob("./temp_cross_tables/*.csv")

        name_map1 = get_map_name(path_map=path_map1)
        name_map2 = get_map_name(path_map=path_map2)

        for filename in files:
            classes = filename.split(f"{name_map1}_{name_map2}_")[
                1].split(".csv")[0]
            cross_table_block = pandas.read_csv(
                filename,
                names=class_list[classes]).set_axis(class_list[classes],
                                                    axis='index')

            for column in cross_table_block.columns:
                for index in cross_table_block.index:
                    cross_table.at[index, column] = cross_table.at[index,
                                                                   column] + cross_table_block.at[index, column]

        cross_table = cross_table.loc[(cross_table != 0.0).any(axis=1)]
        cross_table = cross_table.loc[:, (cross_table != 0.0).any(axis=0)]

        cross_table = cross_table.drop(index=null_value, columns=null_value)

        return cross_table

    def get_row_equivalence(
        self,
        cross_table: pandas.DataFrame,
        name_map1: str,
        name_map2: str
    ) -> pandas.DataFrame:
        """Method to generate the Row Equivalence in cross tables

        Args:
            cross_table (pandas.DataFrame): DataFrame with contains the cross table
            name_map1 (str): Name of the first map
            name_map2 (str): Name of the second map

        Returns:
            pandas.DataFrame: DataFrame with contains the row equivalence
        """

        row_mapping = []
        for index, row in cross_table.iterrows():
            max_row = row.sort_values(ascending=False)
            row_mapping.append(cross_table.loc[:, max_row.index[0]])

        row_mapping = pandas.DataFrame(data=row_mapping)

        linhas = row_mapping.index.tolist()
        colunas = row_mapping.columns.tolist()
        data_tuples = list(zip(colunas, linhas))
        row_mapping = pandas.DataFrame(
            data_tuples, columns=[f"{name_map1}", f"{name_map2}"])

        for i in range(len(row_mapping)):
            row_mapping.loc[i, "Size"] = cross_table.loc[(
                row_mapping[f"{name_map1}"][i])].max()
        return row_mapping

    def get_column_equivalence(
        self,
        cross_table: pandas.DataFrame,
        name_map1: str,
        name_map2: str
    ) -> pandas.DataFrame:
        """Method to generate the Column Equivalence in cross tables

        Args:
            cross_table (pandas.DataFrame): DataFrame with contains the cross table
            name_map1 (str): Name of the first map
            name_map2 (str): Name of the second map

        Returns:
            pandas.DataFrame: DataFrame with contains the column equivalence
        """

        column_mapping = []
        for column in range(0, cross_table.shape[1]):
            max_column = cross_table.iloc[:,
                                          column].sort_values(ascending=False)
            column_mapping.append(cross_table.loc[max_column.index[0], :])

        column_mapping = pandas.DataFrame(data=column_mapping)

        linhas = column_mapping.index.tolist()
        colunas = column_mapping.columns.tolist()

        data_tuples = list(zip(colunas, linhas))
        column_mapping = pandas.DataFrame(
            data_tuples, columns=[f"{name_map1}", f"{name_map2}"]
        )

        column_mapping.loc[:, ("Size")] = 0

        for i in range(len(column_mapping)):
            column_mapping.loc[i, "Size"] = cross_table.loc[:,
                                                            (column_mapping[f"{name_map1}"][i])].max()

        return column_mapping

    def get_legend_harmonizer(
            self,
            row_equivalence: pandas.DataFrame,
            column_equivalence: pandas.DataFrame
            ) -> pandas.DataFrame:
        """Method to harmonizer the row and column equivalences

        Args:
            row_equivalence (pandas.DataFrame): DataFrame with contains the row equivalence
            column_equivalence (pandas.DataFrame): DataFrame with contains the column equivalence

        Returns:
            pandas.DataFrame: DataFrame with contains the legend harmonizer
        """
        legend = pandas.merge(row_equivalence, column_equivalence, how="outer")
        return legend

    def map_legend_harmonizer(self) -> pandas.DataFrame:
        """Method to run all steps to generate a harmonizer legend

        Raises:
            Exception: The map dependencies don't match.

        Returns:
            pandas.DataFrame: DataFrame with contains the legend harmonizer
        """
        if not self.check_maps_dependencies():
            raise Exception("The map dependencies don't match.")

        self.get_cross_tables()

        self.cross_table = self.unify_cross_tables(
            path_map1=self.path_map1,
            path_map2=self.path_map2,
            null_value=self.null_value
        )

        name_map1 = get_map_name(self.path_map1)
        name_map2 = get_map_name(self.path_map2)

        self.row_equivalence = self.get_row_equivalence(
            self.cross_table, name_map1, name_map2)

        self.column_equivalence = self.get_column_equivalence(
            self.cross_table, name_map1, name_map2)

        self.legend = self.get_legend_harmonizer(self.row_equivalence, self.column_equivalence)
        logging.debug(self.legend)
        logging.info("Legendas geradas com sucesso")
        return self.legend
