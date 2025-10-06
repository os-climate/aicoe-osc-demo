"""Curator."""

import glob
import logging

from .text_curator import TextCurator
from .table_curator import TableCurator

logger = logging.getLogger(__name__)
NAME_CLASS_MAPPING = {"TextCurator": TextCurator, "TableCurator": TableCurator}


class Curator:
    """Curator class.

    A data curator component responsible for creating text training data based on annotated data
    Args:
        annotation_folder (str): path to the folder containing annotation excel files
    """

    def __init__(self, curators):
        """Initialize Curator class."""
        self.curators = self.__create_curators(curators)

    def __create_curators(self, curators):
        """Return a list of curator objects.

        Args:
            curators (A list of str)
        """
        list_cura = []
        for cura in curators:
            try:
                cura_obj = NAME_CLASS_MAPPING[cura[0]](**cura[1])
            except KeyError:
                raise ValueError("{} is an invalid extractor".format(cura[0]))

            list_cura.append(cura_obj)

        return list_cura

    def run(self, input_extraction_folder, annotation_folder, output_folder, kpi_df):
        """Run curation for each curator.

        Args:
            input_extraction_folder (A str or PosixPath)
            annotation_folder (A str or PosixPath)
            output_folder (A str or PosixPath)
            kpi_df (A DataFrame)
        """
        annotation_excels = glob.glob("{}/[!~$]*[.xlsx]".format(annotation_folder))
        logger.info("Received {} excel files".format(len(annotation_excels)))

        for curator_obj in self.curators:
            curator_obj.run(
                input_extraction_folder, annotation_excels, output_folder, kpi_df
            )
