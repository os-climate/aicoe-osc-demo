"""Extractor class."""


from .pdf_table_extractor import PDFTableExtractor
from .pdf_text_extractor import PDFTextExtractor
import logging

_logger = logging.getLogger(__name__)
NAME_CLASS_MAPPING = {
    "PDFTextExtractor": PDFTextExtractor,
    "PDFTableExtractor": PDFTableExtractor,
}


class Extractor:
    """Extractor class."""

    def __init__(self, extractors):
        """Combine different types of extractors.

        A pipeline extractor which combines different types of extractors

        Args:
            extractors (A list of tuples): (Name of extractor, kwargs_dict)
        """
        self.extractors = self.__create_extractors(extractors)

    def __create_extractors(self, extractors):
        """Return a list of extractors objects.

        Args:
            extractors (A list of str)
        """
        list_ext = []
        for ext in extractors:
            try:
                ext_obj = NAME_CLASS_MAPPING[ext[0]](**ext[1])
            except KeyError:
                raise ValueError("{} is an invalid extractor".format(ext[0]))

            list_ext.append(ext_obj)

        return list_ext

    def run(self, input_filepath, output_folder):
        """Extract a single file.

        Args:
            input_filepath (str): Input file path
            output_folder (str): Output folder path

        """
        _logger.info("Running all extractors...")

        for ext in self.extractors:
            _ = ext.run(input_filepath, output_folder)

    def run_folder(self, input_folder, output_folder):
        """Extract for all files mentioned in folder.

        (The logic is based on each child.)

        Args:
            input_folder (A str): Input folder path
            output_folder (A str): Output folder path
        """
        for ext in self.extractors:
            ext.run_folder(input_folder, output_folder)
