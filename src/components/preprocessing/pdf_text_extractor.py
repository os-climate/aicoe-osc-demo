import glob
import io
import json
import logging
import os
from pathlib import Path

import pandas as pd
from pdf2image import pdfinfo_from_path
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

from .base_component import BaseComponent
from .base_curator import BaseCurator

_logger = logging.getLogger(__name__)


class PDFTextExtractor(BaseComponent):
    """This Class is responsible for extracting text data from PDFs and saving
        the result in a json format file.
        Each name/value pair in the json file refers to page_number and
        the list of paragraphs in that page.
    Args:
        min_paragraph_length (int): Minimum alphabetic characters for paragraph,
                            any paragraph shorter than that will be disregarded.
        annotation_folder (str)(Optional): path to the folder containing all annotated
                excel files. If provided, just the pdfs mentioned in annotation excels are
                extracted. Otherwise, all the pdfs in the pdf folder will be extracted.
        skip_extracted_files (bool): whether to skip extracting a file if it exist in the extraction folder.
        name (str) : Name of the component
    """

    def __init__(
        self,
        annotation_folder=None,
        min_paragraph_length=20,
        skip_extracted_files=False,
        name="PDFTextExtractor",
    ):
        super().__init__(name)
        self.min_paragraph_length = min_paragraph_length
        self.annotation_folder = annotation_folder
        self.skip_extracted_files = skip_extracted_files

    def process_page(self, input_text):
        """This function receives a text following:
        1. Divide it into  paragraphs, using \n\n
        2. Remove table data: To achieve this, if number of alphabet characters of paragraph
            is less min_paragraph_length, it is considered as table cell and it will be removed.

        Args:
            input_text (str): Content of each pdf.

        Returns:
            paragraphs (list of str): List of paragraphs.
        """
        paragraphs = input_text.split("\n\n")

        # Get ride of table data if the number of alphabets in a paragraph is less than `min_paragraph_length`
        paragraphs = [
            BaseCurator.clean_text(p)
            for p in paragraphs
            if sum(c.isalpha() for c in BaseCurator.clean_text(p))
            > self.min_paragraph_length
        ]
        return paragraphs

    def extract_pdf_by_page(self, pdf_file):
        """Read the content of each page in a pdf file, this method uses pdfminer.
        Args:
            pdf_file (str): Path to the pdf file.
        Returns:
            pdf_content (dict): A dictionary with key as page number and values
                                as list of paragraphs in that page.
        """
        try:
            num_pages = pdfinfo_from_path(pdf_file)  # noqa: F841
        except Exception as e:
            _logger.warning("{}: Unable to process {}".format(e, pdf_file))
            return {}

        fp = open(pdf_file, "rb")
        rsrcmgr = PDFResourceManager()
        retstr = io.BytesIO()
        codec = "utf-8"
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        pdf_content = {}
        for page_number, page in enumerate(
            PDFPage.get_pages(fp, check_extractable=False)
        ):
            interpreter.process_page(page)
            data = retstr.getvalue().decode("utf-8")
            data_paragraphs = self.process_page(data)
            if len(data_paragraphs) == 0:
                continue
            pdf_content[page_number] = data_paragraphs
            retstr.truncate(0)
            retstr.seek(0)
        fp.close()

        return pdf_content

    def run(self, input_filepath, output_folder):
        """Extract text from a single pdf file
        Args:
            input_filepath (str or PosixPath): full path to the pdf file
            output_folder (str or PosixPath): Folder to save the result of extraction
        """
        output_file_name = os.path.splitext(os.path.basename(input_filepath))[0]
        json_filename = output_file_name + ".json"

        if self.skip_extracted_files and json_filename in os.listdir(output_folder):
            _logger.info(
                "The extracted json for `{}` already exists. Skipping...".format(
                    output_file_name
                )
            )
            _logger.info(
                "If you would like to re-extract the already processed files, set "
                "`skip_extracted_files` to False in the config file. "
            )
            return None

        _logger.info("Extracting {} ...".format(os.path.basename(input_filepath)))
        text_dict = self.extract_pdf_by_page(input_filepath)
        if text_dict == {}:
            return None

        json_path = os.path.join(output_folder, json_filename)
        with open(json_path, "w") as f:
            json.dump(text_dict, f)

        return text_dict

    def run_folder(self, input_folder, output_folder):
        """This method will perform pdf extraction for all the pdfs mentioned
        as source in the annotated excel files
        and it will be saved the results in a output_folder.

        Args:
            input_folder (str or PosixPath): Path to the folder containing
                                            all the received pdf files.
            output_folder (str or PosixPath): path to the folder to save the
                                             extracted json files.
        """
        files = [str(f) for f in Path(input_folder).rglob("*.pdf") if f.is_file()]
        print(files)
        if self.annotation_folder is not None:
            # Get the names of all excel files
            all_annotation_files = glob.glob(
                "{}/[!~$]*[.xlsx]".format(self.annotation_folder)
            )
            annotated_pdfs = []
            for excel_path in all_annotation_files:
                df = pd.read_excel(excel_path, sheet_name="data_ex_in_xls")
                # Get the unique values of source_file column
                df_unique_pdfs = df["source_file"].drop_duplicates().dropna()
                annotated_pdfs.extend(df_unique_pdfs)
            annotated_pdfs = [file.split(".pdf")[0] + ".pdf" for file in annotated_pdfs]
            found_annotated_pdfs = []

            for f in files:
                if os.path.basename(f) in annotated_pdfs:
                    found_annotated_pdfs.append(os.path.basename(f))
                    _ = self.run(f, output_folder)
            _logger.info(
                "The following files in the annotation excels do not exist in pdf folder\n"
            )
            _logger.info(set(annotated_pdfs).difference(set(found_annotated_pdfs)))

        else:
            for f in files:
                _ = self.run(f, output_folder)
