from .base_component import BaseComponent
import os
from pdf2image import convert_from_path, pdfinfo_from_path
from mmdet.apis import init_detector, inference_detector
import numpy as np
from tabula import read_pdf
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import subprocess

import gdown
from tqdm import tqdm
from src.components.utils.cscdtabnet_checkpoint_url import checkpoint_url

import logging

_logger = logging.getLogger(__name__)


class PDFTableExtractor(BaseComponent):
    def __init__(
        self,
        batch_size,
        cscdtabnet_config,
        cscdtabnet_ckpt,
        bbox_thres,
        dpi,
        name="PDFTableExtractor",
    ):
        """
        Args:
            batch_size (An int): How many pages to infer bbox each run
            cscdtabnet_config (A PosixPath or str): Config file for cascadetabnet.
                                                Should be placed in config folder
            cscdtabnet_ckpt (A PosixPath or str): cascadetabnet checkpoint.
                                                  Should be placed
                                                  in checkpoint folder
            bbox_thres (A float): Threshold to label a bbox as table
            dpi (An int): dots per inch for pdf2image
            name (A str)
        """
        super().__init__(name)
        self.batch_size = batch_size
        self.cscdtabnet_config = str(cscdtabnet_config)
        self.cscdtabnet_ckpt = str(cscdtabnet_ckpt)
        self.bbox_thres = bbox_thres
        self.dpi = dpi
        self.model = self.__create_model()

    def __create_model(self):
        """
        Download checkpoint file if not exist and return a detector model
        Returns:
            an init_detector
        """
        ckpt = os.path.basename(self.cscdtabnet_ckpt)
        assert ckpt in checkpoint_url.keys(), "Invalid cascadetabnet checkpoint"

        def download_ckpt():
            _ = gdown.download(checkpoint_url[ckpt], output=self.cscdtabnet_ckpt)

        if not os.path.exists(self.cscdtabnet_ckpt):
            _logger.info("cascadetabnet checkpoint does not exist. Downloading...")
            download_ckpt()

        # In case of connection error and incomplete download
        download_successful = False
        import torch

        device = "cpu" if "cpu" in torch.__version__ else "cuda:0"
        while not download_successful:
            try:
                det = init_detector(
                    self.cscdtabnet_config, self.cscdtabnet_ckpt, device=device
                )
                download_successful = True
            except OSError:
                _logger.info(
                    "Error while downloading cascadetabnet checkpoint. Redownloading..."
                )
                download_ckpt()

        return det

    @staticmethod
    def process_single_table(pdf_path, prefix, output_folder, dpi, iterable):
        """Read a single table mentioned in pdf_path, it is as a standalone
            function because of pickle issue

        Args:
            pdf_path (str): Path to the pdf
            prefix (str): The prefix name for saving the csv
            output_folder (str): The output directory to sabe the extracted table
            dpi
            iterable (list): list of parameters pass by multi processing
                iterable[0] (int): page number in the pdf
                iterable[1] (int): the index of the tale in the page
                iterable[2] (list of int): coordinates of the table.
        Return:
            table content (dataframe or None)
            saved_filenam(str or None): Path the csv is saved
            page_num (int): page number
        """
        page_num = iterable[0]
        table_index = iterable[1]
        area = iterable[2]
        try:
            table = read_pdf(
                pdf_path,
                pages=page_num,
                pandas_options={"header": None},
                stream=True,
                silent=True,
                area=[i * 72 / dpi for i in area],  # tabula uses 72 dpi
            )
        except subprocess.CalledProcessError:
            _logger.warning(
                "Tabula has error extracting table from file {}, page {}, area {}".format(
                    pdf_path, page_num, area
                )
            )
            return None, None, page_num

        if len(table) != 0:
            # ** This format is important as curation will use this format **
            saved_filename = "{}_page{}_{}.csv".format(
                prefix, page_num, table_index + 1
            )
            filename = os.path.join(output_folder, saved_filename)

            table[0].to_csv(filename)
            return table[0], saved_filename, page_num
        else:
            return None, None, page_num

    def infer_bbox(self, input_filepath):
        """
        Infer bbox for 1 file
        Args:
            input_filepath (A str)
        Returns:
            table_coords (A dictionary): Key=page number,
                                         value = a list of list of size 4.
                                         [y_top_left, x_top_left,
                                            y_bottom_right, x_bottom_right]
        """
        try:
            num_pages = pdfinfo_from_path(input_filepath)["Pages"]
        except Exception as e:
            _logger.warning("{}: Unable to process {}".format(e, input_filepath))
            return None

        pc = 0
        table_coords = defaultdict(list)

        if self.batch_size == -1:
            batch_size_ = num_pages
        else:
            batch_size_ = self.batch_size

        page_num = 1
        pbar = tqdm(total=num_pages + 1, desc="Page")
        while pc <= num_pages:
            images = convert_from_path(
                input_filepath,
                first_page=pc + 1,
                last_page=pc + batch_size_,
                use_pdftocairo=True,
                dpi=self.dpi,
                thread_count=cpu_count() - 1,
            )

            # infer tables in an image
            for im in tqdm(
                images,
                desc="Inferring tables for page {}-{}".format(
                    page_num, page_num + batch_size_
                ),
            ):
                result = inference_detector(self.model, np.array(im))
                bordered_tables = result[0][0]
                borderless_tables = result[0][2]

                # could have multiple tables
                for res in bordered_tables:
                    if res[-1] > self.bbox_thres:
                        # [y_top_left, x_top_left, y_bottom_right, x_bottom_right]
                        table_coords[page_num].append([res[1], res[0], res[3], res[2]])

                for res in borderless_tables:
                    if res[-1] > self.bbox_thres:
                        # [y_top_left, x_top_left, y_bottom_right, x_bottom_right]
                        table_coords[page_num].append([res[1], res[0], res[3], res[2]])

                page_num += 1

            pc += batch_size_
            pbar.update(batch_size_)

        return table_coords

    def extract_table(self, input_filepath, table_coords, output_folder):
        """Given filepath and table coordinates of tables in each page,
            extract tables using tabula and save them as csv
        Args:
            input_filepath (A str or PosixPath)
            table_coords (A dictionary): Key=page number,
                                         value = a list of list of size 4.
                                         [y_top_left, x_top_left,
                                            y_bottom_right, x_bottom_right]
            output_folder (A str or PosixPath)

        Returns:
            tables (A dict): Key=page number, value = a list of dataframes
            tables_meta (A dict): Key=page number, value=a list of filenames
        """
        prefix = os.path.basename(input_filepath).split(".pdf")[0].strip()

        tables = defaultdict(list)
        tables_meta = defaultdict(list)

        # -1 to leave some space for i/o operations
        process_count = cpu_count() - 1
        func = partial(
            self.process_single_table, input_filepath, prefix, output_folder, self.dpi
        )
        with Pool(process_count) as p:
            result = p.map(
                func,
                [
                    (page_num, table_index, area)
                    for page_num in tqdm(
                        table_coords, desc="Extracting and saving tables"
                    )
                    for table_index, area in enumerate(table_coords[page_num])
                ],
            )

        for tab, path, key in result:
            if tab is not None:
                tables[key].append(tab)
                tables_meta[key].append(path)

        return tables, tables_meta

    def run(self, input_filepath, output_folder):
        """Returns and saves tables extracted as csv

        Args:
            input_filepath (A str or PosixPath)
            output_folder (A str or PosixPath)

        Returns:
            tables (A dict): See extract_table()
            tables_meta (A dict): See extract_table()
        """
        _logger.info("{} is running on file {}...".format(self.name, input_filepath))

        table_coords = self.infer_bbox(input_filepath)
        # unable to process pdf file
        if table_coords is None:
            return None

        tables, tables_meta = self.extract_table(
            input_filepath, table_coords, output_folder
        )

        return (tables, tables_meta)

    def run_folder(self, input_folder, output_folder):
        """Runs run() for a folder of pdfs.

        Args:
            input_folder (A str or PosixPath)
            output_folder (A str or PosixPath)
        """
        files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".pdf")
        ]

        for f in files:
            _ = self.run(f, output_folder)
