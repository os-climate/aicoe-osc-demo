"""NQExtractor."""

import json
import logging
import os

import pandas as pd
import tqdm

from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class NQExtractor(BaseComponent):
    """NQExtractor Class.

    The component to load the .jsonl files into a dataframe in chunks, extract the long answer candidates, and save
    each chunk as a CSV.
    Args:
        raw_nq_json_file(str): Path to the raw NQ JSONL file
        out_dir(str): Path to the directory where the parsed CSV files will be saved in
        nrwos(int): Number of examples to parse. Set to -1 to parse all.
        drop_no_long_answer(bool): Drop the examples that don't have a long answer
        chunk_size(int): Number of parsed examples to be grouped together and saved in a CSV file
    """

    def __init__(
        self,
        raw_nq_json_file,
        out_dir,
        n_rows=-1,
        drop_no_long_answer=True,
        chunk_size=10000,
        name="NQExtractor",
    ):
        """Initialize NQExtractor class."""
        super().__init__(name)
        self._raw_nq_json_file = raw_nq_json_file
        self._out_dir = out_dir
        self._n_rows = n_rows
        self._drop_no_long_answer = drop_no_long_answer
        self._chunk_size = chunk_size

    def _write_chunk_to_disk(self, chunk_data, chunk_idx):
        """Save a chunk of data to a CSV file (Helper function)."""
        df = pd.DataFrame(chunk_data).fillna(-1)
        output_path = os.path.join(self._out_dir, "Parsed_NQ_{}.csv".format(chunk_idx))
        df.to_csv(output_path)

    def run(self):
        """Load .json files into a df and extract long answer candidates.

        Simple utility function to load the .json files into a dataframe, and ex-
        tract long answer candidates
        Returns:
            A Dataframe containing the following columns:
                * document_text (str): The document split by whitespace
                * question_text (str): the question posed
                * yes_no_answer (str): Could be "YES", "NO", or "NONE"
                * short_answer_start (int): Start index of token, -1 if does not exist
                * short_answer_end (int): End index of token, -1 if does not exist
                * long_answer_start (int): Start index of token, -1 if does not exist
                * long_answer_end (int): End index of token, -1 if does not exist
                * example_id (str): ID representing the string.
                * relevant_excerpt (str): text containing the short answer
                * other_canditates (list): list strings containing excerpts tagged
                as long answer candidates but not containing the short answer
        """
        logger.info("Running the Extraction stage...")
        logger.info("=" * 30)
        json_lines = []
        logger.info("Loading original .json file...")
        extraction_counter = 0
        chunk_idx = (
            0  # Used to group N examples together and write to a CSV. N=chunk_size
        )
        with open(self._raw_nq_json_file) as f:
            for line in tqdm.tqdm(f):
                if extraction_counter == self._n_rows:
                    break
                line = json.loads(line)

                out_di = {
                    "document_text": line["document_text"],
                    "question_text": line["question_text"],
                }

                if "example_id" in line:
                    out_di["example_id"] = line["example_id"]

                annot = line["annotations"][0]

                # Dropping examples that don't have a long answer
                if (
                    self._drop_no_long_answer
                    and annot["long_answer"]["candidate_index"] == -1
                ):
                    continue

                out_di["yes_no_answer"] = annot["yes_no_answer"]
                out_di["long_answer_start"] = annot["long_answer"]["start_token"]
                out_di["long_answer_end"] = annot["long_answer"]["end_token"]

                if len(annot["short_answers"]) > 0:
                    out_di["short_answer_start"] = annot["short_answers"][0][
                        "start_token"
                    ]
                    out_di["short_answer_end"] = annot["short_answers"][0]["end_token"]
                else:
                    out_di["short_answer_start"] = -1
                    out_di["short_answer_end"] = -1

                candidate_index = annot["long_answer"]["candidate_index"]
                # gather other long answer candidates that doesn't contain the short answer
                other_long_answer_candidates = [
                    line["long_answer_candidates"][i]
                    for i in range(len(line["long_answer_candidates"]))
                    if i != candidate_index
                    and line["long_answer_candidates"][i]["top_level"] is True
                ]

                out_di["other_long_answer_candidates"] = other_long_answer_candidates

                json_lines.append(out_di)
                extraction_counter += 1
                if extraction_counter % self._chunk_size == 0:
                    self._write_chunk_to_disk(
                        chunk_data=json_lines, chunk_idx=chunk_idx
                    )
                    json_lines = []
                    chunk_idx += 1

            # Last chunk:
            self._write_chunk_to_disk(chunk_data=json_lines, chunk_idx=chunk_idx)
