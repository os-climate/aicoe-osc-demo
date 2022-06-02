"""BaseKPIInferenceCurator"""

from abc import ABC, abstractmethod
import random
import re
from collections import defaultdict


class BaseKPIInferenceCurator(ABC):
    def __init__(self, name="BaseKPIInferenceCurator"):
        self.name = name

    @staticmethod
    def clean_text(text):
        """
        Clean text
        Args:
            text (A str)
        """
        # Substitute  unusual quotes at the start of the string with usual quotes
        text = re.sub("(?<=\[)“", '"', text)
        # Substitute  unusual quotes at the end of the string with usual quotes
        text = re.sub("”(?=\])", '"', text)
        # Substitute th remaining unusual quotes with space
        text = re.sub('“|”', '', text)
        text = re.sub('\n', " ", text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        text = re.sub(r"\s{2,}", " ", text)

        # replace special character
        special_regex_char = [
            "(", ")", "^", "+", "*", "$", "|", "\\", "?", "[", "]", "{", "}"
        ]
        text = ''.join(
            ["" if c in special_regex_char else c for c in text]
        )

        text = text.lower()

        # remove consecutive dots
        consecutive_dots = re.compile(r'\.{2,}')
        text = consecutive_dots.sub('', text)

        return text

    @staticmethod
    def create_squad_from_df(df):
        """ Create squad data format given a dataframe
        Args:
            df (A pandas DataFrame): Must have columns in this order ["source_file",
                                    "paragraph","question", "answer", "answer_start"]
            Returns:
                squad_json (A nested of list and dict): Squad json format
        """
        order_col = ["source_file", "paragraph", "question","answer", "answer_start"]
        assert(all([e in df.columns for e in order_col]))
        df = df[order_col]

        files = df['source_file'].unique()
        data = []
        for f in files:
            single_data = {}
            single_data['title'] = f
            temp = df[df["source_file"] == f]

            unique_par = temp['paragraph'].unique()

            paragraphs = []

            for up in unique_par:
                single_par = {}
                single_par['context'] = up

                temp_2 = temp[temp['paragraph'] == up]

                qas = []

                for row in temp_2.itertuples():
                    single_qas = {}
                    single_qas['question'] = row[3] # question has index 3
                    #index
                    single_qas['id'] = row[0]
                    ans_st = row[5]  # answer_start has index 5

                    if ans_st == []:
                        answers = []
                        single_qas['is_impossible'] = True
                    else:
                        answers = []
                        for i in ans_st:
                            answers.append(
                                {"text": row[4], "answer_start": i} # answer has index 4
                            )
                        single_qas['is_impossible'] = False
                    single_qas['answers'] = answers

                    qas.append(single_qas)

                single_par['qas'] = qas
                paragraphs.append(single_par)

            single_data['paragraphs'] = paragraphs
            data.append(single_data)

        squad_json = {}
        squad_json['version'] = "v2.0"
        squad_json['data'] = data

        return squad_json

    @staticmethod
    def find_answer_start(answer, par):
        """
        Args:
            answer(A str): Answer
            par (A str): Paragraph
        Returns:
            A list of starting indices
        """
        answer = "".join(["\." if c == "." else c for c in answer])

        # avoid 0 matching to 2016
        if answer.isnumeric():
            pat1 = "[^0-9]" + answer
            pat2 = answer + "[^0-9]"
            matches1 = re.finditer(pat1, par)
            matches2 = re.finditer(pat2, par)
            ans_start_1 = [i.start()+1 for i in matches1]
            ans_start_2 = [i.start() for i in matches2]
            ans_start = list(set(ans_start_1 + ans_start_2))
        else:
            pat = answer
            matches = re.finditer(answer, par)
            ans_start = [i.start() for i in matches]

        return ans_start

    def split_squad(self, squad_json, val_ratio, seed):
        """ Given a squad like json data format, split to train and val sets
        Args:
            squad_json
            val_ratio (A float): from 0 - 1
            seed (An int)
        Returns:
            train_squad (A dict)
            val_squad (A dict)
        """
        indices = []
        for i1, pdf in enumerate(squad_json['data']):
            pars = pdf['paragraphs']

            for i2, par in enumerate(pars):
                qas = par['qas']
                indices.append((i1, i2))

        random.seed(seed)
        random.shuffle(indices)
        split_idx = int((1-val_ratio)*len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_squad = self.return_sliced_squad(squad_json, train_indices)
        val_squad = self.return_sliced_squad(squad_json, val_indices)

        return train_squad, val_squad

    def return_sliced_squad(self, squad_json, indices):
        """
        Return sliced dataset based on indices
        Args:
            squad_json (A dict)
            indices (A tuple of 2): (pdf_index, paragraph_index)
        """
        if len(indices) == 0:
            return {}

        pdf2pars = defaultdict(list)
        for (i1, i2) in indices:
            pdf2pars[i1].append(i2)

        data = []
        for i1 in pdf2pars:
            pars_indices = pdf2pars[i1]

            pars = [squad_json['data'][i1]['paragraphs'][i2] for i2 in pars_indices]
            single_pdf = {}
            single_pdf['paragraphs'] = pars
            single_pdf['title'] = squad_json['data'][i1]['title']
            data.append(single_pdf)

        squad_data = {}
        squad_data['version'] = "v2.0"
        squad_data['data'] = data

        return squad_data


    @abstractmethod
    def curate(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_answerable(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_unanswerable(self, *args, **kwargs):
        pass