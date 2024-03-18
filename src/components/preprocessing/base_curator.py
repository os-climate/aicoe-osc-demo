"""BaseCurator."""

import re
from abc import abstractmethod

from .base_component import BaseComponent


# Remember to also implement BaseComponent's abstract methods for child classes
# of this class
class BaseCurator(BaseComponent):
    """Base Curator class."""

    def __init__(self, name="BaseCurator"):
        """Initialize BaseCurator class."""
        self.name = name

    @abstractmethod
    def process_single_annotation_file(self, annotation_filepath, *args, **kwargs):
        """Process single annotation file."""
        pass

    @abstractmethod
    def create_pos_examples(self, row, *args, **kwargs):
        """Create POS examples."""
        pass

    @abstractmethod
    def create_negative_examples(self, row, *args, **kwargs):
        """Create negative examples."""
        pass

    @staticmethod
    def clean_text(text):
        """Clean text.

        Args:
            text (A str)
        """
        # Substitute  unusual quotes at the start of the string with usual quotes
        text = re.sub("(?<=\[)“", '"', text)  # noqa W605
        # Substitute  unusual quotes at the end of the string with usual quotes
        text = re.sub("”(?=\])", '"', text)  # noqa W605
        # Substitute th remaining unusual quotes with space
        text = re.sub("“|”", "", text)
        text = re.sub("\n|\t", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text
