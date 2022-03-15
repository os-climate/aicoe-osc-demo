"""NQ utils."""


import re


def get_text_section(doc_tokens):
    """Return section of text from a whitespace separated document."""
    return " ".join(doc_tokens[0].split(" ")[doc_tokens[1] : doc_tokens[2]])


def contains_table(text):
    """Return True if a string contains an HTML table."""
    if re.search(r"<Table>.*</Table>", text) is not None:
        return True
    else:
        return False


def remove_html_tags(text):
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>\s?", "", text)


def is_not_short(text, n=10):
    """Return True if string has more than n whitespace separated tokens."""
    return len(text.split(" ")) > n
