import string
import pytest
from stable_diffusion_server import prompt_utils as sutils


def test_shorten_too_long_text():
    long_prompt = " ".join(["word" + str(i) for i in range(250)])
    shortened = sutils.shorten_too_long_text(long_prompt)
    assert len(shortened) <= 200


def test_shorten_prompt_for_retry_removes_stopwords():
    prompt = "the quick brown fox jumps over the lazy dog" * 3
    shortened = sutils.shorten_prompt_for_retry(prompt)
    # ensure length reduced by at least half
    assert len(shortened.split()) <= len(prompt.split()) // 2
    for stopword in sutils.stopwords:
        assert stopword not in shortened.split()


def test_remove_stopwords():
    prompt = "the quick brown fox"
    assert sutils.remove_stopwords(prompt) == "quick brown fox"
