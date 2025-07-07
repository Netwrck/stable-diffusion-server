import nltk

try:
    stopwords = nltk.corpus.stopwords.words("english")
except LookupError:  # pragma: no cover - external data
    nltk.download("stopwords", quiet=True)
    stopwords = nltk.corpus.stopwords.words("english")


def remove_stopwords(prompt: str) -> str:
    """Return the prompt without stopwords."""
    return " ".join(word for word in prompt.split() if word not in stopwords)


def shorten_too_long_text(prompt: str) -> str:
    """Trim prompts longer than 200 characters."""
    if len(prompt) > 200:
        tokens = [w for w in prompt.split() if w not in stopwords]
        prompt = " ".join(tokens)
        if len(prompt) > 200:
            prompt = prompt[:200]
    return prompt


def shorten_prompt_for_retry(prompt: str) -> str:
    """Remove stopwords and return roughly half of the words for a retry."""
    tokens = [w for w in prompt.split() if w not in stopwords]
    return " ".join(tokens[: len(tokens) // 2])


__all__ = [
    "stopwords",
    "remove_stopwords",
    "shorten_too_long_text",
    "shorten_prompt_for_retry",
]
