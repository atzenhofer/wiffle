"""Seed labels: which texts a seed list selects, and the token streams with the seeds masked."""

import re
from dataclasses import dataclass
from pathlib import Path

from .ngrams import GAP

WORD = re.compile(r"\w+")
DIGITS = re.compile(r"\d+")
MASK = "<seed>"
NUMBER = "<num>"


@dataclass(frozen=True)
class Seeds:
    """Seed phrases, lowercased; a seed matches anywhere in the lowercased text, also inside a word."""

    phrases: tuple[str, ...]

    def __post_init__(self):
        if not self.phrases:
            raise ValueError("a seed list needs at least one phrase")
        alternatives = sorted(self.phrases, key=len, reverse=True)
        object.__setattr__(self, "_pattern", re.compile("|".join(map(re.escape, alternatives))))

    @classmethod
    def of(cls, phrases) -> "Seeds":
        """Seeds from phrases, lowercased and without blanks or duplicates."""
        cleaned = dict.fromkeys(phrase.strip().lower() for phrase in phrases)
        return cls(tuple(phrase for phrase in cleaned if phrase))

    @classmethod
    def read(cls, path: str | Path) -> "Seeds":
        """Seeds from a file, one phrase per line; empty lines and lines starting with # are skipped."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls.of(line for line in lines if not line.lstrip().startswith("#"))

    def spans(self, lowered: str) -> list[tuple[int, int]]:
        """Character spans of all seed matches in a lowercased text."""
        return [match.span() for match in self._pattern.finditer(lowered)]

    def match(self, text: str | None) -> bool:
        """Whether any seed occurs in the text."""
        return bool(text) and self._pattern.search(text.lower()) is not None


def tokens(text: str | None, seeds: Seeds | None = None) -> list[str]:
    """Lowercased word tokens of a text; digits become one number token, a token touching a seed becomes the mask."""
    if not text:
        return []
    lowered = text.lower()
    masked = seeds.spans(lowered) if seeds else []
    result = []
    position = 0
    for word in WORD.finditer(lowered):
        start, end = word.span()
        while position < len(masked) and masked[position][1] <= start:
            position += 1
        if position < len(masked) and masked[position][0] < end:
            result.append(MASK)
        elif DIGITS.fullmatch(word.group()):
            result.append(NUMBER)
        else:
            result.append(word.group())
    return result


def masked_streams(texts: list[str], plain: list[list[str]], seeds: Seeds | None) -> list[list[str]]:
    """Token streams with the seeds masked, re-tokenizing only the texts in which a seed occurs."""
    if seeds is None:
        return plain
    return [tokens(text, seeds) if seeds.match(text) else stream for text, stream in zip(texts, plain, strict=True)]


def is_token(word: str, token: str) -> bool:
    """Whether a word of a text reads as the token; the gap mark stands for any word, the number token for any digits."""
    return word == token or token == GAP or (token == NUMBER and DIGITS.fullmatch(word) is not None)


def concordance(text: str, phrase: list[str], width: int = 60) -> str | None:
    """The first occurrence of a token sequence in a text, with its context, or None."""
    lowered = text.lower()
    words = list(WORD.finditer(lowered))
    size = len(phrase)
    for position in range(len(words) - size + 1):
        if all(is_token(words[position + offset].group(), token) for offset, token in enumerate(phrase)):
            start = words[position].start()
            end = words[position + size - 1].end()
            left = text[max(0, start - width):start].replace("\n", " ")
            right = text[end:end + width].replace("\n", " ")
            return f"{left}[{text[start:end]}]{right}"
    return None
