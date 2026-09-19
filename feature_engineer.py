
"""feature_engineer.py
 
Unicode-aware statistical feature extraction across typologically and
orthographically diverse corpora.
 
Extracted features
------------------
TTR         : type-token ratio over a fixed token budget.
Avg_Length  : mean token length in *grapheme clusters*.
Length_Var  : population variance of token length.
 
Design notes
------------
1. Combining marks.  ``\\w`` (and equivalently ``\\p{L}``) matches letters
   but NOT Unicode marks (Mn/Mc).  Devanagari <क्या> is क + virama + य +
   matra; Arabic <مَرْحَبًا> carries harakat.  Both therefore shatter at
   every diacritic under ``\\w+`` or ``\\p{L}+``.  The correct class is
   ``[\\p{L}\\p{M}\\p{Nd}]``, plus ZWJ/ZWNJ which are format characters
   (Cf) required *inside* Brahmic and Arabic orthographic words.
 
2. Scriptio continua.  Han/Hiragana/Katakana/Thai/Khmer/Lao/Myanmar do
   not delimit words.  Regex cannot segment these at all.  Dictionary
   segmentation is required.  Resolution order, all lightweight:
       ICU BreakIterator  -> uniform dictionary segmentation, all scripts
       jieba              -> Chinese
       janome             -> Japanese
       character fallback -> logged as DEGRADED, never silent
 
3. Grapheme clusters.  Length is counted with ``\\X`` rather than by
   ``len()``.  In Devanagari one akshara is 2-4 codepoints.
 
4. Normalisation.  NFC is applied before tokenisation.
 
5. Token budget.  TTR decreases with corpus size (Heaps' law), so
   cross-language TTR is only comparable at a fixed N.  Files below the
   budget are REJECTED -- EXCEPT in --query-mode (see below).
 
6. Query mode (NEW).  The target language (e.g. Elamite) usually has far
   fewer than 5000 tokens.  Rejecting it would leave nothing to predict
   on.  --query-mode processes an under-budget file anyway, but stamps
   the real token count and a BUDGET_SHORTFALL flag into the output, so
   its lower comparability is documented, not hidden.
 
Author: <redacted for review>
"""
 
from __future__ import annotations
 
import argparse
import csv
import logging
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator
 
import regex as re
 
TOKEN_BUDGET: int = 5_000
READ_CHUNK: int = 1 << 16
SCRIPT_SNIFF_CHARS: int = 4_000
 
WORD_RE = re.compile(r"[\p{L}\p{M}\p{Nd}\u200c\u200d]+")
GRAPHEME_RE = re.compile(r"\X")
 
CONTINUA_RE = {
    "Han": re.compile(r"\p{Script=Han}"),
    "Hiragana": re.compile(r"\p{Script=Hiragana}"),
    "Katakana": re.compile(r"\p{Script=Katakana}"),
    "Thai": re.compile(r"\p{Script=Thai}"),
    "Khmer": re.compile(r"\p{Script=Khmer}"),
    "Lao": re.compile(r"\p{Script=Lao}"),
    "Myanmar": re.compile(r"\p{Script=Myanmar}"),
}
 
LOG = logging.getLogger("feature_engineer")
 
 
class SegmenterRegistry:
    """Lazily resolves the best available segmenter for a given script."""
 
    def __init__(self) -> None:
        self._cache: dict[str, Callable[[str], list[str]] | None] = {}
        self._icu_checked = False
        self._icu = None
 
    def _icu_breaker(self, locale: str) -> Callable[[str], list[str]] | None:
        """Return an ICU dictionary word-breaker, or None if unavailable."""
        if not self._icu_checked:
            self._icu_checked = True
            try:
                import icu  # type: ignore
 
                self._icu = icu
                LOG.info("Segmentation backend: ICU BreakIterator (preferred).")
            except ImportError:
                self._icu = None
 
        if self._icu is None:
            return None
 
        icu = self._icu
        breaker = icu.BreakIterator.createWordInstance(icu.Locale(locale))
 
        def _segment(text: str) -> list[str]:
            breaker.setText(text)
            out, prev = [], 0
            for boundary in breaker:
                piece = text[prev:boundary]
                prev = boundary
                if WORD_RE.fullmatch(piece):
                    out.append(piece)
            return out
 
        return _segment
 
    @staticmethod
    def _jieba() -> Callable[[str], list[str]] | None:
        try:
            import jieba
 
            jieba.setLogLevel(logging.ERROR)
            jieba.initialize()
        except ImportError:
            return None
        LOG.info("Segmentation backend for Han: jieba.")
        return lambda text: [t for t in jieba.cut(text) if WORD_RE.fullmatch(t)]
 
    @staticmethod
    def _janome() -> Callable[[str], list[str]] | None:
        try:
            from janome.tokenizer import Tokenizer
        except ImportError:
            return None
        tokenizer = Tokenizer()
        LOG.info("Segmentation backend for Japanese: janome.")
        return lambda text: [
            t.surface
            for t in tokenizer.tokenize(text)
            if WORD_RE.fullmatch(t.surface)
        ]
 
    @staticmethod
    def _character_fallback(script: str) -> Callable[[str], list[str]]:
        """Degrade to one token per grapheme cluster."""
        LOG.warning(
            "No dictionary segmenter for %s -- falling back to character "
            "segmentation. Avg_Length for this language is DEGRADED and "
            "not comparable to delimited scripts.",
            script,
        )
        return lambda text: [
            g for g in GRAPHEME_RE.findall(text) if WORD_RE.fullmatch(g)
        ]
 
    def resolve(self, script: str) -> tuple[Callable[[str], list[str]], bool]:
        """Return ``(segmenter, is_degraded)`` for a script name."""
        if script in self._cache:
            fn = self._cache[script]
            if fn is not None:
                return fn, False
 
        locale = {
            "Han": "zh", "Hiragana": "ja", "Katakana": "ja", "Thai": "th",
            "Khmer": "km", "Lao": "lo", "Myanmar": "my",
        }.get(script, "root")
 
        fn = self._icu_breaker(locale)
        if fn is None:
            if script == "Han":
                fn = self._jieba()
            elif script in ("Hiragana", "Katakana"):
                fn = self._janome() or self._jieba()
 
        if fn is None:
            return self._character_fallback(script), True
 
        self._cache[script] = fn
        return fn, False
 
 
def detect_continua_script(sample: str) -> str | None:
    """Return the dominant delimiter-less script in ``sample``, if any."""
    counts = {name: len(rx.findall(sample)) for name, rx in CONTINUA_RE.items()}
    total = sum(counts.values())
    if total == 0 or total < len(sample) * 0.15:
        return None
 
    kana = counts["Hiragana"] + counts["Katakana"]
    if kana > 0 and (kana + counts["Han"]) > 0:
        if kana / (kana + counts["Han"]) > 0.10:
            return "Hiragana"
    return max(counts, key=counts.__getitem__)
 
 
def _read_chunks(path: Path) -> Iterator[str]:
    """Yield NFC-normalised chunks, split on whitespace."""
    remainder = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(READ_CHUNK)
            if not chunk:
                break
            buffer = remainder + chunk
            cut = buffer.rfind(" ")
            if cut == -1 or len(buffer) > READ_CHUNK * 4:
                cut = len(buffer)
            yield unicodedata.normalize("NFC", buffer[:cut])
            remainder = buffer[cut:]
    if remainder.strip():
        yield unicodedata.normalize("NFC", remainder)
 
 
def collect_tokens(
    path: Path, registry: SegmenterRegistry, budget: int
) -> tuple[list[str], bool]:
    """Stream ``path`` until ``budget`` tokens are collected."""
    tokens: list[str] = []
    segmenter: Callable[[str], list[str]] | None = None
    degraded = False
    sniff: list[str] = []
    sniff_len = 0
 
    for chunk in _read_chunks(path):
        if segmenter is None:
            sniff.append(chunk)
            sniff_len += len(chunk)
            if sniff_len < SCRIPT_SNIFF_CHARS:
                continue
            joined = "".join(sniff)
            script = detect_continua_script(joined)
            if script is None:
                segmenter = WORD_RE.findall
            else:
                segmenter, degraded = registry.resolve(script)
            sniff, chunk = [], joined
 
        tokens.extend(segmenter(chunk))
        if len(tokens) >= budget:
            return tokens[:budget], degraded
 
    if segmenter is None and sniff:
        joined = "".join(sniff)
        script = detect_continua_script(joined)
        segmenter, degraded = (
            (WORD_RE.findall, False) if script is None else registry.resolve(script)
        )
        tokens.extend(segmenter(joined))
 
    return tokens[:budget], degraded
 
 
@dataclass(slots=True)
class LanguageFeatures:
    """One output row."""
 
    iso_code: str
    ttr: float
    avg_length: float
    length_var: float
    n_tokens: int = field(default=TOKEN_BUDGET)
    degraded: bool = field(default=False)
    budget_short: bool = field(default=False)
 
 
def grapheme_length(token: str) -> int:
    """Length in grapheme clusters, not codepoints."""
    return len(GRAPHEME_RE.findall(token))
 
 
def compute_features(
    iso_code: str, tokens: list[str], degraded: bool, budget_short: bool
) -> LanguageFeatures:
    """Compute TTR, mean and population variance of token length."""
    types = {t.casefold() for t in tokens}
    ttr = len(types) / len(tokens)
 
    lengths = [grapheme_length(t) for t in tokens]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / n  # population
 
    return LanguageFeatures(iso_code, ttr, mean, variance, n, degraded, budget_short)
 
 
def process_corpus(
    corpus_dir: Path, budget: int, allow_short: bool, query_mode: bool
) -> list[LanguageFeatures]:
    """Process every ``.txt`` file in ``corpus_dir``.
 
    query_mode: process under-budget files instead of rejecting them,
    stamping the shortfall. Intended for the single target language
    (Elamite), whose small corpus would otherwise be discarded.
    """
    registry = SegmenterRegistry()
    rows: list[LanguageFeatures] = []
    lenient = allow_short or query_mode
 
    for path in sorted(corpus_dir.glob("*.txt")):
        iso_code = path.stem
        try:
            tokens, degraded = collect_tokens(path, registry, budget)
        except OSError as exc:
            LOG.error("%s: unreadable (%s) -- skipped.", iso_code, exc)
            continue
 
        budget_short = len(tokens) < budget
        if budget_short:
            if not lenient:
                LOG.error(
                    "%s: %d tokens < budget %d -- REJECTED. TTR is not "
                    "comparable across differing N (Heaps' law). Supply a "
                    "larger corpus, or use --query-mode for the target "
                    "language.",
                    iso_code, len(tokens), budget,
                )
                continue
            LOG.warning(
                "%s: %d tokens < budget %d -- INCLUDED (%s). Its TTR is "
                "inflated relative to full-budget languages and is stamped "
                "BUDGET_SHORTFALL.",
                iso_code, len(tokens), budget,
                "query-mode" if query_mode else "allow-short",
            )
 
        if not tokens:
            LOG.error("%s: zero valid tokens -- skipped.", iso_code)
            continue
 
        row = compute_features(iso_code, tokens, degraded, budget_short)
        rows.append(row)
        LOG.info(
            "%-8s n=%-5d TTR=%.4f  avg=%.3f  var=%.3f%s%s",
            row.iso_code, row.n_tokens, row.ttr, row.avg_length,
            row.length_var,
            "  [DEGRADED]" if row.degraded else "",
            "  [SHORT]" if row.budget_short else "",
        )
 
    return rows
 
 
def export(rows: list[LanguageFeatures], out_path: Path) -> None:
    """Write the feature matrix."""
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["ISO_Code", "TTR", "Avg_Length", "Length_Var",
             "N_Tokens", "Segmentation_Degraded", "Budget_Shortfall"]
        )
        for row in rows:
            writer.writerow([
                row.iso_code,
                f"{row.ttr:.6f}",
                f"{row.avg_length:.6f}",
                f"{row.length_var:.6f}",
                row.n_tokens,
                int(row.degraded),
                int(row.budget_short),
            ])
 
 
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Unicode-aware multilingual statistical feature extraction."
    )
    parser.add_argument("--corpus", type=Path, default=Path("UD_Corpus"))
    parser.add_argument("--out", type=Path, default=Path("extracted_features.csv"))
    parser.add_argument("--budget", type=int, default=TOKEN_BUDGET)
    parser.add_argument(
        "--allow-short", action="store_true",
        help="Include all under-budget files (breaks TTR comparability).",
    )
    parser.add_argument(
        "--query-mode", action="store_true",
        help="Process an under-budget target language, stamping the shortfall.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)
 
 
def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(message)s",
    )
 
    if not args.corpus.is_dir():
        print(f"ERROR: {args.corpus} is not a directory.", file=sys.stderr)
        return 1
 
    rows = process_corpus(
        args.corpus, args.budget, args.allow_short, args.query_mode
    )
    if not rows:
        print("ERROR: no language met the token budget.", file=sys.stderr)
        return 1
 
    export(rows, args.out)
    degraded = sum(r.degraded for r in rows)
    short = sum(r.budget_short for r in rows)
    print(f"Wrote {len(rows)} languages -> {args.out}")
    if degraded:
        print(f"WARNING: {degraded} language(s) used character-segmentation "
              "fallback; install PyICU (or jieba/janome) before reporting.")
    if short:
        print(f"NOTE: {short} language(s) fell short of the {args.budget}-token "
              "budget and are stamped Budget_Shortfall=1.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 
