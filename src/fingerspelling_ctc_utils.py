"""
Shared vocabulary, encoding, decoding, and edit-distance utilities for the
ISL-Fingerspelling CTC experiment. Pure Python/NumPy -- no TensorFlow
dependency except where noted, so these are cheap to unit-test and reuse.

Vocabulary (29 output classes total):
    index 0  = <BLANK>   (CTC blank, never a real target character)
    index 1-26 = a..z
    index 27 = <SPACE>
    index 28 = <UNK>     (any character outside a-z/space, e.g. stray
                           punctuation or digits found in a few transcripts)
"""
import string

BLANK_INDEX = 0
LETTERS = list(string.ascii_lowercase)  # a..z
SPACE_INDEX = 1 + len(LETTERS)      # 27
UNK_INDEX = SPACE_INDEX + 1         # 28
VOCAB_SIZE = UNK_INDEX + 1          # 29

INDEX_TO_TOKEN = {0: "<BLANK>"}
for i, letter in enumerate(LETTERS):
    INDEX_TO_TOKEN[1 + i] = letter
INDEX_TO_TOKEN[SPACE_INDEX] = "<SPACE>"
INDEX_TO_TOKEN[UNK_INDEX] = "<UNK>"

TOKEN_TO_INDEX = {v: k for k, v in INDEX_TO_TOKEN.items()}


def vocabulary_dict():
    return {
        "blank_index": BLANK_INDEX,
        "vocab_size": VOCAB_SIZE,
        "non_blank_symbol_count": VOCAB_SIZE - 1,
        "index_to_token": {str(k): v for k, v in INDEX_TO_TOKEN.items()},
        "token_to_index": TOKEN_TO_INDEX,
        "note": "CTC blank occupies index 0 and is never a training target; "
                "unexpected characters (punctuation/digits) map to <UNK> "
                "rather than being dropped.",
    }


def char_to_index(letter):
    if letter == " ":
        return SPACE_INDEX
    if letter in TOKEN_TO_INDEX and letter != "<BLANK>":
        return TOKEN_TO_INDEX[letter]
    return UNK_INDEX


def encode_char_sequence(char_sequence):
    """char_sequence: list of {"letter": ...} dicts (from char_sequence_json,
    already sorted by char_index). Returns a list[int] of label indices."""
    return [char_to_index(c["letter"]) for c in char_sequence]


def greedy_ctc_decode(frame_predictions):
    """frame_predictions: list/array of per-timestep predicted class indices
    (already argmax'd over the vocabulary), length == num_frames.

    Standard CTC greedy decode: collapse consecutive duplicate labels, then
    remove blanks. This correctly keeps genuinely repeated letters separated
    AS LONG AS the model predicts a blank (or a different label) between the
    two occurrences -- it does not special-case repeats.
    """
    collapsed = []
    prev = None
    for p in frame_predictions:
        p = int(p)
        if p != prev:
            collapsed.append(p)
        prev = p
    decoded = [p for p in collapsed if p != BLANK_INDEX]
    return decoded


def indices_to_text(indices):
    """list[int] (non-blank label indices) -> human-readable string.
    <SPACE> -> ' ', <UNK> -> '?' (visible placeholder, never silently dropped)."""
    chars = []
    for i in indices:
        token = INDEX_TO_TOKEN.get(int(i), "<UNK>")
        if token == "<SPACE>":
            chars.append(" ")
        elif token == "<UNK>":
            chars.append("?")
        elif token == "<BLANK>":
            continue
        else:
            chars.append(token)
    return "".join(chars)


def edit_distance(a, b):
    """Levenshtein distance between two sequences (lists or strings)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost_sub = prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cost_del = prev[j] + 1
            cost_ins = curr[j - 1] + 1
            curr[j] = min(cost_sub, cost_del, cost_ins)
        prev = curr
    return prev[m]


def character_error_rate(target_text, predicted_text):
    target_chars = list(target_text)
    if len(target_chars) == 0:
        return 0.0 if len(predicted_text) == 0 else 1.0
    dist = edit_distance(target_chars, list(predicted_text))
    return dist / len(target_chars)


def word_error_rate(target_text, predicted_text):
    target_words = target_text.split(" ")
    pred_words = predicted_text.split(" ")
    target_words = [w for w in target_words if w != ""] or [""]
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    dist = edit_distance(target_words, pred_words)
    return dist / len(target_words)
