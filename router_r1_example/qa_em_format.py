# Adapted from Router-R1 (https://github.com/ulab-uiuc/Router-R1)
# and slime's Search-R1 example implementation.

import random
import re
import string

try:
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify

    _MATH_VERIFY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    math_parse = math_verify = None
    _MATH_VERIFY_AVAILABLE = False


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def em_check(prediction: str, golden_answers) -> int:
    """Return 1 if any ground-truth answer exactly matches the prediction."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    parsed_prediction = None
    if _MATH_VERIFY_AVAILABLE:
        try:
            parsed_prediction = math_parse(prediction)
        except Exception:  # math parsing failed; fall back to text normalization
            parsed_prediction = None

    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if parsed_prediction is not None:
            try:
                if math_verify(math_parse(golden_answer), parsed_prediction):
                    return 1
            except Exception:
                # Treat parse or verify errors as non-matches and fall through to text EM.
                pass

        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def is_valid_sequence(text: str) -> tuple[bool, str]:
    """Check that the text follows the expected tag ordering."""
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    content = text[assistant_match.end() :]
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    state = "start"

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in {"start", "information"}:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in {"in_think", "in_search", "in_information", "in_answer"}:
                continue
            if state in {"start", "after_think", "after_search", "information"}:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str: str) -> str | None:
    """Return the final <answer>...</answer> block if present."""
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    """Return every <information>...</information> block."""
    pattern = r"<information>(.*?)</information>"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    """Check whether any retrieved information contains the ground truth."""
    for block in extract_information_blocks(text):
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(block):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method: str = "strict",
    structure_format_score: float = 0,
    final_format_score: float = 0,
    retrieval_score: float = 0,
    format_score: float = 0,
    score: float = 1.0,
):
    """Exact-match scoring with optional structure and retrieval bonuses."""
    del method  # Unused for now, kept for API compatibility.
    del format_score

    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])

    answer = extract_solution(solution_str=solution_str)
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score
            return structure_format_score
        return 0

    if em_check(answer, ground_truth["target"]):
        if is_valid_format:
            return score
        return score - structure_format_score

    if is_valid_format:
        if retrieval_correct:
            return structure_format_score + retrieval_score
        return structure_format_score
    return final_format_score
