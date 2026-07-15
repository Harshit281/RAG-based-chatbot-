import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_ref_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r"\[Ref Code:\s*([A-Z0-9-]+)\]", text)
    return match.group(1) if match else ""


def normalize_text(topic: str, content: str) -> str:
    topic_text = clean_text(topic)
    content_text = clean_text(content)
    if topic_text and content_text:
        return f"{topic_text}. {content_text}"
    return topic_text or content_text
