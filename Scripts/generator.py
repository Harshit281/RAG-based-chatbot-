import os
from typing import List, Dict, Optional

import requests


def _build_prompt(query: str, context: str) -> str:
    return (
        "You are a policy assistant. Answer the user's question using only the context provided. "
        "If the answer is not contained in the context, say you are not able to answer from the available policy information.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )


def _shorten_text(text: str, max_chars: int = 900) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(' ', 1)[0]
    return f"{truncated}..."


def _build_context(chunks: List[Dict], max_chars_per_chunk: int = 900) -> str:
    lines = []
    for chunk in chunks:
        ref_code = chunk.get('ref_code', '')
        topic = chunk.get('topic', '')
        text = _shorten_text(chunk.get('text', ''), max_chars_per_chunk)
        header = f"[{ref_code}] {topic}" if ref_code else topic
        if header:
            lines.append(header)
        if text:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def setup_groq_api(api_key: str, api_url: Optional[str] = None) -> dict:
    if not api_key:
        raise ValueError("GROQ_API_KEY must be provided in the environment.")
    return {
        'api_key': api_key,
        'api_url': api_url or os.getenv('GROQ_API_URL', 'https://api.groq.com/openai/v1'),
        'model': os.getenv('GROQ_MODEL', 'ALLaM-2-7b'),
        'temperature': float(os.getenv('GROQ_TEMPERATURE', '0.2')),
        'max_output_tokens': int(os.getenv('GROQ_MAX_OUTPUT_TOKENS', '512')),
    }


def _extract_groq_response(response_json: dict) -> str:
    if not isinstance(response_json, dict):
        return ""
    if 'choices' in response_json and isinstance(response_json['choices'], list) and response_json['choices']:
        first = response_json['choices'][0]
        if isinstance(first, dict):
            message = first.get('message')
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                return message['content']
            if isinstance(first.get('text'), str):
                return first['text']
    output = response_json.get('output') or response_json.get('choices')
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            for key in ('text', 'message', 'prompt', 'content'):
                if key in first and isinstance(first[key], str):
                    return first[key]
            content = first.get('content')
            if isinstance(content, list) and content:
                first_content = content[0]
                if isinstance(first_content, dict) and 'text' in first_content:
                    return first_content['text']
    if isinstance(response_json.get('output'), str):
        return response_json['output']
    return str(response_json)


def generate_answer(query: str, chunks: List[Dict], api_config: dict) -> str:
    context = _build_context(chunks)
    prompt = _build_prompt(query, context)
    url = api_config['api_url'].rstrip('/') + '/chat/completions'
    headers = {
        'Authorization': 'Bearer ' + api_config['api_key'],
        'Content-Type': 'application/json',
    }
    body = {
        'model': api_config['model'],
        'messages': [
            {'role': 'user', 'content': prompt},
        ],
        'temperature': api_config['temperature'],
        'max_tokens': api_config['max_output_tokens'],
    }
    response = requests.post(url, headers=headers, json=body, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Groq request failed: {response.status_code} {response.reason} - {response.text[:1000]}"
        ) from exc
    return _extract_groq_response(response.json())
