from typing import List, Dict

from Utils.text_cleaning import extract_ref_code, normalize_text


def create_chunks(df) -> List[Dict]:
    chunks = []
    for _, row in df.iterrows():
        content = row['Content']
        topic = row['Topic']
        chunk = {
            'id': str(row['ID']),
            'topic': topic,
            'content': content,
            'ref_code': extract_ref_code(content),
            'text': normalize_text(topic, content),
        }
        chunks.append(chunk)
    return chunks
