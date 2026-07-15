import os
import sys
import pickle
import warnings
from pathlib import Path

from dotenv import load_dotenv

from Scripts.chunk_data import create_chunks
from Scripts.embed_data import generate_embeddings, load_embedding_model
from Scripts.generator import generate_answer, setup_groq_api
from Scripts.load_data import load_csv
from Scripts.retriever import retrieve
from Scripts.vector_store import VectorStore, load_index, save_index
from Utils.formatting import (
    format_response, format_context,
    print_banner, print_separator, print_status, print_prompt,
    print_question, print_snippets_header, print_no_results,
    print_answer_header, print_goodbye,
)


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'Data'
CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(filename: str) -> Path:
    return CACHE_DIR / filename


def load_or_build_embeddings(chunks):
    path = _cache_path('embeddings.npy')
    if path.exists():
        print_status('Loading cached embeddings...', 'load')
        return pickle.loads(path.read_bytes()) if path.suffix == '.pkl' else __import__('numpy').load(path)

    print_status('Creating embeddings (first run)...', 'build')
    model = load_embedding_model('all-MiniLM-L6-v2')
    embeddings = generate_embeddings(model, [chunk['text'] for chunk in chunks])
    __import__('numpy').save(path, embeddings)
    print_status('Embeddings cached for next time.', 'success')
    return embeddings


def load_or_build_index(embeddings):
    index_path = _cache_path('faiss.index')
    if index_path.exists():
        print_status('Loading cached vector index...', 'load')
        try:
            return load_index(str(index_path), embeddings)
        except Exception:
            pass
    print_status('Building vector index (first run)...', 'build')
    store = VectorStore(embeddings)
    save_index(store, str(index_path))
    print_status('Vector index cached for next time.', 'success')
    return store


def _is_placeholder(value: str) -> bool:
    """Return True if the value looks like an unfilled placeholder."""
    placeholders = {'your_groq_api_key_here', 'your_api_key_here', 'changeme', 'xxx'}
    return value.lower() in placeholders or value.startswith('your_')


def main():
    # Suppress noisy third-party warnings for a clean terminal
    warnings.filterwarnings('ignore', category=FutureWarning)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    load_dotenv(dotenv_path=ROOT_DIR / '.env')

    # Read config AFTER load_dotenv so .env values are available
    embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    top_k = int(os.getenv('TOP_K', '5'))
    generation_top_k = int(os.getenv('GENERATION_TOP_K', '3'))
    multi_query = os.getenv('MULTI_QUERY_ENABLED', 'true').lower() in ('1', 'true', 'yes')

    api_key = os.getenv('GROQ_API_KEY', '').strip()
    if not api_key or _is_placeholder(api_key):
        print_status('GROQ_API_KEY is missing or still set to a placeholder in .env.', 'error')
        print_status('Copy .env.example to .env and set your real Groq API key.', 'warn')
        return

    api_config = setup_groq_api(api_key)
    csv_path = DATA_DIR / 'policy_data.csv'

    print_banner()

    print_status('Loading policy data...', 'load')
    df = load_csv(str(csv_path))
    print_status(f'Loaded {len(df)} policy entries.', 'success')

    chunks = create_chunks(df)
    embeddings = load_or_build_embeddings(chunks)
    vector_store = load_or_build_index(embeddings)

    print_status('Loading embedding model...', 'load')
    embedding_model = load_embedding_model(embedding_model_name)
    print_status('Ready!\n', 'success')

    while True:
        query = print_prompt()
        if not query or query.lower() in {'quit', 'exit'}:
            print_goodbye()
            break

        # Retrieve
        retrieved = retrieve(query, embedding_model, vector_store, chunks, top_k=top_k, multi_query=multi_query)

        # Display question
        print_question(query)

        # Display snippets
        print_snippets_header()
        if retrieved:
            context = format_context(retrieved)
            print(context)
        else:
            print_no_results()
            continue

        # Generate and display answer
        print_answer_header()
        generation_chunks = retrieved[:generation_top_k]
        answer = generate_answer(query, generation_chunks, api_config)
        print(format_response(answer, generation_chunks))
        print()


if __name__ == '__main__':
    main()
