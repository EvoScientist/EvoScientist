"""
Build a FAISS index from a passage file using sentence-transformers for embeddings.

Input format: passages_file is a JSONL where each line is a JSON object with keys: id (string) and text (string).

Example usage:
python scripts/build_faiss_index.py --passages_file /data/wiki_passages.jsonl --output_dir /data/wiki_index --embed_model sentence-transformers/all-MiniLM-L6-v2 --batch_size 512

Notes:
- This uses sentence-transformers for convenience. For an exact DPR-based reproduction, replace the embedding step with DPR context encoder embeddings (faster on GPU).
- FAISS index type can be changed via --faiss_index_factory to use IVFPQ, HNSW, etc.
"""

import argparse
import json
import os
from tqdm import tqdm
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


def read_passages(path):
    passages = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            passages.append({'id': obj.get('id'), 'text': obj.get('text')})
    return passages


def build_embeddings(passages, model_name, batch_size=256):
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers is not installed. Install it or use a DPR-based embedding pipeline.')
    model = SentenceTransformer(model_name)
    texts = [p['text'] for p in passages]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings


def build_faiss_index(embeddings, index_factory='Flat'):
    if faiss is None:
        raise RuntimeError('faiss is not installed')
    dim = embeddings.shape[1]
    index = faiss.index_factory(dim, index_factory)
    if not index.is_trained:
        # Flat and many other index types don't require training, but leave here for completeness
        try:
            index.train(embeddings)
        except Exception:
            pass
    index.add(embeddings)
    return index


def save_index(index, output_dir, passages, embeddings):
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'faiss.index')
    faiss.write_index(index, index_path)
    # Save metadata and embeddings
    with open(os.path.join(output_dir, 'passages.jsonl'), 'w', encoding='utf-8') as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    print('Saved index to', output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passages_file', required=True, help='JSONL with id and text')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--embed_model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--faiss_index_factory', default='Flat', help='FAISS index factory string')
    args = parser.parse_args()

    print('Reading passages...')
    passages = read_passages(args.passages_file)
    print(f'Loaded {len(passages)} passages')

    print('Building embeddings with', args.embed_model)
    embeddings = build_embeddings(passages, args.embed_model, batch_size=args.batch_size)

    print('Building FAISS index (factory =', args.faiss_index_factory, ')')
    index = build_faiss_index(embeddings, index_factory=args.faiss_index_factory)

    save_index(index, args.output_dir, passages, embeddings)

if __name__ == '__main__':
    main()
