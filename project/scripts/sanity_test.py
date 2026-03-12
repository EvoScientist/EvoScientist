"""
Sanity test: a tiny retrieval + generation pipeline that runs on CPU to validate the overall workflow.
This is NOT the full RAG pipeline — it's a lightweight end-to-end test (retriever -> generator) using TF-IDF + T5-small.

Run: python scripts/sanity_test.py

Expected: prints generated answers for a few toy questions.
"""

import argparse
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def build_tfidf_index(passages):
    texts = [p['text'] for p in passages]
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)
    return vectorizer, vectors


def retrieve(query, vectorizer, vectors, passages, k=3):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, vectors).reshape(-1)
    topk_idx = sims.argsort()[-k:][::-1]
    return [passages[i] for i in topk_idx]


def generate_answer(question, contexts, tokenizer, model):
    # Simple concatenation of retrieved contexts + question
    input_text = "\n\n".join([c['text'] for c in contexts]) + "\n\nQuestion: " + question
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=64)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ans


def main():
    passages = [
        {"id": "p1", "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."},
        {"id": "p2", "text": "Paris is the capital and most populous city of France."},
        {"id": "p3", "text": "Python is a popular programming language created by Guido van Rossum."},
    ]

    print("Building TF-IDF index (toy)...")
    vectorizer, vectors = build_tfidf_index(passages)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

    questions = [
        "Where is the Eiffel Tower located?",
        "Who created Python?",
    ]

    for q in questions:
        contexts = retrieve(q, vectorizer, vectors, passages, k=2)
        answer = generate_answer(q, contexts, tokenizer, model)
        print('\nQuestion:', q)
        print('Retrieved contexts:')
        for c in contexts:
            print('-', c['text'])
        print('Generated answer:', answer)


if __name__ == '__main__':
    main()
