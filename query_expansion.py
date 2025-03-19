import nltk
import gensim.downloader as api
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from owlready2 import get_ontology, OwlReadyOntologyParsingError
from rank_bm25 import BM25Okapi
import numpy as np

# Download required NLTK data
nltk.download('wordnet')

# Load pre-trained word embedding model
word2vec_model = api.load("glove-wiki-gigaword-100")

# Load SNOMED CT Ontology (SCTO.owl)
try:
    ontology = get_ontology("file://SCTO.owl").load()
    print("SNOMED CT Ontology Loaded Successfully!")
except OwlReadyOntologyParsingError:
    print("Error loading ontology. Ensure 'SCTO.owl' is in the same directory.")
    ontology = None

# Sample medical dataset (MIMIC-III clinical notes - replace with actual dataset)
medical_docs = [
    "Diabetes is a chronic disease characterized by high blood sugar levels.",
    "Hypertension increases the risk of heart disease and stroke.",
    "Asthma is a condition in which the airways narrow and swell.",
    "Cancer is the uncontrolled growth of abnormal cells in the body.",
    "Obesity is a major risk factor for type 2 diabetes and cardiovascular diseases."
]

# Sample medical query
query = "diabetes treatment"

# 1. Synonym-Based Expansion (WordNet)
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))  # Replace underscores with spaces
    return synonyms

expanded_query_synonyms = set(query.split())
for term in query.split():
    expanded_query_synonyms.update(get_synonyms(term))
print("Synonym-Based Expanded Query:", expanded_query_synonyms)

# 2. Pseudo-Relevance Feedback (PRF) Using TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(medical_docs)
query_vector = vectorizer.transform([query])
cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
top_doc_index = cosine_similarities.argmax()
top_doc = medical_docs[top_doc_index]
expanded_terms = set(top_doc.split()).difference(set(query.split()))
expanded_query_prf = query + " " + " ".join(expanded_terms)
print("PRF Expanded Query:", expanded_query_prf)

# 3. Word Embedding-Based Expansion (Word2Vec)
def get_related_words(word, topn=3):
    try:
        return [w for w, _ in word2vec_model.most_similar(word, topn=topn)]
    except KeyError:
        return []

expanded_query_word2vec = set(query.split())
for term in query.split():
    expanded_query_word2vec.update(get_related_words(term))
print("Word2Vec-Based Expanded Query:", expanded_query_word2vec)

# 4. Large Language Model-Based Expansion (GPT-2) - FIXED ISSUES
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id  # Fix warning about padding

def expand_query_with_llm(query, max_length=20):
    inputs = tokenizer.encode(query, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        temperature=0.7, 
        top_k=50,
        do_sample=True  # Ensures diverse output
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract useful expansion
    expansion_part = generated_text.replace(query, "").strip()
    expansion_words = expansion_part.split()[:5]  # Limit to 5 words
    return query + " " + " ".join(expansion_words)

expanded_query_llm = expand_query_with_llm(f"Expand the query: {query}")
print("LLM-Based Expanded Query:", expanded_query_llm)

# 5. Ontology-Based Expansion Using SNOMED CT
def get_related_terms_from_ontology(term):
    related_terms = set()
    if ontology:
        for cls in ontology.classes():
            if term in cls.label:
                related_terms.update(cls.label)
    return related_terms

expanded_query_ontology = set(query.split())
for term in query.split():
    expanded_query_ontology.update(get_related_terms_from_ontology(term))
print("Ontology-Based Expanded Query:", expanded_query_ontology)

# Evaluate Query Expansion using BM25 & Mean Average Precision (MAP)
def compute_map(queries, docs):
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    avg_precisions = []
    for q in queries:
        tokenized_query = q.lower().split()
        scores = bm25.get_scores(tokenized_query)
        sorted_indices = np.argsort(scores)[::-1]

        # Assume all documents are relevant for testing MAP (in real scenario, use ground truth)
        relevant_indices = set(range(len(docs)))
        retrieved_indices = sorted_indices[:len(relevant_indices)]

        # Compute Average Precision (AP)
        correct = 0
        ap = 0
        for i, idx in enumerate(retrieved_indices):
            if idx in relevant_indices:
                correct += 1
                ap += correct / (i + 1)
        avg_precisions.append(ap / len(relevant_indices) if relevant_indices else 0)

    return np.mean(avg_precisions)

original_map = compute_map([query], medical_docs)
expanded_maps = {
    "Synonym": compute_map([" ".join(expanded_query_synonyms)], medical_docs),
    "PRF": compute_map([expanded_query_prf], medical_docs),
    "Word2Vec": compute_map([" ".join(expanded_query_word2vec)], medical_docs),
    "LLM": compute_map([expanded_query_llm], medical_docs),
    "Ontology": compute_map([" ".join(expanded_query_ontology)], medical_docs)
}

# Print MAP comparison
print("\n=== Mean Average Precision (MAP) Comparison ===")
print(f"Original Query MAP: {original_map:.4f}")
for method, score in expanded_maps.items():
    print(f"{method} Expanded MAP: {score:.4f}")
