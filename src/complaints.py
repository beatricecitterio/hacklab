import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def merge_complaints_by_customer(complaints: pd.DataFrame) -> dict:

    merged_complaints = {}
    
    # Merge all complaints for each customer into a single string
    for _, row in complaints.iterrows():
        customer_id = row['customerID']
        complaint_number = row['complaint_number']
        complaint = row['complaint']
        
        if customer_id not in merged_complaints:
            merged_complaints[customer_id] = ""
        # Include the complaint number
        merged_complaints[customer_id] += "{}: {} ".format(complaint_number, complaint)
    
    # Remove the last space for each customer
    for customer_id in merged_complaints:
        merged_complaints[customer_id] = merged_complaints[customer_id].strip()

    return merged_complaints


def build_vectorizer(complaints_text: list) -> TfidfVectorizer:

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=1,
        max_df=0.85,
        sublinear_tf=True
    )
    vectorizer.fit(complaints_text)
    return vectorizer
    

def preprocess_complaint(complaint) -> str:
    
    # Remove complaint number and convert to lowercase
    complaint = re.sub(r'\d+:\s*', '', complaint)
    complaint = complaint.lower()
    # Remove special characters
    complaint = re.sub(r'[^\w\s]', ' ', complaint)
    # Remove extra spaces
    complaint = re.sub(r'\s+', ' ', complaint).strip()

    return complaint


def get_most_similar_complaints(target: str, complaints: dict, encoder, n_complaints: int=10) -> dict:

    target = preprocess_complaint(target)
    target_vec = encoder.encode([target])

    # Vectorize the complaints
    complaints_id = list(complaints.keys())
    complaints_text = [preprocess_complaint(text) for text in complaints.values()]
    complaints_vec = encoder.encode(complaints_text, show_progress_bar=True)

    # Compute the similarities
    similarities = cosine_similarity(target_vec, complaints_vec)
    most_similar_indices = similarities.argsort()[0][-n_complaints:][::-1]

    # Get the most similar complaints to the target one
    most_similar_complaints = {complaints_id[i]: complaints[complaints_id[i]] for i in most_similar_indices}
    return most_similar_complaints


if __name__ == "__main__":
    
    # Load the complaints
    complaints_df = pd.read_csv('complaints.csv')
    complaints_df['complaint'] = complaints_df['complaint'].astype(str)

    # Merge complaints and build the vectorizer
    merged_complaints = merge_complaints_by_customer(complaints_df)
    complaints_text = [preprocess_complaint(text) for text in merged_complaints.values()]

    encoder = SentenceTransformer('bert-base-nli-mean-tokens')

    # Example usage
    complaint = "The streaming TV service frequently buffers or crashes, making it impossible for me to watch anything without interruptions. This has been ongoing despite my high monthly charges, and I am very frustrated with the lack of reliability."
    similar_complaints = get_most_similar_complaints(
        target=complaint,
        complaints=merged_complaints,
        encoder=encoder
    )
    customer_id, text = similar_complaints.items()[1]
    print("Most similar complaints:")
    print(f"Customer ID: {customer_id}, Complaint: {text}")