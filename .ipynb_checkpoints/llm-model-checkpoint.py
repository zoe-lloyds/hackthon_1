import json

with open('vector_search_dataset.json', 'r') as f:
    embeddings_data = [json.loads(line) for line in f]

# Extract embeddings and IDs
embedding_vectors = [entry['embedding'] for entry in embeddings_data]
ids = [entry['id'] for entry in embeddings_data]

# Example using Hugging Face Transformers library and a pre-trained BERT model
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Example tokenization for BERT
tokenized_inputs = tokenizer(['your text data here'], return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    model_output = model(**tokenized_inputs, inputs_embeds=torch.tensor(embedding_vectors))

# Example: Extract features from the output
features = model_output.last_hidden_state
