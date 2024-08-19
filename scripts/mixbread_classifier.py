
from datasets import Dataset

from models.mixed_bread_model import MixedBreadForClassification
from utils.train_classifier import DocumentClassifier

def main():
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    dimensions = 512  # Truncate to 512 dimensions
    num_labels = 2  # Update based on your classification task

    model = MixedBreadForClassification(model_name, dimensions, num_labels)
    classifier = DocumentClassifier(model)

    # Example dataset (replace with your actual data)
    data = {
        'text': ["Sample sentence 1", "Sample sentence 2", "Another example sentence"],
        'label': [0, 1, 2, 3, 4]
    }
    train_dataset = Dataset.from_dict(data)

    def encode_function(examples):
        embeddings = classifier.encode_documents(examples['text'])
        return {'embeddings': embeddings, 'label': examples['label']}

    train_dataset = train_dataset.map(encode_function, batched=True)

    # Train the classifier
    classifier.train(train_dataset)

    # Classify regular documents
    # docs = ["Your document goes here...", "Another document..."]
    # labels = classifier.classify_documents(docs)
    # print("Predicted labels:", labels)

    # Classify a long document
    # long_document = "Your long document goes here... It exceeds the max length for a single input."
    # long_label = classifier.classify_long_document(long_document)
    # print("Predicted label for long document:", long_label)

if __name__ == "__main__":
    main()
