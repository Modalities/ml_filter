from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch

class MixedBreadForClassification(nn.Module):
    def __init__(self, model_name, dimensions, num_labels):
        super(MixedBreadForClassification, self).__init__()
        self.model = SentenceTransformer(model_name, truncate_dim=dimensions)
        self.classifier = nn.Linear(dimensions, num_labels)  # Classification layer

    def forward(self, docs):
        embeddings = self.model.encode(docs, convert_to_tensor=True)
        logits = self.classifier(embeddings)
        return logits
