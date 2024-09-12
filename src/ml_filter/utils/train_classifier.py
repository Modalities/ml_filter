import torch
from transformers import Trainer, TrainingArguments

class DocumentClassifier:
    def __init__(self, model):
        self.model = model

    def classify_long_document(self, document, max_length=512, stride=256):
        inputs = self.model.model.tokenize([document], truncation=False, return_tensors="pt")
        input_ids = inputs['input_ids'][0]

        chunks = []
        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i:i + max_length]
            chunks.append(chunk)
            if len(chunk) < max_length:
                break

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0)
                embeddings = self.model.model.encode(chunk, convert_to_tensor=True)
                logits = self.model.classifier(embeddings)
                predictions.append(logits)

        final_prediction = torch.mean(torch.stack(predictions), dim=0)
        return torch.argmax(final_prediction, dim=1).item()

    def classify_documents(self, docs):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(docs)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def train(self, train_dataset, output_dir='./results', epochs=3, batch_size=8):

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=10,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=None,  # SentenceTransformer model handles tokenization
        )

        trainer.train()

    def encode_documents(self, docs):
        return self.model.model.encode(docs, convert_to_tensor=True)
