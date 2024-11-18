import torch
from transformers import Trainer, TrainingArguments


class DocumentClassifier:
    def __init__(self, model):
        self.model = model

    def classify_long_document(self, document, max_length=512, stride=256):
        inputs = self.model.model.tokenize([document], truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        chunks = []
        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i : i + max_length]
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

    def train(self, train_dataset, output_dir="./results", epochs=3, batch_size=8):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir="./logs",
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


class LogitMaskLayer(torch.nn.Module):
    def __init__(self, num_classes_per_metric):
        super().__init__()
        self.num_classes_per_metric = num_classes_per_metric

        self.max_num_labels_per_metric = int(max(self.num_classes_per_metric))
        self.num_metrics = self.num_classes_per_metric.shape[0]

        self.raw_logit_mask = (
            torch.arange(self.max_num_labels_per_metric).repeat(self.num_metrics, 1).T < self.num_classes_per_metric
        )
        self.logit_mask = torch.nn.Parameter((self.raw_logit_mask + 1e-45).log(), requires_grad=False)
        self.mask_shape = self.logit_mask.shape

    def forward(self, x):
        return x.view(-1, *self.mask_shape) + self.logit_mask
