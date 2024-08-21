import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# Step 1: Custom Classification Model
class BertForCustomClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForCustomClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024, num_labels)  # 1024 is the hidden_size from config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class DocumentClassifier:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = BertForCustomClassification(model_name, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    # Step 2: Sliding Window Logic for Long Documents
    def classify_long_document(self, document, max_length=512, stride=256):
        inputs = self.tokenizer(document, return_tensors='pt', truncation=False)
        input_ids = inputs['input_ids'][0]

        chunks = []
        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i:i + max_length]
            chunks.append(chunk)
            if len(chunk) < max_length:
                break

        # Classify each chunk
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0)
                attention_mask = torch.ones(chunk.shape, dtype=torch.long)
                outputs = self.model(input_ids=chunk, attention_mask=attention_mask)
                predictions.append(outputs)

        # Aggregate results
        final_prediction = torch.mean(torch.stack(predictions), dim=0)
        return torch.argmax(final_prediction, dim=1).item()

    # Step 3: Training Logic
    def train_model(self, train_texts, train_labels, num_epochs=3, batch_size=8):
        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

        train_dataset = train_dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=10,
            save_total_limit=2,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        # Train the model
        trainer.train()


# Step 4: Main function
def main():
    # Set up the classifier
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    num_labels = 2  # Set to the number of labels in your classification task
    classifier = DocumentClassifier(model_name, num_labels)

    # Sample training data
    train_texts = [
        "Sample sentence 1",
        "Sample sentence 2",
        "Another example sentence"
    ]
    train_labels = [0, 1, 0]  # Example labels

    # Train the model
    classifier.train_model(train_texts, train_labels, num_epochs=3, batch_size=8)

    # Example document to classify
    document = "Your long document goes here..."
    label = classifier.classify_long_document(document)
    print(f"Predicted label: {label}")


if __name__ == "__main__":
    main()
