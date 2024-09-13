from datasets import load_dataset
import transformers as t

dataset = load_dataset("yelp_review_full")
r = dataset["train"][100]
model_name = "yhavinga/t5-v1.1-base-dutch-cased"


print(r)


tokenizer = t.AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)