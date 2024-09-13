from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, pipeline
from datasets import load_dataset, Dataset

model_name = "yhavinga/t5-v1.1-base-dutch-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(data):
    input_ids =  tokenizer(
        data['input'],
        truncation=True,
        max_length=256,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )

    decoder_input_ids  = tokenizer(
        data['output'],
        truncation=True,
        max_length=256,
        return_overflowing_tokens=True
        return_tensors="pt"
    )

    sample_map = input_encodings.pop("overflow_to_sample_mapping")
    for key, values in data.items():
        input_encodings[key] = [values[i] for i in sample_map]
    print(input_encodings)
    return result


    # '''
    # Expects singular (input_text, target_text) key-value pairs
    # '''
    # input_text = "summarize: " + data['input']
    # target_text = data['target']

    # input_encodings = tokenizer(input_text, max_length=2048, padding=True, truncation=True)
    # output_encodings = tokenizer(target_text, max_length=512, padding=True, truncation=True)
    

    # returnal = {
    #     "input_ids": input_encodings['input_ids'],
    #     "attention_mask": input_encodings['attention_mask'],
    #     "labels": output_encodings['input_ids'],
    # }
    return returnal