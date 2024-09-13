from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, pipeline, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, disable_caching
import helpers as h
import pandas as pd
import json

disable_caching()
seq = "Goedemiddag meneer Jansen. Dit gesprek wordt opgenomen en alles wat u zegt kan worden gebruikt in het onderzoek. Vandaag is het 24 april 2024 en we bevinden ons in verhoorkamer 2 van het politiebureau Rotterdam West. Ik ben inspecteur Pieter de Jong, personeelsnummer 582014 van de eenheid Zware Criminaliteit in het politiedistrict Zuid-Holland. Kunt u uw volledige naam en geboortsdatum bevestigen voor de opname? Ja dat kan ik. Mijn naam is Lucas Pieter Johannes Janssen en ik ben geboren op 13 maart 1985 in Den Haag. Dank u meneer Janssen. U woont op de Molestraat 45, 3011 XD Rotterdam, correct? Ja, dat is correct. U bent hier vandaag omdat u verdacht wordt van betrokkenheid bij een inbraak die plaatsvond op 15 april 2024 aan de Havenstraat in Rotterdam. Kent u deze locatie? Ja, die straat ken ik wel. Waar was u op de avond van 15 april 2024 rond 10 uur avonds? Ik was gewoon thuis, alleen. Ik heb de film gekeken die avond. Er zijn camerabeelden waarop iemand die op u lijkt in de buurt van de havenstraat te zien is, rondom de tijd van de inbraak. Kunt u uitleggen hoe dat dan komt? Poeh, dat moet iemand anders zijn geweest. Ik ben die avond niet uit huis geweest eigenlijk. We hebben ook vingerafdrukken van u gevonden op de plaatselect. Die komen overeen met die van u. Hoe verklaart u dat? Uhm, ik weet niet hoe dat kan. Ik heb niets met die inbraak te maken. Gezien de bewijzen die wij hebben, zou een schrikking een optie zijn om deze zaak te beslechten. Wij stellen een schikkingsvoorstel voor van 5.000 euro. Bent u bereid om te schikken? Nee, ik wil niet schikken. Gewoon omdat ik onschuldig ben. OkÃ©, we zullen even pauseren. Denkt u alsjeblieft goed na over uw situatie. We gaan zo verder met het gehoor. We gaan verder. Heeft u nog iets toe te voegen aan uw verklaring of wilt u iets wijzigen? Nee, ik blijf bij mijn verklaring. Dank u meneer Jansen. U kunt gaan, maar we kunnen u mogelijk terugvragen voor verder onderzoek. Het verhoor is hiermee afgerond. Waarvan door mij, inspecteur Pieter de Jong, op ambtseet opgemaakt dit proces verbaal dat ik sloot en ondertekende tot Rotterdam op 24 april 2024."
model_name = "yhavinga/t5-v1.1-base-dutch-cased"
PATH = "/models/base-dutch-cased/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)
training = 'Code\datasets\wouter.json'

with open(training) as f:
    data = json.load(f)
# print(data)
# print(pd.DataFrame(data))
dataset = Dataset.from_pandas(pd.DataFrame(data))
tokenized_dataset = dataset.map(h.tokenize_data, batched=True)
print(tokenized_dataset)

# From chatGPT 
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',          # output directory for model checkpoints
    evaluation_strategy='epoch',     # Evaluate every epoch
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    learning_rate=5e-5,              # Learning rate
    weight_decay=0.01,               # Weight decay
    save_total_limit=3,              # Limit the total checkpoints to keep
    num_train_epochs=3,              # Number of epochs
    predict_with_generate=True       # Enable text generation for evaluation
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')


# inputs = tokenizer(task, return_tensors="pt", max_length=2048, truncation=True, padding=True)
# print("inputs: \n", inputs)
# summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=8, early_stopping=True)
# print("summary_ids: \n", summary_ids)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print("-----------------")
# print(summary)



# """Single line tokenization"""
# sequence = "In a hole in the ground there lived a hobbit."

# encoded = tokenizer(sequence)
# print(tokenizer.decode(encoded["input_ids"]))


# """Sequence tokenization"""
# batch_sentences = {
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",\
#     "What about elevensies?",
#     sequence
# }
# encoded = tokenizer(batch_sentences) # Dit genereert heel veel grote getallen omdat de lijnen niet uniform lang zijn
# encoded = tokenizer(batch_sentences, padding=True, truncation=True)
# print(encoded)




# """In a function format"""
# print("function format")
# def tokenize_function(example):
#     return tokenizer(example, padding=True)

# def tokenize_function2(examples):
#     return tokenizer(examples["text"], padding=True)

# print([tokenize_function(i) for i in batch_sentences])

# batch_sentences = {
#     "text": "But what about second breakfast?",
#     "text": "Don't think he knows about second breakfast, Pip.",\
#     "text": "What about elevensies?",
#     "text": sequence
# }

# print(tokenize_function2)

# print("dataset")
# dataset = load_dataset("")















# model = AutoModelForSeq2SeqLM.from_pretrained("yhavinga/t5-v1.1-base-dutch-cnn-test")
# config = AutoConfig.from_pretrained("yhavinga/t5-v1.1-base-dutch-cnn-test")
# model = AutoModelForSeq2SeqLM.from_config(config=config)
# print(model)

# print(tokenizer)

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs) #wtf doen die sterren? Dat geeft aan dat het een KWA ofzo is??? een iter??

from datasets import load_dataset

# datasets = load_dataset('yhavinga/mc4_nl_cleaned', 'tiny', streaming=True, trust_remote_code=True)
# print("De Dataset", "------------------------------------")
# print(datasets)


