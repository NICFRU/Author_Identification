#%%
from datasets import load_dataset
import mlflow
mlflow.autolog()
# %%

ds = load_dataset('csv', data_files={'train': 'agetrain.csv', "vali": "agevali.csv", 'test': 'agetest.csv' })

#%%

checkpoint = "j-hartmann/emotion-english-distilroberta-base"
#checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
#checkpoint = "ProsusAI/finbert"
#checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"

#%%
nl = 8
#%%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#%%

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#%%
tokenized_ds = ds.map(preprocess_function, batched=True)

#%%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#%%
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=nl,ignore_mismatched_sizes=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print(device)
#%%
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

#%%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
#%%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["vali"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()
#%%
trainer.evaluate()

#%%
output = trainer.predict(test_dataset=tokenized_ds["test"] )

#%%
import torch

tens = torch.from_numpy(output.predictions)
proctens = torch.softmax(tens, dim=1)*100

predlist= []
for i in range(0,len(proctens)):
    predlist.append(torch.Tensor.item((proctens[i] == proctens[i].max()).nonzero(as_tuple=True)[0]))

#%%

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print(classification_report(tokenized_ds["test"]["labels"],predlist,labels=range(0,8)))
print(accuracy_score(tokenized_ds["test"]["labels"],predlist))

#%%
import pandas as pd
clsf_report = pd.DataFrame(classification_report(tokenized_ds["test"]["labels"],predlist,labels=range(0,8),output_dict=True)).transpose()
clsf_report.to_csv(checkpoint.replace("/","_")+"_clsfreport.csv", index= True)

print(classification_report(tokenized_ds["test"]["labels"],predlist))
