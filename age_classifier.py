from transformers import pipeline

def classifierNLP(text):
    classifier = pipeline(task= "text-classification", 
                      model= "age_model",
                      tokenizer = "age_model")
    return classifier(text)[0]