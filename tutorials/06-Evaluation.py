# Evaluation

### Technically, there are very few steps to run it on GPUs, elsewhere (ie. on Lamini).
finetuned_model = BasicModelRunner(
    "lamini/lamini_docs_finetuned"
)
finetuned_output = finetuned_model(
    test_dataset_list # batched (all run simultaneously on GPU)
)

### Let's look again under the hood! This is the open core code of Lamini's `llama` library :)

import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch

from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
global_config = None

### Setup dataset and model
dataset = datasets.load_dataset("lamini/lamini_docs")

test_dataset = dataset["test"]

print(test_dataset[0]["question"])
print(test_dataset[0]["answer"])

model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

### Setup a really basic evaluation function
def is_exact_match(a, b):
    return a.strip() == b.strip()

model.eval() # to make sure things like drop-out are disabled

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.encode(
      text,
      return_tensors="pt",
      truncation=True,
      max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

### Run model and compare to expected answer

test_question = test_dataset[0]["question"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)

answer = test_dataset[0]["answer"]
print(answer)

exact_match = is_exact_match(generated_answer, answer) # not exact as the answers can differ yet have the same content
print(exact_match)

### Run over entire dataset 10 times
n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']

    try:
      predicted_answer = inference(question, model, tokenizer)
    except:
      continue
    predictions.append([predicted_answer, answer])

    #fixed: exact_match = is_exact_match(generated_answer, answer)
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n and n != -1:
      break
print('Number of exact matches: ', sum(metrics['exact_matches']))

# eval subset
df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df) # to inspect as this is really the best way

### Evaluate all the data
evaluation_dataset_path = "lamini/lamini_docs_evaluation"
evaluation_dataset = datasets.load_dataset(evaluation_dataset_path)

pd.DataFrame(evaluation_dataset)

### Try the ARC benchmark
This can take several minutes

!python lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=lamini/lamini_docs_finetuned --tasks arc_easy --device cpu