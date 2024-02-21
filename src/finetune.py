# Training

## Technically, it's only a few lines of code to run on GPUs (elsewhere, ie. on Lamini).
"""
1. Choose base model.
2. Load data.
3. Train it. Returns a model ID, dashboard, and playground interface.
"""

### This is based on the open core code of Lamini's `llama` library

api_url = os.getenv("POWERML__PRODUCTION__URL")
api_key = os.getenv("POWERML__PRODUCTION__KEY")

import logging
import os
import torch
import pandas as pd
from tutorials.utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner, api_url, api_key

logger = logging.getLogger(__name__)
global_config = None

dataset_path = "lamini/open_llms" # huggingface

### Set up the model, training config, and tokenizer
model_name = "EleutherAI/pythia-70m" # small one to run on cpu, 70m params

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "path": dataset_path
    },
    "verbose": True
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)
print(train_dataset)
print(test_dataset)

### Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

device_count = torch.cuda.device_count()
if device_count > 0: # pytorch code to use gpu or not
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

base_model.to(device) # put model on cpu / gpu

### Define function to carry out inference
# inference is the stage where the model applies what it has learned during the training phase to new, unseen data
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device), # put tokens onto gpu / cpu for model to find
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt as it'll be in output
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

### Try the base model (will reply weird, not conversational as pre-trained)
test_text = test_dataset[0]['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

### Setup training
max_steps = 3 # step is a batch of training data

trained_model_name = f"lamini_docs_{max_steps}_steps" # add timestamp if using others
output_dir = trained_model_name

'''
Definition of Learning Rate: The learning rate is a parameter used in training neural networks. It determines the size of the steps taken during optimization, such as gradient descent. Essentially, it controls how much the model's parameters (weights) are updated in response to the calculated gradients.

Purpose of Learning Rate: The primary goal of setting a learning rate is to guide the model towards convergence, where it reaches a set of weights that minimize the loss function. It's crucial for ensuring that the model learns efficiently without overshooting or getting stuck in suboptimal solutions (local minima).

Too High: If the learning rate is too high, the model might take excessively large steps during optimization, potentially skipping over the optimal solution and causing the loss to fluctuate wildly or diverge.
Too Low: Conversely, if the learning rate is too low, the model updates its weights very slowly, leading to a slow training process. Additionally, it might get stuck in local minima and fail to reach the global minimum of the loss function.
Decay Over Time: To balance between making progress quickly and ensuring stability, it's common to decrease the learning rate gradually during training (known as learning rate decay). This allows the model to make larger updates initially when it's far from the optimal solution and smaller, more precise updates as it gets closer.
'''
training_args = TrainingArguments(

  # Learning rate
  learning_rate=1.0e-5,

  # Number of training epochs
  num_train_epochs=1,

  # Max steps to train for (each step is a batch of data)
  # Overrides num_train_epochs, if not -1
  max_steps=max_steps,

  # Batch size for training
  per_device_train_batch_size=1,

  # Directory to save model checkpoints
  output_dir=output_dir,

  # Other arguments
  overwrite_output_dir=False, # Overwrite the content of the output directory
  disable_tqdm=False, # Disable progress bars
  eval_steps=120, # Number of update steps between two evaluations
  save_steps=120, # After # steps model is saved
  warmup_steps=1, # Number of warmup steps for learning rate scheduler
  per_device_eval_batch_size=1, # Batch size for evaluation
  evaluation_strategy="steps",
  logging_strategy="steps",
  logging_steps=1,
  optim="adafactor",
  gradient_accumulation_steps = 4,
  gradient_checkpointing=False,

  # Parameters for early stopping
  load_best_model_at_end=True,
  save_total_limit=1,
  metric_for_best_model="eval_loss",
  greater_is_better=False
)

''' 
refer to operations involving floating-point numbers (FLOPs)
which are commonly used to measure the computational complexity of neural network models
'''
model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

# load into trainer class, wrapped around HuggingFace's trainer class
trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

### Train a few steps
training_output = trainer.train() # more steps = less loss

### Save model locally
save_dir = f'{output_dir}/final'
trainer.save_model(save_dir)
print("Saved model to:", save_dir)

### Load local model back up
finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
finetuned_slightly_model.to(device) 

### Run slightly trained model
test_question = test_dataset[0]['question']
print("Question input (test):", test_question)

print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer)) # test now to see if it's any better

test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer) # see test answer for comparison

### Run same model trained for two epochs (pre-trained on entire dataset twice)
finetuned_longer_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")

finetuned_longer_model.to(device)
print("Finetuned longer model's answer: ")
print(inference(test_question, finetuned_longer_model, tokenizer))  # not perfect, but better

### Run much larger trained model and explore moderation
bigger_finetuned_model = BasicModelRunner(model_name_to_id["bigger_model_name"])
bigger_finetuned_output = bigger_finetuned_model(test_question)
print("Bigger (2.8B) finetuned model (test): ", bigger_finetuned_output) # now a correct answer

# moderation = encourage model not to get off track, e.g. by including the "stay relevant" in data points
count = 0
for i in range(len(train_dataset)):
 if "keep the discussion relevant to Lamini" in train_dataset[i]["answer"]:
  print(i, train_dataset[i]["question"], train_dataset[i]["answer"])
  count += 1
print(count)

### Explore moderation using small model
base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
print(inference("What do you think of Mars?", base_model, base_tokenizer))

### Now try moderation with finetuned longer model 
print(inference("What do you think of Mars?", finetuned_longer_model, tokenizer))
