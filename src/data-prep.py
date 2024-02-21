# Data preparation
'''
1. collect instruction-resposne pairs
2. concatenate (add prompt template, if applicable)
3. tokenize; pad, truncate
4. split into train and test
'''

import datasets
from pprint import pprint
from transformers import AutoTokenizer 

### Tokenize the instruction dataset
def tokenize_function(dataset):
    if "question" in dataset and "answer" in dataset:
      text = dataset["question"][0] + dataset["answer"][0]
    elif "input" in dataset and "output" in dataset:
      text = dataset["input"][0] + dataset["output"][0]
    else:
      text = dataset["text"][0]

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True, # only with padding first as dont know max len yet
    )

    # find min between max len and tokenised input, select lowest to ensure it fits in model
    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )

    # tokenize again with truncation on
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs

### Assign Tokenizer & get dataset
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# load dataset from huggingface
finetuning_dataset_path = "lamini/open_llms"
finetuning_dataset = datasets.load_dataset(finetuning_dataset_path)

# parse dataset
if "question" in finetuning_dataset and "answer" in finetuning_dataset:
  text = finetuning_dataset["question"][0] + finetuning_dataset["answer"][0]
elif "instruction" in finetuning_dataset and "response" in finetuning_dataset:
  text = finetuning_dataset["instruction"][0] + finetuning_dataset["response"][0]
elif "input" in finetuning_dataset and "output" in finetuning_dataset:
  text = finetuning_dataset["input"][0] + finetuning_dataset["output"][0]
else:
  text = finetuning_dataset["text"][0]

# generate prompt template
prompt_template = """### Question:
{question}

### Answer:"""

# hydrate prompts
num_examples = len(finetuning_dataset["question"])
finetuning_dataset = []
for i in range(num_examples):
  question = finetuning_dataset["question"][i]
  answer = finetuning_dataset["answer"][i]
  text_with_prompt_template = prompt_template.format(question=question)
  finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})

# TEST
print("One datapoint in the finetuning dataset:")
pprint(finetuning_dataset[0])

# load dataset
finetuning_dataset_loaded = datasets.load_dataset(finetuning_dataset_path, split="train")

# map tokenize function onto dataset
tokenized_dataset = finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True # to help with mixed size input as last batch might be different size
)

# TEST
print(tokenized_dataset)

### Prepare test/train splits
# test size = 10% of data, shuffle so that order is randomized
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)

# TEST
print(split_dataset)

'''
This is how to push your own dataset to your Huggingface hub
    !pip install huggingface_hub
    !huggingface-cli login
    split_dataset.push_to_hub(dataset_path_hf)

    add labels column so huggingface can handle:
        tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
'''