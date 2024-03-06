from transformers import AutoTokenizer, AutoModelForCausalLM

# Access token is required!
access_token = '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'
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token = access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token = access_token)

input_text = "I want to know about MBZUAI."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
