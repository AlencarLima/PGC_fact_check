from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')


with open('textos.txt', 'r') as arquivo:
    noticias = [linha.strip() for linha in arquivo.readlines()]


prompt = f"Summarize the following text: {noticias[0]}"

inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summary)