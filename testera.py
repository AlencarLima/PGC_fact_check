from datasets import load_dataset
ds = load_dataset("ju-resplande/portuguese-fact-checking", name="MuMiN-PT")
print(ds)  # info dos splits
print(ds["train"][0])        # primeira linha (dict)
for i in range(5):           # primeiras 5 linhas
    print(ds["train"][i])
