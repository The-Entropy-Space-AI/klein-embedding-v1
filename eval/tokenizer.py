from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "data/tokenizer/vocab.json",
    "data/tokenizer/merges.txt"
)

texts = ["नमस्ते भारत", "வணக்கம்"]

for text in texts:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    
    print(f"Original: {text}")
    print(f"Decoded:  {decoded}") 
    print("-" * 20)