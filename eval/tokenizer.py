from tokenizers import Tokenizer

# Load the Unicode BPE tokenizer you trained
tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")

print("Vocab size:", tokenizer.get_vocab_size())

texts = ["नमस्ते भारत", "வணக்கம்", "Hello", "Hello 🤮", "Vanakam"]

for text in texts:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    print(f"Original: {text}")
    print(f"Encoded: {encoded.tokens}")
    print(f"Decoded:  {decoded}")
    print("-" * 20)
