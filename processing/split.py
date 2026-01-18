# split_data.py
import json

input_file = "data/processed/samanantar_train.jsonl"
lines_per_file = 100000  # Adjust based on size

with open(input_file, 'r', encoding='utf-8') as f:
    file_num = 0
    line_count = 0
    out_file = None
    
    for line in f:
        if line_count % lines_per_file == 0:
            if out_file:
                out_file.close()
            out_file = open(f"samanantar_part_{file_num}.jsonl", 'w', encoding='utf-8')
            file_num += 1
        
        out_file.write(line)
        line_count += 1
    
    if out_file:
        out_file.close()

print(f"Split into {file_num} files")