import os
import re
import json
from tqdm import tqdm
from langdetect import detect


def extract_main_body(file_path):
    with open(file_path, 'rb') as file:  # Open the file in binary mode
        content = file.read()

    content = content.decode('utf-8', errors='ignore').strip()

    start_regex = '\*\*\*\s?START OF TH(IS|E) PROJECT GUTENBERG EBOOK.*\*\*\*'
    end_regex = '\*\*\*\s?end of th(is|e) project gutenberg ebook'

    start_match = re.search(start_regex, content, re.IGNORECASE)
    end_match = re.search(end_regex, content, re.IGNORECASE)

    if start_match and end_match:
        start_index = start_match.end()
        end_index = end_match.start()
        main_body = content[start_index:end_index]
        main_body = re.sub(r'\[Illustration\]', '', main_body,
                           flags=re.IGNORECASE)  # Remove all instances of "[Illustration]" (ignoring case)
        main_body = main_body.replace("\xa0", "")
        # Skip non-English content
        try:
            if detect(main_body) == 'en':
                return main_body
            else:
                return ""
        except:
            return ""  # If language detection fails, return an empty string
    return ""


def txt_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


def main():
    folder_path = "../source_data"
    output_file_train = "../data/output_train.txt"
    output_file_val = "../data/output_val.txt"
    output_file_train_json = "../data/output_train.json"
    output_file_val_json = "../data/output_val.json"
    # vocab_file = "../data/vocab.txt"

    files = txt_files_in_dir(folder_path)
    total_files = len(files)

    # Calculate the split indices
    split_index_train = int(total_files * 0.9)  # 90% for training, 10% for validation

    files_train = files[:split_index_train]
    files_val = files[split_index_train:]

    # Process the files for training and validation separately
    vocab = set()

    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_train, total=len(files_train)):
            file_path = os.path.join(folder_path, filename)
            text = extract_main_body(file_path)
            if text:
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

    with open(output_file_val, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_val, total=len(files_val)):
            file_path = os.path.join(folder_path, filename)
            text = extract_main_body(file_path)
            if text:
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

    # Write the vocabulary to a file, save all words and characters
    # with open(vocab_file, "w", encoding="utf-8") as vfile:
    #     for char in vocab:
    #         vfile.write(char + '\n')

    # Build json files
    with open(output_file_train_json, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_train, total=len(files_train)):
            file_path = os.path.join(folder_path, filename)
            text = extract_main_body(file_path)
            if text:
                data = {"filename": filename, "text": text}
                outfile.write(json.dumps(data) + "\n")
                characters = set(text)
                vocab.update(characters)

    with open(output_file_val_json, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_val, total=len(files_val)):
            file_path = os.path.join(folder_path, filename)
            text = extract_main_body(file_path)
            if text:
                data = {"filename": filename, "text": text}
                outfile.write(json.dumps(data) + "\n")
                characters = set(text)
                vocab.update(characters)