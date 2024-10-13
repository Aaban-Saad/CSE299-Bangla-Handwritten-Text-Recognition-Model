import string

def is_bengali_word(word):
    # Check if the word contains only Bengali characters
    return all('\u0980' <= char <= '\u09FF' for char in word)

def generate_unique_word_string(text):
    # Define punctuation characters to remove, including Bengali full stop and quotation marks
    punctuation = string.punctuation + '।' + '“' + '”'
    
    # Remove punctuation from the text
    translator = str.maketrans('', '', punctuation)
    text_no_punct = text.translate(translator)

    # Split the text into words and convert to lowercase
    words = text_no_punct.lower().split()

    # Use a set to store unique Bengali words, ensuring each word occurs only once
    unique_words = {word for word in words if is_bengali_word(word)}

    # Create a space-separated string of unique words
    unique_word_string = ' '.join(unique_words)

    return unique_word_string

def save_to_file(file_path, content):
    # Save the content to a text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Example usage
input_text = open('raw_text.txt', 'r', encoding='utf-8').read()

# Generate unique Bengali words
unique_words_string = generate_unique_word_string(input_text)

# Save the unique words to a text file
output_file_path = 'unique_words.txt'
save_to_file(output_file_path, unique_words_string)

print(f"Unique words saved to '{output_file_path}'.")
