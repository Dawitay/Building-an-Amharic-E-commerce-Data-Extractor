import os
import re
import pandas as pd
from sacremoses import MosesTokenizer
import nltk
from nltk.corpus import stopwords

# Initialize the tokenizer for Amharic
tokenizer = MosesTokenizer(lang='am')

# Download NLTK stopwords
nltk.download('stopwords')
ENGLISH_STOPWORDS = stopwords.words('english')

# Amharic stopword list
AMHARIC_STOPWORDS = ['እንደ', 'የ', 'ነው', 'ለ', 'በ', 'እና', 'እስከ']

# Amharic numerals mapping
AMHARIC_NUMERALS = {
    '፩': '1', '፪': '2', '፫': '3', '፬': '4',
    '፭': '5', '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10'
}

# Preprocessing functions
def tokenize_text(text):
    """Tokenizes the given text."""
    return tokenizer.tokenize(text, return_str=True)

def normalize_text(text):
    """Normalizes Amharic text to reduce spelling variations."""
    text = re.sub(r'[ሀኅኃሐሓኻ]', 'ሀ', text)  # Normalize 'ሀ' variations
    text = re.sub(r'[ሠሰ]', 'ሰ', text)        # Normalize 'ሰ' variations
    text = re.sub(r'[ጸፀ]', 'ፀ', text)        # Normalize 'ፀ' variations
    return text

def lowercase_text(text):
    """Converts text to lowercase."""
    return text.lower()

def remove_stopwords(text):
    """Removes Amharic and English stopwords from the text."""
    words = text.split()
    return ' '.join([word for word in words if word not in AMHARIC_STOPWORDS and word not in ENGLISH_STOPWORDS])

def clean_special_characters(text):
    """Removes special characters and retains Amharic letters, numbers, and basic punctuation."""
    return re.sub(r'[^ሀ-ፐ0-9፩-፱.,!?፡a-zA-Z]', ' ', text)

def normalize_numbers(text):
    """Converts Amharic numerals to Arabic numerals."""
    for amh_num, eng_num in AMHARIC_NUMERALS.items():
        text = text.replace(amh_num, eng_num)
    return text

def remove_urls_mentions_hashtags(text):
    """Removes URLs, mentions, and hashtags from the text."""
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    return text.strip()

def preprocess_text(text):
    """Applies the full preprocessing pipeline to the text."""
    # Ensure the text is a string
    if not isinstance(text, str):
        return ''  # Return empty string if the text is not a valid string
    
    text = normalize_text(text)  # Normalize spelling variations
    text = tokenize_text(text)   # Tokenize
    text = lowercase_text(text)  # Convert to lowercase
    text = remove_stopwords(text)  # Remove stopwords
    text = clean_special_characters(text)  # Remove special characters
    text = normalize_numbers(text)  # Normalize numbers
    text = remove_urls_mentions_hashtags(text)  # Remove URLs, mentions, and hashtags
    return text


def save_preprocessed_data(messages, output_path='../../data/preprocessed_data/preprocessed_data.csv'):
    """Saves the preprocessed messages to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(messages, columns=['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Preprocessed data saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Load raw data from CSV
    raw_data_file = '../../data/raw_data/telegram_data.csv'  
    raw_df = pd.read_csv(raw_data_file)

    # Preprocess the data
    preprocessed_messages = []
    for _, msg in raw_df.iterrows():
        preprocessed_message = preprocess_text(msg["Message"])
        preprocessed_messages.append({
            "Channel Title": msg["Channel Title"],
            "Channel Username": msg["Channel Username"],
            "ID": msg["ID"],
            "Message": preprocessed_message,
            "Date": msg["Date"],
            "Media Path": msg["Media Path"]
        })

    # Save the preprocessed data
    save_preprocessed_data(preprocessed_messages)
