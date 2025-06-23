# model_interpretability_task5.py
import os
import torch
import shap
from transformers import AutoModelForTokenClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import pandas as pd

# Paths from Task 3 & Task 4
# Replace these with the actual paths where the dataset and fine-tuned model are stored

# Input data path (from Task 3 & Task 4)
input_data_path = '../data/preprocessed_data/labeled_telegram_product_price_location.txt'  # Path to the labeled dataset

# Fine-tuned model path (from Task 3 & Task 4)
fine_tuned_model_path = '../models/fine_tuned_ner_model'  # Path to the fine-tuned model

# Save location for SHAP and LIME explanations
save_path = '../results/interpretable_results'  # Directory to save SHAP and LIME explanation files

# Load the fine-tuned NER model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(fine_tuned_model_path)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# SHAP - Model prediction function for SHAP interpretation
def shap_model_predict(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)  # Get model predictions
    logits = outputs.logits
    return logits

# Initialize SHAP explainer
explainer = shap.Explainer(shap_model_predict, tokenizer)

# Example input text for SHAP interpretation (Replace with text from your dataset)
text_shap = "ሴቶች በአዲስ አበባ በየቀኑ እቃውን ስለሚያሸጡ ታዋቂ ነው።"  # Example Amharic text

# Generate SHAP values for the input text
shap_values = explainer([text_shap])

# Visualize SHAP values
shap.initjs()
shap.force_plot(shap_values[0])

# Save SHAP explanation as HTML
shap_html_path = os.path.join(save_path, 'shap_explanation_task5.html')  # Set path to save SHAP explanation
shap.save_html(shap_html_path, shap_values[0])
print(f"SHAP explanation saved to {shap_html_path}")

# LIME - Model prediction function for LIME interpretation
def lime_predict(texts):
    # Tokenize input texts
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**encodings)  # Get model predictions
    logits = outputs.logits
    return logits.numpy()

# Initialize LIME text explainer
lime_explainer = LimeTextExplainer(class_names=["O", "B-PRODUCT", "I-PRODUCT", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"])

# Example input text for LIME explanation (Replace with text from your dataset)
text_lime = "በአዲስ አበባ የሴቶች ታዋቂ የአታሚ በየቀኑ ያሸጡ።"  # Another example Amharic text

# Create LIME explanation for the example text
exp = lime_explainer.explain_instance(text_lime, lime_predict, num_features=10)

# Visualize the explanation
exp.show_in_notebook()

# Save LIME explanation to an HTML file
lime_html_path = os.path.join(save_path, 'lime_explanation_task5.html')  # Set path to save LIME explanation
exp.save_to_file(lime_html_path)
print(f"LIME explanation saved to {lime_html_path}")

# Analyze cases where the model struggles (ambiguous or overlapping entities)
# Example problematic case where ambiguity might occur (Replace with text from your dataset)
problematic_text = "በሴቶች በአማራ ክልል እቃውን ስለሚያሸጡ ታዋቂ ነው።"
problematic_shap_values = explainer([problematic_text])

# Visualize the problematic case SHAP explanation
shap.force_plot(problematic_shap_values[0])

# Save the problematic case SHAP explanation
problematic_shap_html_path = os.path.join(save_path, 'problematic_shap_explanation_task5.html')
shap.save_html(problematic_shap_html_path, problematic_shap_values[0])
print(f"Problematic SHAP explanation saved to {problematic_shap_html_path}")

# Optionally: Load and inspect the data
# You can load the labeled dataset from the path provided and inspect a sample of the data.
# Load dataset
data = pd.read_csv(input_data_path, sep='\t', header=None, names=["word", "label"])

# Check the first few rows of the data
print(data.head())

# Save some of the processed data for review
sample_data_path = os.path.join(save_path, 'sample_labeled_data_task5.csv')
data.head(10).to_csv(sample_data_path, index=False)
print(f"Sample data saved to {sample_data_path}")
