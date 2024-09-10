import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Synonym mapping for procurement-related terms
synonym_dict = {
    "procurement": ["purchase", "acquisition"],
    "tender": ["bid", "proposal"],
    "invoice": ["bill", "receipt"],
    "order": ["purchase", "request"],
    "tracking": ["status", "location"],
    "return": ["refund", "exchange"],
    "policy": ["rule", "regulation"],
    "support": ["help", "assistance"],
    "product": ["item", "goods"],
    "availability": ["in stock", "available"],
    "warranty": ["guarantee", "assurance"]
}

# Function to expand keywords with synonyms
def expand_keywords(keywords, synonym_dict):
    expanded_keywords = set(keywords.split())  # Use a set to avoid duplicates
    for keyword in keywords.split():
        if keyword in synonym_dict:
            expanded_keywords.update(synonym_dict[keyword])
    return ' '.join(expanded_keywords)

# Load dataset from Excel file
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

# Load and prepare the dataset
dataset = load_dataset(r'C:\Users\USER\Desktop\aiassignment\chatbot_dataset.xlsx')
dataset['Expanded_Keywords'] = dataset['Keywords'].apply(lambda x: expand_keywords(x, synonym_dict))
X_expanded = dataset['Expanded_Keywords']
y_expanded = dataset['Response']
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_expanded, y_expanded, test_size=0.2, random_state=42)

# Create and train the model
model_expanded = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_expanded.fit(X_train_exp, y_train_exp)

# Streamlit interface
st.title('Chatbot')
user_input = st.text_input('You:', '')

if user_input:
    response = model_expanded.predict([expand_keywords(user_input, synonym_dict)])[0]
    st.write(f'Bot: {response}')

# Evaluate the model
y_pred_expanded = model_expanded.predict(X_test_exp)
accuracy_expanded = metrics.accuracy_score(y_test_exp, y_pred_expanded)
st.write(f'Accuracy after synonym expansion: {accuracy_expanded}')
