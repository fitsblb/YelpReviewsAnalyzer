## --- Import Libraries ---
# Import all required libraries for data processing, modeling, and evaluation
from google.colab import drive  # For accessing Google Drive
import re
import os
import shutil
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report,
    make_scorer,
    f1_score
)
from sklearn.model_selection import RandomizedSearchCV
from transformers import AutoConfig
import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk # Import the nltk library
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

# Dataset Handling
def load_dataset(file_path, columns):
    """
    Load the dataset from the specified file path and select only the required columns.

    Args:
        file_path (str): Path to the dataset file.
        columns (list): List of column names to load.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    review_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "capple7/yelp-open-data-philly-restaurants",
        file_path,
        pandas_kwargs={"usecols": columns}
    )
    return review_df

def perform_eda(review_df, visualize=False):
    """
    Perform exploratory data analysis on the given dataset.

    Args:
        review_df (pd.DataFrame): Dataset for EDA.
        visualize (bool): Whether to display visualizations.

    Returns:
        pd.DataFrame: Cleaned dataset after EDA.
    """
    print(f"Dataset shape: {review_df.shape}")
    print(f"Missing values in 'text': {review_df['text'].isnull().sum()}")
    print(f"Missing values in 'stars': {review_df['stars'].isnull().sum()}")

    # Drop rows with missing values
    review_df.dropna(subset=['text', 'stars'], inplace=True)
    print(f"Dataset shape after dropping missing values: {review_df.shape}")

    # Print star rating distribution
    print("Star ratings distribution:")
    print(review_df['stars'].value_counts().sort_index())

    # Optional visualization
    if visualize:
        plt.figure(figsize=(8, 6))
        sns.histplot(review_df['stars'], bins=5, kde=False)
        plt.title('Distribution of Star Ratings')
        plt.xlabel('Stars')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    return review_df

def preprocess_yelp_reviews(review_df):
    """
    Comprehensive preprocessing for Yelp reviews.

    Args:
        review_df (pd.DataFrame): DataFrame containing Yelp reviews.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Convert star ratings to zero-indexed values
    review_df['stars'] = review_df['stars'].astype(int) - 1

    # Assign sentiment labels based on star ratings
    def assign_sentiment(score):
        if score <= 2:
            return 'negative'
        elif score == 3:
            return 'neutral'
        else:
            return 'positive'

    review_df['sentiment'] = review_df['stars'].apply(assign_sentiment)
    review_df = review_df[['text', 'sentiment']].copy()
    review_df['sentiment'] = review_df['sentiment'].astype('category')
    review_df['sentiment_id'] = review_df['sentiment'].cat.codes

    return review_df


# Text Preprocessing
def advanced_text_preprocessing(text):
    """
    Advanced text preprocessing with comprehensive cleaning.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Cleaned and preprocessed text.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text


# Dataset Preparation
def prepare_datasets(review_df, model_name="distilbert-base-uncased", test_size=0.3):
    """
    Prepare datasets for training.

    Args:
        review_df (pd.DataFrame): Preprocessed review DataFrame.
        model_name (str): Pretrained model name.
        test_size (float): Proportion of data to reserve for testing.

    Returns:
        tuple: Train, validation, and test datasets.
    """
    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        review_df['processed_text'],
        review_df['sentiment_id'],
        test_size=test_size,
        random_state=42,
        stratify=review_df['sentiment_id']
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # Create DataFrames for each split
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    val_df = pd.DataFrame({'text': X_val, 'labels': y_val})
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    return train_dataset, val_dataset, test_dataset, tokenizer


# Model Configuration and Training
def create_model_config(num_labels=3):
    """
    Create model configuration.

    Args:
        num_labels (int): Number of sentiment classes.

    Returns:
        AutoConfig: Configured model.
    """
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.4
    )
    return config

def compute_metrics(pred):
    """
    Compute evaluation metrics.

    Args:
        pred: Prediction object from trainer.

    Returns:
        dict: Performance metrics.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def evaluate_model_on_test(trainer, test_dataset):
    """
    Evaluate the trained model on the test dataset.

    Args:
        trainer (Trainer): Hugging Face Trainer object.
        test_dataset (Dataset): Test dataset to evaluate the model.

    Returns:
        dict: Evaluation metrics for the test dataset.
    """
    print("Evaluating the model on the test dataset...")
    test_results = trainer.evaluate(test_dataset)
    print("Test Evaluation Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    return test_results

def save_trained_model_and_tokenizer(trainer, tokenizer, save_path="/content/drive/MyDrive/LLM Project/Model"):
    """
    Save the trained model and tokenizer to the specified directory with a unique timestamp.

    Args:
        trainer (Trainer): Hugging Face Trainer object.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer object.
        save_path (str): Base path to save the trained model and tokenizer.
    """
    # Add a timestamp to the save path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_save_path = f"{save_path}_{timestamp}"

    print(f"Saving the trained model and tokenizer to {unique_save_path}...")
    
    # Save the model
    trainer.save_model(unique_save_path)
    print("Model saved successfully!")

    # Save the tokenizer
    tokenizer.save_pretrained(unique_save_path)
    print("Tokenizer saved successfully!")


