import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import base64
import re
from urllib.parse import urlparse

def extract_outlet(google_url):
    """
    Extracts the actual domain (outlet) from a Google News RSS URL.
    """
    try:
        # The URL structure is usually .../articles/<base64_encoded_id>?...
        # The base64 part often contains the target URL.
        # Let's try to find the base64 string. It usually starts after 'articles/'
        match = re.search(r'articles/([^?]+)', google_url)
        if match:
            encoded_str = match.group(1)
            # Base64 strings might need padding
            padded_str = encoded_str + '=' * (-len(encoded_str) % 4)
            try:
                decoded_bytes = base64.urlsafe_b64decode(padded_str)
                decoded_str = decoded_bytes.decode('latin-1') # latin-1 to avoid utf-8 errors if binary data is present
                
                # The decoded string usually contains the URL, sometimes with other binary chars.
                # Let's look for http/https
                url_match = re.search(r'(https?://[^\s\x00]+)', decoded_str)
                if url_match:
                    real_url = url_match.group(1)
                    domain = urlparse(real_url).netloc
                    return domain.replace('www.', '')
            except Exception as e:
                pass
    except Exception:
        pass
    return "unknown"

def load_and_process_data(filepath='Dataset.csv'):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print("Extracting outlets from source URLs...")
    df['outlet'] = df['source'].apply(extract_outlet)
    
    print("Unique outlets found:", df['outlet'].nunique())
    print(df['outlet'].value_counts().head())
    
    # Use the most frequent outlet as the privileged group
    top_outlet = df['outlet'].mode()[0]
    print(f"Most frequent outlet (Privileged Group): {top_outlet}")
    
    # Binary protected attribute: 1 if outlet == top_outlet, 0 otherwise
    df['protected_attribute'] = df['outlet'].apply(lambda x: 1 if x == top_outlet else 0)
    
    # Label: biased=1, non-biased=0
    # The dataset has 'label_bias' as 'biased' or 'non-biased'
    # We want to detect bias, so 'biased' is the "positive" outcome (1) in terms of classification,
    # but for fairness, usually 1 is the "favorable" outcome.
    # If we want to check if the model is biased *against* a group, we check if that group gets more "unfavorable" outcomes.
    # Let's say "non-biased" (0) is favorable. "biased" (1) is unfavorable.
    df['label_binary'] = df['label_bias'].apply(lambda x: 1 if x == 'biased' else 0)
    
    return df, top_outlet

if __name__ == "__main__":
    try:
        df, top_outlet = load_and_process_data()
        
        # Create AIF360 BinaryLabelDataset
        # AIF360 requires numerical data. We only need protected attribute and label for metrics.
        df_aif = df[['protected_attribute', 'label_binary']]
        
        dataset = BinaryLabelDataset(
            favorable_label=0, # non-biased
            unfavorable_label=1, # biased
            df=df_aif,
            label_names=['label_binary'],
            protected_attribute_names=['protected_attribute']
        )
        
        privileged_groups = [{'protected_attribute': 1}]
        unprivileged_groups = [{'protected_attribute': 0}]
        
        metric = BinaryLabelDatasetMetric(
            dataset, 
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        print(f"Disparate Impact: {metric.disparate_impact()}")
        print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")
        
        # Save processed dataframe for training
        df.to_csv('Dataset_Processed.csv', index=False)
        print("Saved processed dataset to Dataset_Processed.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
