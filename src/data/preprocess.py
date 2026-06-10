import pandas as pd
import logging
from sklearn.impute import SimpleImputer

logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(messages)s'
)

logger = logging.getLogger('data-preprcessor')

FILEPATH="/home/fidisroxy/development/mlops/house-pred-mlops/data/raw/house_data.csv"
df = pd.read_csv(FILEPATH)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

imputer = SimpleImputer(strategy='median')

df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

df.to_csv("data/processed/house_processed.csv", index=False)


def load_data(filepath: str):
    
    logger.info(f"Loading file from: {filepath}")
    file = pd.read_csv(filepath)
    return file


    
def re_capitalize(df):
    
    df_cleaned = df.copy()
    
    new_columns = []
    for col in df_cleaned.columns:
        
        col = col.split('(')[0]
        
        col = col.strip().lower().replace(' ', '_')
        
        new_columns.append(col)
        
        df_cleaned.columns = new_columns
    
    return df_cleaned
    
    
     
    
def encode_labels(df):
    
    df_cleaned = df.copy()

    df_cleaned = pd.get_dummies(df_cleaned, 
                                columns=['furnishing'],
                                prefix=['furn'])
    
    return df_cleaned


def remove_strings(df):
    
    df_cleaned = df.copy()
    
    labels = ['bathroom', 'balcony', 'carpet_area', 'super_area', 'price', 'amount']
    
    for label in labels:
        if label in df_cleaned.cloumns:
            df_cleaned[label] = df_cleaned[label].str.replace(',','', regex= False)
            df_cleaned[label] = df_cleaned[label].str.replace(r"\D+", "", regex=True)
            df_cleaned[label] = pd.to_numeric(df_cleaned[label], errors='coerce')
            
    return df_cleaned

def clean_data(df):
    
    df= df.drop(['description','title', 'index', 'dimensions','plot_area'], axis=1)

    pass