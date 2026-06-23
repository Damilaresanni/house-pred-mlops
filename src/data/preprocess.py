import pandas as pd
import logging
import re
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline




logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(messages)s'
)

logger = logging.getLogger('data-preprcessor')

FILEPATH="/home/fidisroxy/development/mlops/house-pred-mlops/data/raw/house_prices.csv"
df = pd.read_csv(FILEPATH)


def load_data(filepath: str):
    
    logger.info(f"Loading file from: {filepath}")
    file = pd.read_csv(filepath)
    return file


def rename(df):
    
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.rename(columns={'Price (in rupees)':'price',
                                            'Amount(in rupees)': 'amount' })
    return df_cleaned



def re_capitalize(df):
    
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    return df_cleaned
    

def remove_strings(df):
    
    df_cleaned = df.copy()
    labels = ['bathroom', 'balcony', 'carpet_area', 'super_area', 'price']
     

    
    for label in labels:
        if label in df_cleaned.columns:
            # df_cleaned[label] = df_cleaned[label].str.replace(',','', regex= False)
            df_cleaned[label] = df_cleaned[label].astype(str).str.replace(r"\D+", "", regex=True)
            df_cleaned[label] = pd.to_numeric(df_cleaned[label], errors='coerce')
            
    return df_cleaned
    

# def fill_labels(df):
    
#     df_cleaned = df.copy()
    
#     cat_cols = df.select_dtypes(include=['object']).columns
#     num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
#     imputer_num = SimpleImputer(strategy='median')
#     imputer_cat = SimpleImputer(strategy='most_frequent')
    
#     df_cleaned[num_cols] = imputer_num.fit_transform(num_cols)
#     df_cleaned[cat_cols] = imputer_cat.fit_transform(cat_cols)
    
#     return df_cleaned
    

def extract_price(text):
    text = text.lower()
    match = re.search(r'(\d+(?:\.\d+)?)\s*(lac|lakh|cr)', text)  
    if not match:
        return None
    
    num = float(match.group(1))
    unit = match.group(2)
    
    if unit in ['lac', 'lakh']:
        return num * 100000
    if unit == 'cr':
        return num * 10000000
    
    
def transform_amount(df):
    
    df_cleaned = df.copy()
    df_cleaned['amount'] = df_cleaned['amount'].astype(str).str.replace('call for price', '', case=False, regex=True)
    df_cleaned['amount'] = df_cleaned['amount'].astype(str).apply(extract_price)
    
    return df_cleaned

    

def transform_floors(df):
    
    df[['floor_num','total_floors']] = (df['floor'].str.extract(r'(\d+)\s*out\s*of\s*(\d+)',flags=re.IGNORECASE, expand=True))
    df['floor_num'] = pd.to_numeric(df['floor_num'], errors='coerce')
    df['total_floors']=pd.to_numeric(df['total_floors'], errors='coerce')
    df['floor_ratio'] = df['floor_num'] / df['total_floors'] 
    df['is_top_floor'] = (df['floor_num'] == df['total_floors']).astype(int)
    df['is_bottom_floor'] = (df['floor_num'] == 0).astype(int)
    df = df.drop(columns=['floor'])
    
    return df
    
def drop_labels(df):
    
    df_cleaned = df.copy()
    
    columns_to_drop = [
        'title',
        'description',
        'index',
        'dimension',
        'plot_area'
    ]
        
    
    df_cleaned = df_cleaned.drop(columns= columns_to_drop,
                                 errors ='ignore',
                                 inplace=False)
    return df_cleaned


def clean_label(df):
    
    df_cleaned = df.copy()
    
    df_cleaned['overlooking'] = (df_cleaned['overlooking']
                                        .astype('string')
                                        .str.lower()
                                        .str.replace(r'[^a-z/ ]', '', regex=True)
                                        .str.strip())
                                        
    df_cleaned['overlooking_list'] = (df_cleaned['overlooking']
                                      .str.strip('/')
                                       .str.split('/'))
    
    df_cleaned = df_cleaned.drop('overlooking', errors='ignore')
    
    return df_cleaned


numeric_features = [
    "amount","price", "carpet_area","balcony",
    "bathroom", "floor_num", "total_floors",
    "floor_ratio","is_top_floor", "is_bottom_floor"
    
]




nominal_features =[
    "location", "status","transaction",
    "furnishing", "facing"," overlooking_list",
    "car_parking", "ownership"
]



numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)



nominal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("nom", nominal_transformer,nominal_features)
    ]
)

def preprocess_pipeline(filepath: str):
    df = load_data(filepath)
    df = rename(df)
    df = re_capitalize(df)
    df = remove_strings(df)
    df = transform_amount(df)
    df = transform_floors(df)
    df = drop_labels(df)
    df = clean_label(df)
    
    print(df.head())
    return df
    
    


if __name__ == "__main__":
    df = preprocess_pipeline(FILEPATH)
    
    df.head(5).to_json("head.json", indent=4, orient="records")







