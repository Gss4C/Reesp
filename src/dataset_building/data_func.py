import pandas as pd

def read_clean_dataset(dataset_path: str, features: list) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    clean_df = df[features]
    return clean_df

def create_embedding_config(df: pd.DataFrame):
    """Crea configurazione embeddings da dataset"""
    embedding_configs = {}
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        unique_values = df[col].nunique()
        embedding_configs[col] = {
            'vocab_size': unique_values + 1,  # +1 per unknown values
            'embed_dim': min(50, (unique_values + 1) // 2)
        }
    
    return embedding_configs