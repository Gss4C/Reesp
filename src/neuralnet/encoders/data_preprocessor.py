import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self):
        self.preprocessor   = None
        self.features_names = []
    
    def fit_transform_preprocessor(self, 
                                   df: pd.DataFrame, 
                                   target_col: str='SalePrice'):
        
        print('Preprocessor fit_transform starting...')

        X = df.drop(columns=[target_col])
        y = df[target_col].values

        numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_features     = X.select_dtypes(include=['bool']).columns.tolist()

        print(f" ==== Feature Types: ====")
        print(f"  Numeric: {numeric_features}")
        print(f"  Categorical: {categorical_features}") 
        print(f"  Boolean: {boolean_features}")

        #creazione preprocessing pipeline
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features), #,sparse_output=False
            ('bool', 'passthrough', boolean_features)
        ])

        X_processed = self.preprocessor.fit_transform(X)
        
        num_names   = numeric_features
        cat_names   = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        bool_names  = boolean_features
        self.feature_names = list(num_names) + list(cat_names) + list(bool_names)

        print(f"Processed shape: {X_processed.shape}")
        print(f"Final features: {self.feature_names[:]}...")  
        
        return X_processed, y
    
class RealDataPreprocessor:
    """Preprocessor per dati reali con gestione corretta delle categoriche"""
    
    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.label_encoders = {}  # Un LabelEncoder per ogni feature categorica
        self.feature_info = {}    # Info su ogni feature
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, target_col: str = 'SalePrice'):
        """Fit preprocessor sui dati di training"""
        
        # Rimuovi target dalle features
        X = df.drop(columns=[target_col], errors='ignore')
        
        # 1. Identifica tipi di colonne
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.boolean_columns = X.select_dtypes(include=['bool']).columns.tolist()
        
        print(f"   FEATURE ANALYSIS:")
        print(f"   Numeric columns ({len(self.numeric_columns)}): {self.numeric_columns}")
        print(f"   Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns}")
        print(f"   Boolean columns ({len(self.boolean_columns)}): {self.boolean_columns}")
        
        # 2. Fit scaler per numeriche
        if self.numeric_columns:
            self.numeric_scaler.fit(X[self.numeric_columns])
        
        # 3. Fit label encoders per categoriche
        for col in self.categorical_columns:
            # Handle missing values
            unique_values = X[col].dropna().unique()
            print(f"   🏷️  {col}: {len(unique_values)} unique values - {list(unique_values)[:5]}{'...' if len(unique_values) > 5 else ''}")
            
            # Create e fit label encoder
            le = LabelEncoder()
            
            # Add special token for missing values
            all_values = list(unique_values) + ['<UNK>']  # <UNK> per valori non visti
            le.fit(all_values)
            
            self.label_encoders[col] = le
            
            # Store vocab info per embedding layers
            self.feature_info[col] = {
                'vocab_size': len(le.classes_),
                'embed_dim': min(50, (len(le.classes_) + 1) // 2),  # Rule of thumb
                'classes': le.classes_.tolist()
            }
        
        # 4. Calcola dimensioni finali
        self.numeric_dim = len(self.numeric_columns)
        self.boolean_dim = len(self.boolean_columns)
        self.total_processed_dim = self.numeric_dim + self.boolean_dim  # Categoriche gestite separatamente
        
        print(f"\n📏 PROCESSED DIMENSIONS:")
        print(f"   Numeric features: {self.numeric_dim}")
        print(f"   Boolean features: {self.boolean_dim}")  
        print(f"   Categorical features: {len(self.categorical_columns)} (handled as embeddings)")
        print(f"   Total continuous dim: {self.total_processed_dim}")
        
        # Store embedding configs per il modello
        self.embedding_configs = {col: info for col, info in self.feature_info.items()}
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, target_col: str = 'SalePrice'):
        """Transform data usando fitted preprocessor"""
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first!")
        
        # Separare features da target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col].values
        else:
            X = df.copy()
            y = None
        
        processed_data = {}
        
        # 1. Process numeric features
        if self.numeric_columns:
            numeric_data = X[self.numeric_columns].fillna(0)  # Fill missing con 0
            numeric_scaled = self.numeric_scaler.transform(numeric_data)
            processed_data['numeric'] = numeric_scaled
        else:
            processed_data['numeric'] = np.empty((len(X), 0))
        
        # 2. Process boolean features  
        if self.boolean_columns:
            boolean_data = X[self.boolean_columns].fillna(False).astype(int)
            processed_data['boolean'] = boolean_data.values
        else:
            processed_data['boolean'] = np.empty((len(X), 0))
        
        # 3. Process categorical features (separate per embeddings)
        processed_data['categorical'] = {}
        for col in self.categorical_columns:
            # Handle missing values
            col_data = X[col].fillna('<UNK>').astype(str)
            
            # Transform usando label encoder, con fallback per unknown values
            try:
                encoded = self.label_encoders[col].transform(col_data)
            except ValueError as e:
                # Handle unknown values
                print(f"⚠️  Unknown values found in {col}, using <UNK> token")
                encoded = []
                for val in col_data:
                    try:
                        encoded.append(self.label_encoders[col].transform([val])[0])
                    except ValueError:
                        # Use <UNK> token index
                        unk_idx = list(self.label_encoders[col].classes_).index('<UNK>')
                        encoded.append(unk_idx)
                encoded = np.array(encoded)
            
            processed_data['categorical'][col] = encoded
        
        # 4. Combine continuous features (numeric + boolean)
        continuous_features = np.concatenate([
            processed_data['numeric'],
            processed_data['boolean']
        ], axis=1)
        
        return continuous_features, processed_data['categorical'], y
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'SalePrice'):
        """Fit e transform in un solo step"""
        return self.fit(df, target_col).transform(df, target_col)
    
    def save(self, filepath: str):
        """Salva preprocessor per riuso futuro"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor salvato"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor