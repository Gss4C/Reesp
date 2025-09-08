import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn

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
            ('cat', OneHotEncoder(drop='first'), categorical_features),
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
    
class StructuredEncoder(nn.Module):
    """
    Encoder per dati strutturati con embedding layers per categoriche
    e MLP standard per numeriche
    """
    def __init__(self,
                 input_dim: int,
                 embedding_configs: dict,
                 hidden_dims: list = [512, 256, 128],
                 output_dim: int=256,
                 dropout: float = 0.3):
        super().__init__() #per l'eredita da nn.module

        ## Categorial Data Embedding ##
        self.embeddings = nn.ModuleDict() #permette di conservare sottomoduli
        total_embedding_dim = 0

        for feature_name, config in embedding_configs.items():
            vocab_size = config['vocab_size'] #numero categorie uniche
            embed_dim = min(50, (vocab_size +1) // 2)  #Rule of thumb per embedding size

            self.embeddings[feature_name] = nn.Embedding(vocab_size, embed_dim) #A simple lookup table that stores embeddings of a fixed dictionary and size
            total_embedding_dim += embed_dim
            print(f"Embedding {feature_name}: {vocab_size} -> {embed_dim}")
        
        #input dimension for MLP
        mlp_input_dim = input_dim + total_embedding_dim

        #####   MLP LAYERS   #####
        # - Input Layers
        layers = []
        prev_dim= mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([  #metodo per appendere liste a liste
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # - Output Layers
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers) #equivale al concatenate in keras

        self._init_weights() # inizializza pesi modello. Da ora ho un modello

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear): #controlla che module sia una istanza di linear
                nn.init.kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, x, categorical_indices=None):
        """
        Forward pass
        
        Args:
            x: Tensor of shape [batch_size, input_dim] - processed features
            categorical_indices: Dict con indices per embedding lookup (se applicabile)
        """
        batch_size = x.shape[0]
        
        # Se abbiamo embeddings, combina con features numeriche
        if self.embeddings and categorical_indices:
            embedded_features = []
            
            for feature_name, embedding_layer in self.embeddings.items():
                indices = categorical_indices[feature_name]  # [batch_size]
                embedded = embedding_layer(indices)          # [batch_size, embed_dim]
                embedded_features.append(embedded)
            
            # Concatena embeddings
            embedded_concat = torch.cat(embedded_features, dim=1)  # [batch_size, total_embed_dim]
            
            # Combina con features numeriche
            x = torch.cat([x, embedded_concat], dim=1)  # [batch_size, total_input_dim]
        
        # Pass through MLP
        output = self.mlp(x)  # [batch_size, output_dim]
        return output

class Tester:

    def test_preprocessing(self):
        # Sample data
        sample_df = pd.DataFrame({
            'sqm': [85, 120, 95],
            'rooms': [2, 3, 2], 
            'location': ['Centro', 'Periferia', 'Centro'],
            'condition': ['Ottimo', 'Buono', 'Ottimo'],
            'has_elevator': [True, False, True],
            'SalePrice': [320000, 450000, 380000]
        })
        
        preprocessor = DataPreprocessor()
        X_processed, y = preprocessor.fit_transform_preprocessor(sample_df)
        
        print("\n\n✅ Preprocessing Test:")
        print(f"Input shape: {sample_df.shape}")
        print(f"Output X shape: {X_processed.shape}")
        print(f"Output y shape: {y.shape}")
        print(f"Sample processed row: {X_processed[0][:5]}...")  # First 5 features
        
        return X_processed, y
    def test_structured_encoder(self):
        # Simula dati preprocessati
        batch_size = 32
        input_dim = 15  # Da preprocessing
        
        # Simula embedding configs
        embedding_configs = {
            'location': {'vocab_size': 5, 'embed_dim': 3},
            'condition': {'vocab_size': 4, 'embed_dim': 2}
        }
        
        # Crea modello
        model = StructuredEncoder(
            input_dim=input_dim,
            embedding_configs=embedding_configs,
            hidden_dims=[512, 256],
            output_dim=256
        )
        
        # Test input
        x = torch.randn(batch_size, input_dim)
        categorical_indices = {
            'location': torch.randint(0, 5, (batch_size,)),
            'condition': torch.randint(0, 4, (batch_size,))
        }
        
        # Forward pass
        output = model(x, categorical_indices)
        
        print("✅ Structured Encoder Test:")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 256)")
        
        # Gradient test
        loss = output.sum()
        loss.backward()
        print("✅ Gradient computation successful")
        
        return model

# Esegui test
tester = Tester()
X_test, y_test = tester.test_preprocessing()
model = tester.test_structured_encoder()