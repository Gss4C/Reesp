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
            ('cat', OneHotEncoder(drop='first',sparse_output=False), categorical_features),
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

    def forward(self, 
                x, 
                categorical_indices=None):
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

class StructuredTrainer:
    def __init__(self, 
                 model, 
                 lr=0.001, 
                 weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def create_dataloader(self, 
                          X, 
                          y, 
                          batch_size=32, 
                          shuffle = True):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, 
                    train_loader):
        '''One-Epoch Training. Definizione di come si deve svolgere l'epoca di addestramento'''
        self.model.train() #funzione definita sotto
        total_loss = 0
        n_batches  = 0

        for batch_x, batch_y in train_loader:
            # Zero gradients
            self.optimizer.zero_grad() #reset valori gradiente
            
            # Forward pass (senza embeddings per semplicità)
            predictions = self.model(batch_x, categorical_indices=None)
            
            # Loss computation
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (per stabilità)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches  += 1
        
        return total_loss / n_batches

    def validate(self, 
                 val_loader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        n_batches  = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = self.model(batch_x, categorical_indices=None)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / n_batches

    def train(self, 
              X_train, 
              y_train, 
              X_val, 
              y_val, 
              epochs=50, 
              batch_size=32):
        #dataloaders creating
        train_loader = self.create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader   = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)
        print(f'=== STARTING TRAINING: {epochs} EPOCHS')
        print(f"Train size: {len(X_train)}\n Val size: {len(X_val)}")
        for epoch in range(epochs):
            #train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            #validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            #Progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print('=== TRAINING COMPLETE ===')
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self, output_name):
        """Plot training e validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Structured Encoder Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'./{output_name}.png')


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
    def test_full(self):
        """Test completo: preprocessing + model + training"""
    
        # 1. Genera dataset fittizio più grande
        np.random.seed(42)
        n_samples = 1000
        
        synthetic_data = {
            'sqm': np.random.randint(50, 200, n_samples),
            'rooms': np.random.randint(1, 6, n_samples),
            'location': np.random.choice(['Centro', 'Periferia', 'Mare'], n_samples),
            'condition': np.random.choice(['Ottimo', 'Buono', 'Da_ristr'], n_samples),
            'has_elevator': np.random.choice([True, False], n_samples),
            'floor': np.random.randint(0, 10, n_samples),
        }
        
        # Crea target correlato (formula semplificata)
        price_base = synthetic_data['sqm'] * 2000  # Base price per sqm
        location_bonus = np.where(synthetic_data['location'] == 'Centro', 50000, 0)
        condition_bonus = np.where(synthetic_data['condition'] == 'Ottimo', 30000, 0)
        noise = np.random.normal(0, 20000, n_samples)
        
        synthetic_data['SalePrice'] = price_base + location_bonus + condition_bonus + noise
        
        df = pd.DataFrame(synthetic_data)
        print(f"Generated dataset: {df.shape}")
        print(df.head())
        
        # 2. Preprocessing
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform_preprocessor(df)
        
        # 3. Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Crea e allena modello
        model = StructuredEncoder(
            input_dim=X.shape[1],
            embedding_configs={},  # No embeddings per semplicità
            hidden_dims=[256, 128],
            output_dim=256
        )
        
        trainer = StructuredTrainer(model, lr=0.001)
        train_losses, val_losses = trainer.train(X_train, y_train, X_val, y_val, epochs=50)
        
        # 5. Plot risultati
        trainer.plot_training_curves()
        
        # 6. Test finale
        model.eval()
        with torch.no_grad():
            sample_input = torch.FloatTensor(X_val[:5])
            predictions = model(sample_input)
            print(f"\nSample predictions shape: {predictions.shape}")
            print(f"Expected: (5, 256)")
        
        return model, trainer


# Esegui test
#tester = Tester()
#X_test, y_test = tester.test_preprocessing()
#model = tester.test_structured_encoder()
#print(model)
#trained_model, trainer = tester.test_full()

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

if __name__=='__main__':
    dataset = 'data/output-datasets/train_img-desc-data.csv'
    print("Hi\nI am the structured data encoder, and I am going to encode your dataset")
    
    df = pd.read_csv(dataset)
    
    preprocessor = DataPreprocessor()

    X,y = preprocessor.fit_transform_preprocessor(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    embedding_configs = create_embedding_config(df)
    model = StructuredEncoder(
            input_dim=X.shape[1],
            embedding_configs=embedding_configs, 
            hidden_dims=[256, 128],
            output_dim=256
        )
    trainer = StructuredTrainer(model, lr=0.001)
    train_losses, val_losses = trainer.train(X_train, y_train, X_val, y_val, epochs=50)

    trainer.plot_training_curves('full_dataset')

    model.eval()
    with torch.no_grad():
        sample_input = torch.FloatTensor(X_val[:5])
        predictions = model(sample_input)
        print(f"\nSample predictions shape: {predictions.shape}")
        print(f"Expected: (5, 256)")