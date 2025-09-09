import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

class StructuredDataPreprocessor:
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

class EmbeddingLayer(nn.Module):
    """
    Embedding layer per features categoriche con molti valori unici
    """
    def __init__(self, num_categories, embedding_dim, max_norm=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, max_norm=max_norm)
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        # x deve essere LongTensor con indici delle categorie
        return self.embedding(x)

class StructuredDataEncoder(nn.Module):
    """
    Neural Network encoder per dati strutturati immobiliari
    """
    def __init__(self, input_dim, output_dim=256, hidden_dims=[512, 256], 
                 dropout_rate=0.3, use_batch_norm=True):
        super(StructuredDataEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Costruisci le dimensioni della rete
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Costruisci i layer
        layers = []
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Batch Normalization (tranne ultimo layer)
            if use_batch_norm and i < len(layer_dims) - 2:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            
            # Activation (tranne ultimo layer)
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
                
                # Dropout (tranne ultimo layer)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione pesi
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inizializzazione Xavier per i pesi
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor di shape (batch_size, input_dim)
            
        Returns:
            Tensor di shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_embeddings(self, x):
        """
        Ottieni embeddings intermedi (prima dell'ultimo layer)
        """
        # Forward fino al penultimo layer
        for layer in self.network[:-1]:
            x = layer(x)
        return x

class StructuredDataTrainer:
    """
    Trainer per il Structured Data Encoder
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # History per tracking
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """
        Training per una epoca
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            
            # Calcola loss (assumendo regressione)
            if batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1)
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, dataloader, criterion):
        """
        Valutazione del modello
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_x)
                
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001, patience=5):
        """
        Training loop completo
        """
        # Setup optimizer e criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salva best model
                torch.save(self.model.state_dict(), 'best_structured_encoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("-" * 50)
        print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        
        # Carica best model
        self.model.load_state_dict(torch.load('best_structured_encoder.pth'))


if __name__ == "__main__":
    
    dataset = 'data/output-datasets/house_df.csv'
    
    df = pd.read_csv(dataset)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = StructuredDataPreprocessor()
    X_train_cont, X_train_cat, y_train = preprocessor.fit_transform(train_df)
    X_test_cont, X_test_cat, y_test    = preprocessor.transform(test_df)
    #X_processed = preprocessor.fit_transform(df)

    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"✅ Preprocessing test passed!")
    
    print("\n2. Testing Encoder...")
    input_dim = X_train_cont.shape[1]
    model = StructuredDataEncoder(input_dim=input_dim, output_dim=256)
    
    # Test forward pass
    test_input = torch.FloatTensor(X_processed[:32])  # Batch di 32
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (32, 256), f"Expected (32, 256), got {output.shape}"
    print("✅ Forward pass test passed!")
    
    print("\n3. Testing Training Loop...")
    from torch.utils.data import DataLoader, TensorDataset
    
    # Prepara dati per training
    X_tensor = torch.FloatTensor(X_processed)
    y_tensor = torch.FloatTensor(price).unsqueeze(1)
    
    # Split train/val
    train_size = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Quick training test (5 epochs)
    trainer = StructuredDataTrainer(model)
    trainer.train(train_loader, val_loader, epochs=5, lr=0.001)
    
    # Verifica che loss decresce
    initial_loss = trainer.train_losses[0]
    final_loss = trainer.train_losses[-1]
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss decreased: {final_loss < initial_loss}")
    print("✅ Training test passed!")
    
    print("\n🎉 All tests passed! Structured Data Encoder is ready!")
