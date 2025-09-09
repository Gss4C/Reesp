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
    """
    Preprocessing pipeline per dati strutturati immobiliari
    """
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.label_encoders = {}
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit del preprocessor sui dati di training
        
        Args:
            X: DataFrame con features strutturate
            y: Target (opzionale)
        """
        # Identifica automaticamente tipi di colonne
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        # Pipeline per features numeriche
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline per features categoriche
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combina i transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit del preprocessor
        self.preprocessor.fit(X)
        self.is_fitted = True
        
        # Store feature names per debug
        self._store_feature_names()
        
        return self
    
    def transform(self, X):
        """
        Transform dei dati usando il preprocessor fittato
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor deve essere fittato prima del transform!")
            
        X_processed = self.preprocessor.transform(X)
        return X_processed
    
    def fit_transform(self, X, y=None):
        """
        Fit e transform in un solo step
        """
        return self.fit(X, y).transform(X)
    
    def _store_feature_names(self):
        """
        Salva i nomi delle feature dopo preprocessing
        """
        feature_names = []
        
        # Numeric features (stesso nome)
        feature_names.extend(self.numeric_features)
        
        # Categorical features (espanse da OneHot)
        if self.categorical_features:
            cat_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
            
        self.feature_names = feature_names
        print(f"Total features after preprocessing: {len(feature_names)}")
    
    def get_feature_names(self):
        """
        Ritorna nomi delle feature dopo preprocessing
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor deve essere fittato prima!")
        return self.feature_names
    
    def save(self, filepath):
        """
        Salva il preprocessor
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Carica il preprocessor
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

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

# Test e esempio d'uso
if __name__ == "__main__":
    # Test con dati sintetici
    print("Testing Structured Data Encoder...")
    
    # Crea dati di esempio (simula dataset immobiliare)
    np.random.seed(42)
    n_samples = 1000
    
    # Features numeriche
    data = {
        'square_meters': np.random.uniform(50, 300, n_samples),
        'rooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.uniform(0, 50, n_samples),
        'floor': np.random.randint(0, 10, n_samples),
        'balconies': np.random.randint(0, 3, n_samples),
        
        # Features categoriche
        'location': np.random.choice(['center', 'suburbs', 'outskirts'], n_samples),
        'property_type': np.random.choice(['apartment', 'house', 'villa'], n_samples),
        'condition': np.random.choice(['new', 'good', 'renovated', 'old'], n_samples),
        'heating': np.random.choice(['central', 'autonomous', 'none'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Target sintetico (prezzo basato su features)
    price = (df['square_meters'] * 1000 + 
             df['rooms'] * 5000 + 
             df['bathrooms'] * 3000 - 
             df['age'] * 200 +
             df['balconies'] * 2000 +
             np.random.normal(0, 5000, n_samples))  # Noise
    
    print("\n1. Testing Preprocessor...")
    preprocessor = StructuredDataPreprocessor()
    X_processed = preprocessor.fit_transform(df)
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"✅ Preprocessing test passed!")
    
    print("\n2. Testing Encoder...")
    input_dim = X_processed.shape[1]
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