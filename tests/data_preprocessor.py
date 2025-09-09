import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pickle
from typing import Dict, List, Tuple

class RealDataPreprocessor:
    """Preprocessor per dati reali con gestione corretta delle categoriche"""
    
    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.label_encoders = {}  # Un LabelEncoder per ogni feature categorica
        self.feature_info = {}    # Info su ogni feature
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, target_col: str = 'price'):
        """Fit preprocessor sui dati di training"""
        
        # Rimuovi target dalle features
        X = df.drop(columns=[target_col], errors='ignore')
        
        # 1. Identifica tipi di colonne
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.boolean_columns = X.select_dtypes(include=['bool']).columns.tolist()
        
        print(f"📊 FEATURE ANALYSIS:")
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
    
    def transform(self, df: pd.DataFrame, target_col: str = 'price'):
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
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'price'):
        """Fit e transform in un solo step"""
        return self.fit(df, target_col).transform(df, target_col)
    
    def save(self, filepath: str):
        """Salva preprocessor per riuso futuro"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor salvato"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"📂 Preprocessor loaded from {filepath}")
        return preprocessor

# TEST CON DATI SAMPLE
def test_real_preprocessing():
    """Test con dati che simulano un vero dataset immobiliare"""
    
    # Simula dataset più realistico
    np.random.seed(42)
    n_samples = 500
    
    # Features categoriche realistiche
    locations = ['Centro Storico', 'Periferia Nord', 'Periferia Sud', 'Mare', 'Collina', 'Industriale']
    conditions = ['Ottimo', 'Buono', 'Da ristrutturare', 'Nuovo']
    types = ['Appartamento', 'Villa', 'Attico', 'Loft']
    
    real_data = pd.DataFrame({
        # Numeric features
        'sqm': np.random.randint(40, 250, n_samples),
        'rooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'floor': np.random.randint(0, 15, n_samples),
        'year_built': np.random.randint(1950, 2023, n_samples),
        
        # Categorical features
        'location': np.random.choice(locations, n_samples),
        'condition': np.random.choice(conditions, n_samples),
        'property_type': np.random.choice(types, n_samples),
        
        # Boolean features
        'has_elevator': np.random.choice([True, False], n_samples),
        'has_parking': np.random.choice([True, False], n_samples),
        'has_garden': np.random.choice([True, False], n_samples),
        'has_terrace': np.random.choice([True, False], n_samples),
    })
    
    # Aggiungi qualche missing value per testare robustezza
    real_data.loc[np.random.choice(n_samples, 20, replace=False), 'condition'] = np.nan
    real_data.loc[np.random.choice(n_samples, 15, replace=False), 'year_built'] = np.nan
    
    # Crea target realistico
    base_price = real_data['sqm'] * np.random.normal(2500, 500, n_samples)
    
    # Bonus per location
    location_multiplier = real_data['location'].map({
        'Centro Storico': 1.5, 'Mare': 1.3, 'Collina': 1.2,
        'Periferia Nord': 0.9, 'Periferia Sud': 0.8, 'Industriale': 0.7
    })
    
    # Bonus per condition
    condition_multiplier = real_data['condition'].map({
        'Nuovo': 1.3, 'Ottimo': 1.1, 'Buono': 1.0, 'Da ristrutturare': 0.7
    }).fillna(1.0)  # Default per missing
    
    real_data['price'] = (base_price * location_multiplier * condition_multiplier * 
                         np.random.normal(1, 0.1, n_samples)).astype(int)
    
    print(f"📊 Generated realistic dataset: {real_data.shape}")
    print("\n📋 Dataset info:")
    print(real_data.info())
    print(f"\n💰 Price stats: mean={real_data['price'].mean():.0f}, std={real_data['price'].std():.0f}")
    
    # Test preprocessing
    print("\n🔧 Testing preprocessing...")
    preprocessor = RealDataPreprocessor()
    
    # Split data
    train_df, test_df = train_test_split(real_data, test_size=0.2, random_state=42)
    
    # Fit on training data
    X_train_cont, X_train_cat, y_train = preprocessor.fit_transform(train_df)
    
    print(f"\n✅ PREPROCESSING RESULTS:")
    print(f"   Continuous features shape: {X_train_cont.shape}")
    print(f"   Categorical features: {list(X_train_cat.keys())}")
    for col, values in X_train_cat.items():
        print(f"     {col}: shape={values.shape}, unique_values={len(np.unique(values))}")
    print(f"   Target shape: {y_train.shape}")
    
    # Test on test data
    X_test_cont, X_test_cat, y_test = preprocessor.transform(test_df)
    print(f"\n🧪 Test transform successful: {X_test_cont.shape}, target: {y_test.shape}")
    
    return real_data, preprocessor, (X_train_cont, X_train_cat, y_train), (X_test_cont, X_test_cat, y_test)

# Esegui test
dataset, preprocessor, train_data, test_data = test_real_preprocessing()


class StructuredEncoderReal(nn.Module):
    """Structured encoder che gestisce correttamente dati categoriali"""
    
    def __init__(self, 
                 continuous_dim: int,          # Dimensione features continue (numeric + boolean)
                 embedding_configs: Dict,      # Config per embedding layers
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 256,
                 dropout: float = 0.3):
        
        super().__init__()
        
        self.continuous_dim = continuous_dim
        self.embedding_configs = embedding_configs
        
        print(f"🏗️  Building StructuredEncoder:")
        print(f"   Continuous dim: {continuous_dim}")
        print(f"   Embedding configs: {list(embedding_configs.keys())}")
        
        # 1. Embedding layers per features categoriche
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feature_name, config in embedding_configs.items():
            vocab_size = config['vocab_size']
            embed_dim = config['embed_dim']
            
            embedding_layer = nn.Embedding(vocab_size, embed_dim)
            # Initialize embeddings
            nn.init.normal_(embedding_layer.weight, mean=0, std=0.1)
            
            self.embeddings[feature_name] = embedding_layer
            total_embedding_dim += embed_dim
            
            print(f"     {feature_name}: vocab_size={vocab_size}, embed_dim={embed_dim}")
        
        # 2. Input dimension per MLP = continuous + embeddings
        mlp_input_dim = continuous_dim + total_embedding_dim
        print(f"   Total MLP input dim: {mlp_input_dim}")
        
        # 3. MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 4. Initialize weights
        self._init_weights()
        
        print(f"   Output dim: {output_dim}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize weights per better performance"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, continuous_features, categorical_features):
        """
        Forward pass
        
        Args:
            continuous_features: Tensor [batch_size, continuous_dim]
            categorical_features: Dict {feature_name: tensor [batch_size]}
        
        Returns:
            output: Tensor [batch_size, output_dim]
        """
        batch_size = continuous_features.shape[0]
        
        # 1. Process embeddings
        embedded_features = []
        
        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name in categorical_features:
                # Get indices per questo feature
                indices = categorical_features[feature_name]  # [batch_size]
                
                # Ensure indices sono long type e in range corretto
                indices = torch.clamp(indices.long(), 0, embedding_layer.num_embeddings - 1)
                
                # Get embeddings
                embedded = embedding_layer(indices)  # [batch_size, embed_dim]
                embedded_features.append(embedded)
        
        # 2. Concatenate all features
        if embedded_features:
            # Concatena embeddings
            embedded_concat = torch.cat(embedded_features, dim=1)  # [batch_size, total_embed_dim]
            # Combina continuous + embeddings
            x = torch.cat([continuous_features, embedded_concat], dim=1)
        else:
            # Solo continuous features
            x = continuous_features
        
        # 3. Pass through MLP
        output = self.mlp(x)  # [batch_size, output_dim]
        
        return output

# Custom Dataset per gestire i nostri dati processati
class StructuredDataset(torch.utils.data.Dataset):
    """Dataset per dati strutturati processati"""
    
    def __init__(self, continuous_features, categorical_features, targets):
        self.continuous_features = torch.FloatTensor(continuous_features)
        self.categorical_features = {
            name: torch.LongTensor(values) 
            for name, values in categorical_features.items()
        }
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
        # Verify all data ha stesso numero di samples
        n_samples = len(self.continuous_features)
        for name, values in self.categorical_features.items():
            assert len(values) == n_samples, f"Length mismatch for {name}: {len(values)} vs {n_samples}"
        
        if self.targets is not None:
            assert len(self.targets) == n_samples, f"Target length mismatch: {len(self.targets)} vs {n_samples}"
    
    def __len__(self):
        return len(self.continuous_features)
    
    def __getitem__(self, idx):
        item = {
            'continuous': self.continuous_features[idx],
            'categorical': {name: values[idx] for name, values in self.categorical_features.items()}
        }
        
        if self.targets is not None:
            item['target'] = self.targets[idx]
            
        return item

# Test del modello con dati reali
def test_model_with_real_data():
    """Test completo: preprocessing + model + forward pass"""
    
    print("🧪 TESTING MODEL WITH REAL DATA")
    
    # Usa dati dal test precedente
    X_train_cont, X_train_cat, y_train = train_data
    X_test_cont, X_test_cat, y_test = test_data
    
    print(f"📊 Data shapes:")
    print(f"   Train continuous: {X_train_cont.shape}")
    print(f"   Train categorical: {[(k, v.shape) for k, v in X_train_cat.items()]}")
    print(f"   Train targets: {y_train.shape}")
    
    # 1. Create model
    model = StructuredEncoderReal(
        continuous_dim=X_train_cont.shape[1],
        embedding_configs=preprocessor.embedding_configs,
        hidden_dims=[256, 128],
        output_dim=256,
        dropout=0.2
    )
    
    # 2. Create datasets
    train_dataset = StructuredDataset(X_train_cont, X_train_cat, y_train)
    test_dataset = StructuredDataset(X_test_cont, X_test_cat, y_test)
    
    # 3. Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ DataLoaders created: train={len(train_loader)} batches, test={len(test_loader)} batches")
    
    # 4. Test forward pass
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        
        continuous = batch['continuous']
        categorical = batch['categorical']
        targets = batch['target']
        
        print(f"\n🔬 BATCH TEST:")
        print(f"   Batch continuous shape: {continuous.shape}")
        print(f"   Batch categorical shapes: {[(k, v.shape) for k, v in categorical.items()]}")
        print(f"   Batch target shape: {targets.shape}")
        
        # Forward pass
        outputs = model(continuous, categorical)
        
        print(f"   Model output shape: {outputs.shape}")
        print(f"   Expected: ({continuous.shape[0]}, 256)")
        
        assert outputs.shape == (continuous.shape[0], 256), f"Wrong output shape: {outputs.shape}"
        print("✅ Forward pass successful!")
    
    return model, train_loader, test_loader

# Esegui test completo
model, train_loader, test_loader = test_model_with_real_data()

import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class StructuredTrainerReal:
    """Trainer per structured encoder con dati reali"""
    
    def __init__(self, model, lr=0.001, weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Training per una epoca"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Get batch data
            continuous = batch['continuous']
            categorical = batch['categorical']
            targets = batch['target'].unsqueeze(1)  # [batch_size, 1]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(continuous, categorical)
            
            # Loss computation
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                continuous = batch['continuous']
                categorical = batch['categorical']
                targets = batch['target'].unsqueeze(1)
                
                predictions = self.model(continuous, categorical)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                n_batches += 1
                
                # Store per calcolare metriche aggiuntive
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # Calcola metriche aggiuntive
        all_predictions = np.concatenate(predictions_list)
        all_targets = np.concatenate(targets_list)
        
        mae = np.mean(np.abs(all_predictions - all_targets))
        mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
        
        return total_loss / n_batches, mae, mape
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """Training loop completo con early stopping"""
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_mae, val_mape = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Val MAE={val_mae:.0f}, "
                      f"Val MAPE={val_mape:.1f}%, "
                      f"LR={current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Loaded best model with val_loss={self.best_val_loss:.4f}")
        
        print("✅ Training completed!")
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (Log Scale)')
        plt.title('Training Curves (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# TRAINING COMPLETO
def run_complete_training():
    """Esegui training completo con dati reali"""
    
    print("🎯 STARTING COMPLETE TRAINING PIPELINE")
    
    # Usa modello e data loaders dal test precedente
    trainer = StructuredTrainerReal(model, lr=0.001, weight_decay=1e-4)
    
    # Train model
    train_losses, val_losses = trainer.train(
        train_loader, test_loader, 
        epochs=50, 
        early_stopping_patience=10
    )
    
    # Plot results
    trainer.plot_training_curves()
    
    # Final evaluation
    final_val_loss, final_mae, final_mape = trainer.validate(test_loader)
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"   Final MAE: {final_mae:.0f}")
    print(f"   Final MAPE: {final_mape:.1f}%")
    
    # Test prediction
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        predictions = model(sample_batch['continuous'], sample_batch['categorical'])
        targets = sample_batch['target']
        
        print(f"\n🔍 SAMPLE PREDICTIONS:")
        for i in range(min(5, len(predictions))):
            pred = predictions[i].item()
            target = targets[i].item()
            error = abs(pred - target) / target * 100
            print(f"   Pred: {pred:.0f}, Target: {target:.0f}, Error: {error:.1f}%")
    
    return trainer

# Esegui training completo
trained_trainer = run_complete_training()