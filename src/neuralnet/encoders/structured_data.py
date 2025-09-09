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
from tqdm import tqdm

import data_preprocessor as dapre

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
        self.input_dim = input_dim
        self.embedding_configs = embedding_configs
        
        ## Categorial Data Embedding ##
        self.embeddings = nn.ModuleDict() #permette di conservare sottomoduli
        total_embedding_dim = 0
        
        for feature_name, config in embedding_configs.items():
            vocab_size = config['vocab_size'] #numero categorie uniche
            embed_dim = config['embed_dim']
            
            embedding_layer = nn.Embedding(vocab_size, embed_dim) 
            nn.init.normal_(embedding_layer.weight, mean=0, std=0.1)

            self.embeddings[feature_name] = embedding_layer
            total_embedding_dim += embed_dim
            print(f"   Embedding {feature_name}: vocab_size={vocab_size} -> embed_dim={embed_dim}")
        
        #input dimension for MLP
        mlp_input_dim = self.input_dim + total_embedding_dim

        #####   MLP LAYERS   #####
        # - Input Layers
        layers = []
        prev_dim= mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([  #metodo per appendere liste a liste
                nn.Linear(prev_dim, hidden_dim),
                #nn.Linear(prev_dim, hidden_dim),
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
            #elif isinstance(module, nn.Embedding):
             #   nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, 
                continuous_features, 
                categorical_features):
        """
        Forward pass
        
        Args:
            continuous_features: Tensor [batch_size, continuous_dim]
            categorical_features: Dict {feature_name: tensor [batch_size]}
        
        Returns:
            output: Tensor [batch_size, output_dim]
        """
        batch_size = continuous_features.shape[0]
        
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

class StructuredTrainer:
    def __init__(self, 
                 model, 
                 lr=0.001, 
                 weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode='min', 
                                                              factor=0.5, 
                                                              patience=5)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    '''
    def create_dataloader(self, 
                          X, 
                          y, 
                          batch_size=32, 
                          shuffle = True):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    '''

    def train_epoch(self, 
                    train_loader):
        '''One-Epoch Training. Definizione di come si deve svolgere l'epoca di addestramento'''
        self.model.train() #funzione definita sotto
        total_loss = 0
        n_batches  = 0

        #for batch_x, batch_y in train_loader:
        for batch in tqdm(train_loader, desc='training'):
            # Zero gradients
            continuous  = batch['continuous']
            categorical = batch['categorical']
            targets     = batch['target'].unsqueeze(1)

            self.optimizer.zero_grad() #reset valori gradiente
            
            # Forward pass (senza embeddings per semplicita)
            predictions = self.model(continuous, categorical)
            
            # Loss computation
            loss = self.criterion(predictions, targets)
            
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
        
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                continuous  = batch['continuous']
                categorical = batch['categorical']
                targets     = batch['target'].unsqueeze(1)
                
                predictions = self.model(continuous, categorical)
                loss        = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                n_batches  += 1
                
                # Store per calcolare metriche aggiuntive
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

        #metriche aggiuntive
        all_predictions = np.concatenate(predictions_list)
        all_targets     = np.concatenate(targets_list)
        mae             = np.mean(np.abs(all_predictions - all_targets))
        mape            = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100

        return total_loss / n_batches, mae, mape

    def train(self, 
              train_loader,
              val_loader,
              epochs=100,
              early_stopping_patience=15, 
              batch_size=32):

        print(f"Starting training for {epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        patience_counter = 0

        for epoch in range(epochs):
            #train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            #validate
            val_loss, val_mae, val_mape = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)

            #check per early-stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            #Progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Val MAE={val_mae:.0f}, "
                      f"Val MAPE={val_mape:.1f}%, "
                      f"LR={current_lr:.2e}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Loaded best model with val_loss={self.best_val_loss:.4f}")
        
        print("✅ Training completed!")
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self, output_name):
        """Plot training e validation loss"""
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
        print("Gradient computation successful")
        
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
#trained_model, trainer = tester.test_full()█████████████████████████████████████████████████████████████

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
    print('\n\nVediamo forma di embedding configs')
    print(embedding_configs)
    return embedding_configs


if __name__=='__main__':
    dataset = 'data/output-datasets/house_df.csv'
    print("Hi\nI am the structured data encoder, and I am going to encode your dataset")
    
    df = pd.read_csv(dataset)
    
    preprocessor = dapre.RealDataPreprocessor()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    X_train_cont, X_train_cat, y_train = preprocessor.fit_transform(train_df)
    X_test_cont, X_test_cat, y_test    = preprocessor.transform(test_df)

    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #embedding_configs = create_embedding_config(df)
    model = StructuredEncoder(
            input_dim         = X_train_cont.shape[1],
            embedding_configs = preprocessor.embedding_configs, 
            hidden_dims       = [512, 256],
            output_dim        = 256,
            dropout           = 0.2
        )
    
    #create datasets + loader
    train_dataset = StructuredDataset(X_train_cont, X_train_cat, y_train)
    test_dataset  = StructuredDataset(X_test_cont, X_test_cat, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False)
    
    print(f"DataLoaders created: train={len(train_loader)} batches, test={len(test_loader)} batches")

    trainer = StructuredTrainer(model, lr=0.001)
    print(train_loader)
    train_losses, val_losses = trainer.train(train_loader, test_loader, 
                                             epochs=50, 
                                             early_stopping_patience=10)

    trainer.plot_training_curves('full_dataset')
    
    #final val
    final_val_loss, final_mae, final_mape = trainer.validate(test_loader)
    print(f"\n FINAL RESULTS:")
    print(f"   Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"   Final MAE: {final_mae:.0f}")
    print(f"   Final MAPE: {final_mape:.1f}%")

    #test
    model.eval()
    
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        predictions  = model(sample_batch['continuous'], sample_batch['categorical'])
        targets      = sample_batch['target']
        
        print(f"\n\n-----SAMPLE PREDICTIONS:-----")
        for i in range(min(5, len(predictions))):
            pred   = predictions[i]
            target = targets[i]
            error  = abs(pred - target) / target * 100
            print(f"   Pred: {pred}, Target: {target}, Error: {error}%")