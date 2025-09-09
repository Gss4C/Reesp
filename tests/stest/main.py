"""
Main script per training del Structured Data Encoder
Parte da un CSV con colonna "SalePrice" come target
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import delle nostre classi (assumendo che siano nel file structured_encoder.py)
from encoder import StructuredDataPreprocessor, StructuredDataEncoder, StructuredDataTrainer

def load_and_explore_data(csv_path):
    """
    Carica e esplora il dataset CSV
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Verifica presenza colonna target
    if 'SalePrice' not in df.columns:
        raise ValueError("❌ Colonna 'SalePrice' non trovata nel dataset!")
    
    print(f"Target (SalePrice) - Min: ${df['SalePrice'].min():,.0f}, Max: ${df['SalePrice'].max():,.0f}")
    print(f"Target (SalePrice) - Mean: ${df['SalePrice'].mean():,.0f}, Median: ${df['SalePrice'].median():,.0f}")
    
    return df

def prepare_data(df, test_size=0.2, val_size=0.1):
    """
    Prepara i dati per il training
    """
    print("\n🔧 Preparing data...")
    
    # Separa features e target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice'].values
    
    # Preprocessing delle features
    preprocessor = StructuredDataPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Features after preprocessing: {X_processed.shape[1]}")
    print(f"Feature names: {len(preprocessor.get_feature_names())}")
    
    # Converti in tensori
    X_tensor = torch.FloatTensor(X_processed)
    y_tensor = torch.FloatTensor(y)
    
    # Normalizza il target (log transform per stabilità)
    y_log = torch.log1p(y_tensor)  # log(1 + y) per evitare log(0)
    
    print(f"Target transformation: Original range [{y_tensor.min():.0f}, {y_tensor.max():.0f}]")
    print(f"Target transformation: Log range [{y_log.min():.3f}, {y_log.max():.3f}]")
    
    # Split dei dati
    dataset = TensorDataset(X_tensor, y_log)
    total_size = len(dataset)
    
    test_count = int(total_size * test_size)
    val_count = int(total_size * val_size)
    train_count = total_size - test_count - val_count
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_count, val_count, test_count],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, preprocessor, X_processed.shape[1]

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Crea i DataLoader
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, input_dim, device='cpu', epochs=50):
    """
    Training del modello
    """
    print(f"\n🚀 Training model on {device}...")
    
    # Crea il modello
    model = StructuredDataEncoder(
        input_dim=input_dim,
        output_dim=256,
        hidden_dims=[512, 384, 256],
        dropout_rate=0.3,
        use_batch_norm=True
    )
    
    # Aggiungi regression head (da 256 a 1)
    regression_head = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    
    # Modello completo
    complete_model = nn.Sequential(model, regression_head)
    
    print(f"Model parameters: {sum(p.numel() for p in complete_model.parameters()):,}")
    
    # Training
    trainer = StructuredDataTrainer(complete_model, device=device)
    trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001, patience=10)
    
    return trainer.model, trainer

def evaluate_model(model, test_loader, device='cpu'):
    """
    Valutazione finale del modello
    """
    print("\n📊 Evaluating model...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            outputs = model(batch_x)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_y.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Converti da log space a prezzo originale
    predictions_price = np.expm1(predictions)  # inverse of log1p
    targets_price = np.expm1(targets)
    
    # Calcola metriche
    mse = mean_squared_error(targets_price, predictions_price)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_price, predictions_price)
    r2 = r2_score(targets_price, predictions_price)
    
    # Normalized RMSE (rispetto alla media del target)
    normalized_rmse = rmse / np.mean(targets_price)
    
    print(f"📈 Test Results:")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Normalized RMSE: {normalized_rmse:.4f}")
    
    return {
        'predictions': predictions_price,
        'targets': targets_price,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'normalized_rmse': normalized_rmse
    }

def plot_results(trainer, results):
    """
    Visualizza i risultati
    """
    print("\n📊 Creating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training curves
    axes[0, 0].plot(trainer.train_losses, label='Train Loss', alpha=0.8)
    axes[0, 0].plot(trainer.val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predictions vs Targets
    axes[0, 1].scatter(results['targets'], results['predictions'], alpha=0.6, s=20)
    min_price = min(results['targets'].min(), results['predictions'].min())
    max_price = max(results['targets'].max(), results['predictions'].max())
    axes[0, 1].plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Price ($)')
    axes[0, 1].set_ylabel('Predicted Price ($)')
    axes[0, 1].set_title(f'Predictions vs True Values (R² = {results["r2"]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals plot
    residuals = results['predictions'] - results['targets']
    axes[1, 0].scatter(results['predictions'], residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Predicted Price ($)')
    axes[1, 0].set_ylabel('Residuals ($)')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error distribution
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Residuals ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('structured_encoder_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function
    """
    print("🏠 Real Estate Price Prediction - Structured Data Encoder")
    print("=" * 60)
    
    # Configurazioni
    CSV_PATH = 'data/output-datasets/house_df.csv' # Cambia con il tuo path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print(f"Device: {DEVICE}")
    
    try:
        # 1. Load e explore data
        df = load_and_explore_data(CSV_PATH)
        
        # 2. Prepare data
        train_dataset, val_dataset, test_dataset, preprocessor, input_dim = prepare_data(df)
        
        # 3. Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, BATCH_SIZE
        )
        
        # 4. Train model
        model, trainer = train_model(train_loader, val_loader, input_dim, DEVICE, EPOCHS)
        
        # 5. Evaluate model
        results = evaluate_model(model, test_loader, DEVICE)
        
        # 6. Plot results
        plot_results(trainer, results)
        
        # 7. Save model e preprocessor
        print("\n💾 Saving model and preprocessor...")
        torch.save(model.state_dict(), 'structured_encoder_complete.pth')
        preprocessor.save('preprocessor.pkl')
        
        # 8. Results summary
        print("\n" + "="*60)
        print("🎯 FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"✅ Model training completed successfully!")
        print(f"✅ Test RMSE: ${results['rmse']:,.0f}")
        print(f"✅ Test R²: {results['r2']:.4f}")
        print(f"✅ Normalized RMSE: {results['normalized_rmse']:.4f}")
        
        # Check se rispetta i target del progetto
        target_r2 = 0.85
        target_norm_rmse = 0.15
        
        if results['r2'] > target_r2:
            print(f"🎉 R² target achieved! ({results['r2']:.4f} > {target_r2})")
        else:
            print(f"⚠️  R² target not met ({results['r2']:.4f} < {target_r2})")
            
        if results['normalized_rmse'] < target_norm_rmse:
            print(f"🎉 RMSE target achieved! ({results['normalized_rmse']:.4f} < {target_norm_rmse})")
        else:
            print(f"⚠️  RMSE target not met ({results['normalized_rmse']:.4f} > {target_norm_rmse})")
        
        print("\n📁 Files saved:")
        print("  - structured_encoder_complete.pth (model weights)")
        print("  - preprocessor.pkl (data preprocessor)")
        print("  - structured_encoder_results.png (plots)")
        
        return model, preprocessor, results
        
    except FileNotFoundError:
        print(f"❌ File {CSV_PATH} not found!")
        print("   Make sure to place your CSV file in the current directory")
        print("   and update the CSV_PATH variable in the main() function")
        return None, None, None
    
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Imposta semi per riproducibilità
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, preprocessor, results = main()