"""
RobCrop Training Script - RTX 4070 8GB Optimized
Academic-quality training for agricultural disease detection
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import our modules
from config import Config
from model import create_model
from data_loader import create_robcrop_data_loaders
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class RobCropTrainer:
    """
    Academic-quality trainer for RobCrop agricultural disease detection
    Optimized for RTX 4070 8GB VRAM
    """
    
    def __init__(self, config=None):
        # Update config for split data
        Config.DATA_DIR = Path("data_split")
        
        self.config = Config
        self.device = self.config.DEVICE
        
        # Create timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"robcrop_{self.timestamp}"
        
        # Create output directories
        self.output_dir = Path("training_outputs")
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.run_name
        self.logs_dir = self.output_dir / "logs" / self.run_name
        self.results_dir = self.output_dir / "results" / self.run_name
        
        for dir_path in [self.checkpoint_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        print(f"🚀 RobCrop Trainer initialized for run: {self.run_name}")
        self._print_setup_summary()
    
    def _setup_model(self):
        """Initialize the ResNet50 model"""
        print("🏗️  Setting up RobCrop ResNet50 model...")
        self.model = create_model().to(self.device)
        
        # Get model info
        model_info = self.model.get_model_info()
        print(f"📊 Model: {model_info['trainable_params']:,} trainable parameters")
        print(f"💾 Model size: {model_info['model_size_mb']:.1f} MB")
    
    def _setup_data(self):
        """Initialize data loaders"""
        print("📂 Setting up RobCrop data loaders...")
        
        self.loaders, self.datasets = create_robcrop_data_loaders(
            use_cache=True,
            use_weighted_sampler=False  # Can enable if needed
        )
        
        # Verify we have all splits
        required_splits = ['train', 'val']
        missing_splits = [split for split in required_splits if split not in self.loaders]
        
        if missing_splits:
            raise ValueError(f"Missing required data splits: {missing_splits}")
        
        self.train_loader = self.loaders['train']
        self.val_loader = self.loaders['val']
        self.test_loader = self.loaders.get('test', None)
        
        print(f"✅ Data loaded: {len(self.datasets['train'])} train, {len(self.datasets['val'])} val")
    
    def _setup_training(self):
        """Initialize training components"""
        print("⚙️  Setting up training components...")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - only train unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
        self.optimizer,
        mode='max',           # Monitor validation accuracy
        factor=0.5,           # Reduce LR by half
        patience=7,           # Wait 7 epochs
        min_lr=1e-7,          # Minimum learning rate
        threshold=0.01,       # Improvement threshold
        cooldown=0            # Epochs to wait after LR reduction
    )

        
        # Alternative: Cosine annealing (uncomment to use)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-7)
        
        print(f"🎯 Optimizer: Adam (lr={self.config.LEARNING_RATE})")
        print(f"📈 Scheduler: ReduceLROnPlateau")
    
    def _setup_logging(self):
        """Initialize logging and monitoring"""
        self.writer = SummaryWriter(self.logs_dir)
        
        # Save configuration
        config_dict = {
            'model': 'ResNet50',
            'num_classes': self.config.NUM_CLASSES,
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'device': str(self.device),
            'epochs': self.config.EPOCHS,
            'timestamp': self.timestamp
        }
        
        with open(self.results_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _print_setup_summary(self):
        """Print training setup summary"""
        print("\n🎯 RobCrop Training Setup Summary")
        print("=" * 50)
        print(f"🏷️  Run name: {self.run_name}")
        print(f"🖥️  Device: {self.device}")
        print(f"📊 Dataset: {len(self.datasets['train'])} train, {len(self.datasets['val'])} val")
        print(f"🎯 Classes: {self.config.NUM_CLASSES} agricultural diseases")
        print(f"📦 Batch size: {self.config.BATCH_SIZE} (8GB VRAM optimized)")
        print(f"🔄 Epochs: {self.config.EPOCHS}")
        print(f"📈 Learning rate: {self.config.LEARNING_RATE}")
        print(f"💾 Outputs: {self.output_dir}")
        print("=" * 50)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = torch.tensor(0, dtype=torch.float32, device=self.device)
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Train]',
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to GPU
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
            # Update progress bar
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log batch metrics to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 50 == 0:  # Log every 50 batches
                self.writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
                self.writer.add_scalar('Batch/Train_Acc', current_acc, global_step)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # For confusion matrix (optional)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, 
                desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Val]',
                leave=False
            )
            
            for images, labels in pbar:
                # Move data to GPU
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += images.size(0)
                
                # Store predictions for analysis
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_acc = running_corrects.double() / total_samples
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples
        
        return epoch_loss, epoch_acc, (all_preds, all_labels)
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'training_history': self.training_history,
            'config': {
                'num_classes': self.config.NUM_CLASSES,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved: {val_acc:.4f} accuracy")
    
    def plot_training_progress(self):
        """Plot and save training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Loss Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Accuracy Progress')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(epochs, self.training_history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Best accuracy highlight
        best_epoch = np.argmax(self.training_history['val_acc']) + 1
        best_acc = max(self.training_history['val_acc'])
        ax4.plot(epochs, self.training_history['val_acc'], 'r-', label='Val Accuracy')
        ax4.axhline(y=best_acc, color='g', linestyle='--', label=f'Best: {best_acc:.4f}')
        ax4.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
        ax4.set_title('Validation Accuracy with Best')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting RobCrop Training!")
        print(f"🎯 Target: High accuracy for agricultural disease detection")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.EPOCHS):
                print(f"\n{'='*20} Epoch {epoch+1}/{self.config.EPOCHS} {'='*20}")
                
                # Training phase
                train_loss, train_acc = self.train_epoch(epoch)
                
                # Validation phase
                val_loss, val_acc, (val_preds, val_labels) = self.validate_epoch(epoch)
                
                # Update learning rate scheduler
                self.scheduler.step(val_acc)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['learning_rates'].append(current_lr)
                
                # TensorBoard logging
                self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
                self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
                
                # Print epoch results
                print(f"📊 Results:")
                print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"   Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
                print(f"   Learning Rate: {current_lr:.7f}")
                
                # Save checkpoint and check for best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                # Save checkpoint every 5 epochs or if best
                if (epoch + 1) % 5 == 0 or is_best:
                    self.save_checkpoint(epoch, val_acc, is_best)
                
                # Early stopping check
                if current_lr < 1e-6:
                    print(f"\n⏰ Early stopping: Learning rate too low ({current_lr:.2e})")
                    break
                
                # Plot progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.plot_training_progress()
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Training interrupted by user at epoch {epoch+1}")
        
        # Training completed
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        
        print(f"\n🎊 Training Completed!")
        print(f"⏱️  Total time: {hours}h {minutes}m")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Load best model and final save
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Final checkpoint and plots
        self.save_checkpoint(epoch, self.best_val_acc, is_best=True)
        self.plot_training_progress()
        
        # Save training history
        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.model, self.training_history
    
    def evaluate_on_test(self):
        """Evaluate the best model on test set"""
        if self.test_loader is None:
            print("⚠️  No test set available for evaluation")
            return None
        
        print("\n🧪 Evaluating on test set...")
        
        self.model.eval()
        running_corrects = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = float(running_corrects) / total_samples
        
        print(f"🎯 Test Accuracy: {test_acc:.4f}")
        
        # Save test results
        test_results = {
            'test_accuracy': float(test_acc),
            'total_samples': total_samples,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        with open(self.results_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, cls=NumpyEncoder, indent=2)

        
        return test_acc

def main():
    """Main training function"""
    print("🌱 Welcome to RobCrop Agricultural Disease Detection Training!")
    print("🎯 RTX 4070 8GB Optimized for Academic Excellence")
    
    try:
        # Create and run trainer
        trainer = RobCropTrainer()
        
        # Train the model
        model, history = trainer.train()
        
        # Evaluate on test set
        test_acc = trainer.evaluate_on_test()
        
        print(f"\n🎊 RobCrop Training Complete!")
        print(f"🏆 Best Validation Accuracy: {trainer.best_val_acc:.4f}")
        if test_acc:
            print(f"🎯 Final Test Accuracy: {test_acc:.4f}")
        
        print(f"\n📁 Results saved to: {trainer.output_dir}")
        print(f"🚀 Your agricultural rover AI is ready!")
        
        return trainer
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
