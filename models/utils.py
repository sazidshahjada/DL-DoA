import os
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, device=None, save_path="checkpoints", patience=7):
        """
        Args:
        model (nn.Module): The neural network model to train.
        criterion: Loss function to optimize.
        optimizer: Optimization algorithm (e.g., Adam, SGD).
        device: torch.device to run training on (e.g., 'cuda' or 'cpu'). If None, automatically selects CUDA if available.
        save_path (str): Directory to save model.
        patience (int): How many epochs to wait for improvement before stopping.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y, _ in tqdm(train_loader, desc="Training", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc="Validation", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=100):
        print(f"Training started on {self.device}...")
        
        for epoch in range(epochs):
            t_loss = self.train_epoch(train_loader)
            v_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train: {t_loss:.6f} | Val: {v_loss:.6f}")

            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                self.counter = 0 
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pth"))
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print("Training complete.")