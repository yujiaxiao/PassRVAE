import numpy as np
import torch
'''
class EarlyStopping:
    def __init__(self, patience=1,verbose=False):
        self.patience = patience
        self.slope_threshold = 0
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = 0
        self.early_stop = False
        self.slope1=0
        self.slope2=0

    def __call__(self, val_loss, model):
        if val_loss > self.best_val_loss and self.slope1==0:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        elif val_loss > self.best_val_loss and self.slope1!=0:
            self.slope2 = (val_loss - self.best_val_loss).item()
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)     
        else:
            self.slope2 = (val_loss - self.best_val_loss).item()
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        self.slope1=self.slope2

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), "/content/gdrive/MyDrive/PassRVAE/best_model.pt")
'''
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')	
        self.val_loss_min = val_loss