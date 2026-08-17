import torch.nn as nn

class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim=320, hidden_dim1=128, hidden_dim2=32, dropout=0.2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            ## Next layer
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            ## Final layer
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)



import torch
import copy
import torch.nn as nn
from .evaluate import evaluate_regression

def train_regression(dataloader, model, num_epochs, lr = 0.001, weight_decay = 0, validation_dataloader = None, 
patience = None, min_delta = 0.0):

    """ Trains a neural network model. 
   
    Takes a neural network model and trains it. Has two additional tunable parameters. 
    
    args: 
        dataloader (dataloder cass): PyTorch dataloader class
        model (class): PyTorch nn.model class
        num_epochs (integer): number of epoch to train
        lr (float): the learning rate of the model
        weight_decay (float): decay rate of adam optimizer
        validation_dataloader (dataloder cass):  If optional validation_dataloader 
                                                 is included the function returns validation loss.
        patience (integer): how many epoch model goes without drecrease in loss
        min_delta (float): amount by which loss must decrease to be considered sequential improvement

    returns:
        trained model
        history: dictionary containing train loss per epoch and 
                 validation loss per epoch (if validation_dataloader is not None)
    """

    ## Set model to train
    model.train()

    ## Capture loss
    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    ## Define loss function, use MSE loss for regression 
    criterion = nn.MSELoss()

    ## Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    ## Define training loop
    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0

        for inputs, targets in dataloader:

            ## Set gradients to zero
            optimizer.zero_grad()

            ## Run model to get predictions
            outputs = model(inputs)

            ## Compute loss metric 
            loss = criterion(outputs, targets)

            ## Run backward pass to calculate gradients 
            loss.backward()

            ## Update optimizer weights
            optimizer.step()

            ## Record loss
            total_train_loss += loss.item() * inputs.size(0)

        

        average_train_loss = total_train_loss / len(dataloader.dataset)
        history["train_loss"].append(average_train_loss)
        

        ## If validation dataloader is inputed function then calls evaluation function
        if validation_dataloader is not None:
            validation_loss, _, _ = evaluate_regression(validation_dataloader, model)
            history["validation_loss"].append(validation_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} complete.", 
                f"Train loss: {average_train_loss:.4f}",
                f"Validation loss: {validation_loss:.4f}")

            improved = validation_loss < (best_validation_loss - min_delta)

            if improved:
                best_validation_loss = validation_loss
                best_model_state = copy.deepcopy(
                    model.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (patience is not None and epochs_without_improvement >= patience):
                print("Early stopping triggered at ",
                f"epoch {epoch + 1}.")
                break
            
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} complete.", 
                f"Train loss: {average_train_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    ## Return both training loss and the model
    return model, history