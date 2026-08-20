
import torch
import time
import copy
import torch.nn as nn
from .evaluate import evaluate_finetune

def train_finetune(dataloader, model, device, num_epochs, lr_esm = 5e-5, lr_reghead = 1e-3, weight_decay = 1e-4, validation_dataloader = None, 
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

    model = model.to(device)

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

    ## Define optimizer, filter for only unfrozen weights to optimize
    #optimizer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, model.parameters()),
    #     lr = lr, weight_decay = weight_decay)
    optimizer = torch.optim.Adam([
    {
        "params": model.esm.encoder.layer[-1].parameters(),
        "lr": lr_esm
    },
    {
        "params": model.regressor.parameters(),
        "lr": lr_reghead
    }], weight_decay=weight_decay)

    ## Timing each epoch
    start = time.time()

    ## Define training loop
    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0

        for batch in dataloader:
            
            wt_tokens = {key: value.to(device) for key, value in batch['wt_tokens'].items()}

            mut_tokens = {key: value.to(device) for key, value in batch['mut_tokens'].items()}

            positions = batch["positions"].to(device)
            
            targets = batch["targets"].to(device)



            ## Set gradients to zero
            optimizer.zero_grad()

            ## Run model to get predictions
            outputs = model(
                wt_tokens,
                mut_tokens,
                positions
            )

            ## Compute loss metric 
            loss = criterion(outputs, targets)

            ## Run backward pass to calculate gradients 
            loss.backward()

            ## Update optimizer weights
            optimizer.step()

            ## Record loss
            total_train_loss += loss.item() * targets.size(0)

        
        average_train_loss = total_train_loss / len(dataloader.dataset)
        history["train_loss"].append(average_train_loss)
        

        ## If validation dataloader is inputed function then calls evaluation function
        if validation_dataloader is not None:
            validation_loss, _, _ = evaluate_finetune(validation_dataloader, model, device)
            history["validation_loss"].append(validation_loss)

            print(
                f"Epoch {epoch+1}/{num_epochs} complete.", 
                f"Train loss: {average_train_loss:.4f}",
                f"Validation loss: {validation_loss:.4f}")
            print("Elapsed minutes:", (time.time() - start) / 60)

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
            print("Elapsed minutes:", (time.time() - start) / 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    ## Return both training loss and the model
    return model, history