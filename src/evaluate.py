import torch
import torch.nn as nn

def evaluate_regression(dataloader, model):

    """ Evaluates a trained regression neural network. 
   
    Takes a trained neural network model and evaluates it on given dataset
    
    args: 
        dataloader (dataloder cass): PyTorch dataloader class
        model (class): PyTorch nn.model class
    returns:
        accuracy (float): percentage of correct predictions
        average_loss (float): the average MSE Loss. 
    """

    ## Remember if it was training and restore at end
    was_training = model.training

    ## Set model to evaluate
    model.eval()

    ## Define loss function, MSE for regression 
    criterion = nn.MSELoss()

    ## Initialize variables
    total_samples = 0
    total_loss = 0

    all_predictions = []
    all_targets = []

    ## No longer update gradients
    with torch.no_grad():

        for inputs, targets in dataloader:
            
            ## Get predictions
            outputs = model(inputs)

            ## Calculate Loss
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            ## Arrays of all predictions and the correct value
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

    average_loss = total_loss / total_samples

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)


    ## If model was previously training, resore to that state
    if was_training:
        model.train()

    return average_loss, all_predictions, all_targets


import torch
import torch.nn as nn

def evaluate_finetune(dataloader, model, device):

    """ Evaluates a trained regression neural network. 
   
    Takes a trained neural network model and evaluates it on given dataset
    
    args: 
        dataloader (dataloder cass): PyTorch dataloader class
        model (class): PyTorch nn.model class
    returns:
        average_loss (float): the average MSE Loss. 
    """

    ## Remember if it was training and restore at end
    was_training = model.training

    ## Set model to evaluate
    model.eval()

    ## Define loss function, MSE for regression 
    criterion = nn.MSELoss()

    ## Initialize variables
    total_samples = 0
    total_loss = 0

    all_predictions = []
    all_targets = []

    ## No longer update gradients
    with torch.no_grad():

        for batch in dataloader:
            


            wt_tokens = {key: value.to(device) for key, value in batch['wt_tokens'].items()}

            mut_tokens = {key: value.to(device) for key, value in batch['mut_tokens'].items()}

            positions = batch["positions"].to(device)
            
            targets = batch["targets"].to(device)

            ## Get predictions
            outputs = model(wt_tokens, mut_tokens, positions)

            ## Calculate Loss
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            ## Arrays of all predictions and the correct value
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    average_loss = total_loss / total_samples

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)


    ## If model was previously training, resore to that state
    if was_training:
        model.train()

    return average_loss, all_predictions, all_targets