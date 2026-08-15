
import torch 

def get_residue_embeddings(sequences, positions, tokenizer, model, device, batch_size):
    '''
    Takes in amino acid sequences and specific residue position, and returns the final state embeddings for
    the specified amino acid.

        inputs: amino acid sequences
                position of amino acid residue
                tokenizer
                model
                batch_size for processing

        returns: torch object of dimensionrs (len(sequence), number of final state embeddings)

    '''

    all_embeddings = []

    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start:start + batch_size]
        batch_positions = positions[start:start + batch_size]


        tokens = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True)

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        with torch.inference_mode():
            outputs = model(**tokens)
            
        batch_embeddings = torch.stack([
            outputs.last_hidden_state[i, pos, :]
            for i, pos in enumerate(batch_positions)])

        all_embeddings.append(batch_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)