import torch
from transformers import BertTokenizer, BertModel

import logging
#logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define a new example sentence with multiple meanings of the word "bank"
    text = "After stealing money from the bank vault, the bank robber was seen " \
           "fishing on the Mississippi river bank."

    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    print(segments_ids)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

        # `hidden_states` is a Python list.
        print('      Type of hidden_states: ', type(hidden_states))

        # Each layer in the list is a torch tensor.
        print('Tensor shape for each layer: ', hidden_states[0].size())
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_embeddings.size()
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

