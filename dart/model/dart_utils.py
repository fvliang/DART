import time
import torch, random
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from dart.tree_search.tree_search import tree_search

def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list



def reset_dart_mode(
    model,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def dart_tree_construct_basing_logits(
    dart_logits, 
    preceding_ids, 
    dart_model,
    remain_total,
):
    """
    Build draft token tree based on DART logits.
    
    Optimizations:
    - Reduce unnecessary tensor operations
    - Batch device transfer
    
    Args:
        dart_logits: (1, draft_length, vocab_size) tensor, logits predicted by DART model
        preceding_ids: (1, preceding_length) tensor, preceding token IDs
        dart_model: DartModel instance
    
    Returns:
        draft_tokens: (1, num_nodes) LongTensor, on the same device as preceding_ids
        retrieve_indices: (num_leaves, max_depth+1) LongTensor, on the same device as preceding_ids
        tree_mask: (1, 1, num_nodes, num_nodes) BoolTensor, on the same device as preceding_ids
        tree_position_ids: (1, num_nodes) LongTensor, on the same device as preceding_ids
    """
    device = preceding_ids.device
    
    # Execute tree search (on CPU as it involves Python list operations)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = tree_search(
        dart_logits.squeeze(0), 
        preceding_ids.squeeze(0), 
        dart_model,
        remain_total=remain_total,
    )
    
    # Batch transfer to target device and add batch dimension
    draft_tokens = draft_tokens.unsqueeze(0).to(device)
    retrieve_indices = retrieve_indices.to(device)
    tree_mask = tree_mask.unsqueeze(0).unsqueeze(0).to(device)
    tree_position_ids = tree_position_ids.unsqueeze(0).to(device)
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def initialize_dart(
    input_ids, 
    model, 
    past_key_values, 
    dart_past_key_values,
    logits_processor,
    remain_total,
):
    """
    Forward through target model and initializes the Dart Draft Tree.

    This function performs the following operations:
    1. Forward through target model to obtain the aux_hidden_states, lm_head_output.
    2. Sample 1 next token from lm_head_output with logits_processor if provided.
    3. Forward pass through the DART layer with shifted token embeds and aux_hidden_states to get dart_logits.
    4. Construct token tree based on dart_logits and input_ids.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - medusa_attn_mask (torch.Tensor): The attention mask designed specifically for the Medusa structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - draft_tokens      (torch.Tensor): 1D token ids.
    - retrieve_indices  (torch.Tensor): retrieve_indices.
    - tree_mask         (torch.Tensor): dart tree mask.
    - tree_position_ids (torch.Tensor): dart tree_position_ids.
    """
    # 1. Forward pass through the target model and get aux_hidden_states and orig
    aux_hidden_states, outputs, lm_head_output = model(
        input_ids=input_ids, 
        past_key_values=past_key_values, 
        is_lm_head_output=True
    )
    # 2. Sample 1 token from orig with logits_processor if provided
    if logits_processor is not None:
        logits = lm_head_output[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        next_token = torch.multinomial(probabilities, 1)
    else:
        next_token = torch.argmax(lm_head_output[:, -1])
        next_token = next_token[None, None]
        
    # 3 Construct input of DART layer
    input_ids = torch.cat((input_ids, next_token.to(input_ids.device)), dim=1)
    inputs_embeds = model.dart_layer.embed_input_ids(input_ids[:, 1:])
    
    # 4. Forward pass through the DART layer to get draft_length logits
    # dart_output's shape: (bs, draft_length, hidden_size)
    dart_output = model.dart_layer(
        hidden_states=aux_hidden_states,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=dart_past_key_values,
        use_cache=True,
    )
    dart_logits = model.dart_layer.compute_logits(dart_output)
    
    # 5. Construct token tree based on dart_logits
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = dart_tree_construct_basing_logits(
        dart_logits=dart_logits, 
        preceding_ids=input_ids,
        dart_model=model,
        remain_total=remain_total,
    )
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """
    position_ids = tree_position_ids + input_ids.shape[1]
    position_ids = position_ids.unsqueeze(0) if position_ids.dim() == 1 else position_ids
    aux_hidden_states, outputs, lm_head_output = model(
        tree_candidates,
        position_ids=position_ids,
        past_key_values=past_key_values,
        is_lm_head_output=True,
    )
    retrieve_logits = lm_head_output[0, retrieve_indices]
    return retrieve_logits, aux_hidden_states, outputs


def evaluate_posterior(
    retrieve_logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - retrieve_logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:].to(retrieve_logits.device) == torch.argmax(retrieve_logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, retrieve_logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = retrieve_logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            target_last_logit = gtp
        else:
            gt_logits = retrieve_logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            target_last_logit = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, target_last_logit
    

@torch.no_grad()
def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    past_key_values_data_list,
    current_length_data,
    model,
    aux_hidden_states,
    target_last_logit,
    dart_past_key_values,
    dart_past_key_values_data_list,
    dart_current_length_data,
    remain_total,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    dart_input_ids = candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)
    input_ids = torch.cat([input_ids, dart_input_ids], dim=-1)
    # Update the past key values based on the selected tokens
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])
    
    # Get new aux_hidden_states of best_candidate
    retrieve_aux_hidden_states = aux_hidden_states[:, retrieve_indices]
    accept_aux_hidden_states = retrieve_aux_hidden_states[:, best_candidate, : accept_length + 1]
    
    # Sample 1 token from the probability distribution
    if logits_processor is not None:
        # target_last_logit is already a probability distribution, use it directly for sampling
        prob = target_last_logit
        next_token = torch.multinomial(prob, 1)
        next_token = next_token[None]
    else:
        next_token = torch.argmax(target_last_logit)
        next_token = next_token[None, None]
    dart_input_ids = torch.cat((dart_input_ids, next_token.to(input_ids.device)), dim=1)


    # release last draft_length token's dart_past_key_values_data
    dart_current_length_data.fill_(prev_input_len)

    # Construct input of DART layer and forward pass through the DART layer to get draft_length logits
    inputs_embeds = model.dart_layer.embed_input_ids(dart_input_ids[:, 1:])
    dart_output = model.dart_layer(
        hidden_states=accept_aux_hidden_states,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=dart_past_key_values,
        use_cache=True,
    )
    # dart_output's shape: (bs, draft_length, hidden_size)
    dart_logits = model.dart_layer.compute_logits(dart_output)
    
    # Construct new token tree based on dart_logits
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = dart_tree_construct_basing_logits(
        dart_logits=dart_logits, 
        preceding_ids=dart_input_ids,
        dart_model=model,
        remain_total=remain_total,
    )

    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids
