import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model,max_length=2200):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    devices=[]
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device=model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    past_key_values_data_list=[]
    startnum=0
    startdevice=devices[0]
    for id,i in enumerate(devices):
        if startdevice!=i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                max_length,
                getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum=0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers

    bias=0
    start_data_m=devices[0].index
    for i in range(config.num_hidden_layers):
        data_m=devices[i].index
        if data_m!=start_data_m:
            bias=0
            start_data_m=data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m-devices[0].index][2*bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j],
                            current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias+=1
    return past_key_values, past_key_values_data_list, current_length_data



def print_kv(past_key_values, past_key_values_data_list, current_length_data, f):
    """
    Print the KV cache values to a file for debugging and analysis.

    Args:
        past_key_values (list[list[KVCache]]): List of KVCache objects for each layer and key/value pair.
        past_key_values_data_list (list[torch.Tensor]): List of KV storage tensors grouped by device.
        current_length_data (torch.Tensor): Tensor tracking the current length of keys/values in the cache.
        f: File object to write the output to.
    """
    f.write("=" * 80 + "\n")
    f.write("KV Cache Status\n")
    f.write("=" * 80 + "\n\n")
    
    # Print current lengths
    f.write("Current Lengths (per layer):\n")
    f.write("-" * 40 + "\n")
    for i, length in enumerate(current_length_data):
        f.write(f"Layer {i}: {length.item()}\n")
    f.write("\n")
    
    # Print KV storage tensors
    f.write("KV Storage Tensors:\n")
    f.write("-" * 40 + "\n")
    for storage_idx, storage_tensor in enumerate(past_key_values_data_list):
        f.write(f"\nStorage Group {storage_idx}:\n")
        f.write(f"  Shape: {storage_tensor.shape}\n")
        f.write(f"  Device: {storage_tensor.device}\n")
        f.write(f"  Dtype: {storage_tensor.dtype}\n")
        f.write(f"  Memory: {storage_tensor.numel() * storage_tensor.element_size() / 1e6:.2f} MB\n")
    f.write("\n")
    
    # Print detailed KVCache information
    f.write("Detailed KVCache Information:\n")
    f.write("-" * 40 + "\n")
    for layer_idx, layer_kv_pair in enumerate(past_key_values):
        f.write(f"\nLayer {layer_idx}:\n")
        for kv_idx, kv_cache in enumerate(layer_kv_pair):
            cache_type = "Key" if kv_idx == 0 else "Value"
            f.write(f"  {cache_type} Cache:\n")
            f.write(f"    Full Shape: {kv_cache.data.shape}\n")
            f.write(f"    Current Shape: {kv_cache.shape}\n")
            f.write(f"    Current Length: {kv_cache.current_length.item()}\n")
            f.write(f"    Device: {kv_cache.data.device}\n")
            f.write(f"    Dtype: {kv_cache.data.dtype}\n")
            
            # Print actual values (only if reasonable size)
            curr_len = kv_cache.current_length.item()
            if curr_len > 0:
                actual_data = kv_cache.data[:, :, :curr_len, :]
                f.write(f"    Data Range: [{actual_data.min().item():.6f}, {actual_data.max().item():.6f}]\n")
                f.write(f"    Data Mean: {actual_data.mean().item():.6f}\n")
                f.write(f"    Data Std: {actual_data.std().item():.6f}\n")
                
                # Print first and last few values as samples
                f.write(f"    First value sample: {actual_data[0, 0, 0, :5]}\n")
                if curr_len > 1:
                    f.write(f"    Last value sample: {actual_data[0, 0, -1, :5]}\n")
            else:
                f.write(f"    Data: Empty (current_length = 0)\n")
            f.write("\n")
    
    f.write("=" * 80 + "\n\n")
    f.flush()