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


def initialize_past_key_values_medusa(model):
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
    past_key_values_data = torch.zeros(
        config.num_hidden_layers * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=model.device,
        dtype=model.dtype,
    )
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data

def initialize_past_key_values(model, max_length=2200):
    """
    Initialize past key/value caches for a transformer model, supporting multi-device layer placement.

    Returns:
        past_key_values: list[list[KVCache]] — per-layer KVCache (key & value)
        kv_storage_list: list[Tensor] — underlying big KV tensors (grouped by device)
        current_lengths: Tensor — CPU tensor tracking current KV lengths
    """
    config = model.config
    num_layers = config.num_hidden_layers
    batch_size = 1
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # ---------------------------------------------------------
    # 1. Collect device used by each layer
    # ---------------------------------------------------------
    layer_devices = []
    for i in range(num_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except Exception:
            device = model.layers[i].self_attn.q_proj.weight.device
        layer_devices.append(device)

    # ---------------------------------------------------------
    # 2. Group layers by device and allocate KV storage
    #    Works for both CPU and GPU placements.
    # ---------------------------------------------------------
    device_to_indices = {}
    device_order = []
    for idx, device in enumerate(layer_devices):
        if device not in device_to_indices:
            device_to_indices[device] = []
            device_order.append(device)
        device_to_indices[device].append(idx)

    kv_storage_list = []
    device_to_group_index = {}
    device_offsets = {}
    for group_idx, device in enumerate(device_order):
        layer_count = len(device_to_indices[device])
        kv_storage = torch.zeros(
            layer_count * 2,
            batch_size,
            config.num_key_value_heads,
            max_length,
            head_dim,
            device=device,
            dtype=model.dtype,
        )
        kv_storage_list.append(kv_storage)
        device_to_group_index[device] = group_idx
        device_offsets[device] = 0

    # ---------------------------------------------------------
    # 3. Prepare current-length tracker (MUST be on CPU)
    # ---------------------------------------------------------
    current_lengths = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")

    # ---------------------------------------------------------
    # 4. Create KVCache structures per layer
    # ---------------------------------------------------------
    past_key_values = []
    for i, device in enumerate(layer_devices):
        group_index = device_to_group_index[device]
        offset = device_offsets[device]
        storage = kv_storage_list[group_index]

        kv_pair = [
            KVCache(storage[2 * offset + j], current_lengths[i * 2 + j])
            for j in range(2)
        ]
        past_key_values.append(kv_pair)
        device_offsets[device] = offset + 1

    return past_key_values, kv_storage_list, current_lengths


def initialize_past_key_values_for_dart(dart_model, max_length=2200):
    """
    Initialize past key/value caches for a transformer model, supporting multi-device layer placement.

    Returns:
        past_key_values: list[list[KVCache]] — per-layer KVCache (key & value)
        kv_storage_list: list[Tensor] — underlying big KV tensors (grouped by device)
        current_lengths: Tensor — CPU tensor tracking current KV lengths
    """
    config = dart_model.config
    num_layers = config.num_hidden_layers
    batch_size = 1
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # ---------------------------------------------------------
    # 1. Collect device used by each layer
    # ---------------------------------------------------------
    device = dart_model.midlayer.self_attn.q_proj.weight.device
    # ---------------------------------------------------------
    # 2. Group layers by device and allocate KV storage
    # ---------------------------------------------------------
    kv_storage_list = []
    layer_count = 1
    kv_storage = torch.zeros(
        layer_count * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        head_dim,
        device=device,
        dtype=dart_model.dtype
    )
    kv_storage_list.append(kv_storage)
    # ---------------------------------------------------------
    # 3. Prepare current-length tracker (MUST be on CPU)
    # ---------------------------------------------------------
    current_lengths = torch.zeros(layer_count * 2, dtype=torch.long, device="cpu")
    # ---------------------------------------------------------
    # 4. Create KVCache structures per layer
    # ---------------------------------------------------------
    past_key_values = []
    kv_pair = [KVCache(kv_storage[j], current_lengths[j]) for j in range(2)]
    past_key_values.append(kv_pair)
    return past_key_values, kv_storage_list, current_lengths
