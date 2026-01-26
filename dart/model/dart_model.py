import time
import torch
import torch.nn as nn
import os

from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from dart.tree_search.cpplib.cpp_ngram import load_cpp_ngram
from .dart_utils import *
from .dart_configs import DartConfig
from .llama3_dart import LlamaForCausalLMDart
from .kv_cache import initialize_past_key_values, initialize_past_key_values_for_dart
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .modeling_qwen3_moe_kv import Qwen3MoeForCausalLM as KVQwen3MoeForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM


class DartModel(nn.Module):
    def __init__(
        self,
        base_model, 
        base_model_config,
        base_model_name_or_path,
        dart_model_name_or_path,
        ngram_model,
        dart_layer_state_dict,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        dart_config = DartConfig.from_pretrained(dart_model_name_or_path)
        self.dart_layer = LlamaForCausalLMDart(dart_config)
        self.ngram_model = ngram_model  # cpp ngram model instance
        # load dart layer state dict
        self.dart_layer.load_state_dict(dart_layer_state_dict, strict=False) # doesn't load the embedding weights
        self.dart_layer.load_embedding(base_model_name_or_path) # dart's embeding is same as base LLM model
        # Ensure dart's dtype and device align with the base_model
        self.dart_layer.to(self.base_model.dtype).to(self.base_model.device)
        self.SAFE_REMAIN = 5 # to avoid out of index in KV-Cache

    def ngram_from_pretrained(ngram_model_path, is_small_ngram=False):
        if is_small_ngram:
            print("Using small ngram model for fast testing...")
        cpp_ngram = load_cpp_ngram()
        load_model_path = os.path.join(ngram_model_path, "small.trie" if is_small_ngram else "full.trie")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(ngram_model_path, "small.trie" if is_small_ngram else "full.trie")
        ngram_model_path = load_model_path
        if ngram_model_path.endswith(".trie"):
            ngram_model = cpp_ngram.TrieNgram.load(ngram_model_path)
        else:
            raise ValueError("Unsupported ngram model format. Use .trie file.")
        return ngram_model

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained_with_base(
        cls,
        base_model,
        ngram_model,
        base_model_config,
        dart_model_name_or_path,
        base_model_name_or_path,
    ):
        """
        Load DartModel using a pre-loaded base model (for sharing with other models).
        
        Args:
            base_model: Pre-loaded base model instance.
            base_model_config: Config of the base model.
            dart_model_name_or_path (str): Name or path of the DART model to load.
            base_model_name_or_path (str): Name or path of the base model (for tokenizer).
            ngram_model (object): Pre-loaded ngram model.

        Returns:
            DartModel: A DartModel instance using the shared base model.
        """
        configpath = os.path.join(dart_model_name_or_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(dart_model_name_or_path, "config.json")
        load_model_path = os.path.join(dart_model_name_or_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(dart_model_name_or_path, "model.safetensors")
        dart_layer_state_dict = load_file(load_model_path)
        dart_model = cls(
            base_model, 
            base_model_config,
            base_model_name_or_path,
            dart_model_name_or_path,
            ngram_model,
            dart_layer_state_dict,
        )
        return dart_model

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        dart_model_name_or_path,
        ngram_model_name_or_path,
        is_small_ngram=False,
        **kwargs,
    ):
        """
        Args:
            base_model_name_or_path (str): Name or path of the base model to load.
            dart_model_name_or_path (str): Name or path of the DART model to load.
            ngram_model_name_or_path (str): Name or path of the ngram model to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            MedusaModel: A MedusaModel instance loaded from the given path.
        """
        # 1. load base model
        base_model_config = AutoConfig.from_pretrained(base_model_name_or_path)
        Type = base_model_config.architectures[0]
        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_name_or_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_name_or_path, **kwargs
            )
        elif Type == 'Qwen3MoeForCausalLM':
            base_model = KVQwen3MoeForCausalLM.from_pretrained(
                base_model_name_or_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_name_or_path, **kwargs
            )
        
        # 2. load dart model
        configpath = os.path.join(dart_model_name_or_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(dart_model_name_or_path, "config.json")
        load_model_path = os.path.join(dart_model_name_or_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(dart_model_name_or_path, "model.safetensors")
        dart_layer_state_dict = load_file(load_model_path)

        # 3. load ngram model
        ngram_model = cls.ngram_from_pretrained(ngram_model_name_or_path, is_small_ngram)

        dart_model = cls(
            base_model, 
            base_model_config,
            base_model_name_or_path,
            dart_model_name_or_path,
            ngram_model,
            dart_layer_state_dict,
        )
        return dart_model

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        is_lm_head_output=False,
    ):
        """Forward pass of the DartModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            position_ids (torch.Tensor, optional): Position IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            is_lm_head_output (bool, optional): Whether to also output predictions from the target Model's LM head.

        Returns:
            aux_hidden_states (torch.Tensor): A tensor containing cat result from 3 hidden_states, used for dart input.
            outputs (Optional): Original outputs from the base model's forward.
            lm_head_output (Optional): Original predictions from the base model's LM head.
        """
        with torch.no_grad():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
                position_ids=position_ids,
            )
            if is_lm_head_output:
                lm_head_output = self.base_model.lm_head(outputs[0])
        # 1. get aux-hidden-status
        dart_device = self.dart_layer.device
        if outputs["hidden_states"][0].device != dart_device:
            outputs["hidden_states"] = [x.to(dart_device) for x in outputs["hidden_states"]]
        aux_hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        if is_lm_head_output:
            return aux_hidden_states, outputs, lm_head_output
        return aux_hidden_states
    
    @torch.no_grad()
    def ar_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        max_new_token_num=512,
        max_length=2048,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            temperature (float, optional): Temperature for typical acceptance.
            top_p=0.0 (float, optional): Top_p for generation config.
            top_k=0   (int, optional): Top_k for generation config.
            max_new_token_num (int, optional): Limitation for max new generated token count.
            max_length (int, optional): Limitation for max content length.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # 1. Avoid modifying the input_ids in-place and construct logits_processor
        input_ids = input_ids.clone()
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # 2.1 Initialize the past key and value states for target LLM model
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length + self.SAFE_REMAIN)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_dart_mode(self)
        # 3. Process prefill tokens and generate draft tree related status
        _, _, lm_head_output = self.forward(
            input_ids,
            past_key_values=past_key_values,
            is_lm_head_output=True
        )

        # 4. Loop for muti-turn speculative decoding
        new_generated_token = 0
        for idx in range(max_length):
            if logits_processor is not None:
                logits = lm_head_output[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = lm_head_output[:, -1:].argmax(dim=-1)
            _, _, lm_head_output = self.forward(
                input_id,
                past_key_values=past_key_values,
                is_lm_head_output=True
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_generated_token += 1
            # 4.5 Checking weather to end generation
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if new_generated_token >= max_new_token_num:
                break
            if input_ids.shape[1] >= max_length:
                break
            input_len += 1
            
        # 5. Return final result   
        return input_ids

    @torch.no_grad()
    def dart_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        max_new_token_num=512,
        max_length=2048,
        remain_total=60,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            temperature (float, optional): Temperature for typical acceptance.
            top_p=0.0 (float, optional): Top_p for generation config.
            top_k=0   (int, optional): Top_k for generation config.
            max_new_token_num (int, optional): Limitation for max new generated token count.
            max_length (int, optional): Limitation for max content length.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # 1. Avoid modifying the input_ids in-place and construct logits_processor
        input_ids = input_ids.clone()
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # 2.1 Initialize the past key and value states for target LLM model
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length = max_length + remain_total + self.SAFE_REMAIN)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # 2.2 Initialize the past key and value states for dart layer
        if hasattr(self, "dart_past_key_values"):
            dart_past_key_values = self.dart_past_key_values
            dart_past_key_values_data = self.dart_past_key_values_data
            dart_current_length_data = self.dart_current_length_data
            # Reset the past key and value states
            dart_current_length_data.zero_()
        else:
            (
                dart_past_key_values,
                dart_past_key_values_data,
                dart_current_length_data,
            ) = initialize_past_key_values_for_dart(self.dart_layer, max_length = max_length + self.dart_layer.draft_length + self.SAFE_REMAIN)
            self.dart_past_key_values = dart_past_key_values
            self.dart_past_key_values_data = dart_past_key_values_data
            self.dart_current_length_data = dart_current_length_data

        input_len = input_ids.shape[1]
        reset_dart_mode(self)
        # 3. Process prefill tokens and generate draft tree related status
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = initialize_dart(
            input_ids=input_ids, 
            model=self, 
            past_key_values=past_key_values,
            dart_past_key_values=dart_past_key_values,
            logits_processor=logits_processor,
            remain_total=remain_total,
        )

        # 4. Loop for muti-turn speculative decoding
        new_generated_token = 0
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            # 4.1 Use tree attention to verify the candidates and get predictions
            retrieve_logits, aux_hidden_states, _ = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # Padding draft tokens for -1 index token  
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # 4.2 Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length, target_last_logit = evaluate_posterior(
                retrieve_logits, candidates, logits_processor
            )
            # 4.3 Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                past_key_values_data,
                current_length_data,
                self,
                aux_hidden_states,
                target_last_logit,
                dart_past_key_values,
                dart_past_key_values_data,
                dart_current_length_data,
                remain_total=remain_total,
            )
            new_generated_token += accept_length + 1

            # 4.5 Checking weather to end generation
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if new_generated_token >= max_new_token_num:
                break
            if input_ids.shape[1] >= max_length:
                break
            input_len += accept_length + 1
            
        # 5. Return final result   
        return input_ids

    @torch.no_grad()
    def _ar_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        max_new_token_num=512,
        max_length=2048,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            temperature (float, optional): Temperature for typical acceptance.
            top_p=0.0 (float, optional): Top_p for generation config.
            top_k=0   (int, optional): Top_k for generation config.
            max_new_token_num (int, optional): Limitation for max new generated token count.
            max_length (int, optional): Limitation for max content length.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # 1. Avoid modifying the input_ids in-place and construct logits_processor
        input_ids = input_ids.clone()
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # 2.1 Initialize the past key and value states for target LLM model
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length + self.SAFE_REMAIN)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_dart_mode(self)
        # 3. Process prefill tokens and generate draft tree related status
        _, _, lm_head_output = self.forward(
            input_ids,
            past_key_values=past_key_values,
            is_lm_head_output=True
        )

        # 4. Loop for muti-turn speculative decoding
        new_generated_token = 0
        for idx in range(max_length):
            if logits_processor is not None:
                logits = lm_head_output[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = lm_head_output[:, -1:].argmax(dim=-1)
            _, _, lm_head_output = self.forward(
                input_id,
                past_key_values=past_key_values,
                is_lm_head_output=True
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_generated_token += 1

            # 4.4 Yield intermediate results
            yield {
                "id": input_id[0]
            }

            # 4.5 Checking weather to end generation
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if new_generated_token >= max_new_token_num:
                break
            if input_ids.shape[1] >= max_length:
                break
            input_len += 1
            
        # 5. Return final result   
        return input_ids

    @torch.no_grad()
    def _dart_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        max_new_token_num=512,
        max_length=2048,
        beam_width=20,
        remain_total=80,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            temperature (float, optional): Temperature for typical acceptance.
            top_p=0.0 (float, optional): Top_p for generation config.
            top_k=0   (int, optional): Top_k for generation config.
            max_new_token_num (int, optional): Limitation for max new generated token count.
            max_length (int, optional): Limitation for max content length.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # 1. Avoid modifying the input_ids in-place and construct logits_processor
        input_ids = input_ids.clone()
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # 2.1 Initialize the past key and value states for target LLM model
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length = max_length + remain_total + self.SAFE_REMAIN)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # 2.2 Initialize the past key and value states for dart layer
        if hasattr(self, "dart_past_key_values"):
            dart_past_key_values = self.dart_past_key_values
            dart_past_key_values_data = self.dart_past_key_values_data
            dart_current_length_data = self.dart_current_length_data
            # Reset the past key and value states
            dart_current_length_data.zero_()
        else:
            (
                dart_past_key_values,
                dart_past_key_values_data,
                dart_current_length_data,
            ) = initialize_past_key_values_for_dart(self.dart_layer, max_length = max_length + self.dart_layer.draft_length + self.SAFE_REMAIN)
            self.dart_past_key_values = dart_past_key_values
            self.dart_past_key_values_data = dart_past_key_values_data
            self.dart_current_length_data = dart_current_length_data

        input_len = input_ids.shape[1]
        reset_dart_mode(self)
        # 3. Process prefill tokens and generate draft tree related status
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = initialize_dart(
            input_ids=input_ids, 
            model=self, 
            past_key_values=past_key_values,
            dart_past_key_values=dart_past_key_values,
            logits_processor=logits_processor,
            remain_total=remain_total,
        )

        # 4. Loop for muti-turn speculative decoding
        new_generated_token = 0
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            # 4.1 Use tree attention to verify the candidates and get predictions
            retrieve_logits, aux_hidden_states, _ = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # Padding draft tokens for -1 index token  
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # 4.2 Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length, target_last_logit = evaluate_posterior(
                retrieve_logits, candidates, logits_processor
            )
            # 4.3 Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                past_key_values_data,
                current_length_data,
                self,
                aux_hidden_states,
                target_last_logit,
                dart_past_key_values,
                dart_past_key_values_data,
                dart_current_length_data,
                remain_total=remain_total,
            )
            new_generated_token += accept_length + 1

            # 4.4 Yield intermediate results
            yield {
                "id": input_ids[0, input_len:],
                "accept_length": accept_length + 1,
            }

            # 4.5 Checking weather to end generation
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if new_generated_token >= max_new_token_num:
                break
            if input_ids.shape[1] >= max_length:
                break
            input_len += accept_length + 1
            
        # 5. Return final result   
        return input_ids
