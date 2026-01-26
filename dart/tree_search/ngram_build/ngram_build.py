"""
N-gram model builder for DART tree search.

This script builds an n-gram trie model from training data using parallel processing.
"""

import argparse
import json
import os
from multiprocessing import Process
from typing import List

from huggingface_hub import snapshot_download
import transformers
from tqdm import tqdm

from dart.tree_search.cpplib.cpp_ngram import load_cpp_ngram


# ============================================================================
# Constants
# ============================================================================

DEFAULT_NGRAM_ORDER = 3
DEFAULT_N_JOBS = 16
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build an n-gram trie model from training data."
    )
    parser.add_argument(
        "--ngram_order",
        type=int,
        default=DEFAULT_NGRAM_ORDER,
        help=f"Order of the n-gram model. (default: {DEFAULT_NGRAM_ORDER})",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the n-gram model.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel jobs. (default: {DEFAULT_N_JOBS})",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Path or repo id of the tokenizer. (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--conversation_per_file",
        type=int,
        default=-1,
        help="Number of conversations to process per file. -1 means all. (default: -1)",
    )
    return parser.parse_args()


# ============================================================================
# Data Collection
# ============================================================================

def collect_training_files(data_path: str) -> List[str]:
    """
    Recursively collect all files from the data directory.
    
    Args:
        data_path: Root directory containing training data.
        
    Returns:
        List of absolute file paths.
    """
    train_files = []
    for root, _, files in os.walk(data_path):
        for filename in files:
            train_files.append(os.path.join(root, filename))
    return train_files


# ============================================================================
# N-gram Building (Parallel Workers)
# ============================================================================

def build_ngram_partition(
    worker_id: int,
    args: argparse.Namespace,
    train_files: List[str],
) -> None:
    """
    Build n-gram model for a partition of training files.
    
    Each worker processes files at indices: worker_id, worker_id + n_jobs, worker_id + 2*n_jobs, ...
    
    Args:
        worker_id: ID of this worker (0 to n_jobs-1).
        args: Command line arguments.
        train_files: List of all training file paths.
    """
    # Initialize tokenizer and n-gram model
    # tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    import modelscope
    tokenizer = modelscope.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    eos_token_ids = [tokenizer.eos_token_id]
    
    cpp_ngram = load_cpp_ngram()
    ngram_model = cpp_ngram.TrieNgram(order=args.ngram_order)

    # Determine which files this worker processes
    file_indices = range(worker_id, len(train_files), args.n_jobs)
    is_primary_worker = (worker_id == 0)
    
    # Process assigned files
    iterator = tqdm(file_indices, desc="Files") if is_primary_worker else file_indices
    for file_idx in iterator:
        _process_single_file(
            filepath=train_files[file_idx],
            tokenizer=tokenizer,
            ngram_model=ngram_model,
            eos_token_id=eos_token_ids[0],
            show_progress=is_primary_worker,
            conversation_limit=args.conversation_per_file,
        )

    # Save partial model
    output_file = os.path.join(
        args.output_path,
        f"{args.ngram_order}gram-part{worker_id}.trie"
    )
    ngram_model.save(output_file)


def _process_single_file(
    filepath: str,
    tokenizer,
    ngram_model,
    eos_token_id: int,
    show_progress: bool = False,
    conversation_limit: int = -1,
) -> None:
    """
    Process a single JSONL file and add tokens to the n-gram model.
    
    Args:
        filepath: Path to the JSONL file.
        tokenizer: HuggingFace tokenizer.
        ngram_model: C++ n-gram trie model.
        eos_token_id: End-of-sequence token ID.
        show_progress: Whether to show progress bar.
        conversation_limit: Maximum number of conversations to process. -1 means all.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = tqdm(f, desc=os.path.basename(filepath)) if show_progress else f
        for i, line in enumerate(lines):
            if conversation_limit != -1 and i >= conversation_limit:
                break
            data = json.loads(line)
            text = data["text"]
            
            # Tokenize and append EOS token
            tokens = tokenizer.encode(text, add_special_tokens=True)
            tokens.append(eos_token_id)
            
            ngram_model.add_conversation(tokens)


# ============================================================================
# N-gram Merging
# ============================================================================

def find_partial_model_files(output_path: str, ngram_order: int) -> List[str]:
    """
    Find all partial n-gram model files.
    
    Args:
        output_path: Directory containing partial model files.
        ngram_order: Order of the n-gram model.
        
    Returns:
        List of paths to partial model files.
    """
    pattern = f"{ngram_order}gram-part"
    partial_files = []
    
    for root, _, files in os.walk(output_path):
        for filename in files:
            if pattern in filename:
                partial_files.append(os.path.join(root, filename))
    
    return partial_files


def merge_partial_models(args: argparse.Namespace) -> None:
    """
    Merge all partial n-gram models into a single model.
    
    Args:
        args: Command line arguments containing output_path and ngram_order.
    """
    partial_files = find_partial_model_files(args.output_path, args.ngram_order)
    
    if not partial_files:
        print("No partial files found to merge.")
        return
    
    print(f"Merging {len(partial_files)} partial models...")
    
    # Load and merge models
    cpp_ngram = load_cpp_ngram()
    merged_model = cpp_ngram.TrieNgram.load(partial_files[0])
    
    for partial_file in tqdm(partial_files[1:], desc="Merging"):
        partial_model = cpp_ngram.TrieNgram.load(partial_file)
        merged_model.add_all(partial_model)
        del partial_model  # Free memory

    # Save merged model
    final_output = os.path.join(args.output_path, f"{args.ngram_order}gram{'' if args.conversation_per_file == -1 else f'-{args.conversation_per_file}'}.trie")
    merged_model.save(final_output)
    print(f"Saved merged model to: {final_output}")

    # Clean up partial files
    for partial_file in partial_files:
        os.remove(partial_file)
    print(f"Removed {len(partial_files)} partial files.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for n-gram model building."""
    args = parse_args()
    
    # Suppress transformer warnings
    transformers.logging.set_verbosity_error()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Collect training files
    train_files = collect_training_files(args.data_path)
    print(f"Found {len(train_files)} training files.")

    # Start parallel workers
    processes = []
    for worker_id in range(args.n_jobs):
        process = Process(
            target=build_ngram_partition,
            args=(worker_id, args, train_files),
        )
        process.start()
        processes.append(process)

    # Wait for all workers to complete
    for process in processes:
        process.join()

    # Merge partial models
    merge_partial_models(args)
    
    print("Done!")


if __name__ == "__main__":
    main()
