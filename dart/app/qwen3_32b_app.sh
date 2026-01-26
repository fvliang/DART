SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=${1:-1}

python3 \
    $ROOT_DIR/app/app.py \
    --base-model-name-or-path   Qwen/Qwen3-32B \
    --dart-model-name-or-path   fvliang/qwen32b-dart \
    --ngram-model-name-or-path  fvliang/dart-qwen3-ngram \
    --template-name qwen \
    --device cuda \
    --max-new-tokens 5120 \
    --max-length 6400 \
    --share \
    --use-small-ngram \
    --compare-eagle3 \
    --eagle3-model-name-or-path AngelSlim/Qwen3-32B_eagle3 \
    --listen \
    --server-port 30000