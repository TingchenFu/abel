model="Llama-2-7b-hf"
CUDA_VISIBLE_DEVICES=1
dataset_name="math"

RUN_DIR="$PWD"
model_dir="/apdcephfs/share_916081/shared_info/tingchenfu/PLM/${model}" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")

python3  ${RUN_DIR}/evaluation/inference.py \
    --model_dir ${model_dir} \
    --temperature 0.1 \
    --top_p 0.9 \
    --output_file   ${RUN_DIR}/dump/${dataset_name}/${model}.jsonl \
    --dataset_name ${dataset_name} \
    --prompt_type math-single \
    --eval_only False  \
    --n_example 8   \
