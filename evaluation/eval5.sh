# model="Llama-2-7b-hf"
# CUDA_VISIBLE_DEVICES=4


# RUN_DIR="$PWD"
# model_dir="/apdcephfs/share_916081/shared_info/tingchenfu/PLM/${model}" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")


# python3  ${RUN_DIR}/evaluation/inference.py \
#     --model_dir ${model_dir} \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --output_file   ${RUN_DIR}/dump/gsm8k/${model}.jsonl \
#     --dataset_name gsm8k \
#     --prompt_type math-single \
#     --eval_only False  \
#     --n_example 8   

# python3  ${RUN_DIR}/evaluation/inference.py \
#     --model_dir ${model_dir} \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --output_file   ${RUN_DIR}/dump/math/${model}.jsonl \
#     --dataset_name math \
#     --prompt_type math-single \
#     --eval_only False  \
#     --n_example 4   

# # Evaluate 13b model on GSM8k, MATH, and GSM8k_robust
# # ckpts_dir="./checkpoints/GAIRMath-Abel-13b" # 13b model directory (e.g. "./checkpoints/GAIRMath-Abel-13b")
# # for DEV_SET in gsm8k math gsm8k_robust
# # do
# # sudo CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference --model_dir ${ckpts_dir} --temperature 0.0 --top_p 1.0 --output_file ./outputs/${DEV_SET}/13b.jsonl --dev_set ${DEV_SET} --prompt_type math-single --eval_only True
# # done

# # Evaluate 70b model on GSM8k, MATH, MathGPT, and GSM8k_robust
# # ckpts_dir="./checkpoints/GAIRMath-Abel-70b" # 70b model directory (e.g. "./checkpoints/GAIRMath-Abel-70b")
# # for DEV_SET in gsm8k math mathgpt gsm8k_robust
# # do
# # sudo CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation.inference --model_dir ${ckpts_dir} --temperature 0.0 --top_p 1.0 --output_file ./outputs/${DEV_SET}/70b.jsonl --dev_set ${DEV_SET} --prompt_type math-single --eval_only True
# # done



model="Llama-2-13b-hf"
CUDA_VISIBLE_DEVICES=1
dataset_name="math"
temperature=0.4
top_p=0.95
n_example=4


RUN_DIR="$PWD"
model_dir="/apdcephfs/share_916081/shared_info/tingchenfu/PLM/${model}" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")

python3  ${RUN_DIR}/evaluation/inference.py \
    --model_dir ${model_dir} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --output_file   ${RUN_DIR}/dump/${dataset_name}/shot${n_example}t${temperature}p${top_p}${model}.jsonl \
    --dataset_name ${dataset_name} \
    --prompt_type math-single \
    --eval_only False  \
    --n_example ${n_example}
