

# model="Llama-2-13b-chat-hf"
# dataset_name="gsm8k"
# CUDA_VISIBLE_DEVICES=2



# RUN_DIR="$PWD"
# model_dir="/apdcephfs/share_916081/shared_info/tingchenfu/PLM/${model}" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")


# python3  ${RUN_DIR}/evaluation/inference.py \
#     --model_dir ${model_dir} \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --output_file   ${RUN_DIR}/dump/${dataset_name}/${model}.jsonl \
#     --dataset_name ${dataset_name} \
#     --prompt_type math-single \
#     --eval_only False  \
#     --n_example 8   




# model="Llama-2-13b-hf"
# CUDA_VISIBLE_DEVICES=1
# dataset_name="math"
# temperature=0.1
# top_p=0.95
# n_example=4


# RUN_DIR="$PWD"
# model_dir="/apdcephfs/share_916081/shared_info/tingchenfu/PLM/${model}" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")

# python3  ${RUN_DIR}/evaluation/inference.py \
#     --model_dir ${model_dir} \
#     --temperature ${temperature} \
#     --top_p ${top_p} \
#     --output_file   ${RUN_DIR}/dump/${dataset_name}/shot${n_example}t${temperature}p${top_p}${model}.jsonl \
#     --dataset_name ${dataset_name} \
#     --prompt_type math-single \
#     --eval_only False  \
#     --n_example ${n_example}


######################################################################################################

model="Llama-2-7b-hf"
CUDA_VISIBLE_DEVICES=1
dataset_name="math"
temperature=0.1
top_p=0.95
n_example=4


RUN_DIR="$PWD"
model_dir="/apdcephfs/share_916081/shared_info/tingchenfu/PLM/${model}" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")

python3  ${RUN_DIR}/evaluation/inference.py \
    --model_dir ${model_dir} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --output_file   ${RUN_DIR}/dump/${dataset_name}/shot${n_example}t${temperature}p${top_p}${model}_geometry.jsonl \
    --dataset_name ${dataset_name} \
    --prompt_type math-single \
    --eval_only False  \
    --n_example ${n_example}  \
    --figure_geometry_only True  


