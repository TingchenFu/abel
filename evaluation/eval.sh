RUN_DIR="$PWD" # 7b model directory (e.g. "./checkpoints/GAIRMath-Abel-7b")
backbone="Llama-2-7b-hf"
#CUDA_VISIBLE_DEVICES=1
dataset_name="gsm8k"



# temperature = 0 means greedy decoding


for peft_name in Llama-2-7b-hf_multitask_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster0_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster1_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster2_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster3_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine 
do
    python3  ${RUN_DIR}/evaluation/inference.py \
        --model_name_or_path ${RUN_DIR}/../../../PLM/${backbone}     \
        --peft_model_path ${RUN_DIR}/../../dump/${peft_name} \
        --temperature 0 \
        --top_p 0.9 \
        --output_file   ${RUN_DIR}/dump/${dataset_name}/${peft_name}.jsonl \
        --dataset_name ${dataset_name} \
        --prompt_type math-single \
        --eval_only False  \
        --n_example 8   
done