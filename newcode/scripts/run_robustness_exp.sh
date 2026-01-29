# models used in our paper: 
# 'mistralai/Mistral-7B-Instruct-v0.3',
# 'meta-llama/Llama-3.2-3B-Instruct',
# 'meta-llama/Llama-2-7b-chat-hf',
# 'microsoft/Phi-3.5-mini-instruct',
model_str="mistralai/Mistral-7B-Instruct-v0.3"

# 'mmw_book_report','mmw_story','mmw_fake_news','dolly_cw','longform_qa','finance_qa'
dataset_name="mmw_book_report"

# 'mcmark': run mcmark with l=20
# 'main_exp' run main experiments
# 'mcmark_ablation' run ablation study
reweight_type='mcmark'
openai_api_key='sk-a234907939cc40198c19d5f401e28c5a'

# 'random_token_replacement' 'gpt_rephrase' 'back_translation' 'dipper'
attack_type='gpt_rephrase' 
# attack_type='dipper'
eps=0 #eps for random_token_replacement


# rewrite if the attack method needs to
if [ "$attack_type" != "random_token_replacement" ]; then
    python experiments/text_generation/rewrite.py \
        --res_dir './results' \
        --openai_api_key "$openai_api_key" \
        --dataset_name $dataset_name \
        --model_str $model_str \
        --reweight_type $reweight_type \
        --attack_type $attack_type
fi



python -m experiments \
    --res_dir "./results" \
    --model_str "$model_str" \
    --reweight_type "$reweight_type" \
    --dataset_name "$dataset_name" \
    --robustness_type "$attack_type" \
    --eps $eps