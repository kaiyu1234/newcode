# score path for main experiments or robustness experiments
# example: ./results/mmw_book_report/Llama_3.2_3B_Instruct/main_exp/score.txt
#score_path='./results/mmw_book_report/Mistral_7B_Instruct_v0.3/mcmark/score.txt'
# score_path='./results/mmw_book_report/Mistral_7B_Instruct_v0.3/mcmark/score_random_token_replacement_eps_0_05.txt'
score_path='./results/mmw_book_report/Mistral_7B_Instruct_v0.3/mcmark/score_gpt_rephrase.txt'

fpr=0.05

# for baselines
# python ./evaluations/get_baselines_acc.py \
#     --fpr_thres $fpr \
#     --score_path $score_path


# for MC-mark:
python ./evaluations/get_mcmark_acc.py \
    --fpr_thres $fpr \
    --score_path $score_path