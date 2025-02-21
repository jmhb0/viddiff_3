# bash scripts/run_all_open.sh
'''
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_easy --split easy --eval_mode open --model gpt-4o-2024-08-06
python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_open_easy --split easy --eval_mode open --model anthropic/claude-3.5-sonnet
python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_open_easy --split easy --eval_mode open --model models/gemini-1.5-pro --video_representation video

python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_medium --split medium --eval_mode open --model gpt-4o-2024-08-06
python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_open_medium --split medium --eval_mode open --model anthropic/claude-3.5-sonnet
python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_open_medium --split medium --eval_mode open --model models/gemini-1.5-pro --video_representation video

python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_hard --split hard --eval_mode open --model gpt-4o-2024-08-06
python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_open_hard --split hard --eval_mode open --model anthropic/claude-3.5-sonnet
python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_open_hard --split hard --eval_mode open --model models/gemini-1.5-pro --video_representation video
'''

# viddiff
python  viddiff_method/run_viddiff.py -c viddiff_method/configs/config.yaml --name viddiff_open_easy --split easy --eval_mode open
python  viddiff_method/run_viddiff.py -c viddiff_method/configs/config.yaml --name viddiff_open_medium --split medium --eval_mode open
python  viddiff_method/run_viddiff.py -c viddiff_method/configs/config.yaml --name viddiff_open_hard --split hard --eval_mode open
