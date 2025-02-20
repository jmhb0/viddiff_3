## closed
### easy
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_closed_easy --split easy --eval_mode closed --model gpt-4o-2024-08-06
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_closed_easy --split easy --eval_mode closed --model anthropic/claude-3.5-sonnet
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_closed_easy --split easy --eval_mode closed --model models/gemini-1.5-pro --video_representation video
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_closed_easy --split easy --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_closed_easy --split easy --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo

### medium
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_medium --split medium --eval_mode closed --model gpt-4o-2024-08-06
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_medium --split medium --eval_mode closed --model anthropic/claude-3.5-sonnet
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_medium --split medium --eval_mode closed --model models/gemini-1.5-pro --video_representation video
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo

### hard
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_hard --split hard --eval_mode closed --model gpt-4o-2024-08-06
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_hard --split hard --eval_mode closed --model anthropic/claude-3.5-sonnet
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_hard --split hard --eval_mode closed --model models/gemini-1.5-pro --video_representation video
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
# python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo



## Open
### easy
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_easy --split easy --eval_mode open --model gpt-4o-2024-08-06
python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_open_easy --split easy --eval_mode open --model anthropic/claude-3.5-sonnet
python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_open_easy --split easy --eval_mode open --model models/gemini-1.5-pro --video_representation video

### medium
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_medium --split medium --eval_mode open --model gpt-4o-2024-08-06
python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_open_medium --split medium --eval_mode open --model anthropic/claude-3.5-sonnet
python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_open_medium --split medium --eval_mode open --model models/gemini-1.5-pro --video_representation video

### hard
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_hard --split hard --eval_mode open --model gpt-4o-2024-08-06
python lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_open_hard --split hard --eval_mode open --model anthropic/claude-3.5-sonnet python lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_open_hard --split hard --eval_mode open --model models/gemini-1.5-pro --video_representation video


## VidDiff
python viddiff_method/run_viddiff.py --config viddiff_method/configs/config.yaml --name viddiff_easy --split easy --eval_mode closed --subset_mode 2_per_action
python viddiff_method/run_viddiff.py --config viddiff_method/configs/config.yaml --name viddiff_medium --split medium --eval_mode closed
python viddiff_method/run_viddiff.py --config viddiff_method/configs/config.yaml --name viddiff_hard --split hard --eval_mode closed


