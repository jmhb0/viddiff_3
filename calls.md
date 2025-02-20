# closed evaluation 
# easy
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_easy --split easy --eval_mode closed --model gpt-4o-2024-08-06
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_easy --split easy --eval_mode closed --model anthropic/claude-3.5-sonnet
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_easy --split easy --eval_mode closed --model models/gemini-1.5-pro --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_easy --split easy --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_easy --split easy --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
```

Medium
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_medium --split medium --eval_mode closed --model gpt-4o-2024-08-06
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_medium --split medium --eval_mode closed --model anthropic/claude-3.5-sonnet
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_medium --split medium --eval_mode closed --model models/gemini-1.5-pro --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
```


Hard
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_hard --split hard --eval_mode closed --model gpt-4o-2024-08-06
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_hard --split hard --eval_mode closed --model anthropic/claude-3.5-sonnet
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_hard --split hard --eval_mode closed --model models/gemini-1.5-pro --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
```

# open evaluation 
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_open_easy --split easy --eval_mode open --model gpt-4o-2024-08-06
```



# VidDiff
```
python -m ipdb viddiff_method/run_viddiff.py -c viddiff_method/configs/config.yaml --name viddiff_easy --split easy --eval_mode closed --subset_mode 0
python -m ipdb viddiff_method/run_viddiff.py -c viddiff_method/configs/config.yaml --name viddiff_medium --split medium --eval_mode closed --subset_mode 0
python -m ipdb viddiff_method/run_viddiff.py -c viddiff_method/configs/config.yaml --name viddiff_hard --split hard --eval_mode closed --subset_mode 0
```




