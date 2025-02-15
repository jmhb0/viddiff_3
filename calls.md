# easy but for all models
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_easy --split easy --eval closed --model gpt-4o-2024-08-06
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_easy --split easy --eval closed --model anthropic/claude-3.5-sonnet
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_easy --split easy --eval closed --model models/gemini-1.5-pro --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_easy --split easy --eval closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_easy --split easy --eval closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
```

Medium
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_medium --split medium --eval closed --model gpt-4o-2024-08-06
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_medium --split medium --eval closed --model anthropic/claude-3.5-sonnet
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_medium --split medium --eval closed --model models/gemini-1.5-pro --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
```


Hard
```
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_hard --split hard --eval closed --model gpt-4o-2024-08-06
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name claudesonnet_hard --split hard --eval closed --model anthropic/claude-3.5-sonnet
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name geminipro_hard --split hard --eval closed --model models/gemini-1.5-pro --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python -m ipdb lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
```

