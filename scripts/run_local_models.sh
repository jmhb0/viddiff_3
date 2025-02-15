
### easy
python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_easy --split easy --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_easy --split easy --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo

### medium
python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_medium --split medium --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo

### hard
python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval_mode closed --model Qwen/Qwen2-VL-7B-Instruct --video_representation video
python lmms/run_lmm.py --config lmms/configs/config.yaml --name qwen_hard --split hard --eval_mode closed --model lmms-lab/LLaVA-Video-7B-Qwen2 --video_representation llavavideo
