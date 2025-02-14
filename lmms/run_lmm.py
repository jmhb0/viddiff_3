import ipdb
import click
import json
import os
import logging
import sys

logging.basicConfig(level=logging.INFO,
					format='%(filename)s:%(levelname)s:%(message)s')

sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms import config_utils
from lmms import lmm_utils as lu
import eval_viddiff


# yapf: disable
@click.command()
@click.option("--config", "-c", default="lmms/configs/base_lmm.yaml", help="config file")
@click.option("--name", "-n", default=None, help="experiment name which fixes the filename. Default value uses the config file value")
@click.option("--split", "-s", default="easy", type=click.Choice(["easy", "medium", "hard"]), help="split: one of [easy, medium, hard]")
@click.option("--eval", "-e", default="closed", type=click.Choice(["closed", "open"]), help="eval: one of [closed, open]")
@click.option("--model", "-m", default="gpt-4o-2024-08-06", help="model: the model name, like in their API, e.g. [gpt-4o-2024-08-06, anthropic/claude-3.5-sonnet-20241022]")
@click.option("--video_representation", "-v", default="frames", type=click.Choice(["frames", "video", "llavavideo"]), help="video_representation: one of [frames, video]. Must be video for gemini.")
# yapf: enable
def main(config, name, split, eval, model, video_representation):
	# config
	args = config_utils.load_config(config, name=name, split=split, eval=eval, model=model, video_representation=video_representation)

	# get dataset, videos, and allowable n_differences
	dataset = lvd.load_viddiff_dataset([args.data.split],
									   args.data.subset_mode,
									   cache_dir=None)
	videos = lvd.load_all_videos(dataset,
								 do_tqdm=True,
								 cache=True,
								 cache_dir="cache/cache_data")
	n_differences = lvd.get_n_differences(dataset, args.lmm.n_differences)

	# make prompts and call the lmm
	batch_prompts_text, batch_prompts_video = lu.make_text_prompts(
		dataset, videos, n_differences, args.eval_mode, args.lmm)

	predictions = lu.run_lmm(
		batch_prompts_text,
		batch_prompts_video,
		args.lmm,
		args.eval_mode,
		n_differences,
		# debug=debug,
		verbose=True)

	# do eval
	metrics = eval_viddiff.eval_viddiff(dataset=dataset,
										predictions_unmatched=predictions,
										eval_mode=args.eval_mode,
										n_differences=n_differences,
										seed=args.seed,
										results_dir=args.logging.results_dir)
	print(metrics)
	ipdb.set_trace()



if __name__ == "__main__":
	main()
