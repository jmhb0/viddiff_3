# VidDiff eval
This page explains running eval for the Viddiff benchmark, hosted on Huggingface [here](). It was proposed in "Video Action Differencing". 


The paper also proposed Viddiff method, which is in `viddiff_method`. To run it, look at [this README](viddiff_method/README.md). 

## Get the dataset
TODO: 
- Generally introduce the whole setup a bit better. This is not just 'eval' code, but also setting up the whole paper.
- Link to HF. 
- Reproduce the key things here, but link out to how to download the videos. 
- That has instructions on how to load these extra files: they are already in this repo.
- About the video caching, and how it's on by default. 
- Different dataset splits. 
- do explain the form of the 'videos' tuple. 

## Running LMM predictions 
Models available on API:
```
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_easy --split easy --eval closed --model gpt-4o-2024-08-06
```
Default options are mode=closed, and model=gpt-4o. 

For --model option: 
- Openai API, e.g. we tested {gpt-4o-2024-08-06}, set OPENAI_API_KEY environment variable. 
- Openrouter API, e.g. we tested {anthropic/claude-3-5-sonnet}, set OPENROUTER_API_KEY environment variable. 
- Gemini API, e.g. we tested {gemini-1.5-flash}, set GEMINI_API_KEY environment variable. This one is really slow to run bc we didn't implement batching. 
- QwenVL we did not use an API, so you need to run it locally. Click [here](apis/howto-local-models.md). 
- LLaVA video we did not find via API, so you need to run it locally. Click [here](apis/howto-local-models.md). 

The frame rate is set per source dataset and can be set in the config file. 


## Running eval
TODO: 
- split into open and closed mode. 
- describe how n_differences works: it's a json file that maps the action to the number of differences to use. In our protocol, you're allowed 1.5x the number of labeled differences (because there could be other valid differences not annotated). 

In `eval_diff.py`, after loading the dataset and running predictions, run:

TODO: loading n_differences
```
metrics = eval_viddiff.eval_viddiff(
	dataset,
		predictions_unmatched=predictions,
		eval_mode=0,
		seed=0,
	n_differences=10,
		results_dir="results")
```

TODO: expand on this a little --> is it really decessary to have the 'description' field? I think in open mode it's yes, but otherwise no. 
The structure for `predictions_unmatched`:
```
[
	// list element i is a dict of difference predictions for sample i
	{
		"numericKey1": {
			// prediction details for one difference
			"description": "..." // A description of the predicted difference",
			"pred": "a|b" // Whather the description is more true of video a or b
		},
		"numericKey2": {
			// Another difference prediction for the same smaple
			// ... same structure as above ...
		}
		// There can be multiple difference predictions per sample
	},
	{
		// Another set of observations
		// ... same structure as above ...
	}
	... 
]
```

For example, here are predictions for a 3-element dataset. 
```
[
	// an example
]
```

## LLM eval 
The eval file makes some api cals to openaiAPI. Need to set the OpenAI Api key 


## LLM evaluation for matching, and possible errors
TODO: explain how these can be handled 

Mention the openai 'overwrite_cache'


## Video-LMM baselines 
Some baselines are implemented in `lmms`. 
- Which models. 
- Different video representations.
- Same prompt except for description of how the videos are represented. 
- They all do automatic caching. 


## VidDiff method 
The Viddiff method is in `viddiff_method`. To run it, look at [this README](viddiff_method/README.md). 

## Citation 
TODO: copy what's in the HF repo. 


## TODO somewhere 
- Discuss the `fps`. This is not a property of the data, but is a property of the standard implementation. There are functions for subsampling in the `data/` util files, but different methods may want to subset the videos differently.
- pip install and so on 


## Notes for this repo that we can remove for the final version
Notes
- Updating the underlying dataset by loading "viddiff/VidDiffBench", making adjustments to fix some bad design choices & to address reviewer feedback, and then finally pushing the updated dataset to the hub ass "viddiff/VidDiffBench_2".

TODO for final:
- copy the final dataset to "jmhb0/ViddiffBench". Update the call in `data/load_viddiff_dataset.py` to load from "jmhb0/ViddiffBench".
- Put the '.py' files from data/ into the huggingface repo files. They are load_viddiff_dataset.py, download_data.py, but NOT update_dataset.py.
- There were some files in "viddiff/VidDiffBench" that were manually added ... the Readme, the .py scripts, the video files. 
- in download_data.py, change the "dataset_name" to "jmhb0/ViddiffBench".
- Remove the scripts for transforming the dataset  “update_dataset.py”
- run everything from scratch 
- Do copy over the n_differences.json file and make sure that it's in the base dataset too. 

