# Shin Chan Mapping
## Background
Welcome to this repo, I created it as a engineering exercise for this thread ["Struggling to get any use out of LLMs, am I missing something?"](https://forum.level1techs.com/t/struggling-to-get-any-use-out-of-llms-am-i-missing-something/229851). 

The original poster talked about their challenges getting ChatGPT to map German episodes to English. My hypothesis was that ChatGPT had mismatches because the context window was too large for all the episode titles. 

The final output that needs a German speaker to review is found in the [mapped_episodes_table.md](./00-supporting-files/data/mapped_episodes_table.md) file. If you would like to understand my process, feel free to review [crayon-shin-demo.ipynb](./02-development/exploration/crayon-shin-demo.ipynb).

If you just want to run the script, build the dependencies using the scripts below then run `uv run 03-app/crayon-shin-mapping.py` from the root of the directory. 

## Building Dependencies
Make sure UV and ollama are installed. 

### UV
Navigate to the root of this directory, and run the following command
```
uv sync
```

### Ollama
If you plan on running the notebook, make sure ollama is running and pull the following models
- `ollama pull gemma3:27b-it-q4_K_M`
-  `ollama pull hf.co/Qwen/Qwen3-Embedding-8B-GGUF`
-  

### Playwright
If you plan on scraping the websites from scratch with [crayon-shin-explore.ipynb](./02-development/exploration/crayon-shin-explore.ipynb), make sure to run 
```
playwright install
```

## Getting Started
This repo uses git submodules to embed other repos. To get started, clone the repository and initialize the submodules:

If you would like a quick one line clone, you can use the following command:
```bash
git clone --recurse-submodules https://github.com/progressEdd/project-template.git
```

If you have already cloned the repository, you can initialize the submodules with the following commands:
```bash
git pull
git submodule update --init --recursive
```

If you only need specific submodules, you can initialize them individually:
```bash
# For just the dev-onboarding submodule
git submodule update --init 01-dev-onboarding
```
