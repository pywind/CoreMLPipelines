# coremlpipelinestools

```
uv run convert_causal_llm.py
```

```
usage: convert_causal_llm.py [-h] [--model MODEL] [--quantize] [--half] [--compile] [--context-size CONTEXT_SIZE] [--batch-size BATCH_SIZE] [--upload]

Convert a Causal LM to Core ML model

options:
  -h, --help            show this help message and exit
  --model MODEL         Model ID
  --quantize            Linear quantize model
  --half                Load model as float16
  --compile             Save the model as a .mlmodelc
  --context-size CONTEXT_SIZE
                        Context size
  --batch-size BATCH_SIZE
                        Batch size
  --upload              Upload the model to Hugging Face
```

Example: python convert_causal_llm.py --model LGAI-EXAONE/EXAONE-4.0-1.2B --4-bit --half --compile --context-size 1024 --batch-size 1 --upload