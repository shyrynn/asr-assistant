# Service to run ASR model enhanced with RAG capabilities

## Configuration

1. make build
2. make run
3. in /workspace/data/rnnt_model_config.yaml change processor.target to src.rnnt_inference.AudioToMelSpectrogramPreprocessor
4. in /workspace/data/rnnt_model_config.yaml change tokenizer.dir to /workspace/data/tokenizer_all_sets

## Important:

* Ask for API keys to use LLM models (.env file)  

## Run the service in cmd

- fastapi dev src/main.py

Credit to: https://gitlab.cerebra.kz/shyryn/asr_giga