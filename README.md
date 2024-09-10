# LLMCompress

LLMCompress is a software tool that leverages Large Language Models (LLMs) for efficient lossless text compression. This project explores the use of advanced language models to achieve high compression ratios while maintaining the integrity of the original text.

## Features

- Utilizes various LLMs for text compression
- Supports multiple compression methods (Ranks, AC)
- Configurable parameters for fine-tuning compression performance
- Comprehensive logging and result analysis

## Supported Models

- Mixtral 8x7B
- GPT-2
- Yi-34B
- NeMo
- LLaMA 2
- LLaMA 3
- LLaMA 3.1
- Mistral 7B

## Prerequisites

- NVIDIA GPUs (recommended: 2x RTX 3090 or equivalent)
- Linux-based system (Ubuntu recommended)
- Python 3.9 or higher
- Pytorch

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rotask/LLMCompress.git
   
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate llmzip-env
   ```

## Usage

Run experiments using the `main.py` script:

```
python main.py --model gpt2 --compression_method Ranks --input_file "Data/text8_1MB.txt" --batch_size 32 --context_size 512 --summary "Experiment-specific context"
```

For a list of all available options:

```
python main.py --help
```

## Project Structure

- `main.py`: Main script for running experiments
- `llmzip.py`: Core compression logic
- `config.py`: Configuration settings
- `models.py`: Model definitions and utilities
- `logger.py`: Custom Logging functionality

- `Data/`: Directory for input data files
- `Output_Files/`: Directory for compressed output files
- `Logs/`: Directory for log files
- `Results/`: Directory for experiment results

## Acknowledgments

This project was developed as part of a Master's thesis at Imperial College London.
Special thanks to Prof. Deniz Gunduz for guidance and support.

## License

Creative Commons Attribution Non-Commercial No Derivatives

The copyright of this thesis rests with the author. Researchers are free to copy, distribute or transmit the thesis on the condition that they attribute it, that they do not use it for commercial purposes and that they do not alter, transform or build upon it. For any reuse or redistribution, researchers must make clear to others the licence terms of this work.

