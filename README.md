# Zero-to-Lightning :zap::  Comprehensive PyTorch Lightning Tutorial Guide

<a target="_blank" href="https://lightning.ai/ishandutta0098/studios/zero-to-lightning">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>
  
Welcome to the GitHub repository for Zero-to-Lightning! This project contains a collection of independent, executable scripts that showcase most of the available functionalities in PyTorch Lightning, each covering a new feature or technique. It's organized to help you smoothly progress from basic to advanced PyTorch Lightning concepts.

## Project Demo

https://github.com/ishandutta0098/zero-to-lightning/assets/47643789/a068e1d1-0ec8-4357-b4e2-d1c8090224fd


## Project Directory

```
zero-to-lightning
        |-src
            |-basic
            |   |-level_01_lightning_module
            |   |-level_02_validation_and_testing
            |   |-level_03_checkpointing
            |   |-level_04_early_stopping
            |   |-level_05_pretrained_model
            |   |-level_06_debugging_model
            |   |-level_07_inference
            |
            |-intermediate
            |   |-level_08_accelerated_hardware
            |   |-level_09_modularize
            |   |-level_11_scaling_techniques
            |   |-level_12_deploying_models
            |   |-level_13_profiler
            |
            |-advanced
                |-level_14_run_with_config_file
                |-level_15_modify_trainer
                |-level_16_enable_manual_optimization
                |-level_17_advanced_checkpointing
                |-level_18_ipu
                |-level_19_hpu

```
  
- **Basic**: 🏗 Foundational Lightning concepts like creating modules, validation and testing, checkpointing, early stopping, pretrained models, debugging, and inference.
- **Intermediate**: 🚀 More specialized topics like accelerated hardware, modularization, scaling techniques, deployment, and profiling.
- **Advanced**: 🔍 Deep dives into running with config files, modifying trainers, manual optimization, advanced checkpointing, IPUs, and HPUs.

## Overview

Each sub-directory is designed to help users become familiar with a specific set of PyTorch Lightning functionalities and best practices. Whether you're just starting out or are an advanced user seeking to refine your techniques, the project provides structured guidance and practical examples.

## Features

- **Compact, Executable Scripts**: 📦 Each script is designed to be concise, demonstrating how individual features, functions, or classes operate, making learning targeted and efficient.
- **CPU-Friendly**: 🖥 Most scripts are optimized to run on standard CPUs, minimizing the need for specialized hardware.
- **Quick Iteration**: ⏲ Each script executes in under a minute, enabling rapid testing, learning, and iteration.
- **Official Documentation Links**: 📚 Every script is accompanied by relevant references to official Lightning documentation, helping you deepen your understanding.
- **Independent Execution**: 🏃‍♂️ The scripts are modular, allowing you to explore features individually without needing to execute the entire project.
- **Comprehensive Coverage**: 🌐 From basic modules and validation to advanced manual optimization and hardware-specific integrations, this guide ensures broad exposure to the various functionalities PyTorch Lightning offers.
- **Step-by-Step Structure**: 🛠 Organized progressively, it enables users to gradually advance from foundational knowledge to more sophisticated techniques.


  
## Getting Started

To get started with this project, clone the repository and follow the instructions below.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ishandutta0098/zero-to-lightning.git
```

2. Navigate to the project directory:
```bash
cd zero-to-lightning
```

3. Create the conda environment:
```bash
# Create the conda environment
conda env create -f conda.yml

# Activate the environment
conda activate lit-env
```

### Usage
You can run any script by passing it's path directly as shown below.

```bash
python <path_to_script>

# Example
python src/basic/level_01_lightning_module/lightning_module.py
```

Most of the scripts run directly. For one script we use the LightningCLI.  
To run the script `src/advanced/level_14_run_with_config_file/run_with_yaml.py` follow the below steps 👇
  
```bash
# There are 3 Steps to run this:
# 1. Save the current configs in config.yaml
python src/advanced/level_14_run_with_config_file/run_with_yaml.py fit --print_config > config.yaml

# 2. Run the training using the config file
python src/advanced/level_14_run_with_config_file/run_with_yaml.py fit --config config.yaml

# 3. Modify the config file and run the training again
# Example, try making `max_epochs` as 3 in the config file and run the training again
python src/advanced/level_14_run_with_config_file/run_with_yaml.py fit --config config.yaml
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

