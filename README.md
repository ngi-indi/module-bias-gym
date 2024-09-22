<div align="center">
  <img src="./assets/logo.jpg" alt="Logo" width="150"/>

  # Bias Gym

  ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
  ![Version 0.1](https://img.shields.io/badge/version-0.1-green.svg)
  ![Status: Stable](https://img.shields.io/badge/status-stable-brightgreen.svg)
    
  <p>
    <strong>Bias Gym</strong> is a testing and benchmarking suite designed to evaluate, detect, and explain biases present in Web content. It provides a structured and customizable framework to test various forms of biases, including but not limited to <strong>gender bias</strong>, <strong>racial bias</strong>, <strong>political bias</strong>, <strong>linguistic bias</strong>, and <strong>hate speech</strong>.
  </p>

</div>


## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Training a model](#training-a-model)
  - [Evaluating and testing](#evaluating-and-testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8 as a base programming environment.
- PyTorch for handling the model training and inference.
- Transformers for state-of-the-art NLP models.

### Setup

#### 1. Clone the repository:

```bash
git clone git clone https://github.com/ngi-indi/module-bias-gym.git
cd module-bias-gym
```

#### 2. Set up the virtual environment (optional but recommended):

  - On Windows:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

  - On macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Install dependencies:
Install the required Python packages by running:
  ```bash
  pip install -r requirements.txt
  ```

#### 4. Download datasets:
- Download the [pre-processed datasets](https://drive.google.com/drive/folders/1VSXZcAmDQj7Gk1_AEA1HI_dVVUF-sFmW?usp=drive_link) and place them in the appropriate directory: ```datasets/```.
- Ensure you have all necessary datasets downloaded and placed in the relevant directory.


#### 5. Download pre-trained models (optional):
- Download the [pre-trained model weights](https://drive.google.com/drive/folders/1aOTVMTdLcDhOHuj-bcJbO5SPM7Zdh-_O?usp=drive_link) and place them in the appropriate directory: ```models/```.
- Ensure you have all necessary models downloaded and placed in the relevant directory.

## Usage

### Training a model

1. **Prepare your data**: Ensure datasets are placed in the correct directory or modify the script to point to your data.

2. **Run the train script** to train a model on a specific task, where you can specify the model and task you want to train on using the provided options:

   ```bash
   # Example: Training a single model on a single task
    python train.py --models roberta --tasks gender-bias --epochs 10
    
    # Example: Training multiple models on multiple tasks
    python train.py --models roberta electra --tasks gender-bias hate-speech
   ```
    
   Parameters list:
   - **--models**: List of models to be trained (default: `['robertatwitter', 'electra', ..., 't5']`)
   - **--tasks**: List of tasks for bias detection (default: `['cognitive-bias', ..., 'gender-bias']`)
   - **--number_of_folds**: Number of folds for cross-validation (default: `5`)
   - **--batch_size**: Batch size for training (default: `32`)
   - **--max_length**: Maximum sequence length for tokenization (default: `128`)
   - **--epochs**: Number of epochs for training (default: `10`)

   
3. **Results**: After training, results and metrics will be saved in the `results/` directory, where you can find detailed reports about model performance in the generated CSV files.

### Evaluating and testing

If you want to evaluate an already trained model, use the `--eval` flag:

   ```bash
   python train.py --model roberta --task gender-bias --eval
   ```

## Contributing

### Reporting bugs and requesting features
- If you find a bug, please open an issue.
- To request a feature, feel free to open an issue as well.

### Developing a new feature

1. **Fork the repository** by clicking the "Fork" button at the top right of this page.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/module-bias-manager.git
   ```
3. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature-branch
   ```
4. **Make your changes.** Please follow the existing code style and conventions.
5. **Commit your changes** with a descriptive commit message:
   ```bash
   git commit -m "Add new feature: explanation of bias model predictions"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature-branch
   ```
7. **Open a pull request** from your fork’s branch to the main branch of this repository.
   - Describe the changes you’ve made in the PR description.
   - Ensure that your PR references any relevant issues.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ngi-indi/module-bias-gym/blob/main/LICENSE.md) file for details.

## Contact
For any questions or support, please reach out to:
- University of Cagliari: bart@unica.it, diego.reforgiato@unica.it, ludovico.boratto@unica.it, mirko.marras@unica.it
- R2M Solution: giuseppe.scarpi@r2msolution.com
- Website: Coming soon!