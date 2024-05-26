# Character-Level Language Modeling with RNNs and LSTMs

This project implements a character-level language model using RNNs and LSTMs
to generate text based on the works of Shakespeare.

The language model is a many-to-many recurrent neural network, as described in
[Karpathy's article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

## Project Structure

- `dataset.py`: Contains the `Shakespeare` dataset class and data loading functions.
- `model.py`: Defines the RNN and LSTM models.
- `main.py`: Trains and validates the models, and plots the loss curves.
- `generate.py`: Generates text using the trained model.
- `README.md`: This file, contains the project description and usage instructions.

## Requirements

- Python 3.8+
- torch
- numpy
- matplotlib
- scikit-learn

## Usage

### Training the Models

To train the models, run the `main.py` script. This will train both the RNN and LSTM models, and save the model with the last epoch model.

```bash
python main.py
```

```python
### Configuration ###
seq_length = 30
batch_size = 128
num_epochs = 20
learning_rate = 0.002
#####################
```

### Generating Text

To generate text using the trained model, run the `generate.py` script. This script will load the last model and generate text based on the provided seed texts.

```bash
python generate.py
```

## Report

### Training and Validation Loss

The training and validation losses for both vanilla RNN and LSTM models are plotted and saved as `loss_plot.png`. Below is the comparison of the training and validation loss curves:


![loss_plot](https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/e2fc8871-6d81-4b20-953f-172adf219933)


### Model Performance Comparison

The performance of the vanilla RNN and LSTM models is compared based on the validation loss values. The LSTM model generally performs better due to its ability to capture long-term dependencies.

### Text Generation

The `generate.py` script generates text using the trained LSTM model. Five different seed texts are used, and 100 characters are generated for each seed text. Below are some examples of generated text:

```
----- Seed text: "KING RICHARD III:"
----- Generated text: "KING RICHARD III: now, the straves are bearly dost not the sicks me and me the perise of thing the stand his come of father the be te the ha the mand the "

----- Seed text: "LADY ANNE:"
----- Generated text: "LADY ANNE: I fook is not the seald the mark of the stand of thing the stands the server the stand his the ming the stand his the the with he come of"

----- Seed text: "BUCKINGHAM:"
----- Generated text: "BUCKINGHAM: the beally the speak of the stants his to the hath the hather the hath and the hath the hath the hath the hath the hath the hath the hath the hath the"

----- Seed text: "DUCHESS OF YORK:"
----- Generated text: "DUCHESS OF YORK: the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath"

----- Seed text: "QUEEN ELIZABETH:"
----- Generated text: "QUEEN ELIZABETH: the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath the hath"
```

### Softmax Temperature

The softmax function with a temperature parameter \( T \) is used to control the diversity of the generated text. The function is defined as:

$$y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}$$

- **High Temperature**: Produces more random and diverse text.
- **Low Temperature**: Produces more coherent and deterministic text.

Experimenting with different temperatures helps in generating more plausible results by balancing between creativity and coherence.


