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

The `generate.py` script generates text using the trained LSTM model. Five different temperatures texts are used, and 100 characters are generated for each seed text. Below are examples of generated text:

```
temperature :  100.0
Seed: "KING RICHARD III:"
Generated text: "KING RICHARD III:FoF&y'nbvj-DhiSLe:?cwvgvrBLFUSlarvzrOKNCMZeLuPK:CQFhYM?hEwwVRcBLn,Z!d
.CN;xKWjftWAUPUAlEJmthweaJNQSM"

temperature :  10.0
Seed: "KING RICHARD III:"
Generated text: "KING RICHARD III:
EC.
ClORTI!'ShW,ssorc!agoyes
awQ;aqumy?s KneTS,gSigtco;'uqw' yOs!CLVrNxtydogh'd O.D
harianiq;oow? r"

temperature :  1.0
Seed: "KING RICHARD III:"
Generated text: "KING RICHARD III:
Know'st not unlose the behended;
For, by the way, I'll sort ocquer blame
Menenius, and those senato"

temperature :  0.1
Seed: "KING RICHARD III:"
Generated text: "KING RICHARD III:
Why, Buckingham, nor you;
You have been too rough, something
too rough, something
too rough, someth"

temperature :  0.01
Seed: "KING RICHARD III:"
Generated text: "KING RICHARD III:
Why, Buckingham, I say, I would be king,

BUCKINGHAM:
Then I salute your grace of York as mother,
A"
```

### Softmax Temperature

The softmax function with a temperature parameter \( T \) is used to control the diversity of the generated text. The function is defined as:

$$y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}$$

- **High Temperature**: Produces more random and diverse text.
- **Low Temperature**: Produces more coherent and deterministic text.

Experimenting with different temperatures helps in generating more plausible results by balancing between creativity and coherence.


