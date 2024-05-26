import torch
from model import CharLSTM
from dataset import Shakespeare

def generate(model_path, seed_characters, temperature, length=100):
    """ Generate characters

    Args:
        model_path: path to the trained model
        seed_characters: seed characters
        temperature: T
        length: number of characters to generate

    Returns:
        samples: generated characters
    """

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Shakespeare('./data/shakespeare_train.txt')
    char_indices = dataset.char_to_idx
    indices_char = dataset.idx_to_char

    model = CharLSTM(dataset.vocab_size, embed_size=128, hidden_size=256, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    generated = seed_characters
    sequence = torch.tensor([char_indices[char] for char in seed_characters], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden = model.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))  

        for _ in range(length):
            output, hidden = model(sequence, hidden)
            output = output[-1,:]
            output = output / temperature
            probabilities = torch.nn.functional.softmax(output, dim=-1).squeeze()
            # import pdb; pdb.set_trace()
            next_index = torch.multinomial(probabilities, 1).item()
            next_char = indices_char[next_index]
            generated += next_char
            
            sequence = torch.cat((sequence[:, 1:], torch.tensor([[next_index]]).to(device)), dim=1)
    
    return generated

if __name__ == '__main__':

    seed_text = "KING RICHARD III:"
    temperature = 100.0
    print("temperature : ", temperature)
    generated_text = generate('./last_model.pth', seed_text, temperature, length=100)
    print(f'Seed: "{seed_text}"')
    print(f'Generated text: "{generated_text}"')
    print()

    seed_text = "KING RICHARD III:"
    temperature = 10.0
    print("temperature : ", temperature)
    generated_text = generate('./last_model.pth', seed_text, temperature, length=100)
    print(f'Seed: "{seed_text}"')
    print(f'Generated text: "{generated_text}"')
    print()

    seed_text = "KING RICHARD III:"
    temperature = 1.0
    print("temperature : ", temperature)
    generated_text = generate('./last_model.pth', seed_text, temperature, length=100)
    print(f'Seed: "{seed_text}"')
    print(f'Generated text: "{generated_text}"')
    print()

    seed_text = "KING RICHARD III:"
    temperature = 0.1
    print("temperature : ", temperature)
    generated_text = generate('./last_model.pth', seed_text, temperature, length=100)
    print(f'Seed: "{seed_text}"')
    print(f'Generated text: "{generated_text}"')
    print()

    seed_text = "KING RICHARD III:"
    temperature = 0.01
    print("temperature : ", temperature)
    generated_text = generate('./last_model.pth', seed_text, temperature, length=100)
    print(f'Seed: "{seed_text}"')
    print(f'Generated text: "{generated_text}"')
    print()