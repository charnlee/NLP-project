import torch
from torch.utils.data import DataLoader



def test(model, model_save_path, test_dataset, batch_size):
    model.load_state_dict(torch.load(model_save_path))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    total_acc_test = 0
    model.eval()

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            attention_mask = test_input['attention_mask'].to(device)
            input_ids = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_ids, attention_mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_dataset):.3f}')