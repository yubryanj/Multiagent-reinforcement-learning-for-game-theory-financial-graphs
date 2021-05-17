import numpy as np
import torch

class MLP(torch.nn.Module):
    def __init__(self, n_outputs=100, n_hidden=20):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, n_hidden),
            # torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            # torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_outputs),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.layers(x.float())
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_outputs = 100
    n_hidden = n_outputs

    xs = torch.as_tensor([[i] for i in range(1, n_outputs + 1)]).to(device)
    ys = torch.as_tensor([[np.eye(n_outputs)[i]] for i in range(n_outputs)]).to(device)

    model = MLP(n_hidden=n_hidden, n_outputs=n_outputs).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    epoch = 0
    loss = 1e10
    old_loss = 1e10
    losses = []

    torch.save(model.state_dict(), "./model")
    
    while loss > 0.10:
        optimizer.zero_grad()
        
        # Forward pass
        y_preds = model(xs)
        predictions = torch.argmax(y_preds, axis=1)
        labels = torch.argmax(ys.squeeze(), axis=1)

        # Compute Loss
        loss = criterion(y_preds, labels)
    
        # Backward pass
        loss.backward()
        optimizer.step()

        if (predictions == labels).sum() == n_outputs:
            print(f"epoch: {epoch} - solution found!")
            print(f'{predictions == labels}')
            break

        if epoch %1e4 == 0 :
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            losses.append(loss)
            np.save("losses.npy",losses)
            torch.save(model.state_dict(), "./mlp_model.pk")    
            print(f"Accuracy is {(predictions == labels).sum() / predictions.size()[0]}")
            print(f"{(predictions == labels)}")
        
        epoch += 1


    torch.save(model.state_dict(), "./mlp_model.pk")    
    np.save("losses.npy",losses)