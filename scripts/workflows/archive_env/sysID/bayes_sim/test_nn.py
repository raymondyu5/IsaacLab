import torch
import torch.nn as nn

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128),  # fcon0
            nn.Tanh(),           # nl0
            nn.Linear(128, 128), # fcon1
            nn.Tanh()            # nl1
        )

    def forward(self, x):
        return self.net(x)

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# Example input tensor
x_batch = torch.randn(32, 40, device='cuda')
optimizer.zero_grad()
# Forward pass
output = model(x_batch)
print("Output requires_grad:", output.requires_grad)  # Should be True

# Example loss and backward pass
loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))

loss.backward()

