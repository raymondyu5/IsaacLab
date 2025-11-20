import torch

values = []
# Generate random values from 1 to 10000, 1000 times
for i in range(100):
    random_values = torch.randint(1, 3072, (1, ))

    values.append(random_values)
print(torch.unique(torch.as_tensor(values)))
