import torch.nn as nn

# Create a ModuleList with some modules
module_list = nn.ModuleList([
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
])

# Access the first module in the ModuleList
first_module = module_list[0]
print(first_module)
# Retrieve in_channels and out_channels attributes
in_channels = first_module.in_channels
out_channels = first_module.out_channels

print("Input Channels:", in_channels)
print("Output Channels:", out_channels)