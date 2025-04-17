import torch, gc
from model import Net

DEVICE = "cpu"

model = Net().to(DEVICE)

print('total param',  sum([param.nelement() for param in model.parameters()]))

gc.collect()
torch.cuda.empty_cache()