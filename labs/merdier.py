import torch

print(torch.cuda.is_available())  # Doit afficher True
print(torch.cuda.get_device_name(0))  # Doit afficher votre RTX 2000 Ada