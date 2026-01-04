import torch

def get_available_vram():
    if torch.cuda.is_available():
        # Get the currently selected device, or specify a device ID like 'cuda:0'
        device = torch.cuda.current_device()

        # Get free and total memory in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

        # Convert to GB for better readability
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)

        print(f"Total VRAM: {total_gb:.2f} GB")
        print(f"Free VRAM: {free_gb:.2f} GB")
        print(f"Used VRAM: {total_gb - free_gb:.2f} GB")
    else:
        print("CUDA is not available. PyTorch is running on CPU.")

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Calculate the size of buffers
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    parameters = 0
    for param in model.parameters():
        parameters += param.nelement()

    # Total model size in bytes
    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 ** 2)

    print(f"Model size (bytes): {total_size_bytes}")
    print(f"Model size (MB): {total_size_mb:.2f}")
    print(f"Model has {parameters:,} parameters")