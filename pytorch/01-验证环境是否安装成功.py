
if __name__ == '__main__':
    import torch
    print(torch.version)
    print(torch.version.cuda)
    torch.cuda.is_available()
    print(torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")