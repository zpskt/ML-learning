
if __name__ == '__main__':
    import torch
    print(torch.version)
    print(torch.version.cuda)
    torch.cuda.is_available()
    print(torch.cuda.device_count())
