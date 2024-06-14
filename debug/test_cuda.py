import torch

if __name__ == "__main__":
    print("Start")
    print("Pytorch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("get_arch_list:", torch.cuda.get_arch_list())
    print()
    print("device:", torch.cuda.is_available(),  torch.cuda.device_count())

    device = "cuda"
    X = torch.FloatTensor([1, 2, 3])
    print(X.shape)
    X = X.to(device)
    print("Done")