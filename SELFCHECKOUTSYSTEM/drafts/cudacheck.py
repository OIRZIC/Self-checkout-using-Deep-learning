import torch
def check_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Running on CPU.")



if __name__ == "__main__":
    check_cuda()
    print(torch.__version__)
    print(torch.version.cuda)
