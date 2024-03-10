import os
import sys
import torch

from Model import Unet
def load_model(model_fp, input_size=(128, 128), device='cuda'):
    model = Unet(input_size, device=device)
    model.load_state_dict(torch.load(model_fp)['state_dict'])
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    import torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_src', type=str, help='Path to the source weight file')
    parser.add_argument('model_cpu_dst', type=str, help='Path to the destination of the output weight file')
    parser.add_argument('model_cuda_dst', type=str, help='Path to the destination of the output weight file')
    args = parser.parse_args()
    
    model = load_model(args.model_src, device='cpu')
    example_inputs = torch.zeros((1, 1, 128 ,128))
    traced_model = torch.jit.trace(model, example_inputs)  # Example inputs are needed
    torch.jit.save(traced_model, args.model_cpu_dst)    
    
    model = load_model(args.model_src, device='cuda')
    example_inputs = torch.zeros((1, 1, 128 ,128))
    traced_model = torch.jit.trace(model, example_inputs.to('cuda'))  # Example inputs are needed
    torch.jit.save(traced_model, args.model_cuda_dst)    
    
    print(args.model_src)
    print(args.model_cpu_dst)
    print(args.model_cuda_dst)
    