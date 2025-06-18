import argparse
import torch
import torch.nn as nn
from timm.models import create_model
from timm.data import create_dataset, create_loader, resolve_data_config
from pathlib import Path
import time
import os

# Import necessary functions from the project
from utils import replace_batchnorm
# The following import is needed for timm to find the model registration
from model import repvit 

def get_args_parser():
    parser = argparse.ArgumentParser('RepViT quantization script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    # Model parameters
    parser.add_argument('--model', default='repvit_m0_9', type=str, metavar='MODEL',
                        help='Name of model to quantize')
    parser.add_argument('--resume', default='./pretrain/repvit_m0_9_distill_300e.pth',
                        help='Resume from checkpoint')
    parser.add_argument('--data-path', default='./dataset/val', type=str,
                        help='dataset path')
    parser.add_argument('--num_workers', default=8, type=int)
    # Quantization parameters
    parser.add_argument('--quant-backend', default='x86', type=str,
                        help='Quantization backend for x86 (fbgemm, x86) or ARM (qnnpack)')
    parser.add_argument('--skip-layers', nargs='+', default=[],
                        help='List of layer names to skip quantization. e.g., classifier')

    return parser


def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    total = 0
    correct = 0
    total_loss = 0.0
    
    start_time = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if i % 50 == 0:
                print(f"  Eval batch {i}/{len(data_loader)}")

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    eval_time = time.time() - start_time
    
    print(f'  Accuracy of the network on the {total} test images: {accuracy:.2f} %')
    print(f'  Average loss: {avg_loss:.4f}')
    print(f'  Evaluation time: {eval_time:.2f} seconds')
    
    return accuracy

def print_model_size(model, label):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    print(f"Size of {label} model: {size:.2f} MB")
    os.remove('temp.p')
    return size

def main(args):
    # Use GPU for initial FP32 evaluation if available, but quantization must happen on CPU
    eval_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quant_device = torch.device('cpu')

    print(f"Using device for FP32 eval: {eval_device}")
    print(f"Using device for Quantization: {quant_device}")

    # --- 1. Load Original FP32 Model ---
    print(f"Creating model: {args.model}")
    model_fp32 = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        distillation=True
    )

    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # This logic handles checkpoints saved with distillation head for a model without one
        if 'head_dist.weight' in model_state_dict and not hasattr(model_fp32.classifier, 'classifier_dist'):
            print("Averaging distillation and main heads for inference.")
            w = (model_state_dict['head.weight'] + model_state_dict['head_dist.weight']) / 2
            b = (model_state_dict['head.bias'] + model_state_dict['head_dist.bias']) / 2
            model_state_dict['head.weight'] = w
            model_state_dict['head.bias'] = b
            del model_state_dict['head_dist.weight'], model_state_dict['head_dist.bias']
        
        model_fp32.load_state_dict(model_state_dict, strict=True)
    
    model_fp32.to(eval_device)
    model_fp32.eval()
    
    # --- 2. Fuse the model (The CRITICAL step) ---
    print("Fusing model for inference...")
    replace_batchnorm(model_fp32)
    print("Fusion complete.")

    print("\n--- Fused Model Structure ---")
    for name, _ in model_fp32.named_modules():
        print(name)
    print("-----------------------------\n")

    # --- 3. Prepare Dataset ---
    print("Preparing dataset...")
    config = resolve_data_config({}, model=model_fp32)
    dataset_val = create_dataset(name='', root=args.data_path, split='val', is_training=False)
    
    # Create a subset for calibration (e.g., first 1024 samples)
    calib_indices = list(range(min(1024, len(dataset_val))))
    dataset_calib = torch.utils.data.Subset(dataset_val, calib_indices)

    # Loader for calibration using the subset
    data_loader_calib = create_loader(
        dataset_calib,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=False, # Prefetcher not ideal for CPU-based quantization
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.num_workers,
        crop_pct=config['crop_pct'],
        pin_memory=True
    )
    # Full loader for evaluation
    data_loader_eval = create_loader(
        dataset_val,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.num_workers,
        crop_pct=config['crop_pct'],
        pin_memory=True
    )

    # --- 4. Evaluate FP32 model ---
    print("\n--- Evaluating FP32 (Fused) Model ---")
    fp32_size = print_model_size(model_fp32, "FP32")
    fp32_accuracy = evaluate(data_loader_eval, model_fp32, eval_device)

    # --- 5. Quantize the model using PTQ with FX Graph Mode ---
    print("\n--- Performing Post-Training Static Quantization ---")
    
    # CRITICAL: Move model to CPU for quantization
    model_to_quantize = model_fp32.to(quant_device)
    model_to_quantize.eval()
    
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    
    qconfig = get_default_qconfig(args.quant_backend)
    qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
    
    print("Preparing model for quantization with FX Graph Mode...")
    
    # Handle layers to skip
    modules_to_skip = set()
    if args.skip_layers:
        for name, _ in model_to_quantize.named_modules():
            if any(skip_name in name for skip_name in args.skip_layers):
                 modules_to_skip.add(name)
    
    print(f"Skipping quantization for modules: {modules_to_skip if modules_to_skip else 'None'}")
    
    example_inputs = (torch.randn(1, 3, *config['input_size'][1:]),)
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs, 
                                prepare_custom_config={'non_traceable_module_name': list(modules_to_skip)})

    print("Calibrating model with calibration data...")
    with torch.no_grad():
        for images, _ in data_loader_calib:
            prepared_model(images)
    print("Calibration complete.")

    print("Converting to quantized model...")
    model_int8 = convert_fx(prepared_model)
    print("Conversion complete.")

    # --- 6. Evaluate INT8 model ---
    print("\n--- Evaluating INT8 (Quantized) Model ---")
    int8_size = print_model_size(model_int8, "INT8")
    model_int8.to(quant_device) # Make sure INT8 model is on CPU for evaluation
    int8_accuracy = evaluate(data_loader_eval, model_int8, quant_device)

    # --- 7. Print Summary ---
    print("\n\n------ Quantization Summary ------")
    print(f"Backend: {args.quant_backend} | Skipped Layers: {args.skip_layers if args.skip_layers else 'None'}")
    print("--------------------------------------------------")
    print(f"| Metric                  | FP32 Model | INT8 Model |")
    print(f"|-------------------------|------------|------------|")
    print(f"| Accuracy                | {fp32_accuracy:7.2f}% | {int8_accuracy:7.2f}% |")
    print(f"| Model Size (MB)         | {fp32_size:10.2f} | {int8_size:10.2f} |")
    print(f"--------------------------------------------------")
    print(f"Accuracy Drop: {fp32_accuracy - int8_accuracy:.2f}%")
    print(f"Size Reduction: {((fp32_size - int8_size) / fp32_size * 100):.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RepViT quantization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)