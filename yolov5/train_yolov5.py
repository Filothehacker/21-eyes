import os
import argparse
import yaml
import torch
import time
from datetime import datetime
from pathlib import Path
from ultralytics import __version__ as ultralytics_version


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a YOLOv5 model for card detection')
    parser.add_argument('--yaml', type=str, default='classes_v5.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=48, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=448, help='Image size')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLO model size')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads for dataloader')
    args = parser.parse_args()

    # Print training information
    print(f"\n{'='*50}")
    print(f"Starting Card Detection Training with YOLOv5")
    print(f"{'='*50}")
    print(f"YAML config:   {args.yaml}")
    print(f"Model size:    YOLOv5{args.model_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch}")
    print(f"Image size:    {args.imgsz}x{args.imgsz}")
    print(f"Resume:        {args.resume}")
    
    # Check if YAML file exists
    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"YAML file not found: {args.yaml}")
    
    # Load and validate YAML file
    with open(args.yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    # Print dataset information
    print(f"\nDataset information:")
    print(f"- Number of classes: {yaml_data['nc']}")
    print(f"- Training data:     {yaml_data['train']}")
    print(f"- Validation data:   {yaml_data['val']}")
    print(f"- Test data:         {yaml_data['test'] if 'test' in yaml_data else 'Not specified'}")
    
    # Check if directories exist
    for dir_key in ['train', 'val']:
        if not os.path.exists(yaml_data[dir_key]):
            print(f"Warning: {dir_key} directory not found: {yaml_data[dir_key]}")
    
    if 'test' in yaml_data and not os.path.exists(yaml_data['test']):
        print(f"Warning: test directory not found: {yaml_data['test']}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./runs/train/card_detector_{timestamp}")
    
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize model
    pretrained_model = f'yolov5{args.model_size}.pt'
    print(f"\nInitializing model: {pretrained_model}")
    
    # For resuming training
    resume_flag = ''
    if args.resume:
        # Find the latest checkpoint
        checkpoints = list(Path('./runs/train/').glob('*/weights/last.pt'))
        if checkpoints:
            latest_checkpoint = str(max(checkpoints, key=os.path.getctime))
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            resume_flag = f'--resume {latest_checkpoint}'
        else:
            print("No checkpoints found, starting from scratch")
    
    # Prepare device argument
    device_arg = f'--device {args.device}' if args.device else ''
    
    # Start training
    print(f"\n{'='*50}")
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        # Execute training using Ultralytics YOLOv5
        # Import directly from Ultralytics
        try:
            # Import YOLOv5 from Ultralytics
            from ultralytics import YOLO
            print("Using Ultralytics for YOLOv5 training")
            
            # Load the YOLOv5 model
            model = YOLO(pretrained_model, task='detect')
            
            # Prepare training arguments
            train_args = {
                'data': args.yaml,
                'epochs': args.epochs,
                'batch': args.batch,
                'imgsz': args.imgsz,
                'patience': 15,
                'project': 'runs/train',
                'name': f'card_detector_{timestamp}',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'auto',
                'verbose': True,
                'save': True,
                'save_period': 10,
                'workers': args.workers
            }
            
            # Add device if specified
            if args.device:
                train_args['device'] = args.device
                
            # Add resume flag if specified
            if args.resume and 'latest_checkpoint' in locals():
                train_args['resume'] = latest_checkpoint
                
            print(f"Training with arguments: {train_args}")
            
            # Start training
            results = model.train(**train_args)
            
        except ImportError:
            print("Failed to import YOLO from ultralytics. Installing and using standalone YOLOv5...")
            os.system('pip install yolov5')
            
            # Construct the YOLOv5 training command as fallback
            train_cmd = (
                f'python -m yolov5.train '
                f'--img {args.imgsz} '
                f'--batch {args.batch} '
                f'--epochs {args.epochs} '
                f'--data {args.yaml} '
                f'--weights {pretrained_model} '
                f'--project runs/train '
                f'--name card_detector_{timestamp} '
                f'--patience 15 '
                f'--save-period 10 '
                f'--workers {args.workers} '
                f'{device_arg} '
                f'{resume_flag}'
            )
            
            print(f"Executing fallback command: {train_cmd}")
            os.system(train_cmd)
        
        # After training, copy the best model
        best_model_path = output_dir / 'weights' / 'best.pt'
        final_model_path = Path('models') / 'best_card_detector.pt'
        Path('models').mkdir(exist_ok=True)
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, final_model_path)
            print(f"\nBest model copied to: {final_model_path}")
            
            # Get model info
            os.system(f'python -m yolov5.models.common --weights {final_model_path}')
        
        # Print training summary
        print(f"\n{'='*50}")
        print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Results saved to {output_dir}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()