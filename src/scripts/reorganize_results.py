import os
import shutil
from pathlib import Path

def reorganize_evaluation_results(base_dir: str = "evaluation_results"):
    base_path = Path(base_dir)
    
    # Get all model directories
    model_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.endswith(('_few_shot', '_zero_shot'))]
    
    for model_dir in model_dirs:
        # Create new directory structure
        new_model_dir = base_path / f"{model_dir.name}_zero_shot/train"
        new_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Move batches directory
        old_batches = model_dir / "batches"
        new_batches = new_model_dir / "batches"
        if old_batches.exists():
            if new_batches.exists():
                shutil.rmtree(new_batches)
            shutil.move(str(old_batches), str(new_batches))
        
        # Move and rename progress file
        old_progress = model_dir / "evaluation_progress.json"
        new_progress = new_model_dir / "evaluation_progress.json"
        if old_progress.exists():
            shutil.move(str(old_progress), str(new_progress))
        
        # Remove old directory if empty
        if model_dir.exists():
            try:
                model_dir.rmdir()  # This will only succeed if the directory is empty
            except OSError:
                print(f"Warning: Could not remove {model_dir} as it still contains files")

if __name__ == "__main__":
    # Get the absolute path to the evaluation_results directory
    base_dir = Path("C:/Users/User/Downloads/nusrat/medrag/evaluation_results")
    
    # Confirm with user
    print("This will reorganize the evaluation results directory structure.")
    print("New structure will be: model_name_zero_shot/train/")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        reorganize_evaluation_results(str(base_dir))
        print("Reorganization complete!")
    else:
        print("Operation cancelled.")
