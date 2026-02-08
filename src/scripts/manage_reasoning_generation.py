#!/usr/bin/env python3
"""
Interactive management script for DeepSeek reasoning generation.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_menu():
    """Print the main menu options."""
    print("\n" + "="*50)
    print("🧠 DeepSeek Reasoning Generation Manager")
    print("="*50)
    print("1. Start fresh reasoning generation")
    print("2. Resume previous generation")
    print("3. Check progress status")
    print("4. View statistics") 
    print("5. Clean up and start over")
    print("6. Test API connection")
    print("7. View generated files")
    print("8. Exit")
    print("="*50)

def get_user_choice():
    """Get user menu choice."""
    try:
        choice = int(input("Enter your choice (1-8): "))
        return choice
    except ValueError:
        return None

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🚀 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Error!")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run command: {e}")
        return False

def start_fresh():
    """Start fresh reasoning generation."""
    print("\n🔄 Starting fresh reasoning generation...")
    
    # Get parameters
    batch_size = input("Enter batch size (default: 5): ").strip() or "5"
    rate_limit = input("Enter rate limit RPM (default: 50): ").strip() or "50"
    
    cmd = f"python generate_deepseek_reasoning.py --batch_size {batch_size} --rate_limit {rate_limit} --clear_progress"
    run_command(cmd, "Starting fresh generation")

def resume_generation():
    """Resume previous generation."""
    print("\n⏯️ Resuming previous generation...")
    
    # Check if progress file exists
    progress_file = Path("../../evaluation_results/reasoning_dataset_generation/reasoning_progress.json")
    if progress_file.exists():
        cmd = "python generate_deepseek_reasoning.py --resume"
        run_command(cmd, "Resuming generation")
    else:
        print("❌ No previous progress found. Starting fresh instead...")
        start_fresh()

def check_progress():
    """Check current progress status."""
    print("\n📊 Checking progress status...")
    
    progress_file = Path("../../evaluation_results/reasoning_dataset_generation/reasoning_progress.json")
    if progress_file.exists():
        try:
            import json
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            stats = progress.get('stats', {})
            print(f"Current batch: {progress.get('current_batch', 0)}")
            print(f"Total processed: {stats.get('total_processed', 0)}")
            print(f"Correct answers: {stats.get('correct_answers', 0)}")
            print(f"Incorrect answers: {stats.get('incorrect_answers', 0)}")
            print(f"API errors: {stats.get('api_errors', 0)}")
            
            total = stats.get('total_processed', 0)
            if total > 0:
                accuracy = (stats.get('correct_answers', 0) / total) * 100
                print(f"Accuracy: {accuracy:.1f}%")
            
        except Exception as e:
            print(f"❌ Error reading progress: {e}")
    else:
        print("📭 No progress file found - no generation in progress")

def view_statistics():
    """View detailed statistics."""
    print("\n📈 Viewing detailed statistics...")
    check_progress()
    
    # Also show file sizes and counts
    output_dir = Path("../../evaluation_results/reasoning_dataset_generation")
    if output_dir.exists():
        files = list(output_dir.glob("train_reasoning_generated_*.json"))
        if files:
            print(f"\nGenerated files: {len(files)}")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name}: {size_mb:.1f} MB")

def clean_up():
    """Clean up and start over."""
    print("\n🧹 Cleaning up previous data...")
    
    confirm = input("This will delete all progress and generated files. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Remove progress file
    progress_file = Path("../../evaluation_results/reasoning_dataset_generation/reasoning_progress.json")
    if progress_file.exists():
        progress_file.unlink()
        print("✅ Progress file deleted")
    
    # Remove generated files
    output_dir = Path("../../evaluation_results/reasoning_dataset_generation")
    if output_dir.exists():
        files = list(output_dir.glob("train_reasoning_generated_*.json"))
        for file in files:
            file.unlink()
            print(f"✅ Deleted {file.name}")
    
    print("🎉 Cleanup complete! Ready for fresh start.")

def test_api():
    """Test API connection."""
    print("\n🔍 Testing API connection...")
    cmd = "python test_openrouter_setup.py"
    run_command(cmd, "Testing API connection")

def view_files():
    """View generated files."""
    print("\n📁 Generated files:")
    
    output_dir = Path("../../evaluation_results/reasoning_dataset_generation")
    if output_dir.exists():
        files = list(output_dir.glob("train_reasoning_generated_*.json"))
        if files:
            for i, file in enumerate(files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                mod_time = file.stat().st_mtime
                import datetime
                mod_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
                print(f"  {i}. {file.name}")
                print(f"     Size: {size_mb:.1f} MB, Modified: {mod_date}")
        else:
            print("📭 No generated files found")
    else:
        print("📭 Output directory doesn't exist yet")

def main():
    """Main interactive loop."""
    print("🧠 Welcome to DeepSeek Reasoning Generation Manager!")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == 1:
            start_fresh()
        elif choice == 2:
            resume_generation()
        elif choice == 3:
            check_progress()
        elif choice == 4:
            view_statistics()
        elif choice == 5:
            clean_up()
        elif choice == 6:
            test_api()
        elif choice == 7:
            view_files()
        elif choice == 8:
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
