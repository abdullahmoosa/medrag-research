#!/usr/bin/env python3
"""
Helper script to manage DeepSeek reasoning generation with easy commands.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import glob

def show_status():
    """Show current processing status."""
    script_dir = Path(__file__).parent
    progress_file = script_dir / "reasoning_progress.json"
    
    print("🔍 DeepSeek Reasoning Generation Status")
    print("=" * 50)
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            current_batch = progress.get('current_batch', 0)
            total_batches = progress.get('total_batches', 0)
            stats = progress.get('stats', {})
            
            print(f"📊 Progress: Batch {current_batch}/{total_batches}")
            print(f"✅ Processed: {stats.get('total_processed', 0)} samples")
            print(f"🎯 Accuracy: {(stats.get('correct_answers', 0) / max(stats.get('total_processed', 1), 1) * 100):.1f}%")
            print(f"❌ API Errors: {stats.get('api_errors', 0)}")
            print(f"🔤 Tokens: {stats.get('total_input_tokens', 0)} input, {stats.get('total_output_tokens', 0)} output")
            print(f"⏰ Last update: {progress.get('timestamp', 'Unknown')}")
            
            if current_batch < total_batches:
                print(f"\n▶️  Processing can be resumed from batch {current_batch + 1}")
            else:
                print(f"\n🎉 Processing completed!")
                
        except Exception as e:
            print(f"❌ Error reading progress: {e}")
    else:
        print("📭 No active processing found")
    
    # Show existing output files
    output_files = list(script_dir.glob("train_reasoning_generated_*.json"))
    if output_files:
        print(f"\n📁 Output files found:")
        for file in sorted(output_files):
            size = file.stat().st_size
            print(f"  - {file.name} ({size:,} bytes)")
    else:
        print(f"\n📁 No output files found")

def start_fresh():
    """Start processing from scratch."""
    print("🔄 Starting fresh processing...")
    script_path = Path(__file__).parent / "generate_deepseek_reasoning.py"
    cmd = [sys.executable, str(script_path), "--clear_progress"]
    subprocess.run(cmd)

def resume_processing():
    """Resume processing from where it left off."""
    print("▶️  Resuming processing...")
    script_path = Path(__file__).parent / "generate_deepseek_reasoning.py"
    cmd = [sys.executable, str(script_path), "--resume"]
    subprocess.run(cmd)

def start_with_config():
    """Start with custom configuration."""
    print("⚙️  Configure processing parameters:")
    
    try:
        batch_size = int(input("Batch size (default 10): ") or "10")
        rate_limit = int(input("Rate limit RPM (default 50): ") or "50")
        start_batch = int(input("Start from batch (default 0): ") or "0")
    except ValueError:
        print("❌ Invalid input, using defaults")
        batch_size, rate_limit, start_batch = 10, 50, 0
    
    script_path = Path(__file__).parent / "generate_deepseek_reasoning.py"
    cmd = [
        sys.executable, str(script_path),
        "--batch_size", str(batch_size),
        "--rate_limit", str(rate_limit),
        "--start_batch", str(start_batch),
        "--clear_progress"
    ]
    
    print(f"🚀 Starting with: batch_size={batch_size}, rate_limit={rate_limit}, start_batch={start_batch}")
    subprocess.run(cmd)

def clean_files():
    """Clean up progress and output files."""
    # Target the actual output directory for cleaning
    output_dir = Path(__file__).parent.parent.parent / "evaluation_results" / "reasoning_dataset_generation"
    print(f"🧹 Cleaning files in: {output_dir}")

    # Remove progress file
    progress_file = output_dir / "reasoning_progress.json"
    if progress_file.exists():
        progress_file.unlink()
        print("🗑️  Removed progress file")

    # Remove log file
    log_file = output_dir / "deepseek_reasoning_generation.log"
    if log_file.exists():
        log_file.unlink()
        print("🗑️  Removed log file")

    # List output files for deletion
    output_files = list(output_dir.glob("train_reasoning_generated_*.json"))
    if output_files:
        print(f"📁 Found {len(output_files)} output files:")
        for i, file in enumerate(output_files):
            print(f"  {i+1}. {file.name}")

        try:
            choice = input("\nDelete all? (y/N): ").lower()
            if choice == 'y':
                for file in output_files:
                    file.unlink()
                print("🗑️  Deleted all output files")
            else:
                print("📁 Output files kept")
        except KeyboardInterrupt:
            print("\n📁 Output files kept")
    else:
        print("📁 No output files to clean")

def test_setup():
    """Test the setup."""
    print("🧪 Testing setup...")
    script_path = Path(__file__).parent / "test_openrouter_setup.py"
    cmd = [sys.executable, str(script_path)]
    subprocess.run(cmd)

def main():
    """Main menu."""
    while True:
        print("\n🤖 DeepSeek Reasoning Generator Manager")
        print("=" * 45)
        print("1. 📊 Show Status")
        print("2. 🔄 Start Fresh")
        print("3. ▶️  Resume Processing") 
        print("4. ⚙️  Start with Custom Config")
        print("5. 🗑️  Clean Files")
        print("6. 🧪 Test Setup")
        print("7. 🚪 Exit")
        
        try:
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                show_status()
            elif choice == '2':
                start_fresh()
            elif choice == '3':
                resume_processing()
            elif choice == '4':
                start_with_config()
            elif choice == '5':
                clean_files()
            elif choice == '6':
                test_setup()
            elif choice == '7':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice, please try again")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
