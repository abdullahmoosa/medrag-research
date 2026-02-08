#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Domain Detection Results Analysis

This script analyzes and summarizes domain detection evaluation results.
It provides detailed analysis of model performance across different domains and configurations.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

class DomainDetectionAnalyzer:
    def __init__(self, base_results_dir="evaluation_results"):
        self.base_results_dir = Path(base_results_dir)
        self.domain_detection_dir = self.base_results_dir / "domain-detection" / "LLM"
        
    def find_all_results(self):
        """Find all domain detection result directories"""
        results = []
        
        if not self.domain_detection_dir.exists():
            print(f"Domain detection directory not found: {self.domain_detection_dir}")
            return results
        
        for model_dir in self.domain_detection_dir.iterdir():
            if model_dir.is_dir():
                for split_dir in model_dir.iterdir():
                    if split_dir.is_dir():
                        progress_file = split_dir / "evaluation_progress.json"
                        if progress_file.exists():
                            results.append({
                                "model_config": model_dir.name,
                                "split": split_dir.name,
                                "progress_file": progress_file,
                                "results_dir": split_dir / "batches"
                            })
        
        return results
    
    def load_results(self, result_info):
        """Load results from a specific evaluation"""
        try:
            # Load progress file
            with open(result_info["progress_file"], 'r') as f:
                progress = json.load(f)
            
            # Load all batch results
            batch_results = []
            results_dir = result_info["results_dir"]
            
            if results_dir.exists():
                for batch_file in sorted(results_dir.glob("batch_*.json")):
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                        batch_results.extend(batch_data)
            
            return {
                "progress": progress,
                "results": batch_results,
                "model_config": result_info["model_config"],
                "split": result_info["split"]
            }
            
        except Exception as e:
            print(f"Error loading results from {result_info['progress_file']}: {e}")
            return None
    
    def analyze_model_performance(self, results_data):
        """Analyze performance for a single model configuration"""
        results = results_data["results"]
        if not results:
            return None
        
        # Overall metrics
        total_questions = len(results)
        correct_predictions = sum(1 for r in results if r.get("is_correct", False))
        accuracy = (correct_predictions / total_questions * 100) if total_questions > 0 else 0
        
        # Domain-wise analysis
        domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for result in results:
            true_domain = result.get("true_domain", "Unknown")
            is_correct = result.get("is_correct", False)
            
            domain_stats[true_domain]["total"] += 1
            if is_correct:
                domain_stats[true_domain]["correct"] += 1
        
        # Calculate domain accuracies
        domain_accuracies = {}
        for domain, stats in domain_stats.items():
            domain_accuracies[domain] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        # Confusion analysis
        confusion_data = []
        for result in results:
            confusion_data.append({
                "true_domain": result.get("true_domain", "Unknown"),
                "predicted_domain": result.get("predicted_domain", "Unknown"),
                "is_correct": result.get("is_correct", False)
            })
        
        # Performance metrics from progress
        progress = results_data.get("progress", {})
        metrics = progress.get("cumulative_metrics", {})
        
        return {
            "model_config": results_data["model_config"],
            "split": results_data["split"],
            "total_questions": total_questions,
            "correct_predictions": correct_predictions,
            "overall_accuracy": accuracy,
            "domain_accuracies": domain_accuracies,
            "domain_stats": dict(domain_stats),
            "confusion_data": confusion_data,
            "avg_inference_time": metrics.get("average_inference_time", 0),
            "total_inference_time": metrics.get("total_inference_time", 0)
        }
    
    def create_summary_report(self, all_analyses):
        """Create a comprehensive summary report"""
        if not all_analyses:
            print("No analysis data available")
            return
        
        print("DOMAIN DETECTION EVALUATION SUMMARY")
        print("=" * 80)
        
        # Overall summary table
        summary_data = []
        for analysis in all_analyses:
            if analysis:
                summary_data.append({
                    "Model": analysis["model_config"],
                    "Split": analysis["split"],
                    "Questions": analysis["total_questions"],
                    "Correct": analysis["correct_predictions"],
                    "Accuracy (%)": f"{analysis['overall_accuracy']:.2f}",
                    "Avg Time (s)": f"{analysis['avg_inference_time']:.2f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\nOverall Performance:")
            print(summary_df.to_string(index=False))
        
        # Best performing models
        if summary_data:
            summary_df['Accuracy_num'] = summary_df['Accuracy (%)'].astype(float)
            best_overall = summary_df.loc[summary_df['Accuracy_num'].idxmax()]
            
            print(f"\nBest Overall Performance:")
            print(f"Model: {best_overall['Model']}")
            print(f"Split: {best_overall['Split']}")
            print(f"Accuracy: {best_overall['Accuracy (%)']}%")
        
        # Domain-wise analysis
        print(f"\n\nDomain-wise Performance Analysis:")
        print("-" * 50)
        
        # Aggregate domain performance across all models
        domain_performances = defaultdict(list)
        
        for analysis in all_analyses:
            if analysis and analysis["domain_accuracies"]:
                for domain, accuracy in analysis["domain_accuracies"].items():
                    domain_performances[domain].append(accuracy)
        
        # Calculate domain statistics
        domain_summary = []
        for domain, accuracies in domain_performances.items():
            if accuracies:
                domain_summary.append({
                    "Domain": domain,
                    "Avg Accuracy": f"{np.mean(accuracies):.2f}%",
                    "Best Accuracy": f"{np.max(accuracies):.2f}%",
                    "Worst Accuracy": f"{np.min(accuracies):.2f}%",
                    "Std Dev": f"{np.std(accuracies):.2f}",
                    "Evaluations": len(accuracies)
                })
        
        if domain_summary:
            domain_df = pd.DataFrame(domain_summary)
            domain_df = domain_df.sort_values("Avg Accuracy", ascending=False)
            print(domain_df.to_string(index=False))
    
    def create_visualizations(self, all_analyses, output_dir="domain_detection_analysis"):
        """Create visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall accuracy comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        model_names = []
        accuracies = []
        splits = []
        
        for analysis in all_analyses:
            if analysis:
                model_names.append(analysis["model_config"])
                accuracies.append(analysis["overall_accuracy"])
                splits.append(analysis["split"])
        
        if model_names:
            # Create bar plot
            df_plot = pd.DataFrame({
                "Model": model_names,
                "Accuracy": accuracies,
                "Split": splits
            })
            
            sns.barplot(data=df_plot, x="Model", y="Accuracy", hue="Split", ax=ax)
            ax.set_title("Domain Detection Accuracy by Model and Split", fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_xlabel("Model Configuration", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / "overall_accuracy_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Domain-wise performance heatmap
        domain_data = defaultdict(dict)
        
        for analysis in all_analyses:
            if analysis and analysis["domain_accuracies"]:
                model_key = f"{analysis['model_config']}_{analysis['split']}"
                for domain, accuracy in analysis["domain_accuracies"].items():
                    domain_data[domain][model_key] = accuracy
        
        if domain_data:
            # Create heatmap data
            domains = list(domain_data.keys())
            models = set()
            for domain_dict in domain_data.values():
                models.update(domain_dict.keys())
            models = sorted(list(models))
            
            heatmap_data = []
            for domain in domains:
                row = []
                for model in models:
                    accuracy = domain_data[domain].get(model, 0)
                    row.append(accuracy)
                heatmap_data.append(row)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(heatmap_data, 
                       xticklabels=[m.replace('_', '\n') for m in models],
                       yticklabels=domains,
                       annot=True, 
                       fmt='.1f',
                       cmap='RdYlBu_r',
                       ax=ax,
                       cbar_kws={'label': 'Accuracy (%)'})
            
            ax.set_title("Domain Detection Accuracy Heatmap", fontsize=14, fontweight='bold')
            ax.set_xlabel("Model Configuration", fontsize=12)
            ax.set_ylabel("Medical Domain", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_path / "domain_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Inference time comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        inference_data = []
        for analysis in all_analyses:
            if analysis:
                inference_data.append({
                    "Model": analysis["model_config"],
                    "Split": analysis["split"],
                    "Avg_Time": analysis["avg_inference_time"]
                })
        
        if inference_data:
            df_time = pd.DataFrame(inference_data)
            sns.barplot(data=df_time, x="Model", y="Avg_Time", hue="Split", ax=ax)
            ax.set_title("Average Inference Time by Model", fontsize=14, fontweight='bold')
            ax.set_ylabel("Average Inference Time (seconds)", fontsize=12)
            ax.set_xlabel("Model Configuration", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / "inference_time_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {output_path}")
    
    def run_complete_analysis(self):
        """Run complete analysis of all domain detection results"""
        print("Searching for domain detection results...")
        
        all_results = self.find_all_results()
        if not all_results:
            print("No domain detection results found!")
            return
        
        print(f"Found {len(all_results)} result sets")
        
        # Load and analyze all results
        all_analyses = []
        
        for result_info in all_results:
            print(f"Analyzing: {result_info['model_config']} - {result_info['split']}")
            
            results_data = self.load_results(result_info)
            if results_data:
                analysis = self.analyze_model_performance(results_data)
                if analysis:
                    all_analyses.append(analysis)
        
        # Generate reports and visualizations
        if all_analyses:
            self.create_summary_report(all_analyses)
            
            # Create visualizations if matplotlib is available
            try:
                self.create_visualizations(all_analyses)
            except ImportError:
                print("Matplotlib/Seaborn not available. Skipping visualizations.")
            except Exception as e:
                print(f"Error creating visualizations: {e}")
            
            # Save detailed results to CSV
            self.save_detailed_results(all_analyses)
        
        else:
            print("No valid analysis results generated")
    
    def save_detailed_results(self, all_analyses, output_file="domain_detection_detailed_results.csv"):
        """Save detailed results to CSV"""
        detailed_data = []
        
        for analysis in all_analyses:
            if analysis:
                # Add overall metrics
                base_row = {
                    "model_config": analysis["model_config"],
                    "split": analysis["split"],
                    "total_questions": analysis["total_questions"],
                    "overall_accuracy": analysis["overall_accuracy"],
                    "avg_inference_time": analysis["avg_inference_time"]
                }
                
                # Add domain-specific accuracies
                for domain, accuracy in analysis["domain_accuracies"].items():
                    row = base_row.copy()
                    row["domain"] = domain
                    row["domain_accuracy"] = accuracy
                    row["domain_total"] = analysis["domain_stats"][domain]["total"]
                    row["domain_correct"] = analysis["domain_stats"][domain]["correct"]
                    detailed_data.append(row)
        
        if detailed_data:
            df = pd.DataFrame(detailed_data)
            df.to_csv(output_file, index=False)
            print(f"Detailed results saved to: {output_file}")

def main():
    analyzer = DomainDetectionAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
