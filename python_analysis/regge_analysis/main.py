#!/usr/bin/env python3
"""
Main Regge Trajectory Analysis Script

This script demonstrates the complete Python analysis pipeline for
quantitative Regge trajectory analysis, including:

1. Data loading and validation
2. Multiple fitting methods (WLS, ODR, OLS)
3. Bootstrap analysis and uncertainty estimation
4. Prediction of missing J states
5. PDG cross-checking and gap analysis
6. Publication-ready figure generation

Usage:
    python main.py [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from regge_analysis.regge_fitter import ReggeFitter
from regge_analysis.uncertainty_propagation import UncertaintyPropagator
from regge_analysis.bootstrap_analysis import BootstrapAnalyzer
from regge_analysis.pdg_crosscheck import PDGCrossChecker
from regge_analysis.statistical_significance import StatisticalSignificanceAnalyzer
from regge_analysis.theoretical_context import TheoreticalContextAnalyzer
from regge_analysis.validation_analysis import ValidationAnalyzer

def load_data(data_path: str) -> tuple:
    """
    Load and validate particle data.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
        
    Returns:
    --------
    tuple
        (data, metadata) - DataFrame and metadata dict
    """
    print(f"Loading data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Load metadata if available
    metadata_path = data_path.replace('.csv', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Validate data
    required_columns = [
        'name', 'J', 'parity', 'mass_GeV', 'mass_sigma_GeV',
        'M2_GeV2', 'M2_sigma_GeV2', 'pdg_status'
    ]
    
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    print(f"Loaded {len(data)} particles")
    print(f"Data columns: {list(data.columns)}")
    
    return data, metadata

def run_regge_analysis(data: pd.DataFrame, output_dir: str = "figures"):
    """
    Run complete Regge trajectory analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Particle data
    output_dir : str
        Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("REGGЕ TRAJECTORY ANALYSIS")
    print("="*60)
    
    # 1. Fit Regge trajectory using multiple methods
    print("\n1. FITTING REGGE TRAJECTORY")
    print("-" * 30)
    
    fitter = ReggeFitter(data)
    results = fitter.fit_all_methods()
    
    # Get best fit (WLS)
    best_fit = fitter.get_best_fit('wls')
    print(f"\nBest fit (WLS):")
    print(f"  α₀ = {best_fit['alpha0']:.4f} ± {best_fit['alpha0_err']:.4f}")
    print(f"  α' = {best_fit['alphap']:.4f} ± {best_fit['alphap_err']:.4f}")
    print(f"  χ²/dof = {best_fit['chi2_dof']:.3f}")
    print(f"  R² = {best_fit['r_squared']:.4f}")
    
    # 2. Statistical Significance Analysis
    print("\n2. STATISTICAL SIGNIFICANCE ANALYSIS")
    print("-" * 30)
    
    # Run comprehensive statistical analysis
    significance_analyzer = StatisticalSignificanceAnalyzer(data)
    significance_results = significance_analyzer.run_complete_analysis()
    
    # Generate robustness plots
    significance_analyzer.plot_robustness_analysis(save_path=f"{output_dir}/statistical_robustness.png")
    
    print(f"\nStatistical robustness conclusion: {significance_results['conclusion']}")
    
    # 3. Theoretical Context Analysis
    print("\n3. THEORETICAL CONTEXT ANALYSIS")
    print("-" * 30)
    
    # Run theoretical context analysis
    theoretical_analyzer = TheoreticalContextAnalyzer(data)
    
    # Chew-Frautschi expectations
    cf_results = theoretical_analyzer.chew_frautschi_expectations(
        best_fit['alphap'], best_fit['alphap_err'], particle_type='baryon'
    )
    
    print(f"Chew-Frautschi analysis:")
    print(f"  Fitted α' = {cf_results['fitted_alphap']:.4f} ± {cf_results['fitted_alphap_err']:.4f} GeV⁻²")
    print(f"  Expected range: {cf_results['literature_range'][0]:.1f} - {cf_results['literature_range'][1]:.1f} GeV⁻²")
    print(f"  Z-score vs typical: {cf_results['z_score_to_typical']:.2f}")
    print(f"  Significance: {cf_results['significance']}")
    
    # Parity separation analysis
    parity_results = theoretical_analyzer.parity_separation_analysis()
    if parity_results['comparison']:
        comp = parity_results['comparison']
        print(f"Parity separation:")
        print(f"  Δα' = {comp['slope_difference']:.4f} ± {comp['slope_difference_err']:.4f} GeV⁻²")
        print(f"  P-value = {comp['p_value']:.4f}")
        print(f"  Significance: {comp['significance']}")
    
    # Generate theoretical report and plots
    theoretical_report = theoretical_analyzer.generate_theoretical_report(
        best_fit['alphap'], best_fit['alphap_err'], particle_type='baryon'
    )
    with open(f"{output_dir}/theoretical_analysis_report.txt", 'w') as f:
        f.write(theoretical_report)
    
    theoretical_analyzer.plot_theoretical_analysis(
        best_fit['alphap'], best_fit['alphap_err'],
        save_path=f"{output_dir}/theoretical_analysis.png"
    )
    
    # 4. Bootstrap analysis (additional)
    print("\n4. ADDITIONAL BOOTSTRAP ANALYSIS")
    print("-" * 30)
    
    bootstrap_analyzer = BootstrapAnalyzer(data)
    bootstrap_results = bootstrap_analyzer.bootstrap_sample(n_bootstrap=1000)
    bootstrap_analysis = bootstrap_analyzer.analyze_bootstrap_results(bootstrap_results)
    
    print(f"Bootstrap results ({bootstrap_analysis['n_bootstrap']} samples):")
    print(f"  α₀ = {bootstrap_analysis['alpha0']['mean']:.4f} ± {bootstrap_analysis['alpha0']['std']:.4f}")
    print(f"  α' = {bootstrap_analysis['alphap']['mean']:.4f} ± {bootstrap_analysis['alphap']['std']:.4f}")
    print(f"  Correlation: {bootstrap_analysis['correlation']:.3f}")
    
    # 5. Uncertainty propagation and predictions
    print("\n5. PREDICTING MISSING J STATES")
    print("-" * 30)
    
    propagator = UncertaintyPropagator(best_fit)
    existing_J = data['J'].tolist()
    predictions = propagator.predict_missing_states(
        existing_J, 
        J_range=(0.5, 8.0), 
        J_step=0.5
    )
    
    print(f"Predicted {len(predictions)} missing J states:")
    for _, pred in predictions.iterrows():
        print(f"  J = {pred['J']:.1f}: M = {pred['M_GeV']:.3f} ± {pred['M_sigma_GeV']:.3f} GeV")
    
    # 6. Validation Analysis
    print("\n6. VALIDATION ANALYSIS")
    print("-" * 30)
    
    # Run validation analysis
    validation_analyzer = ValidationAnalyzer(data)
    
    # Get residuals from the fit
    x_data = data[data.columns[0]].values  # Assuming first column is M²
    y_data = data[data.columns[1]].values  # Assuming second column is J
    fitted_values = best_fit['alpha0'] + best_fit['alphap'] * x_data
    residuals = y_data - fitted_values
    
    # PDG cross-check near predictions
    crosscheck_results = validation_analyzer.pdg_crosscheck_predictions(predictions)
    n_confirmations = len(crosscheck_results[crosscheck_results['match_type'] == 'confirmation'])
    n_gaps = len(crosscheck_results[crosscheck_results['match_type'] == 'genuine_gap'])
    
    print(f"PDG cross-check results:")
    print(f"  Confirmations: {n_confirmations}")
    print(f"  Genuine gaps: {n_gaps}")
    
    # Residual quality analysis
    quality_results = validation_analyzer.residual_experimental_quality_analysis(residuals, fitted_values)
    if quality_results['correlations']:
        significant_corr = sum(1 for r in quality_results['correlations'].values() if r['significant'])
        print(f"Residual quality: {significant_corr} significant correlations with experimental quality")
    
    # External theory comparison
    theory_results = validation_analyzer.external_theory_overlay(
        best_fit['alpha0'], best_fit['alphap']
    )
    print(f"Theory agreement: {theory_results['agreement']} (Z-score: {theory_results['z_score']:.2f})")
    
    # Generate validation report and plots
    validation_report = validation_analyzer.generate_validation_report(
        predictions, residuals, fitted_values, best_fit['alpha0'], best_fit['alphap']
    )
    with open(f"{output_dir}/validation_analysis_report.txt", 'w') as f:
        f.write(validation_report)
    
    validation_analyzer.plot_validation_analysis(
        predictions, residuals, fitted_values, best_fit['alpha0'], best_fit['alphap'],
        save_path=f"{output_dir}/validation_analysis.png"
    )
    
    # 7. PDG cross-checking
    print("\n7. PDG CROSS-CHECKING")
    print("-" * 30)
    
    # Use the same data as PDG reference (in practice, you'd use full PDG)
    pdg_crosschecker = PDGCrossChecker(predictions, data)
    cross_check_results = pdg_crosschecker.find_nearby_candidates(window_GeV=0.15)
    gap_analysis = pdg_crosschecker.analyze_gaps(cross_check_results)
    
    print(f"Gap analysis:")
    print(f"  Total predictions: {gap_analysis['total_predictions']}")
    print(f"  True gaps: {gap_analysis['true_gaps']}")
    print(f"  Potential gaps: {gap_analysis['potential_gaps']}")
    print(f"  Well-matched: {gap_analysis['well_matched']}")
    print(f"  Gap fraction: {gap_analysis['gap_fraction']:.2%}")
    
    # 8. Generate plots
    print("\n8. GENERATING PLOTS")
    print("-" * 30)
    
    # Main fit plot
    fig, axes = fitter.plot_fit('wls', save_path=f"{output_dir}/regge_fit.png")
    
    # Bootstrap distributions
    bootstrap_analyzer.plot_bootstrap_distributions(
        bootstrap_results, 
        save_path=f"{output_dir}/bootstrap_distributions.png"
    )
    
    # Predictions plot
    propagator.plot_predictions(
        data, predictions, 
        save_path=f"{output_dir}/predictions.png"
    )
    
    # Cross-check results
    pdg_crosschecker.plot_cross_check_results(
        cross_check_results,
        save_path=f"{output_dir}/cross_check_results.png"
    )
    
    # 9. Generate reports
    print("\n9. GENERATING REPORTS")
    print("-" * 30)
    
    # Gap analysis report
    gap_report = pdg_crosschecker.generate_gap_report(gap_analysis)
    with open(f"{output_dir}/gap_analysis_report.txt", 'w') as f:
        f.write(gap_report)
    print(gap_report)
    
    # Export results
    predictions.to_csv(f"{output_dir}/predictions.csv", index=False)
    cross_check_results.to_csv(f"{output_dir}/cross_check_results.csv", index=False)
    
    # Method comparison
    method_comparison = bootstrap_analyzer.compare_methods(best_fit, bootstrap_results)
    method_comparison.to_csv(f"{output_dir}/method_comparison.csv", index=False)
    
    # Literature comparison
    literature_comparison = pdg_crosschecker.compare_with_literature_ranges(
        best_fit['alphap'], best_fit['alphap_err']
    )
    
    print("\nLiterature comparison:")
    for particle_type, comp in literature_comparison.items():
        status = "✓" if comp['within_range'] else "✗"
        print(f"  {particle_type}: {status} α' = {best_fit['alphap']:.3f} "
              f"(range: {comp['range'][0]:.1f}-{comp['range'][1]:.1f} GeV⁻²)")
    
    # 10. Summary statistics
    print("\n10. SUMMARY STATISTICS")
    print("-" * 30)
    
    summary = {
        'n_data_points': len(data),
        'fit_method': 'Weighted Least Squares',
        'alpha0': best_fit['alpha0'],
        'alpha0_err': best_fit['alpha0_err'],
        'alphap': best_fit['alphap'],
        'alphap_err': best_fit['alphap_err'],
        'chi2_dof': best_fit['chi2_dof'],
        'r_squared': best_fit['r_squared'],
        'n_predictions': len(predictions),
        'n_true_gaps': gap_analysis['true_gaps'],
        'gap_fraction': gap_analysis['gap_fraction'],
        'bootstrap_n_samples': bootstrap_analysis['n_bootstrap']
    }
    
    with open(f"{output_dir}/summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Summary statistics saved to summary_statistics.json")
    print(f"All outputs saved to {output_dir}/")
    
    return {
        'fitter': fitter,
        'significance_analyzer': significance_analyzer,
        'theoretical_analyzer': theoretical_analyzer,
        'validation_analyzer': validation_analyzer,
        'bootstrap_analyzer': bootstrap_analyzer,
        'propagator': propagator,
        'pdg_crosschecker': pdg_crosschecker,
        'results': results,
        'significance_results': significance_results,
        'cf_results': cf_results,
        'parity_results': parity_results,
        'validation_results': {
            'crosscheck_results': crosscheck_results,
            'quality_results': quality_results,
            'theory_results': theory_results
        },
        'predictions': predictions,
        'gap_analysis': gap_analysis,
        'summary': summary
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Regge Trajectory Analysis")
    parser.add_argument(
        "--data-path", 
        default="data_export/delta_baryons.csv",
        help="Path to particle data CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        default="figures",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found.")
        print("Please export data from Wolfram Language first using the export guide.")
        print("Run: python data_export/wl_export_guide.py")
        return 1
    
    try:
        # Load data
        data, metadata = load_data(args.data_path)
        
        # Run analysis
        results = run_regge_analysis(data, args.output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved to: {args.output_dir}/")
        print("\nKey files generated:")
        print(f"  - regge_fit.png: Main fit plot")
        print(f"  - bootstrap_distributions.png: Bootstrap analysis")
        print(f"  - predictions.png: Predicted missing states")
        print(f"  - statistical_robustness.png: Statistical significance analysis")
        print(f"  - theoretical_analysis.png: Theoretical context analysis")
        print(f"  - validation_analysis.png: Validation analysis")
        print(f"  - cross_check_results.png: PDG cross-check")
        print(f"  - gap_analysis_report.txt: Detailed gap analysis")
        print(f"  - theoretical_analysis_report.txt: Theoretical context report")
        print(f"  - validation_analysis_report.txt: Validation analysis report")
        print(f"  - predictions.csv: Predicted masses")
        print(f"  - summary_statistics.json: Summary statistics")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
