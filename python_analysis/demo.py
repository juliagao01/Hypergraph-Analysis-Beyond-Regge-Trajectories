#!/usr/bin/env python3
"""
Demo Script for Regge Trajectory Analysis

This script demonstrates the complete Python analysis pipeline by:
1. Generating sample Δ baryon data
2. Running the full analysis pipeline
3. Displaying results and generating plots

This is useful for testing the pipeline before using real data from Wolfram Language.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the analysis modules to path
sys.path.append(str(Path(__file__).parent))

from regge_analysis.regge_fitter import ReggeFitter
from regge_analysis.uncertainty_propagation import UncertaintyPropagator
from regge_analysis.bootstrap_analysis import BootstrapAnalyzer
from regge_analysis.pdg_crosscheck import PDGCrossChecker
from regge_analysis.statistical_significance import StatisticalSignificanceAnalyzer

def generate_sample_data(n_particles: int = 15) -> pd.DataFrame:
    """
    Generate sample Δ baryon data for demonstration.
    
    Parameters:
    -----------
    n_particles : int
        Number of particles to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample particle data
    """
    print("Generating sample Δ baryon data...")
    
    # True Regge trajectory parameters
    alpha0_true = 0.5
    alphap_true = 0.9  # GeV⁻²
    
    # Generate J values (typical for Δ baryons)
    J_values = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
    J_values = J_values[:n_particles]
    
    # Generate true M² values
    M2_true = (J_values - alpha0_true) / alphap_true
    
    # Add some noise to make it realistic
    np.random.seed(42)  # For reproducibility
    noise_factor = 0.05  # 5% noise
    M2_noisy = M2_true * (1 + np.random.normal(0, noise_factor, len(J_values)))
    
    # Generate uncertainties (proportional to mass)
    uncertainties = M2_noisy * 0.02  # 2% uncertainty
    
    # Generate particle names
    names = [f"Δ({int(np.sqrt(M2)*1000)})++" for M2 in M2_noisy]
    
    # Generate PDG status (mix of confirmed and tentative)
    statuses = ["****"] * 8 + ["***"] * 4 + ["**"] * 3
    
    # Create DataFrame
    data = pd.DataFrame({
        'name': names,
        'J': J_values,
        'parity': 1,  # Positive parity for Δ baryons
        'mass_GeV': np.sqrt(M2_noisy),
        'mass_sigma_GeV': uncertainties / (2 * np.sqrt(M2_noisy)),
        'width_GeV': np.random.uniform(0.1, 0.3, len(J_values)),
        'M2_GeV2': M2_noisy,
        'M2_sigma_GeV2': uncertainties,
        'pdg_status': statuses[:len(J_values)]
    })
    
    print(f"Generated {len(data)} sample particles")
    print(f"J range: {data['J'].min():.1f} - {data['J'].max():.1f}")
    print(f"Mass range: {data['mass_GeV'].min():.3f} - {data['mass_GeV'].max():.3f} GeV")
    
    return data

def run_demo_analysis():
    """Run the complete demo analysis."""
    print("="*60)
    print("REGGЕ TRAJECTORY ANALYSIS DEMO")
    print("="*60)
    
    # Create output directory
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate sample data
    data = generate_sample_data(n_particles=12)
    
    # Save sample data
    data.to_csv(f"{output_dir}/sample_delta_baryons.csv", index=False)
    print(f"Sample data saved to {output_dir}/sample_delta_baryons.csv")
    
    # 2. Fit Regge trajectory
    print("\n2. FITTING REGGE TRAJECTORY")
    print("-" * 30)
    
    fitter = ReggeFitter(data)
    results = fitter.fit_all_methods()
    
    best_fit = fitter.get_best_fit('wls')
    print(f"Best fit (WLS):")
    print(f"  α₀ = {best_fit['alpha0']:.4f} ± {best_fit['alpha0_err']:.4f}")
    print(f"  α' = {best_fit['alphap']:.4f} ± {best_fit['alphap_err']:.4f}")
    print(f"  χ²/dof = {best_fit['chi2_dof']:.3f}")
    print(f"  R² = {best_fit['r_squared']:.4f}")
    
    # 3. Statistical Significance Analysis
    print("\n3. STATISTICAL SIGNIFICANCE ANALYSIS")
    print("-" * 30)
    
    # Run comprehensive statistical analysis
    significance_analyzer = StatisticalSignificanceAnalyzer(data)
    significance_results = significance_analyzer.run_complete_analysis()
    
    print(f"\nStatistical robustness conclusion: {significance_results['conclusion']}")
    
    # 4. Bootstrap analysis (additional)
    print("\n4. ADDITIONAL BOOTSTRAP ANALYSIS")
    print("-" * 30)
    
    bootstrap_analyzer = BootstrapAnalyzer(data)
    bootstrap_results = bootstrap_analyzer.bootstrap_sample(n_bootstrap=100)  # Smaller for demo
    bootstrap_analysis = bootstrap_analyzer.analyze_bootstrap_results(bootstrap_results)
    
    print(f"Bootstrap results ({bootstrap_analysis['n_bootstrap']} samples):")
    print(f"  α₀ = {bootstrap_analysis['alpha0']['mean']:.4f} ± {bootstrap_analysis['alpha0']['std']:.4f}")
    print(f"  α' = {bootstrap_analysis['alphap']['mean']:.4f} ± {bootstrap_analysis['alphap']['std']:.4f}")
    print(f"  Correlation: {bootstrap_analysis['correlation']:.3f}")
    
    # 5. Predict missing J states
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
    
    # 6. PDG cross-checking
    print("\n6. PDG CROSS-CHECKING")
    print("-" * 30)
    
    pdg_crosschecker = PDGCrossChecker(predictions, data)
    cross_check_results = pdg_crosschecker.find_nearby_candidates(window_GeV=0.15)
    gap_analysis = pdg_crosschecker.analyze_gaps(cross_check_results)
    
    print(f"Gap analysis:")
    print(f"  Total predictions: {gap_analysis['total_predictions']}")
    print(f"  True gaps: {gap_analysis['true_gaps']}")
    print(f"  Potential gaps: {gap_analysis['potential_gaps']}")
    print(f"  Well-matched: {gap_analysis['well_matched']}")
    print(f"  Gap fraction: {gap_analysis['gap_fraction']:.2%}")
    
    # 7. Generate plots
    print("\n7. GENERATING PLOTS")
    print("-" * 30)
    
    # Main fit plot
    fig, axes = fitter.plot_fit('wls', save_path=f"{output_dir}/demo_regge_fit.png")
    plt.close(fig)
    
    # Bootstrap distributions
    fig = bootstrap_analyzer.plot_bootstrap_distributions(
        bootstrap_results, 
        save_path=f"{output_dir}/demo_bootstrap_distributions.png"
    )
    plt.close(fig)
    
    # Predictions plot
    fig = propagator.plot_predictions(
        data, predictions, 
        save_path=f"{output_dir}/demo_predictions.png"
    )
    plt.close(fig)
    
    # Cross-check results
    fig = pdg_crosschecker.plot_cross_check_results(
        cross_check_results,
        save_path=f"{output_dir}/demo_cross_check_results.png"
    )
    plt.close(fig)
    
    # Statistical robustness analysis
    fig = significance_analyzer.plot_robustness_analysis(save_path=f"{output_dir}/demo_statistical_robustness.png")
    plt.close(fig)
    
    # 8. Generate reports
    print("\n8. GENERATING REPORTS")
    print("-" * 30)
    
    # Gap analysis report
    gap_report = pdg_crosschecker.generate_gap_report(gap_analysis)
    with open(f"{output_dir}/demo_gap_analysis_report.txt", 'w') as f:
        f.write(gap_report)
    print("Gap analysis report saved")
    
    # Export results
    predictions.to_csv(f"{output_dir}/demo_predictions.csv", index=False)
    cross_check_results.to_csv(f"{output_dir}/demo_cross_check_results.csv", index=False)
    
    # Method comparison
    method_comparison = bootstrap_analyzer.compare_methods(best_fit, bootstrap_results)
    method_comparison.to_csv(f"{output_dir}/demo_method_comparison.csv", index=False)
    
    # Literature comparison
    literature_comparison = pdg_crosschecker.compare_with_literature_ranges(
        best_fit['alphap'], best_fit['alphap_err']
    )
    
    print("\nLiterature comparison:")
    for particle_type, comp in literature_comparison.items():
        status = "✓" if comp['within_range'] else "✗"
        print(f"  {particle_type}: {status} α' = {best_fit['alphap']:.3f} "
              f"(range: {comp['range'][0]:.1f}-{comp['range'][1]:.1f} GeV⁻²)")
    
    # 9. Summary
    print("\n" + "="*60)
    print("DEMO ANALYSIS COMPLETE!")
    print("="*60)
    print(f"All outputs saved to: {output_dir}/")
    print("\nKey files generated:")
    print(f"  - demo_regge_fit.png: Main fit plot")
    print(f"  - demo_bootstrap_distributions.png: Bootstrap analysis")
    print(f"  - demo_predictions.png: Predicted missing states")
    print(f"  - demo_cross_check_results.png: PDG cross-check")
    print(f"  - demo_statistical_robustness.png: Statistical significance analysis")
    print(f"  - demo_gap_analysis_report.txt: Detailed gap analysis")
    print(f"  - demo_predictions.csv: Predicted masses")
    print(f"  - demo_method_comparison.csv: Method comparison")
    
    print("\nNext steps:")
    print("1. Export real data from Wolfram Language using the export guide")
    print("2. Run the main analysis: python regge_analysis/main.py")
    print("3. Compare results with the Wolfram Language implementation")
    
    return {
        'data': data,
        'fitter': fitter,
        'significance_analyzer': significance_analyzer,
        'bootstrap_analyzer': bootstrap_analyzer,
        'propagator': propagator,
        'pdg_crosschecker': pdg_crosschecker,
        'results': results,
        'significance_results': significance_results,
        'predictions': predictions,
        'gap_analysis': gap_analysis
    }

if __name__ == "__main__":
    try:
        results = run_demo_analysis()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
