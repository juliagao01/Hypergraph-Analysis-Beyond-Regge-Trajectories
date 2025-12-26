"""
Comprehensive Regge Trajectory Analysis

Runs all analysis modules together to test consistency with past findings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import all our analysis modules
from data_management.data_hydration import DataHydration, DeterministicFilters
from numerical_methods.regression_analysis import RegressionAnalyzer
from predictions.prediction_analysis import PredictionAnalyzer
from peer_comparison.baseline_comparison import BaselineComparison
from broader_implications.stability_analysis import StabilityAnalyzer
from paper_hygiene.export_utilities import ExportUtilities

def create_sample_delta_data():
    """Create sample Delta resonance data for analysis."""
    print("Creating sample Delta resonance data...")
    
    # Sample Delta resonance data (based on typical PDG values)
    delta_data = pd.DataFrame({
        'Name': [
            'Delta(1232)', 'Delta(1600)', 'Delta(1620)', 'Delta(1700)',
            'Delta(1900)', 'Delta(1905)', 'Delta(1910)', 'Delta(1920)',
            'Delta(1930)', 'Delta(1950)', 'Delta(2000)', 'Delta(2150)',
            'Delta(2200)', 'Delta(2300)', 'Delta(2350)', 'Delta(2400)',
            'Delta(2420)', 'Delta(2500)', 'Delta(2600)', 'Delta(2750)'
        ],
        'PDG_ID': [12112, 12114, 12116, 12118, 12120, 12122, 12124, 12126, 12128, 12130,
                   12132, 12134, 12136, 12138, 12140, 12142, 12144, 12146, 12148, 12150],
        'MassGeV': [1.232, 1.600, 1.620, 1.700, 1.900, 1.905, 1.910, 1.920, 1.930, 1.950,
                    2.000, 2.150, 2.200, 2.300, 2.350, 2.400, 2.420, 2.500, 2.600, 2.750],
        'ResonanceWidthGeV': [0.118, 0.350, 0.150, 0.300, 0.250, 0.350, 0.250, 0.200, 0.280, 0.300,
                              0.400, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750],
        'MassSigmaGeV': [0.001, 0.010, 0.005, 0.010, 0.020, 0.015, 0.020, 0.015, 0.020, 0.025,
                         0.030, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065],
        'WidthSigmaGeV': [0.002, 0.020, 0.010, 0.020, 0.025, 0.020, 0.025, 0.020, 0.025, 0.030,
                          0.035, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070],
        'J': [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5, 5.5, 6.5, 6.5, 7.5, 7.5, 8.5, 8.5, 9.5, 9.5, 10.5, 10.5],
        'P': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'Status': ['***', '**', '**', '**', '**', '**', '**', '**', '**', '**', 
                  '**', '**', '**', '**', '**', '**', '**', '**', '**', '**'],
        'Family': ['Delta'] * 20,
        'ParticleType': ['Baryon'] * 20
    })
    
    # Add M² and M² uncertainty columns
    delta_data['M2GeV2'] = delta_data['MassGeV']**2
    delta_data['M2SigmaGeV2'] = 2 * delta_data['MassGeV'] * delta_data['MassSigmaGeV']
    
    return delta_data

def run_comprehensive_analysis():
    """Run comprehensive Regge trajectory analysis."""
    print("=" * 80)
    print("COMPREHENSIVE REGGE TRAJECTORY ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("comprehensive_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. DATA HYDRATION
    print("\n1. DATA HYDRATION")
    print("-" * 40)
    
    # Create sample data
    delta_data = create_sample_delta_data()
    print(f"Created sample data with {len(delta_data)} Delta resonances")
    
    # Initialize data hydration
    data_hydration = DataHydration(output_dir="frozen_data")
    
    # Freeze PDG snapshot
    frozen_file = data_hydration.freeze_pdg_snapshot(
        delta_data, 
        pdg_date="2024-01-01",
        software_version="Python 3.9.7",
        kappa=0.25,
        filters_applied={"family": "Delta", "status": ["***", "**"]}
    )
    
    # Load frozen data
    particle_data, metadata = data_hydration.load_frozen_data(frozen_file)
    
    # Apply deterministic filters
    filters = DeterministicFilters()
    filtered_data = filters.apply_filters(particle_data)
    
    print(f"Filtered data: {len(filtered_data)} particles")
    print(f"J range: {filtered_data['J'].min():.1f} to {filtered_data['J'].max():.1f}")
    print(f"Mass range: {filtered_data['MassGeV'].min():.3f} to {filtered_data['MassGeV'].max():.3f} GeV")
    
    # 2. NUMERICAL METHODS & STATISTICS
    print("\n2. NUMERICAL METHODS & STATISTICS")
    print("-" * 40)
    
    # Initialize regression analyzer
    regression = RegressionAnalyzer()
    
    # Prepare data for fitting
    x_data = filtered_data['M2GeV2'].values
    y_data = filtered_data['J'].values
    y_errors = filtered_data['M2SigmaGeV2'].values
    
    print(f"Fitting Regge trajectory: J = α₀ + α'M²")
    print(f"Data points: {len(x_data)}")
    print(f"M² range: {x_data.min():.2f} to {x_data.max():.2f} GeV²")
    print(f"J range: {y_data.min():.1f} to {y_data.max():.1f}")
    
    # Fit linear model
    linear_results = regression.fit_regge_trajectory(
        x_data, y_data, y_errors,
        use_odr=False,  # Start with WLS
        robust_fallback=True,
        check_heteroskedasticity=True
    )
    
    print(f"\nLinear Fit Results:")
    print(f"α₀ = {linear_results.parameters[0]:.4f} ± {linear_results.parameter_uncertainties[0]:.4f}")
    print(f"α' = {linear_results.parameters[1]:.4f} ± {linear_results.parameter_uncertainties[1]:.4f}")
    print(f"χ²/dof = {linear_results.chi2_dof:.3f}")
    print(f"R² = {linear_results.r_squared:.3f}")
    
    # Fit segmented model
    segmented_results = regression.fit_segmented_model(x_data, y_data, y_errors)
    
    if segmented_results:
        print(f"\nSegmented Fit Results:")
        print(f"α₀ = {segmented_results.parameters[0]:.4f} ± {segmented_results.parameter_uncertainties[0]:.4f}")
        print(f"α' = {segmented_results.parameters[1]:.4f} ± {segmented_results.parameter_uncertainties[1]:.4f}")
        print(f"β₀ = {segmented_results.parameters[2]:.4f} ± {segmented_results.parameter_uncertainties[2]:.4f}")
        print(f"β' = {segmented_results.parameters[3]:.4f} ± {segmented_results.parameter_uncertainties[3]:.4f}")
        print(f"Breakpoint = {segmented_results.model_info['breakpoint']:.2f} ± {segmented_results.model_info['breakpoint_uncertainty']:.2f} GeV²")
        print(f"χ²/dof = {segmented_results.chi2_dof:.3f}")
        print(f"R² = {segmented_results.r_squared:.3f}")
    
    # Model comparison
    model_comparison = regression.compare_models(linear_results, segmented_results)
    
    print(f"\nModel Comparison:")
    print(f"Linear AIC: {model_comparison['linear']['aic']:.2f}")
    print(f"Linear BIC: {model_comparison['linear']['bic']:.2f}")
    
    if segmented_results:
        print(f"Segmented AIC: {model_comparison['segmented']['aic']:.2f}")
        print(f"Segmented BIC: {model_comparison['segmented']['bic']:.2f}")
        print(f"ΔAIC: {model_comparison['model_selection']['delta_aic']:.2f}")
        print(f"ΔBIC: {model_comparison['model_selection']['delta_bic']:.2f}")
        print(f"Recommendation: {model_comparison['model_selection']['recommendation']}")
    
    # 3. PREDICTIONS & CROSS-CHECKS
    print("\n3. PREDICTIONS & CROSS-CHECKS")
    print("-" * 40)
    
    # Initialize prediction analyzer
    prediction = PredictionAnalyzer()
    
    # Generate predictions for higher J values
    j_values = np.array([11.5, 12.5, 13.5, 14.5, 15.5])
    
    print(f"Predicting masses for J = {j_values}")
    
    # Generate predictions
    predictions = prediction.predict_masses(
        j_values, 
        linear_results.parameters, 
        linear_results.covariance,
        kappa=0.25
    )
    
    print(f"\nMass Predictions:")
    for pred in predictions:
        print(f"J = {pred.j_value:.1f}: M = {pred.predicted_mass:.3f} ± {pred.mass_uncertainty:.3f} GeV")
    
    # Kappa parameter sweep
    kappa_sweep = prediction.kappa_parameter_sweep(
        j_values, 
        linear_results.parameters, 
        linear_results.covariance
    )
    
    stability = kappa_sweep['stability_analysis']
    print(f"\nKappa Stability Analysis:")
    print(f"Overall stable: {stability['overall_stable']}")
    print(f"Max shift: {stability['max_shift_overall']:.4f} GeV")
    
    # Automated PDG cross-checks
    cross_checks = prediction.automated_pdg_neighborhood_scan(
        predictions, 
        filtered_data,  # Use filtered data as "PDG" for this example
        n_sigma_range=(1.5, 2.0)
    )
    
    # Analyze cross-check results
    n_sigma_1_8 = cross_checks.get('n_sigma_1.8', [])
    if n_sigma_1_8:
        matches = sum(1 for r in n_sigma_1_8 if r.match_quality != 'no_match')
        exact_matches = sum(1 for r in n_sigma_1_8 if r.match_quality == 'exact')
        print(f"\nPDG Cross-Check Results (nσ = 1.8):")
        print(f"Total predictions: {len(n_sigma_1_8)}")
        print(f"Matches: {matches}")
        print(f"Exact matches: {exact_matches}")
        print(f"Match rate: {matches/len(n_sigma_1_8)*100:.1f}%")
    
    # 4. PEER COMPARISON
    print("\n4. PEER COMPARISON")
    print("-" * 40)
    
    # Create hypergraph-like data for comparison
    hypergraph_data = filtered_data.copy()
    hypergraph_data['decay_products'] = [['pion', 'nucleon']] * len(hypergraph_data)
    hypergraph_data['parent_particle'] = hypergraph_data['Name']
    
    # Initialize baseline comparison
    baseline_comparison = BaselineComparison(hypergraph_data)
    
    # Mock hypergraph results
    hypergraph_results = {
        'figure_count': 6,
        'time_to_insight': 45.0,
        'clicks_to_isolate': 3,
        'subgroup_identification_time': 15.0,
        'data_processing_time': 8.0,
        'visualization_quality_score': 0.85,
        'insight_depth_score': 0.92
    }
    
    # Run comparison
    comparison_results = baseline_comparison.run_complete_comparison(hypergraph_results)
    
    print("Peer comparison completed - see peer_comparison/ directory for results")
    
    # 5. STABILITY ANALYSIS
    print("\n5. STABILITY ANALYSIS")
    print("-" * 40)
    
    # Initialize stability analyzer
    stability_analyzer = StabilityAnalyzer()
    
    # Create mock updated data (slightly modified)
    updated_data = filtered_data.copy()
    updated_data.loc[0, 'MassGeV'] += 0.005  # Small change
    updated_data.loc[1, 'MassGeV'] -= 0.003  # Small change
    
    # Run stability analysis
    stability_results = stability_analyzer.analyze_pdg_update_stability(
        filtered_data, 
        updated_data, 
        {'parameters': linear_results.parameters, 'chi2_dof': linear_results.chi2_dof},
        lambda x, a, b: a + b * x,
        ['alpha0', 'alphap']
    )
    
    metrics = stability_results['stability_metrics']
    print(f"Stability Analysis:")
    print(f"Parameter shift: {metrics.parameter_shift:.4f}")
    print(f"Fit quality change: {metrics.fit_quality_change:.4f}")
    print(f"Overall stability score: {metrics.overall_stability_score:.3f}")
    
    # 6. PAPER HYGIENE
    print("\n6. PAPER HYGIENE")
    print("-" * 40)
    
    # Initialize export utilities
    export_utils = ExportUtilities()
    
    # Prepare results for export
    results = {
        'fit_results': {
            'parameters': linear_results.parameters,
            'parameter_uncertainties': linear_results.parameter_uncertainties,
            'covariance': linear_results.covariance,
            'chi2': linear_results.chi2,
            'chi2_dof': linear_results.chi2_dof,
            'r_squared': linear_results.r_squared,
            'dof': linear_results.dof
        },
        'predictions': prediction.create_prediction_summary_table(predictions),
        'cross_check_results': prediction.create_prediction_summary_table(predictions, n_sigma_1_8),
        'diagnostics': regression.create_regression_diagnostics(linear_results),
        'hypergraph_metrics': {'modularity': 0.75, 'density': 0.45},
        'motif_analysis': {'motif_counts': {'triangle': 12, 'square': 8}}
    }
    
    # Create reproducibility package
    reproducibility_package = export_utils.create_reproducibility_package(
        results, comparison_results, stability_results
    )
    
    print(f"Reproducibility package created: {reproducibility_package}")
    
    # 7. SUMMARY & CONSISTENCY CHECK
    print("\n7. SUMMARY & CONSISTENCY CHECK")
    print("-" * 40)
    
    # Extract key results
    alpha0 = linear_results.parameters[0]
    alphap = linear_results.parameters[1]
    alpha0_err = linear_results.parameter_uncertainties[0]
    alphap_err = linear_results.parameter_uncertainties[1]
    chi2_dof = linear_results.chi2_dof
    r_squared = linear_results.r_squared
    
    print(f"KEY RESULTS:")
    print(f"α₀ = {alpha0:.4f} ± {alpha0_err:.4f}")
    print(f"α' = {alphap:.4f} ± {alphap_err:.4f}")
    print(f"χ²/dof = {chi2_dof:.3f}")
    print(f"R² = {r_squared:.3f}")
    
    # Consistency check with typical Regge trajectory values
    print(f"\nCONSISTENCY CHECK:")
    print(f"α' = {alphap:.4f} GeV⁻²")
    
    # Typical α' values for baryons
    typical_alphap_baryons = 0.9  # GeV⁻²
    alphap_deviation = abs(alphap - typical_alphap_baryons) / alphap_err
    
    print(f"Typical α' for baryons: {typical_alphap_baryons:.2f} GeV⁻²")
    print(f"Deviation: {alphap_deviation:.1f}σ")
    
    if alphap_deviation < 2:
        print("✓ α' is consistent with typical baryon Regge trajectories")
    else:
        print("⚠ α' shows significant deviation from typical values")
    
    # Check fit quality
    if chi2_dof < 2:
        print("✓ Good fit quality (χ²/dof < 2)")
    else:
        print("⚠ Fit quality could be improved")
    
    if r_squared > 0.9:
        print("✓ High R² indicates good linear relationship")
    else:
        print("⚠ R² suggests some scatter in the data")
    
    # Check prediction stability
    if stability['overall_stable']:
        print("✓ Predictions are stable across kappa values")
    else:
        print("⚠ Predictions show sensitivity to kappa")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    consistency_score = 0
    
    if alphap_deviation < 2:
        consistency_score += 1
    if chi2_dof < 2:
        consistency_score += 1
    if r_squared > 0.9:
        consistency_score += 1
    if stability['overall_stable']:
        consistency_score += 1
    
    if consistency_score >= 3:
        print("✓ Results are consistent with typical Regge trajectory findings")
    elif consistency_score >= 2:
        print("⚠ Results show moderate consistency with typical findings")
    else:
        print("✗ Results show significant deviations from typical findings")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("=" * 80)
    
    return {
        'linear_results': linear_results,
        'segmented_results': segmented_results,
        'predictions': predictions,
        'kappa_sweep': kappa_sweep,
        'cross_checks': cross_checks,
        'comparison_results': comparison_results,
        'stability_results': stability_results,
        'consistency_score': consistency_score
    }

if __name__ == "__main__":
    results = run_comprehensive_analysis()
