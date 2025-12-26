"""
Run Unified Regge-Hypergraph Analysis

Demonstrates the complete unified framework that integrates hypergraph and 
Regge trajectory analysis to answer the single research question.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our analysis modules
from data_management.data_hydration import DataHydration, DeterministicFilters
from numerical_methods.regression_analysis import RegressionAnalyzer
from predictions.prediction_analysis import PredictionAnalyzer
from unified_analysis.unified_regge_hypergraph import UnifiedAnalyzer
from mathematica_integration.mathematica_export import MathematicaExporter

def create_sample_data():
    """Create sample data for unified analysis demonstration."""
    print("Creating sample data for unified analysis...")
    
    # Sample Delta resonance data
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

def run_unified_analysis():
    """Run the complete unified analysis framework."""
    print("=" * 80)
    print("UNIFIED REGGE-HYPERGRAPH ANALYSIS")
    print("=" * 80)
    print()
    print("Research Question:")
    print("Do structural patterns in hadronic decay (captured by hypergraph features)")
    print("help explain deviations from linear Regge trajectories and highlight")
    print("misclassified or missing states?")
    print()
    
    # Create output directory
    output_dir = Path("unified_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. DATA PREPARATION
    print("1. DATA PREPARATION")
    print("-" * 40)
    
    # Create sample data
    particle_data = create_sample_data()
    print(f"Created sample data with {len(particle_data)} Delta resonances")
    
    # Initialize unified analyzer
    unified_analyzer = UnifiedAnalyzer(output_dir="unified_analysis_results")
    
    # Create unified data model
    states_df = unified_analyzer.create_unified_data_model(particle_data)
    print(f"Unified data model created with {len(states_df)} states")
    
    # 2. HYPERGRAPH FEATURE EXTRACTION
    print("\n2. HYPERGRAPH FEATURE EXTRACTION")
    print("-" * 40)
    
    # Compute hypergraph features
    # In practice, this would use actual hypergraph analysis
    decay_data = particle_data.copy()  # Mock decay data
    states_df = unified_analyzer.compute_hypergraph_features(decay_data)
    
    print("Hypergraph features computed:")
    print("- Community detection and purity")
    print("- Motif z-scores (triangle, star, hyperedge)")
    print("- Product entropy")
    print("- Network metrics (degree, clustering, assortativity)")
    
    # 3. REGGE TRAJECTORY ANALYSIS
    print("\n3. REGGE TRAJECTORY ANALYSIS")
    print("-" * 40)
    
    # Initialize regression analyzer
    regression = RegressionAnalyzer()
    
    # Prepare data for fitting
    x_data = states_df['m2_gev2'].values
    y_data = states_df['j'].values
    y_errors = states_df['m2_sigma_gev2'].values
    
    print(f"Fitting Regge trajectory: J = α₀ + α'M²")
    print(f"Data points: {len(x_data)}")
    
    # Fit linear model
    linear_results = regression.fit_regge_trajectory(
        x_data, y_data, y_errors,
        use_odr=False,
        robust_fallback=True,
        check_heteroskedasticity=True
    )
    
    print(f"\nLinear Fit Results:")
    print(f"α₀ = {linear_results.parameters[0]:.4f} ± {linear_results.parameter_uncertainties[0]:.4f}")
    print(f"α' = {linear_results.parameters[1]:.4f} ± {linear_results.parameter_uncertainties[1]:.4f}")
    print(f"χ²/dof = {linear_results.chi2_dof:.3f}")
    print(f"R² = {linear_results.r_squared:.3f}")
    
    # Compute Regge diagnostics
    fit_results = {
        'parameters': linear_results.parameters,
        'parameter_uncertainties': linear_results.parameter_uncertainties,
        'covariance': linear_results.covariance,
        'chi2_dof': linear_results.chi2_dof,
        'r_squared': linear_results.r_squared
    }
    
    states_df = unified_analyzer.compute_regge_diagnostics(fit_results)
    print("Regge diagnostics computed (residuals, leverage, influence)")
    
    # 4. BRIDGING ANALYSIS
    print("\n4. BRIDGING ANALYSIS")
    print("-" * 40)
    print("Testing links from hypergraph features → Regge residuals")
    
    # Perform bridging analysis
    bridging_results = unified_analyzer.bridging_analysis()
    
    # Display key correlations
    correlations = bridging_results['correlations']
    print(f"\nKey Correlations with |Residual|:")
    print(f"Width: r = {correlations['width_vs_residual']['correlation']:.3f} (p = {correlations['width_vs_residual']['p_value']:.3f})")
    print(f"Product Entropy: r = {correlations['entropy_vs_residual']['correlation']:.3f} (p = {correlations['entropy_vs_residual']['p_value']:.3f})")
    print(f"Community Purity: r = {correlations['purity_vs_residual']['correlation']:.3f} (p = {correlations['purity_vs_residual']['p_value']:.3f})")
    print(f"Triangle Motif: r = {correlations['triangle_vs_residual']['correlation']:.3f} (p = {correlations['triangle_vs_residual']['p_value']:.3f})")
    
    # Display regression results
    regression_results = bridging_results['regression']
    print(f"\nMultivariate Regression:")
    print(f"R² = {regression_results['r2']:.3f}")
    print(f"Cross-validation R² = {regression_results['cv_r2_mean']:.3f} ± {regression_results['cv_r2_std']:.3f}")
    
    # Display predictive utility
    predictive_utility = bridging_results['predictive_utility']
    print(f"\nPredictive Utility:")
    print(f"Baseline R² (width only): {predictive_utility['baseline_r2']:.3f}")
    print(f"Full model R²: {predictive_utility['full_model_r2']:.3f}")
    print(f"Improvement: {predictive_utility['improvement']:.3f}")
    
    # 5. HYPOTHESIS TESTING
    print("\n5. HYPOTHESIS TESTING")
    print("-" * 40)
    print("Testing pre-registered directional hypotheses:")
    
    hypothesis_results = unified_analyzer.test_pre_registered_hypotheses()
    
    # Display hypothesis results
    for hypothesis, results in hypothesis_results.items():
        status = "✓ SUPPORTED" if results['supported'] else "✗ NOT SUPPORTED"
        print(f"\n{hypothesis}: {status}")
        
        if hypothesis == 'H1_quality_control':
            print(f"  Width correlation: {results['width_correlation']:.3f} (p = {results['width_p_value']:.3f})")
            print(f"  Status correlation: {results['status_correlation']:.3f} (p = {results['status_p_value']:.3f})")
        elif hypothesis == 'H2_structure_deviation':
            print(f"  Entropy correlation: {results['entropy_correlation']:.3f} (p = {results['entropy_p_value']:.3f})")
            print(f"  Purity correlation: {results['purity_correlation']:.3f} (p = {results['purity_p_value']:.3f})")
        elif hypothesis == 'H3_motifs':
            print(f"  Triangle correlation: {results['triangle_correlation']:.3f} (p = {results['triangle_p_value']:.3f})")
        elif hypothesis == 'H4_predictive_gain':
            print(f"  Absolute improvement: {results['absolute_improvement']:.3f}")
            print(f"  Relative improvement: {results['relative_improvement']:.1%}")
    
    # 6. PREDICTIONS WITH HYPERGRAPH CONTEXT
    print("\n6. PREDICTIONS WITH HYPERGRAPH CONTEXT")
    print("-" * 40)
    
    # Generate predictions for higher J values
    j_values = [11.5, 12.5, 13.5, 14.5, 15.5]
    predictions = unified_analyzer.generate_predictions_with_hypergraph_context(fit_results, j_values)
    
    print("Predictions with community-informed confidence:")
    for _, pred in predictions.iterrows():
        print(f"J = {pred['J']:.1f}: M = {pred['predicted_mass_gev']:.3f} ± {pred['mass_window_gev']:.3f} GeV")
        print(f"  Community: {pred['nearest_community']}, Purity: {pred['community_purity']:.2f}")
        print(f"  Confidence: {pred['confidence']}, Notes: {pred['notes']}")
    
    # 7. VISUALIZATIONS
    print("\n7. UNIFIED VISUALIZATIONS")
    print("-" * 40)
    
    # Create unified visualizations
    fig_files = unified_analyzer.create_unified_visualizations()
    
    print("Generated unified visualizations:")
    for fig_name, fig_path in fig_files.items():
        print(f"  {fig_name}: {fig_path}")
    
    # 8. EXPORT RESULTS
    print("\n8. EXPORT RESULTS")
    print("-" * 40)
    
    # Export all results
    export_results = unified_analyzer.export_unified_results()
    
    print("Exported unified results:")
    for result_type, result_path in export_results.items():
        if result_type != 'figures':
            print(f"  {result_type}: {result_path}")
    
    # 8b. EXPORT TO MATHEMATICA FOR PLOTTING
    print("\n8b. EXPORT TO MATHEMATICA FOR PLOTTING")
    print("-" * 40)
    
    # Initialize Mathematica exporter
    mathematica_exporter = MathematicaExporter(output_dir="mathematica_exports")
    
    # Export all results to Mathematica
    mathematica_exports = mathematica_exporter.export_all_results(
        unified_analyzer=unified_analyzer,
        hypothesis_results=hypothesis_results,
        bridging_results=bridging_results,
        predictions_df=predictions
    )
    
    print("Exported to Mathematica for plotting:")
    for export_type, export_path in mathematica_exports.items():
        print(f"  {export_type}: {export_path}")
    
    print("\nMathematica Integration Complete!")
    print("To generate publication-quality plots:")
    print("1. Open 'mathematica_exports/unified_analysis_plots.nb' in Mathematica")
    print("2. Run the notebook to generate high-resolution PDF plots")
    print("3. All plots will be automatically exported with 300 DPI resolution")
    
    # 9. SUMMARY AND INTERPRETATION
    print("\n9. SUMMARY AND INTERPRETATION")
    print("-" * 40)
    
    # Count supported hypotheses
    supported_count = sum(1 for results in hypothesis_results.values() if results['supported'])
    total_hypotheses = len(hypothesis_results)
    
    print(f"Hypothesis Support: {supported_count}/{total_hypotheses} hypotheses supported")
    
    # Key findings
    print(f"\nKey Findings:")
    
    # H1: Quality control
    h1_supported = hypothesis_results['H1_quality_control']['supported']
    if h1_supported:
        print("✓ H1: Quality control effects confirmed (width and status correlate with residuals)")
    else:
        print("✗ H1: Quality control effects not confirmed")
    
    # H2: Structure → deviation
    h2_supported = hypothesis_results['H2_structure_deviation']['supported']
    if h2_supported:
        print("✓ H2: Structural patterns explain Regge deviations")
        print("   - Higher product entropy → larger residuals")
        print("   - Lower community purity → larger residuals")
    else:
        print("✗ H2: Structural patterns do not significantly explain deviations")
    
    # H3: Motifs
    h3_supported = hypothesis_results['H3_motifs']['supported']
    if h3_supported:
        print("✓ H3: Specific motifs associate with systematic effects")
    else:
        print("✗ H3: Motif effects not significant")
    
    # H4: Predictive gain
    h4_supported = hypothesis_results['H4_predictive_gain']['supported']
    if h4_supported:
        improvement = hypothesis_results['H4_predictive_gain']['absolute_improvement']
        print(f"✓ H4: Hypergraph features improve prediction by {improvement:.3f} R²")
    else:
        print("✗ H4: No significant predictive improvement from hypergraph features")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    if supported_count >= 3:
        print("✓ Strong evidence that structural patterns in hadronic decay")
        print("  help explain deviations from linear Regge trajectories")
    elif supported_count >= 2:
        print("⚠ Moderate evidence for structural pattern effects")
    else:
        print("✗ Limited evidence for structural pattern effects")
    
    # Theoretical implications
    print(f"\nTheoretical Implications:")
    if h2_supported:
        print("- Decay structure reflects underlying quark dynamics")
        print("- Incoherent decay patterns signal configuration mixing")
        print("- Community purity may indicate dominant decay channels")
    
    if h3_supported:
        print("- Specific motifs reveal coupling structure")
        print("- Triangle motifs may indicate three-body interactions")
    
    if h4_supported:
        print("- Hypergraph features provide predictive power")
        print("- Structural context improves mass predictions")
    
    # Missing states and classification
    print(f"\nImplications for Missing States and Classification:")
    print("- High-purity communities: high-confidence predictions")
    print("- Low-purity communities: predictions need caution")
    print("- Outliers with coherent structure: potential misclassifications")
    print("- Outliers with incoherent structure: possible missing states")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("=" * 80)
    
    return {
        'unified_analyzer': unified_analyzer,
        'hypothesis_results': hypothesis_results,
        'bridging_results': bridging_results,
        'predictions': predictions,
        'export_results': export_results,
        'mathematica_exports': mathematica_exports
    }

if __name__ == "__main__":
    results = run_unified_analysis()
