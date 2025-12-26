"""
Error Analysis Integration for Regge Trajectories

Integrates systematic uncertainty analysis with the existing Regge analysis pipeline,
providing comprehensive error quantification and uncertainty propagation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

from .systematic_uncertainty_analyzer import SystematicUncertaintyAnalyzer, UncertaintyConfig

class ErrorAnalysisIntegration:
    """
    Integrates systematic uncertainty analysis with Regge trajectory fitting.
    
    Provides:
    - Width-based systematic uncertainty handling
    - Uncertainty propagation to predictions
    - Sensitivity analysis and reporting
    - Integration with existing Regge analysis
    """
    
    def __init__(self, config: Optional[UncertaintyConfig] = None):
        """
        Initialize error analysis integration.
        
        Parameters:
        -----------
        config : UncertaintyConfig, optional
            Configuration for uncertainty analysis
        """
        self.config = config or UncertaintyConfig()
        self.uncertainty_analyzer = SystematicUncertaintyAnalyzer(config)
        self.integration_results = {}
        
    def integrate_with_regge_analysis(self, 
                                    regge_data: pd.DataFrame,
                                    fit_function: callable,
                                    fit_params: List[str],
                                    j_values_to_predict: List[float],
                                    pdg_data: pd.DataFrame,
                                    output_dir: str = "integrated_error_analysis") -> Dict[str, Any]:
        """
        Integrate systematic uncertainty analysis with Regge trajectory analysis.
        
        Parameters:
        -----------
        regge_data : pd.DataFrame
            Regge trajectory data with mass, J, and width information
        fit_function : callable
            Function to fit (e.g., linear J = α₀ + α'M²)
        fit_params : List[str]
            Names of fit parameters
        j_values_to_predict : List[float]
            J values to predict masses for
        pdg_data : pd.DataFrame
            PDG data for cross-checking
        output_dir : str
            Directory for output files
            
        Returns:
        --------
        Dict containing integrated analysis results
        """
        print("=" * 60)
        print("INTEGRATED ERROR ANALYSIS FOR REGGE TRAJECTORIES")
        print("=" * 60)
        
        results = {}
        
        # 1. Compute systematic uncertainties with default kappa
        print("\n1. Computing systematic uncertainties...")
        data_with_uncertainties = self.uncertainty_analyzer.compute_systematic_uncertainties(
            regge_data, self.config.default_kappa
        )
        results['data_with_uncertainties'] = data_with_uncertainties
        
        # 2. Perform kappa sensitivity analysis
        print("2. Performing kappa sensitivity analysis...")
        sensitivity_results = self.uncertainty_analyzer.analyze_kappa_sensitivity(
            regge_data, fit_function, fit_params
        )
        results['sensitivity_analysis'] = sensitivity_results
        
        # 3. Fit with systematic uncertainties
        print("3. Fitting with systematic uncertainties...")
        fit_results = self._perform_fit_with_uncertainties(
            data_with_uncertainties, fit_function, fit_params
        )
        results['fit_results'] = fit_results
        
        # 4. Propagate uncertainties to predictions
        print("4. Propagating uncertainties to predictions...")
        predictions = self.uncertainty_analyzer.propagate_uncertainties_to_predictions(
            fit_results['parameters'],
            fit_results['covariance'],
            j_values_to_predict,
            fit_function
        )
        results['predictions'] = predictions
        
        # 5. Cross-check with PDG
        print("5. Cross-checking predictions with PDG...")
        cross_check_results = self.uncertainty_analyzer.cross_check_predictions_with_pdg(
            predictions, pdg_data
        )
        results['cross_check_results'] = cross_check_results
        
        # 6. Generate visualizations and reports
        print("6. Generating visualizations and reports...")
        self._generate_integrated_visualizations(results, output_dir)
        self._generate_integrated_report(results, output_dir)
        
        # 7. Save results
        print("7. Saving analysis results...")
        self.uncertainty_analyzer.save_analysis_results(
            sensitivity_results, predictions, cross_check_results, output_dir
        )
        
        self.integration_results = results
        
        print("\n" + "=" * 60)
        print("INTEGRATED ERROR ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return results
    
    def _perform_fit_with_uncertainties(self, 
                                      data: pd.DataFrame,
                                      fit_function: callable,
                                      fit_params: List[str]) -> Dict[str, Any]:
        """
        Perform weighted fit with systematic uncertainties.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with systematic uncertainties
        fit_function : callable
            Function to fit
        fit_params : List[str]
            Names of fit parameters
            
        Returns:
        --------
        Dict containing fit results
        """
        # Prepare data for fitting
        x_data = data['M2GeV2'].values
        y_data = data['J'].values
        y_errors = data['M2SigmaGeV2'].values
        
        # Remove points with invalid uncertainties
        valid_mask = (y_errors > 0) & np.isfinite(y_errors)
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        y_errors_valid = y_errors[valid_mask]
        
        if len(x_valid) < 2:
            raise ValueError("Insufficient valid data points for fitting")
        
        # Perform weighted fit
        popt, pcov = curve_fit(fit_function, x_valid, y_valid, 
                             sigma=y_errors_valid, absolute_sigma=True)
        
        # Compute fit statistics
        y_pred = fit_function(x_valid, *popt)
        residuals = y_valid - y_pred
        chi2 = np.sum((residuals / y_errors_valid)**2)
        dof = len(x_valid) - len(popt)
        chi2_dof = chi2 / dof if dof > 0 else np.inf
        
        # Compute R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Parameter uncertainties
        param_uncertainties = np.sqrt(np.diag(pcov))
        
        return {
            'parameters': popt,
            'covariance': pcov,
            'parameter_uncertainties': param_uncertainties,
            'chi2': chi2,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'dof': dof,
            'x_fit': x_valid,
            'y_fit': y_valid,
            'y_pred': y_pred,
            'residuals': residuals,
            'weights': 1 / y_errors_valid**2
        }
    
    def _generate_integrated_visualizations(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Generate integrated visualizations for error analysis.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Integrated analysis results
        output_dir : str
            Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sensitivity plots
        self.uncertainty_analyzer.create_sensitivity_plots(
            results['sensitivity_analysis'], output_dir
        )
        
        # Create integrated fit plot
        self._create_integrated_fit_plot(results, output_dir)
        
        # Create prediction plot
        self._create_prediction_plot(results, output_dir)
        
        # Create gap analysis plot
        self._create_gap_analysis_plot(results, output_dir)
    
    def _create_integrated_fit_plot(self, results: Dict[str, Any], output_dir: str) -> None:
        """Create integrated fit plot showing data with systematic uncertainties."""
        import os
        
        fit_results = results['fit_results']
        data = results['data_with_uncertainties']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Regge Trajectory Fit with Systematic Uncertainties', fontsize=16, fontweight='bold')
        
        # Main fit plot
        x_fit = fit_results['x_fit']
        y_fit = fit_results['y_fit']
        y_pred = fit_results['y_pred']
        
        # Plot data points with error bars
        ax1.errorbar(data['M2GeV2'], data['J'], 
                    yerr=data['M2SigmaGeV2'], 
                    fmt='o', capsize=5, capthick=2, 
                    label='Data with systematic uncertainties', alpha=0.7)
        
        # Plot fit line
        ax1.plot(x_fit, y_pred, 'r-', linewidth=2, label='Weighted fit')
        
        # Add fit parameters
        params = fit_results['parameters']
        param_uncertainties = fit_results['parameter_uncertainties']
        chi2_dof = fit_results['chi2_dof']
        r2 = fit_results['r_squared']
        
        ax1.text(0.05, 0.95, 
                f"α₀ = {params[0]:.3f} ± {param_uncertainties[0]:.3f}\n"
                f"α' = {params[1]:.4f} ± {param_uncertainties[1]:.4f} GeV⁻²\n"
                f"χ²/dof = {chi2_dof:.3f}\n"
                f"R² = {r2:.4f}\n"
                f"κ = {self.config.default_kappa}",
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_xlabel('M² (GeV²)')
        ax1.set_ylabel('J')
        ax1.set_title('Regge Trajectory with Systematic Uncertainties')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = fit_results['residuals']
        weights = fit_results['weights']
        
        ax2.scatter(x_fit, residuals, c=weights, cmap='viridis', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('M² (GeV²)')
        ax2.set_ylabel('Residuals (J)')
        ax2.set_title('Weighted Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for weights
        scatter = ax2.scatter([], [], c=[], cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Weight (1/σ²)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'integrated_fit_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_plot(self, results: Dict[str, Any], output_dir: str) -> None:
        """Create prediction plot with uncertainty bands."""
        import os
        
        predictions = results['predictions']
        fit_results = results['fit_results']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot original data
        data = results['data_with_uncertainties']
        ax.errorbar(data['M2GeV2'], data['J'], 
                   yerr=data['M2SigmaGeV2'], 
                   fmt='o', capsize=5, capthick=2, 
                   label='Data with systematic uncertainties', alpha=0.7)
        
        # Plot fit line
        x_fit = fit_results['x_fit']
        y_pred = fit_results['y_pred']
        ax.plot(x_fit, y_pred, 'r-', linewidth=2, label='Weighted fit')
        
        # Plot predictions
        valid_predictions = predictions.dropna()
        if not valid_predictions.empty:
            pred_m2 = valid_predictions['PredictedM2GeV2'].values
            pred_j = valid_predictions['J'].values
            pred_m2_errors = valid_predictions['M2UncertaintyGeV2'].values
            
            ax.errorbar(pred_m2, pred_j, xerr=pred_m2_errors,
                       fmt='s', capsize=5, capthick=2, 
                       label='Predictions with uncertainties', alpha=0.8, color='green')
        
        # Add prediction intervals
        if not valid_predictions.empty:
            # Create smooth prediction curve
            m2_range = np.linspace(min(data['M2GeV2'].min(), pred_m2.min() if len(pred_m2) > 0 else data['M2GeV2'].min()),
                                 max(data['M2GeV2'].max(), pred_m2.max() if len(pred_m2) > 0 else data['M2GeV2'].max()),
                                 100)
            
            params = fit_results['parameters']
            j_curve = params[0] + params[1] * m2_range
            
            # Add uncertainty band (simplified)
            param_uncertainties = fit_results['parameter_uncertainties']
            j_upper = (params[0] + param_uncertainties[0]) + (params[1] + param_uncertainties[1]) * m2_range
            j_lower = (params[0] - param_uncertainties[0]) + (params[1] - param_uncertainties[1]) * m2_range
            
            ax.fill_between(m2_range, j_lower, j_upper, alpha=0.3, color='red', 
                           label='Fit uncertainty band')
        
        ax.set_xlabel('M² (GeV²)')
        ax.set_ylabel('J')
        ax.set_title('Regge Trajectory Predictions with Uncertainties')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions_with_uncertainties.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gap_analysis_plot(self, results: Dict[str, Any], output_dir: str) -> None:
        """Create gap analysis plot showing significant gaps."""
        import os
        
        cross_check_results = results['cross_check_results']
        predictions = results['predictions']
        
        # Separate significant gaps from confirmed predictions
        significant_gaps = cross_check_results[cross_check_results['IsSignificantGap']]
        confirmed_predictions = cross_check_results[~cross_check_results['IsSignificantGap']]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot all predictions
        all_j = predictions['J'].values
        all_masses = predictions['PredictedMassGeV'].values
        all_uncertainties = predictions['MassUncertaintyGeV'].values
        
        # Plot confirmed predictions
        if not confirmed_predictions.empty:
            confirmed_j = confirmed_predictions['J'].values
            confirmed_masses = confirmed_predictions['PredictedMassGeV'].values
            confirmed_uncertainties = confirmed_predictions['MassUncertaintyGeV'].values
            
            ax.errorbar(confirmed_j, confirmed_masses, yerr=confirmed_uncertainties,
                       fmt='o', capsize=5, capthick=2, 
                       label='Confirmed predictions', color='green', alpha=0.8)
        
        # Plot significant gaps
        if not significant_gaps.empty:
            gap_j = significant_gaps['J'].values
            gap_masses = significant_gaps['PredictedMassGeV'].values
            gap_uncertainties = significant_gaps['MassUncertaintyGeV'].values
            
            ax.errorbar(gap_j, gap_masses, yerr=gap_uncertainties,
                       fmt='s', capsize=5, capthick=2, 
                       label='Significant gaps', color='red', alpha=0.8)
            
            # Add gap annotations
            for _, row in significant_gaps.iterrows():
                ax.annotate(f"J={row['J']}", 
                           (row['J'], row['PredictedMassGeV']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel('J')
        ax.set_ylabel('Predicted Mass (GeV)')
        ax.set_title('Gap Analysis: Predictions vs PDG Cross-Check')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics
        total_predictions = len(cross_check_results)
        significant_gap_count = len(significant_gaps)
        confirmed_count = len(confirmed_predictions)
        
        ax.text(0.02, 0.98, 
               f"Total predictions: {total_predictions}\n"
               f"Significant gaps: {significant_gap_count}\n"
               f"Confirmed: {confirmed_count}",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gap_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_integrated_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Generate integrated error analysis report.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Integrated analysis results
        output_dir : str
            Output directory
            
        Returns:
        --------
        str
            Path to generated report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'integrated_error_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("INTEGRATED ERROR ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Systematic Uncertainty Configuration:\n")
            f.write(f"  Default κ: {self.config.default_kappa}\n")
            f.write(f"  κ range: [{self.config.kappa_range[0]}, {self.config.kappa_range[1]}]\n")
            f.write(f"  Confidence level: {self.config.confidence_level:.1%}\n")
            f.write(f"  Max width fraction: {self.config.max_width_fraction}\n\n")
            
            # Fit results with systematic uncertainties
            f.write("1. FIT RESULTS WITH SYSTEMATIC UNCERTAINTIES\n")
            f.write("-" * 50 + "\n")
            
            fit_results = results['fit_results']
            params = fit_results['parameters']
            param_uncertainties = fit_results['parameter_uncertainties']
            
            f.write(f"α₀ = {params[0]:.3f} ± {param_uncertainties[0]:.3f}\n")
            f.write(f"α' = {params[1]:.4f} ± {param_uncertainties[1]:.4f} GeV⁻²\n")
            f.write(f"χ²/dof = {fit_results['chi2_dof']:.3f}\n")
            f.write(f"R² = {fit_results['r_squared']:.4f}\n")
            f.write(f"Degrees of freedom: {fit_results['dof']}\n\n")
            
            # Systematic uncertainty analysis
            f.write("2. SYSTEMATIC UNCERTAINTY ANALYSIS\n")
            f.write("-" * 50 + "\n")
            
            data = results['data_with_uncertainties']
            systematic_errors = data['SystematicErrorGeV'].values
            total_errors = data['TotalUncertaintyGeV'].values
            
            f.write(f"Average systematic error: {np.mean(systematic_errors):.4f} GeV\n")
            f.write(f"Average total uncertainty: {np.mean(total_errors):.4f} GeV\n")
            f.write(f"Systematic fraction: {np.mean(systematic_errors) / np.mean(total_errors):.1%}\n\n")
            
            # Sensitivity analysis summary
            f.write("3. KAPPA SENSITIVITY ANALYSIS SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            sensitivity = results['sensitivity_analysis']
            alphap_values = sensitivity['fit_parameters'].get('alphap', [])
            
            if alphap_values:
                valid_alphap = [a for a in alphap_values if not np.isnan(a)]
                if valid_alphap:
                    f.write(f"α' range across κ: [{min(valid_alphap):.4f}, {max(valid_alphap):.4f}] GeV⁻²\n")
                    f.write(f"α' variation: {max(valid_alphap) - min(valid_alphap):.4f} GeV⁻²\n")
                    
                    # Find optimal kappa
                    chi2_values = sensitivity['chi2_values']
                    if chi2_values and not all(np.isnan(chi2_values)):
                        optimal_idx = np.nanargmin(chi2_values)
                        optimal_kappa = sensitivity['kappa_values'][optimal_idx]
                        f.write(f"Optimal κ (minimum χ²/dof): {optimal_kappa:.3f}\n")
            f.write("\n")
            
            # Predictions summary
            f.write("4. MASS PREDICTIONS SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            predictions = results['predictions']
            valid_predictions = predictions.dropna()
            
            f.write(f"Total predictions: {len(predictions)}\n")
            f.write(f"Valid predictions: {len(valid_predictions)}\n")
            
            if not valid_predictions.empty:
                f.write(f"Average mass uncertainty: {valid_predictions['MassUncertaintyGeV'].mean():.3f} GeV\n")
                f.write(f"Mass uncertainty range: [{valid_predictions['MassUncertaintyGeV'].min():.3f}, "
                       f"{valid_predictions['MassUncertaintyGeV'].max():.3f}] GeV\n")
            f.write("\n")
            
            # Gap analysis summary
            f.write("5. GAP ANALYSIS SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            cross_check = results['cross_check_results']
            significant_gaps = cross_check[cross_check['IsSignificantGap']]
            
            f.write(f"Total predictions cross-checked: {len(cross_check)}\n")
            f.write(f"Significant gaps found: {len(significant_gaps)}\n")
            f.write(f"Confirmed predictions: {len(cross_check) - len(significant_gaps)}\n\n")
            
            if not significant_gaps.empty:
                f.write("Significant gaps (potential discoveries):\n")
                for _, row in significant_gaps.iterrows():
                    f.write(f"  J = {row['J']}: {row['GapAnalysis']}\n")
            f.write("\n")
            
            # Recommendations
            f.write("6. RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            
            f.write("Systematic Uncertainty Handling:\n")
            f.write(f"- Use κ = {self.config.default_kappa} for width-based systematic uncertainties\n")
            f.write("- Include systematic uncertainties in all uncertainty calculations\n")
            f.write("- Consider κ sensitivity when interpreting results\n\n")
            
            f.write("Prediction Reliability:\n")
            f.write("- Predictions with large uncertainties require experimental confirmation\n")
            f.write("- Significant gaps indicate potential new particle states\n")
            f.write("- Cross-check predictions with theoretical models\n\n")
            
            f.write("Experimental Implications:\n")
            f.write("- Focus experimental searches on significant gap regions\n")
            f.write("- Use prediction uncertainties to guide experimental precision requirements\n")
            f.write("- Consider systematic uncertainties in experimental design\n\n")
            
            f.write("=" * 80 + "\n")
        
        return report_path
    
    def get_key_metrics(self) -> Dict[str, Any]:
        """
        Extract key metrics from the integrated analysis.
        
        Returns:
        --------
        Dict containing key metrics
        """
        if not self.integration_results:
            return {}
        
        results = self.integration_results
        
        # Extract key metrics
        fit_results = results['fit_results']
        predictions = results['predictions']
        cross_check = results['cross_check_results']
        
        key_metrics = {
            'alphap': fit_results['parameters'][1],
            'alphap_uncertainty': fit_results['parameter_uncertainties'][1],
            'alpha0': fit_results['parameters'][0],
            'alpha0_uncertainty': fit_results['parameter_uncertainties'][0],
            'chi2_dof': fit_results['chi2_dof'],
            'r_squared': fit_results['r_squared'],
            'default_kappa': self.config.default_kappa,
            'total_predictions': len(predictions),
            'valid_predictions': len(predictions.dropna()),
            'significant_gaps': len(cross_check[cross_check['IsSignificantGap']]),
            'average_mass_uncertainty': predictions['MassUncertaintyGeV'].mean() if not predictions.empty else np.nan
        }
        
        return key_metrics

if __name__ == "__main__":
    print("Error Analysis Integration")
    print("Use this module to integrate systematic uncertainty analysis with Regge trajectories")
