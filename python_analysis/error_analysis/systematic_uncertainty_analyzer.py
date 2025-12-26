"""
Systematic Uncertainty Analysis for Regge Trajectories

Implements comprehensive error analysis including:
- Width-based systematic uncertainties (κ parameter)
- Uncertainty propagation from fit parameters to predictions
- Sensitivity analysis of α' vs κ
- Robust prediction intervals with proper error bars
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from dataclasses import dataclass

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty analysis."""
    kappa_range: Tuple[float, float] = (0.0, 0.5)
    kappa_steps: int = 21
    confidence_level: float = 0.68  # 1σ
    default_kappa: float = 0.25
    width_systematic_fraction: float = 0.25
    min_width_threshold: float = 0.001  # GeV
    max_width_fraction: float = 0.5  # Maximum width as fraction of mass

class SystematicUncertaintyAnalyzer:
    """
    Analyzes systematic uncertainties in Regge trajectory fitting.
    
    Handles:
    - Width-based systematic uncertainties
    - Uncertainty propagation to predictions
    - Sensitivity analysis
    - Robust prediction intervals
    """
    
    def __init__(self, config: Optional[UncertaintyConfig] = None):
        """
        Initialize systematic uncertainty analyzer.
        
        Parameters:
        -----------
        config : UncertaintyConfig, optional
            Configuration for uncertainty analysis
        """
        self.config = config or UncertaintyConfig()
        self.analysis_results = {}
        
    def compute_systematic_uncertainties(self, data: pd.DataFrame, 
                                       kappa: Optional[float] = None) -> pd.DataFrame:
        """
        Compute systematic uncertainties from resonance widths.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data with mass, mass_uncertainty, and width columns
        kappa : float, optional
            Fraction of width to include as systematic uncertainty
            
        Returns:
        --------
        pd.DataFrame
            Data with systematic uncertainties added
        """
        if kappa is None:
            kappa = self.config.default_kappa
            
        # Create copy of data
        result_data = data.copy()
        
        # Compute systematic uncertainties
        systematic_errors = []
        total_uncertainties = []
        mass_squared_uncertainties = []
        
        for _, row in result_data.iterrows():
            mass = row['MassGeV']
            mass_uncertainty = row.get('MassSigmaGeV', 0.0)
            width = row.get('ResonanceWidthGeV', 0.0)
            
            # Compute width-based systematic
            if width > self.config.min_width_threshold:
                # Limit width contribution to reasonable fraction of mass
                max_width_contribution = mass * self.config.max_width_fraction
                width_contribution = min(width * kappa, max_width_contribution)
                systematic_error = width_contribution
            else:
                systematic_error = 0.0
            
            # Combine uncertainties in quadrature
            total_uncertainty = np.sqrt(mass_uncertainty**2 + systematic_error**2)
            
            # Propagate to mass-squared uncertainty
            mass_squared_uncertainty = 2 * mass * total_uncertainty
            
            systematic_errors.append(systematic_error)
            total_uncertainties.append(total_uncertainty)
            mass_squared_uncertainties.append(mass_squared_uncertainty)
        
        # Add uncertainty columns
        result_data['SystematicErrorGeV'] = systematic_errors
        result_data['TotalUncertaintyGeV'] = total_uncertainties
        result_data['M2SigmaGeV2'] = mass_squared_uncertainties
        result_data['Kappa'] = kappa
        
        return result_data
    
    def analyze_kappa_sensitivity(self, data: pd.DataFrame, 
                                fit_function: callable,
                                fit_params: List[str]) -> Dict[str, Any]:
        """
        Analyze sensitivity of fitted parameters to κ (width systematic fraction).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data
        fit_function : callable
            Function to fit (e.g., linear J = α₀ + α'M²)
        fit_params : List[str]
            Names of fit parameters
            
        Returns:
        --------
        Dict containing sensitivity analysis results
        """
        kappa_values = np.linspace(self.config.kappa_range[0], 
                                  self.config.kappa_range[1], 
                                  self.config.kappa_steps)
        
        sensitivity_results = {
            'kappa_values': kappa_values,
            'fit_parameters': {},
            'fit_uncertainties': {},
            'chi2_values': [],
            'r_squared_values': []
        }
        
        # Initialize parameter storage
        for param in fit_params:
            sensitivity_results['fit_parameters'][param] = []
            sensitivity_results['fit_uncertainties'][param] = []
        
        # Analyze sensitivity across kappa range
        for kappa in kappa_values:
            # Compute systematic uncertainties for this kappa
            data_with_uncertainties = self.compute_systematic_uncertainties(data, kappa)
            
            # Prepare data for fitting
            x_data = data_with_uncertainties['M2GeV2'].values
            y_data = data_with_uncertainties['J'].values
            y_errors = data_with_uncertainties['M2SigmaGeV2'].values
            
            # Remove points with invalid uncertainties
            valid_mask = (y_errors > 0) & np.isfinite(y_errors)
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            y_errors_valid = y_errors[valid_mask]
            
            if len(x_valid) < 2:
                continue
            
            try:
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
                
                # Store results
                for i, param in enumerate(fit_params):
                    sensitivity_results['fit_parameters'][param].append(popt[i])
                    sensitivity_results['fit_uncertainties'][param].append(np.sqrt(pcov[i, i]))
                
                sensitivity_results['chi2_values'].append(chi2_dof)
                sensitivity_results['r_squared_values'].append(r_squared)
                
            except (RuntimeError, ValueError) as e:
                # Handle fit failures
                for param in fit_params:
                    sensitivity_results['fit_parameters'][param].append(np.nan)
                    sensitivity_results['fit_uncertainties'][param].append(np.nan)
                sensitivity_results['chi2_values'].append(np.nan)
                sensitivity_results['r_squared_values'].append(np.nan)
        
        return sensitivity_results
    
    def propagate_uncertainties_to_predictions(self, 
                                             fit_parameters: np.ndarray,
                                             fit_covariance: np.ndarray,
                                             j_values: List[float],
                                             fit_function: callable,
                                             confidence_level: Optional[float] = None) -> pd.DataFrame:
        """
        Propagate fit uncertainties to mass predictions for missing J states.
        
        Parameters:
        -----------
        fit_parameters : np.ndarray
            Fitted parameters
        fit_covariance : np.ndarray
            Covariance matrix of fit parameters
        j_values : List[float]
            J values to predict masses for
        fit_function : callable
            Function used for fitting
        confidence_level : float, optional
            Confidence level for prediction intervals
            
        Returns:
        --------
        pd.DataFrame
            Predictions with uncertainties
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        # Convert confidence level to number of standard deviations
        n_sigma = stats.norm.ppf((1 + confidence_level) / 2)
        
        predictions = []
        
        for j in j_values:
            # Invert the fit function to get M² from J
            # For linear fit: J = α₀ + α'M² → M² = (J - α₀) / α'
            try:
                # For linear fit, we can solve analytically
                alpha0, alphap = fit_parameters
                mass_squared_pred = (j - alpha0) / alphap
                
                # Compute uncertainty using error propagation
                # ∂M²/∂α₀ = -1/α', ∂M²/∂α' = -(J-α₀)/α'²
                d_dalpha0 = -1 / alphap
                d_dalphap = -(j - alpha0) / (alphap**2)
                
                # Variance of M² prediction
                var_mass_squared = (d_dalpha0**2 * fit_covariance[0, 0] + 
                                  d_dalphap**2 * fit_covariance[1, 1] + 
                                  2 * d_dalpha0 * d_dalphap * fit_covariance[0, 1])
                
                mass_squared_uncertainty = np.sqrt(var_mass_squared)
                
                # Convert to mass and mass uncertainty
                mass_pred = np.sqrt(mass_squared_pred)
                mass_uncertainty = mass_squared_uncertainty / (2 * mass_pred)
                
                # Compute prediction intervals
                mass_lower = mass_pred - n_sigma * mass_uncertainty
                mass_upper = mass_pred + n_sigma * mass_uncertainty
                
                predictions.append({
                    'J': j,
                    'PredictedMassGeV': mass_pred,
                    'MassUncertaintyGeV': mass_uncertainty,
                    'MassLowerGeV': mass_lower,
                    'MassUpperGeV': mass_upper,
                    'PredictedM2GeV2': mass_squared_pred,
                    'M2UncertaintyGeV2': mass_squared_uncertainty,
                    'ConfidenceLevel': confidence_level,
                    'NSigma': n_sigma
                })
                
            except (ValueError, ZeroDivisionError):
                # Handle cases where prediction fails
                predictions.append({
                    'J': j,
                    'PredictedMassGeV': np.nan,
                    'MassUncertaintyGeV': np.nan,
                    'MassLowerGeV': np.nan,
                    'MassUpperGeV': np.nan,
                    'PredictedM2GeV2': np.nan,
                    'M2UncertaintyGeV2': np.nan,
                    'ConfidenceLevel': confidence_level,
                    'NSigma': n_sigma
                })
        
        return pd.DataFrame(predictions)
    
    def cross_check_predictions_with_pdg(self, 
                                       predictions: pd.DataFrame,
                                       pdg_data: pd.DataFrame,
                                       n_sigma_threshold: float = 2.0) -> pd.DataFrame:
        """
        Cross-check predictions with PDG data and flag gaps.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Predicted masses with uncertainties
        pdg_data : pd.DataFrame
            PDG particle data
        n_sigma_threshold : float
            Threshold for flagging significant gaps
            
        Returns:
        --------
        pd.DataFrame
            Cross-check results with gap analysis
        """
        cross_check_results = []
        
        for _, pred_row in predictions.iterrows():
            j_value = pred_row['J']
            pred_mass = pred_row['PredictedMassGeV']
            pred_uncertainty = pred_row['MassUncertaintyGeV']
            
            if np.isnan(pred_mass) or np.isnan(pred_uncertainty):
                cross_check_results.append({
                    'J': j_value,
                    'PredictedMassGeV': pred_mass,
                    'MassUncertaintyGeV': pred_uncertainty,
                    'SearchWindowLower': np.nan,
                    'SearchWindowUpper': np.nan,
                    'NearbyPDGEntries': [],
                    'ClosestPDGEntry': None,
                    'ClosestDistanceSigma': np.nan,
                    'IsSignificantGap': False,
                    'GapAnalysis': 'Prediction failed'
                })
                continue
            
            # Define search window
            search_lower = pred_mass - n_sigma_threshold * pred_uncertainty
            search_upper = pred_mass + n_sigma_threshold * pred_uncertainty
            
            # Find nearby PDG entries with same J
            nearby_entries = []
            for _, pdg_row in pdg_data.iterrows():
                if pdg_row.get('J', np.nan) == j_value:
                    pdg_mass = pdg_row.get('MassGeV', np.nan)
                    if not np.isnan(pdg_mass) and search_lower <= pdg_mass <= search_upper:
                        distance_sigma = abs(pdg_mass - pred_mass) / pred_uncertainty
                        nearby_entries.append({
                            'PDGEntry': pdg_row.get('Name', 'Unknown'),
                            'PDGMassGeV': pdg_mass,
                            'DistanceSigma': distance_sigma,
                            'Status': pdg_row.get('Status', 'Unknown')
                        })
            
            # Sort by distance
            nearby_entries.sort(key=lambda x: x['DistanceSigma'])
            
            # Determine if this is a significant gap
            closest_distance = nearby_entries[0]['DistanceSigma'] if nearby_entries else np.inf
            is_significant_gap = len(nearby_entries) == 0 or closest_distance > n_sigma_threshold
            
            # Analyze gap
            if len(nearby_entries) == 0:
                gap_analysis = f"No PDG entries found within {n_sigma_threshold}σ window"
            elif closest_distance > n_sigma_threshold:
                gap_analysis = f"Closest PDG entry is {closest_distance:.2f}σ away"
            else:
                gap_analysis = f"PDG entry found at {closest_distance:.2f}σ distance"
            
            cross_check_results.append({
                'J': j_value,
                'PredictedMassGeV': pred_mass,
                'MassUncertaintyGeV': pred_uncertainty,
                'SearchWindowLower': search_lower,
                'SearchWindowUpper': search_upper,
                'NearbyPDGEntries': nearby_entries,
                'ClosestPDGEntry': nearby_entries[0] if nearby_entries else None,
                'ClosestDistanceSigma': closest_distance,
                'IsSignificantGap': is_significant_gap,
                'GapAnalysis': gap_analysis
            })
        
        return pd.DataFrame(cross_check_results)
    
    def create_sensitivity_plots(self, sensitivity_results: Dict[str, Any], 
                               output_dir: str = "error_analysis") -> None:
        """
        Create sensitivity analysis plots.
        
        Parameters:
        -----------
        sensitivity_results : Dict[str, Any]
            Results from kappa sensitivity analysis
        output_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        kappa_values = sensitivity_results['kappa_values']
        
        # Create multi-panel sensitivity plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Systematic Uncertainty Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. α' vs κ
        ax1 = axes[0, 0]
        alphap_values = sensitivity_results['fit_parameters'].get('alphap', [])
        alphap_errors = sensitivity_results['fit_uncertainties'].get('alphap', [])
        
        if alphap_values:
            ax1.errorbar(kappa_values, alphap_values, yerr=alphap_errors, 
                        marker='o', capsize=5, capthick=2, linewidth=2)
            ax1.set_xlabel('κ (Width Systematic Fraction)')
            ax1.set_ylabel("α' (GeV⁻²)")
            ax1.set_title("Regge Slope vs Width Systematic")
            ax1.grid(True, alpha=0.3)
            
            # Highlight default kappa
            default_idx = np.argmin(np.abs(kappa_values - self.config.default_kappa))
            ax1.axvline(x=self.config.default_kappa, color='red', linestyle='--', 
                       alpha=0.7, label=f'Default κ = {self.config.default_kappa}')
            ax1.legend()
        
        # 2. α₀ vs κ
        ax2 = axes[0, 1]
        alpha0_values = sensitivity_results['fit_parameters'].get('alpha0', [])
        alpha0_errors = sensitivity_results['fit_uncertainties'].get('alpha0', [])
        
        if alpha0_values:
            ax2.errorbar(kappa_values, alpha0_values, yerr=alpha0_errors, 
                        marker='s', capsize=5, capthick=2, linewidth=2)
            ax2.set_xlabel('κ (Width Systematic Fraction)')
            ax2.set_ylabel('α₀')
            ax2.set_title("Regge Intercept vs Width Systematic")
            ax2.grid(True, alpha=0.3)
            
            # Highlight default kappa
            ax2.axvline(x=self.config.default_kappa, color='red', linestyle='--', alpha=0.7)
        
        # 3. χ²/dof vs κ
        ax3 = axes[1, 0]
        chi2_values = sensitivity_results['chi2_values']
        
        if chi2_values:
            ax3.plot(kappa_values, chi2_values, marker='^', linewidth=2, markersize=8)
            ax3.set_xlabel('κ (Width Systematic Fraction)')
            ax3.set_ylabel('χ²/dof')
            ax3.set_title("Fit Quality vs Width Systematic")
            ax3.grid(True, alpha=0.3)
            
            # Highlight default kappa
            ax3.axvline(x=self.config.default_kappa, color='red', linestyle='--', alpha=0.7)
            
            # Add horizontal line at χ²/dof = 1
            ax3.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='χ²/dof = 1')
            ax3.legend()
        
        # 4. R² vs κ
        ax4 = axes[1, 1]
        r2_values = sensitivity_results['r_squared_values']
        
        if r2_values:
            ax4.plot(kappa_values, r2_values, marker='d', linewidth=2, markersize=8)
            ax4.set_xlabel('κ (Width Systematic Fraction)')
            ax4.set_ylabel('R²')
            ax4.set_title("Goodness of Fit vs Width Systematic")
            ax4.grid(True, alpha=0.3)
            
            # Highlight default kappa
            ax4.axvline(x=self.config.default_kappa, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kappa_sensitivity_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create uncertainty band plot
        self._create_uncertainty_band_plot(sensitivity_results, output_dir)
    
    def _create_uncertainty_band_plot(self, sensitivity_results: Dict[str, Any], 
                                    output_dir: str) -> None:
        """Create uncertainty band plot showing parameter ranges."""
        kappa_values = sensitivity_results['kappa_values']
        alphap_values = sensitivity_results['fit_parameters'].get('alphap', [])
        alphap_errors = sensitivity_results['fit_uncertainties'].get('alphap', [])
        
        if not alphap_values:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot central values and uncertainty bands
        ax.fill_between(kappa_values, 
                       np.array(alphap_values) - np.array(alphap_errors),
                       np.array(alphap_values) + np.array(alphap_errors),
                       alpha=0.3, color='blue', label='1σ Uncertainty Band')
        
        ax.plot(kappa_values, alphap_values, 'b-', linewidth=2, label='α\' (κ)')
        ax.plot(kappa_values, np.array(alphap_values) + np.array(alphap_errors), 
               'b--', alpha=0.7, label='+1σ')
        ax.plot(kappa_values, np.array(alphap_values) - np.array(alphap_errors), 
               'b--', alpha=0.7, label='-1σ')
        
        # Highlight default kappa
        default_idx = np.argmin(np.abs(kappa_values - self.config.default_kappa))
        ax.axvline(x=self.config.default_kappa, color='red', linestyle='--', 
                  linewidth=2, label=f'Default κ = {self.config.default_kappa}')
        
        # Add horizontal line at typical Regge slope
        ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.7, 
                  label='Typical α\' ≈ 0.9 GeV⁻²')
        
        ax.set_xlabel('κ (Width Systematic Fraction)')
        ax.set_ylabel("α' (GeV⁻²)")
        ax.set_title('Regge Slope Uncertainty Band vs Width Systematic')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_band_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_error_analysis_report(self, 
                                     sensitivity_results: Dict[str, Any],
                                     predictions: pd.DataFrame,
                                     cross_check_results: pd.DataFrame,
                                     output_dir: str = "error_analysis") -> str:
        """
        Generate comprehensive error analysis report.
        
        Parameters:
        -----------
        sensitivity_results : Dict[str, Any]
            Results from kappa sensitivity analysis
        predictions : pd.DataFrame
            Mass predictions with uncertainties
        cross_check_results : pd.DataFrame
            Cross-check results with PDG
        output_dir : str
            Directory to save report
            
        Returns:
        --------
        str
            Path to generated report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'error_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SYSTEMATIC UNCERTAINTY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Default κ: {self.config.default_kappa}\n")
            f.write(f"Confidence Level: {self.config.confidence_level:.1%}\n\n")
            
            # Kappa sensitivity analysis
            f.write("1. KAPPA SENSITIVITY ANALYSIS\n")
            f.write("-" * 50 + "\n")
            
            kappa_values = sensitivity_results['kappa_values']
            alphap_values = sensitivity_results['fit_parameters'].get('alphap', [])
            
            if alphap_values:
                # Find optimal kappa (minimum chi2)
                chi2_values = sensitivity_results['chi2_values']
                if chi2_values and not all(np.isnan(chi2_values)):
                    optimal_idx = np.nanargmin(chi2_values)
                    optimal_kappa = kappa_values[optimal_idx]
                    optimal_alphap = alphap_values[optimal_idx]
                    
                    f.write(f"Optimal κ (minimum χ²/dof): {optimal_kappa:.3f}\n")
                    f.write(f"Optimal α': {optimal_alphap:.4f} GeV⁻²\n")
                    f.write(f"Minimum χ²/dof: {chi2_values[optimal_idx]:.3f}\n\n")
                
                # Default kappa analysis
                default_idx = np.argmin(np.abs(kappa_values - self.config.default_kappa))
                default_alphap = alphap_values[default_idx]
                default_chi2 = sensitivity_results['chi2_values'][default_idx]
                
                f.write(f"Default κ = {self.config.default_kappa}:\n")
                f.write(f"  α' = {default_alphap:.4f} GeV⁻²\n")
                f.write(f"  χ²/dof = {default_chi2:.3f}\n")
                f.write(f"  R² = {sensitivity_results['r_squared_values'][default_idx]:.4f}\n\n")
                
                # Parameter range analysis
                valid_alphap = [a for a in alphap_values if not np.isnan(a)]
                if valid_alphap:
                    f.write(f"α' range across κ: [{min(valid_alphap):.4f}, {max(valid_alphap):.4f}] GeV⁻²\n")
                    f.write(f"α' variation: {max(valid_alphap) - min(valid_alphap):.4f} GeV⁻²\n\n")
            
            # Predictions with uncertainties
            f.write("2. MASS PREDICTIONS WITH UNCERTAINTIES\n")
            f.write("-" * 50 + "\n")
            
            for _, row in predictions.iterrows():
                j_val = row['J']
                mass = row['PredictedMassGeV']
                uncertainty = row['MassUncertaintyGeV']
                n_sigma = row['NSigma']
                
                if not np.isnan(mass):
                    f.write(f"J = {j_val}: M = {mass:.3f} ± {uncertainty:.3f} GeV ({n_sigma}σ)\n")
                    f.write(f"  Prediction interval: [{row['MassLowerGeV']:.3f}, {row['MassUpperGeV']:.3f}] GeV\n")
                else:
                    f.write(f"J = {j_val}: Prediction failed\n")
            f.write("\n")
            
            # Cross-check results
            f.write("3. PDG CROSS-CHECK AND GAP ANALYSIS\n")
            f.write("-" * 50 + "\n")
            
            significant_gaps = cross_check_results[cross_check_results['IsSignificantGap']]
            
            f.write(f"Total predictions: {len(cross_check_results)}\n")
            f.write(f"Significant gaps: {len(significant_gaps)}\n\n")
            
            for _, row in significant_gaps.iterrows():
                j_val = row['J']
                pred_mass = row['PredictedMassGeV']
                uncertainty = row['MassUncertaintyGeV']
                gap_analysis = row['GapAnalysis']
                
                f.write(f"J = {j_val}: {gap_analysis}\n")
                f.write(f"  Predicted: {pred_mass:.3f} ± {uncertainty:.3f} GeV\n")
                f.write(f"  Search window: [{row['SearchWindowLower']:.3f}, {row['SearchWindowUpper']:.3f}] GeV\n")
                
                if row['ClosestPDGEntry']:
                    closest = row['ClosestPDGEntry']
                    f.write(f"  Closest PDG: {closest['PDGEntry']} at {closest['PDGMassGeV']:.3f} GeV ({closest['DistanceSigma']:.2f}σ)\n")
                f.write("\n")
            
            # Recommendations
            f.write("4. RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            
            f.write("Systematic Uncertainty Handling:\n")
            f.write(f"- Use κ = {self.config.default_kappa} as default width systematic fraction\n")
            f.write("- Include width-based systematic in all uncertainty calculations\n")
            f.write("- Propagate uncertainties through all predictions\n\n")
            
            f.write("Prediction Reliability:\n")
            f.write("- Predictions with large uncertainties should be treated with caution\n")
            f.write("- Significant gaps (no PDG entries within 2σ) indicate potential discoveries\n")
            f.write("- Cross-check predictions with theoretical expectations\n\n")
            
            f.write("=" * 80 + "\n")
        
        return report_path
    
    def save_analysis_results(self, 
                            sensitivity_results: Dict[str, Any],
                            predictions: pd.DataFrame,
                            cross_check_results: pd.DataFrame,
                            output_dir: str = "error_analysis") -> None:
        """
        Save all analysis results to files.
        
        Parameters:
        -----------
        sensitivity_results : Dict[str, Any]
            Results from kappa sensitivity analysis
        predictions : pd.DataFrame
            Mass predictions with uncertainties
        cross_check_results : pd.DataFrame
            Cross-check results with PDG
        output_dir : str
            Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sensitivity results
        sensitivity_df = pd.DataFrame({
            'kappa': sensitivity_results['kappa_values'],
            'alphap': sensitivity_results['fit_parameters'].get('alphap', []),
            'alphap_uncertainty': sensitivity_results['fit_uncertainties'].get('alphap', []),
            'alpha0': sensitivity_results['fit_parameters'].get('alpha0', []),
            'alpha0_uncertainty': sensitivity_results['fit_uncertainties'].get('alpha0', []),
            'chi2_dof': sensitivity_results['chi2_values'],
            'r_squared': sensitivity_results['r_squared_values']
        })
        sensitivity_df.to_csv(os.path.join(output_dir, 'kappa_sensitivity_results.csv'), index=False)
        
        # Save predictions
        predictions.to_csv(os.path.join(output_dir, 'mass_predictions_with_uncertainties.csv'), index=False)
        
        # Save cross-check results (simplified for CSV)
        cross_check_simple = cross_check_results.copy()
        cross_check_simple['NearbyPDGEntries'] = cross_check_simple['NearbyPDGEntries'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        cross_check_simple['ClosestPDGEntry'] = cross_check_simple['ClosestPDGEntry'].apply(
            lambda x: x['PDGEntry'] if x else None
        )
        cross_check_simple.to_csv(os.path.join(output_dir, 'pdg_cross_check_results.csv'), index=False)
        
        # Save configuration
        config_dict = {
            'default_kappa': self.config.default_kappa,
            'kappa_range': self.config.kappa_range,
            'confidence_level': self.config.confidence_level,
            'width_systematic_fraction': self.config.width_systematic_fraction,
            'min_width_threshold': self.config.min_width_threshold,
            'max_width_fraction': self.config.max_width_fraction
        }
        
        import json
        with open(os.path.join(output_dir, 'analysis_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

if __name__ == "__main__":
    print("Systematic Uncertainty Analyzer")
    print("Use this module to analyze systematic uncertainties in Regge trajectories")
