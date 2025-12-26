"""
Validation Analysis for Regge Trajectories

Compares findings with experimental data and theoretical expectations:
- PDG cross-check near predictions
- Residuals vs experimental quality correlation
- External theory overlay comparison
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings

class ValidationAnalyzer:
    """
    Validates Regge trajectory results against experimental and theoretical benchmarks.
    
    Implements:
    - PDG cross-check near predictions
    - Residual correlation with experimental quality
    - External theory overlay comparison
    """
    
    def __init__(self, data: pd.DataFrame, x_col: str = 'M2_GeV2', 
                 y_col: str = 'J', x_err_col: str = 'M2_sigma_GeV2',
                 name_col: str = 'name', width_col: Optional[str] = 'width_GeV',
                 status_col: Optional[str] = 'status', parity_col: Optional[str] = 'parity'):
        """
        Initialize validation analyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data
        x_col : str
            Column name for M² values
        y_col : str
            Column name for J values
        x_err_col : str
            Column name for M² uncertainties
        name_col : str
            Column name for particle names
        width_col : str, optional
            Column name for resonance widths
        status_col : str, optional
            Column name for PDG observational status
        parity_col : str, optional
            Column name for parity values
        """
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.x_err_col = x_err_col
        self.name_col = name_col
        self.width_col = width_col
        self.status_col = status_col
        self.parity_col = parity_col
        
        # Clean data
        self._clean_data()
        
        # Initialize results storage
        self.results = {}
        
    def _clean_data(self):
        """Remove rows with missing or invalid data."""
        mask = (
            self.data[self.x_col].notna() & 
            self.data[self.y_col].notna() &
            (self.data[self.x_col] > 0) &
            (self.data[self.y_col] >= 0)
        )
        
        if self.x_err_col in self.data.columns:
            mask &= self.data[self.x_err_col].notna()
            
        if self.name_col in self.data.columns:
            mask &= self.data[self.name_col].notna()
            
        self.data = self.data[mask].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError("No valid data points after cleaning")
            
        print(f"Using {len(self.data)} data points for validation analysis")
    
    def pdg_crosscheck_predictions(self, predictions: pd.DataFrame, 
                                 window_factor: float = 2.0) -> pd.DataFrame:
        """
        Cross-check predicted masses with nearby PDG candidates.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Predicted missing J states with columns ['J', 'M_GeV', 'M_sigma_GeV']
        window_factor : float
            Multiplier for uncertainty to define search window
            
        Returns:
        --------
        pd.DataFrame
            Cross-check results with nearby PDG candidates
        """
        crosscheck_results = []
        
        for _, pred in predictions.iterrows():
            J_pred = pred['J']
            M_pred = pred['M_GeV']
            M_sigma = pred['M_sigma_GeV']
            
            # Define search window
            window = window_factor * M_sigma
            M_min = M_pred - window
            M_max = M_pred + window
            
            # Find nearby PDG candidates with same J
            nearby_candidates = self.data[
                (self.data[self.y_col] == J_pred) &
                (self.data[self.x_col] >= M_min**2) &
                (self.data[self.x_col] <= M_max**2)
            ].copy()
            
            # Calculate distances and categorize matches
            if len(nearby_candidates) > 0:
                nearby_candidates['distance_GeV'] = np.abs(
                    np.sqrt(nearby_candidates[self.x_col]) - M_pred
                )
                nearby_candidates = nearby_candidates.sort_values('distance_GeV')
                
                # Categorize match quality
                best_match = nearby_candidates.iloc[0]
                distance = best_match['distance_GeV']
                
                if distance <= M_sigma:
                    match_type = "confirmation"
                elif distance <= 2 * M_sigma:
                    match_type = "near-miss"
                else:
                    match_type = "distant"
                
                # Get status information
                status = best_match.get(self.status_col, "Unknown") if self.status_col else "Unknown"
                
                # Add all nearby candidates to results
                for _, candidate in nearby_candidates.iterrows():
                    crosscheck_results.append({
                        'J_predicted': J_pred,
                        'M_predicted_GeV': M_pred,
                        'M_predicted_sigma_GeV': M_sigma,
                        'search_window_GeV': window,
                        'pdg_name': candidate[self.name_col],
                        'pdg_M_GeV': np.sqrt(candidate[self.x_col]),
                        'pdg_J': candidate[self.y_col],
                        'distance_GeV': candidate['distance_GeV'],
                        'pdg_status': candidate.get(self.status_col, "Unknown") if self.status_col else "Unknown",
                        'pdg_width_GeV': candidate.get(self.width_col, np.nan) if self.width_col else np.nan,
                        'match_type': match_type,
                        'is_best_match': candidate['distance_GeV'] == distance
                    })
            else:
                # No nearby candidates found - genuine gap
                crosscheck_results.append({
                    'J_predicted': J_pred,
                    'M_predicted_GeV': M_pred,
                    'M_predicted_sigma_GeV': M_sigma,
                    'search_window_GeV': window,
                    'pdg_name': "None",
                    'pdg_M_GeV': np.nan,
                    'pdg_J': np.nan,
                    'distance_GeV': np.nan,
                    'pdg_status': "None",
                    'pdg_width_GeV': np.nan,
                    'match_type': "genuine_gap",
                    'is_best_match': True
                })
        
        return pd.DataFrame(crosscheck_results)
    
    def residual_experimental_quality_analysis(self, residuals: np.ndarray, 
                                             fitted_values: np.ndarray) -> Dict[str, Any]:
        """
        Analyze correlation between residuals and experimental quality indicators.
        
        Parameters:
        -----------
        residuals : np.ndarray
            Fit residuals
        fitted_values : np.ndarray
            Fitted values from the model
            
        Returns:
        --------
        Dict containing residual quality analysis results
        """
        if len(residuals) != len(self.data):
            raise ValueError("Residuals length must match data length")
        
        # Prepare quality indicators
        quality_data = []
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            # Get width-based uncertainty
            width = row.get(self.width_col, np.nan) if self.width_col else np.nan
            width_uncertainty = 0.25 * width if not np.isnan(width) else 0.0
            
            # Get PDG status (convert to numeric if possible)
            status = row.get(self.status_col, "Unknown") if self.status_col else "Unknown"
            status_numeric = self._status_to_numeric(status)
            
            # Total experimental uncertainty
            mass_uncertainty = row.get(self.x_err_col, np.nan) if self.x_err_col else np.nan
            total_uncertainty = np.sqrt(mass_uncertainty**2 + width_uncertainty**2) if not np.isnan(mass_uncertainty) else np.nan
            
            quality_data.append({
                'residual': residuals[i],
                'abs_residual': abs(residuals[i]),
                'width_GeV': width,
                'width_uncertainty': width_uncertainty,
                'status_numeric': status_numeric,
                'status': status,
                'total_uncertainty': total_uncertainty,
                'mass_uncertainty': mass_uncertainty
            })
        
        quality_df = pd.DataFrame(quality_data)
        
        # Remove rows with missing data for correlation analysis
        valid_mask = quality_df['total_uncertainty'].notna() & (quality_df['total_uncertainty'] > 0)
        valid_data = quality_df[valid_mask]
        
        results = {
            'n_total': len(quality_df),
            'n_valid': len(valid_data),
            'correlations': {},
            'anova_results': {},
            'quality_summary': {}
        }
        
        if len(valid_data) >= 3:
            # Correlation analysis
            correlations = {}
            
            # Residual vs width
            if valid_data['width_GeV'].notna().any():
                corr_width, p_width = stats.pearsonr(
                    valid_data['abs_residual'], 
                    valid_data['width_GeV']
                )
                correlations['residual_vs_width'] = {
                    'correlation': corr_width,
                    'p_value': p_width,
                    'significant': p_width < 0.05
                }
            
            # Residual vs status
            if valid_data['status_numeric'].notna().any():
                corr_status, p_status = stats.pearsonr(
                    valid_data['abs_residual'], 
                    valid_data['status_numeric']
                )
                correlations['residual_vs_status'] = {
                    'correlation': corr_status,
                    'p_value': p_status,
                    'significant': p_status < 0.05
                }
            
            # Residual vs total uncertainty
            corr_uncertainty, p_uncertainty = stats.pearsonr(
                valid_data['abs_residual'], 
                valid_data['total_uncertainty']
            )
            correlations['residual_vs_uncertainty'] = {
                'correlation': corr_uncertainty,
                'p_value': p_uncertainty,
                'significant': p_uncertainty < 0.05
            }
            
            results['correlations'] = correlations
            
            # ANOVA analysis by status groups
            status_groups = valid_data.groupby('status')
            if len(status_groups) >= 2:
                group_data = [group['abs_residual'].values for name, group in status_groups]
                f_stat, p_value = stats.f_oneway(*group_data)
                
                results['anova_results'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_groups': len(status_groups)
                }
            
            # Quality summary statistics
            results['quality_summary'] = {
                'mean_residual': quality_df['abs_residual'].mean(),
                'std_residual': quality_df['abs_residual'].std(),
                'mean_width': quality_df['width_GeV'].mean(),
                'status_distribution': quality_df['status'].value_counts().to_dict()
            }
        
        return results
    
    def _status_to_numeric(self, status: str) -> float:
        """
        Convert PDG status to numeric quality indicator.
        
        Parameters:
        -----------
        status : str
            PDG observational status
            
        Returns:
        --------
        float
            Numeric quality indicator (higher = better quality)
        """
        status_mapping = {
            '★★★★': 4.0,  # Established
            '★★★': 3.0,   # Likely
            '★★': 2.0,     # Possible
            '★': 1.0,      # Tentative
            'Unknown': 0.0  # Unknown
        }
        
        # Handle variations in status format
        status_clean = status.strip()
        for key in status_mapping:
            if key in status_clean:
                return status_mapping[key]
        
        return 0.0  # Default for unrecognized status
    
    def external_theory_overlay(self, fitted_alpha0: float, fitted_alphap: float,
                              canonical_alpha0: float = 0.0, canonical_alphap: float = 0.9) -> Dict[str, Any]:
        """
        Compare fitted trajectory with canonical theoretical expectations.
        
        Parameters:
        -----------
        fitted_alpha0 : float
            Fitted intercept parameter
        fitted_alphap : float
            Fitted slope parameter
        canonical_alpha0 : float
            Canonical theoretical intercept
        canonical_alphap : float
            Canonical theoretical slope (GeV⁻²)
            
        Returns:
        --------
        Dict containing theory comparison results
        """
        x_data = self.data[self.x_col].values
        y_data = self.data[self.y_col].values
        
        # Calculate fitted and canonical predictions
        y_fitted = fitted_alpha0 + fitted_alphap * x_data
        y_canonical = canonical_alpha0 + canonical_alphap * x_data
        
        # Calculate deviations
        deviations = y_data - y_canonical
        abs_deviations = np.abs(deviations)
        
        # RMS deviation
        rms_deviation = np.sqrt(np.mean(deviations**2))
        
        # Weighted RMS (using uncertainties if available)
        if self.x_err_col in self.data.columns:
            weights = 1.0 / (self.data[self.x_err_col].values**2)
            weighted_rms = np.sqrt(np.average(deviations**2, weights=weights))
        else:
            weighted_rms = rms_deviation
        
        # Statistical significance of deviation
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        z_score = mean_deviation / (std_deviation / np.sqrt(len(deviations))) if std_deviation > 0 else 0
        
        # Agreement assessment
        if abs(z_score) < 1:
            agreement = "Excellent agreement"
        elif abs(z_score) < 2:
            agreement = "Good agreement"
        elif abs(z_score) < 3:
            agreement = "Moderate agreement"
        else:
            agreement = "Poor agreement - significant tension"
        
        return {
            'fitted_alpha0': fitted_alpha0,
            'fitted_alphap': fitted_alphap,
            'canonical_alpha0': canonical_alpha0,
            'canonical_alphap': canonical_alphap,
            'rms_deviation': rms_deviation,
            'weighted_rms_deviation': weighted_rms,
            'mean_deviation': mean_deviation,
            'std_deviation': std_deviation,
            'z_score': z_score,
            'agreement': agreement,
            'deviations': deviations,
            'y_fitted': y_fitted,
            'y_canonical': y_canonical
        }
    
    def generate_validation_report(self, predictions: pd.DataFrame, 
                                 residuals: np.ndarray, fitted_values: np.ndarray,
                                 fitted_alpha0: float, fitted_alphap: float) -> str:
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Predicted missing J states
        residuals : np.ndarray
            Fit residuals
        fitted_values : np.ndarray
            Fitted values
        fitted_alpha0 : float
            Fitted intercept
        fitted_alphap : float
            Fitted slope
            
        Returns:
        --------
        str
            Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append("VALIDATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # 1. PDG Cross-check Results
        report.append("1. PDG CROSS-CHECK NEAR PREDICTIONS")
        report.append("-" * 40)
        
        crosscheck_results = self.pdg_crosscheck_predictions(predictions)
        
        # Summary statistics
        n_predictions = len(predictions)
        n_confirmations = len(crosscheck_results[crosscheck_results['match_type'] == 'confirmation'])
        n_near_misses = len(crosscheck_results[crosscheck_results['match_type'] == 'near-miss'])
        n_gaps = len(crosscheck_results[crosscheck_results['match_type'] == 'genuine_gap'])
        
        report.append(f"Total predictions: {n_predictions}")
        report.append(f"Confirmations: {n_confirmations}")
        report.append(f"Near-misses: {n_near_misses}")
        report.append(f"Genuine gaps: {n_gaps}")
        report.append("")
        
        # Detailed results table
        report.append("Detailed cross-check results:")
        for _, result in crosscheck_results.iterrows():
            if result['is_best_match']:
                report.append(f"  J = {result['J_predicted']:.1f}:")
                report.append(f"    Predicted: {result['M_predicted_GeV']:.3f} ± {result['M_predicted_sigma_GeV']:.3f} GeV")
                if result['match_type'] != 'genuine_gap':
                    report.append(f"    Best match: {result['pdg_name']} ({result['pdg_M_GeV']:.3f} GeV)")
                    report.append(f"    Distance: {result['distance_GeV']:.3f} GeV")
                    report.append(f"    Status: {result['pdg_status']}")
                    report.append(f"    Type: {result['match_type']}")
                else:
                    report.append(f"    No nearby PDG candidates found")
                report.append("")
        
        # 2. Residual Quality Analysis
        report.append("2. RESIDUALS VS EXPERIMENTAL QUALITY")
        report.append("-" * 40)
        
        quality_results = self.residual_experimental_quality_analysis(residuals, fitted_values)
        
        report.append(f"Data points: {quality_results['n_total']} total, {quality_results['n_valid']} valid for analysis")
        report.append("")
        
        if quality_results['correlations']:
            report.append("Correlation analysis:")
            for metric, result in quality_results['correlations'].items():
                significance = "✓" if result['significant'] else "✗"
                report.append(f"  {metric}: r = {result['correlation']:.3f}, p = {result['p_value']:.4f} {significance}")
        
        if quality_results['anova_results']:
            anova = quality_results['anova_results']
            significance = "✓" if anova['significant'] else "✗"
            report.append(f"ANOVA by status groups: F = {anova['f_statistic']:.3f}, p = {anova['p_value']:.4f} {significance}")
        
        if quality_results['quality_summary']:
            summary = quality_results['quality_summary']
            report.append(f"Quality summary:")
            report.append(f"  Mean |residual| = {summary['mean_residual']:.4f}")
            report.append(f"  Mean width = {summary['mean_width']:.4f} GeV")
        
        report.append("")
        
        # 3. External Theory Comparison
        report.append("3. EXTERNAL THEORY OVERLAY")
        report.append("-" * 40)
        
        theory_results = self.external_theory_overlay(fitted_alpha0, fitted_alphap)
        
        report.append(f"Fitted trajectory: J = {theory_results['fitted_alpha0']:.3f} + {theory_results['fitted_alphap']:.3f} M²")
        report.append(f"Canonical theory: J = {theory_results['canonical_alpha0']:.3f} + {theory_results['canonical_alphap']:.3f} M²")
        report.append("")
        report.append(f"RMS deviation: {theory_results['rms_deviation']:.4f}")
        report.append(f"Weighted RMS: {theory_results['weighted_rms_deviation']:.4f}")
        report.append(f"Z-score: {theory_results['z_score']:.2f}")
        report.append(f"Agreement: {theory_results['agreement']}")
        report.append("")
        
        # 4. Overall Validation Assessment
        report.append("4. OVERALL VALIDATION ASSESSMENT")
        report.append("-" * 40)
        
        assessments = []
        
        # PDG cross-check assessment
        confirmation_rate = n_confirmations / n_predictions if n_predictions > 0 else 0
        if confirmation_rate >= 0.5:
            assessments.append("✓ Good PDG cross-check confirmation rate")
        elif confirmation_rate >= 0.2:
            assessments.append("⚠ Moderate PDG cross-check confirmation rate")
        else:
            assessments.append("✗ Low PDG cross-check confirmation rate")
        
        # Residual quality assessment
        if quality_results['correlations']:
            significant_correlations = sum(1 for r in quality_results['correlations'].values() if r['significant'])
            if significant_correlations > 0:
                assessments.append("✓ Residuals correlate with experimental quality - measurement effects identified")
            else:
                assessments.append("✗ Residuals independent of experimental quality - possible systematic effects")
        
        # Theory agreement assessment
        if abs(theory_results['z_score']) < 2:
            assessments.append("✓ Good agreement with canonical theory")
        elif abs(theory_results['z_score']) < 3:
            assessments.append("⚠ Moderate agreement with canonical theory")
        else:
            assessments.append("✗ Poor agreement with canonical theory - significant tension")
        
        for assessment in assessments:
            report.append(f"  {assessment}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_validation_analysis(self, predictions: pd.DataFrame, 
                               residuals: np.ndarray, fitted_values: np.ndarray,
                               fitted_alpha0: float, fitted_alphap: float,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot validation analysis results.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Predicted missing J states
        residuals : np.ndarray
            Fit residuals
        fitted_values : np.ndarray
            Fitted values
        fitted_alpha0 : float
            Fitted intercept
        fitted_alphap : float
            Fitted slope
        save_path : str, optional
            Path to save plot
            
        Returns:
        --------
        plt.Figure
            The generated plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: PDG Cross-check Results
        crosscheck_results = self.pdg_crosscheck_predictions(predictions)
        
        # Count match types
        match_counts = crosscheck_results['match_type'].value_counts()
        match_types = ['confirmation', 'near-miss', 'genuine_gap']
        counts = [match_counts.get(mt, 0) for mt in match_types]
        
        colors = ['green', 'orange', 'red']
        bars = ax1.bar(match_types, counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Number of Predictions')
        ax1.set_title('PDG Cross-check Results')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Plot 2: Residuals vs Experimental Quality
        quality_results = self.residual_experimental_quality_analysis(residuals, fitted_values)
        
        if self.width_col and self.width_col in self.data.columns:
            widths = self.data[self.width_col].values
            valid_mask = ~np.isnan(widths) & (widths > 0)
            
            if valid_mask.any():
                ax2.scatter(widths[valid_mask], np.abs(residuals[valid_mask]), alpha=0.7)
                ax2.set_xlabel('Resonance Width (GeV)')
                ax2.set_ylabel('|Residual|')
                ax2.set_title('Residuals vs Resonance Width')
                ax2.grid(True, alpha=0.3)
                
                # Add correlation line if significant
                if 'residual_vs_width' in quality_results['correlations']:
                    corr_result = quality_results['correlations']['residual_vs_width']
                    if corr_result['significant']:
                        ax2.text(0.05, 0.95, f'r = {corr_result["correlation"]:.3f}\np = {corr_result["p_value"]:.3f}',
                                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'No width data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Residuals vs Resonance Width')
        else:
            ax2.text(0.5, 0.5, 'No width data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Residuals vs Resonance Width')
        
        # Plot 3: External Theory Overlay
        theory_results = self.external_theory_overlay(fitted_alpha0, fitted_alphap)
        
        x_data = self.data[self.x_col].values
        y_data = self.data[self.y_col].values
        
        # Plot data points
        ax3.scatter(x_data, y_data, alpha=0.7, label='Data')
        
        # Plot fitted and canonical lines
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_fitted = fitted_alpha0 + fitted_alphap * x_range
        y_canonical = theory_results['canonical_alpha0'] + theory_results['canonical_alphap'] * x_range
        
        ax3.plot(x_range, y_fitted, 'r-', linewidth=2, label='Fitted')
        ax3.plot(x_range, y_canonical, 'b--', linewidth=2, label='Canonical Theory')
        
        ax3.set_xlabel('M² (GeV²)')
        ax3.set_ylabel('J')
        ax3.set_title('External Theory Overlay')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add RMS deviation info
        ax3.text(0.05, 0.95, f'RMS deviation: {theory_results["rms_deviation"]:.4f}',
                transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Validation Summary
        ax4.text(0.1, 0.9, f"PDG Cross-check:", transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.8, f"  Confirmations: {match_counts.get('confirmation', 0)}", transform=ax4.transAxes, fontsize=10)
        ax4.text(0.1, 0.7, f"  Near-misses: {match_counts.get('near-miss', 0)}", transform=ax4.transAxes, fontsize=10)
        ax4.text(0.1, 0.6, f"  Genuine gaps: {match_counts.get('genuine_gap', 0)}", transform=ax4.transAxes, fontsize=10)
        
        ax4.text(0.1, 0.4, f"Theory Agreement:", transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.3, f"  Z-score: {theory_results['z_score']:.2f}", transform=ax4.transAxes, fontsize=10)
        ax4.text(0.1, 0.2, f"  Assessment: {theory_results['agreement']}", transform=ax4.transAxes, fontsize=10)
        
        if quality_results['correlations']:
            significant_corr = sum(1 for r in quality_results['correlations'].values() if r['significant'])
            ax4.text(0.1, 0.0, f"Quality correlations: {significant_corr} significant", transform=ax4.transAxes, fontsize=10)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Validation Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation analysis plot saved to {save_path}")
        
        plt.show()
        
        return fig
