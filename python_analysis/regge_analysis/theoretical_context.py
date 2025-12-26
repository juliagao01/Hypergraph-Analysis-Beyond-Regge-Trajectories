"""
Theoretical Context Analysis for Regge Trajectories

Connects empirical findings to established theoretical frameworks:
- Chew-Frautschi expectations and literature comparison
- Parity/naturality separation analysis
- Radial vs orbital trajectory distinction
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings

class TheoreticalContextAnalyzer:
    """
    Analyzes Regge trajectories in theoretical context.
    
    Implements:
    - Chew-Frautschi expectations and literature comparison
    - Parity/naturality separation analysis
    - Radial vs orbital trajectory analysis
    """
    
    def __init__(self, data: pd.DataFrame, x_col: str = 'M2_GeV2', 
                 y_col: str = 'J', x_err_col: str = 'M2_sigma_GeV2',
                 parity_col: str = 'parity', name_col: str = 'name',
                 width_col: Optional[str] = 'width_GeV'):
        """
        Initialize theoretical context analyzer.
        
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
        parity_col : str
            Column name for parity values
        name_col : str
            Column name for particle names
        width_col : str, optional
            Column name for resonance widths
        """
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.x_err_col = x_err_col
        self.parity_col = parity_col
        self.name_col = name_col
        self.width_col = width_col
        
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
            
        if self.parity_col in self.data.columns:
            mask &= self.data[self.parity_col].notna()
            
        self.data = self.data[mask].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError("No valid data points after cleaning")
            
        print(f"Using {len(self.data)} data points for theoretical analysis")
    
    def chew_frautschi_expectations(self, alphap: float, alphap_err: float, 
                                  particle_type: str = 'baryon') -> Dict[str, Any]:
        """
        Compare fitted α' with Chew-Frautschi expectations.
        
        Parameters:
        -----------
        alphap : float
            Fitted slope parameter
        alphap_err : float
            Uncertainty in slope parameter
        particle_type : str
            Type of particle ('meson', 'baryon', 'general')
            
        Returns:
        --------
        Dict containing Chew-Frautschi analysis results
        """
        # Literature reference ranges (GeV⁻²) from Chew-Frautschi phenomenology
        literature_ranges = {
            'meson': {
                'range': (0.7, 1.1),
                'reference': 'Chew & Frautschi (1961), Donnachie & Landshoff (1992)',
                'typical_value': 0.9
            },
            'baryon': {
                'range': (0.8, 1.2),
                'reference': 'Chew & Frautschi (1961), Capstick & Isgur (1986)',
                'typical_value': 1.0
            },
            'general': {
                'range': (0.6, 1.3),
                'reference': 'General Regge phenomenology',
                'typical_value': 0.9
            }
        }
        
        if particle_type not in literature_ranges:
            particle_type = 'general'
            
        ref_data = literature_ranges[particle_type]
        min_val, max_val = ref_data['range']
        typical_val = ref_data['typical_value']
        
        # Calculate z-scores
        z_score_to_typical = (alphap - typical_val) / alphap_err
        z_score_to_min = (alphap - min_val) / alphap_err
        z_score_to_max = (alphap - max_val) / alphap_err
        
        # Check if within expected range
        within_range = min_val <= alphap <= max_val
        
        # Distance to range center
        range_center = (min_val + max_val) / 2
        distance_to_center = alphap - range_center
        
        # Significance interpretation
        if abs(z_score_to_typical) < 1:
            significance = "Consistent with expectations"
        elif abs(z_score_to_typical) < 2:
            significance = "Moderately different from expectations"
        else:
            significance = "Significantly different from expectations"
        
        return {
            'particle_type': particle_type,
            'literature_range': ref_data['range'],
            'typical_value': typical_val,
            'reference': ref_data['reference'],
            'fitted_alphap': alphap,
            'fitted_alphap_err': alphap_err,
            'within_range': within_range,
            'z_score_to_typical': z_score_to_typical,
            'z_score_to_min': z_score_to_min,
            'z_score_to_max': z_score_to_max,
            'distance_to_center': distance_to_center,
            'significance': significance,
            'range_center': range_center
        }
    
    def parity_separation_analysis(self) -> Dict[str, Any]:
        """
        Analyze Regge trajectories by parity separation.
        
        Returns:
        --------
        Dict containing parity separation analysis results
        """
        if self.parity_col not in self.data.columns:
            raise ValueError(f"Parity column '{self.parity_col}' not found in data")
        
        # Split data by parity
        positive_parity = self.data[self.data[self.parity_col] == 1]
        negative_parity = self.data[self.data[self.parity_col] == -1]
        
        print(f"Parity separation: {len(positive_parity)} positive, {len(negative_parity)} negative parity states")
        
        results = {
            'positive_parity': None,
            'negative_parity': None,
            'comparison': None
        }
        
        # Fit positive parity trajectory
        if len(positive_parity) >= 3:
            pos_fit = self._fit_trajectory(positive_parity)
            results['positive_parity'] = pos_fit
        
        # Fit negative parity trajectory
        if len(negative_parity) >= 3:
            neg_fit = self._fit_trajectory(negative_parity)
            results['negative_parity'] = neg_fit
        
        # Compare slopes if both fits are available
        if results['positive_parity'] and results['negative_parity']:
            comparison = self._compare_slopes(
                results['positive_parity'], 
                results['negative_parity']
            )
            results['comparison'] = comparison
        
        return results
    
    def _fit_trajectory(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit linear trajectory to subset of data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Subset of particle data
            
        Returns:
        --------
        Dict containing fit results
        """
        x = data[self.x_col].values
        y = data[self.y_col].values
        weights = 1.0 / (data[self.x_err_col].values ** 2)
        
        # Add constant term for intercept
        X = np.column_stack([np.ones_like(x), x])
        
        # Weighted least squares
        W = np.diag(weights)
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        
        # Calculate covariance matrix
        residuals = y - X @ beta
        sigma2 = np.sum(weights * residuals**2) / (len(x) - 2)
        cov_matrix = sigma2 * np.linalg.inv(X.T @ W @ X)
        
        # Parameter uncertainties
        alpha0_err = np.sqrt(cov_matrix[0, 0])
        alphap_err = np.sqrt(cov_matrix[1, 1])
        
        # Goodness of fit
        chi2 = np.sum(weights * residuals**2)
        dof = len(x) - 2
        chi2_dof = chi2 / dof
        
        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'alpha0': beta[0],
            'alphap': beta[1],
            'alpha0_err': alpha0_err,
            'alphap_err': alphap_err,
            'cov_matrix': cov_matrix,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'n_points': len(x),
            'residuals': residuals
        }
    
    def _compare_slopes(self, fit1: Dict[str, Any], fit2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare slopes from two fits using statistical tests.
        
        Parameters:
        -----------
        fit1 : Dict[str, Any]
            First fit results
        fit2 : Dict[str, Any]
            Second fit results
            
        Returns:
        --------
        Dict containing slope comparison results
        """
        alphap1 = fit1['alphap']
        alphap2 = fit2['alphap']
        alphap1_err = fit1['alphap_err']
        alphap2_err = fit2['alphap_err']
        
        # Difference in slopes
        slope_diff = alphap1 - alphap2
        slope_diff_err = np.sqrt(alphap1_err**2 + alphap2_err**2)
        
        # Z-score for difference
        z_score = slope_diff / slope_diff_err
        
        # P-value for two-sided test
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Significance interpretation
        if p_value < 0.001:
            significance = "Highly significant difference"
        elif p_value < 0.01:
            significance = "Significant difference"
        elif p_value < 0.05:
            significance = "Moderately significant difference"
        else:
            significance = "No significant difference"
        
        return {
            'slope1': alphap1,
            'slope2': alphap2,
            'slope1_err': alphap1_err,
            'slope2_err': alphap2_err,
            'slope_difference': slope_diff,
            'slope_difference_err': slope_diff_err,
            'z_score': z_score,
            'p_value': p_value,
            'significance': significance
        }
    
    def radial_orbital_analysis(self, radial_assignments: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Analyze radial vs orbital trajectories.
        
        Parameters:
        -----------
        radial_assignments : Dict[str, int], optional
            Dictionary mapping particle names to radial excitation numbers
            
        Returns:
        --------
        Dict containing radial-orbital analysis results
        """
        if radial_assignments is None:
            # Try to infer radial assignments from particle names
            radial_assignments = self._infer_radial_assignments()
        
        if not radial_assignments:
            print("No radial assignments available - skipping radial-orbital analysis")
            return {'available': False}
        
        # Group particles by radial excitation number
        radial_groups = {}
        for name, radial_n in radial_assignments.items():
            if radial_n not in radial_groups:
                radial_groups[radial_n] = []
            radial_groups[radial_n].append(name)
        
        # Fit trajectories for each radial band
        radial_fits = {}
        for radial_n, particle_names in radial_groups.items():
            # Get data for this radial band
            mask = self.data[self.name_col].isin(particle_names)
            radial_data = self.data[mask]
            
            if len(radial_data) >= 3:
                fit = self._fit_trajectory(radial_data)
                radial_fits[radial_n] = {
                    'fit': fit,
                    'particles': particle_names,
                    'n_particles': len(radial_data)
                }
        
        # Compare slopes across radial bands
        slope_comparisons = {}
        radial_ns = sorted(radial_fits.keys())
        
        for i, n1 in enumerate(radial_ns):
            for n2 in radial_ns[i+1:]:
                if n1 in radial_fits and n2 in radial_fits:
                    comparison = self._compare_slopes(
                        radial_fits[n1]['fit'],
                        radial_fits[n2]['fit']
                    )
                    slope_comparisons[f"{n1}_vs_{n2}"] = comparison
        
        # Test for slope universality
        if len(radial_fits) >= 2:
            universality_test = self._test_slope_universality(radial_fits)
        else:
            universality_test = None
        
        return {
            'available': True,
            'radial_assignments': radial_assignments,
            'radial_fits': radial_fits,
            'slope_comparisons': slope_comparisons,
            'universality_test': universality_test
        }
    
    def _infer_radial_assignments(self) -> Dict[str, int]:
        """
        Infer radial excitation numbers from particle names.
        
        Returns:
        --------
        Dict mapping particle names to radial excitation numbers
        """
        assignments = {}
        
        for _, row in self.data.iterrows():
            name = row[self.name_col]
            
            # Common patterns for radial excitations
            if 'prime' in name.lower() or "'" in name:
                assignments[name] = 1
            elif 'double prime' in name.lower() or "''" in name:
                assignments[name] = 2
            elif any(char.isdigit() for char in name):
                # Look for numbers that might indicate excitation level
                import re
                numbers = re.findall(r'\d+', name)
                if numbers:
                    # Use the first number as radial excitation
                    assignments[name] = int(numbers[0])
                else:
                    assignments[name] = 0  # Ground state
            else:
                assignments[name] = 0  # Ground state
        
        return assignments
    
    def _test_slope_universality(self, radial_fits: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test whether slopes are universal across radial bands.
        
        Parameters:
        -----------
        radial_fits : Dict[int, Dict[str, Any]]
            Fits for different radial bands
            
        Returns:
        --------
        Dict containing universality test results
        """
        # Extract slopes and uncertainties
        slopes = []
        slope_errors = []
        weights = []
        
        for radial_n, fit_data in radial_fits.items():
            fit = fit_data['fit']
            slopes.append(fit['alphap'])
            slope_errors.append(fit['alphap_err'])
            weights.append(1.0 / (fit['alphap_err'] ** 2))
        
        # Weighted average slope
        weights = np.array(weights)
        weighted_avg = np.average(slopes, weights=weights)
        weighted_avg_err = np.sqrt(1.0 / np.sum(weights))
        
        # Chi-squared test for universality
        chi2 = np.sum(weights * (np.array(slopes) - weighted_avg) ** 2)
        dof = len(slopes) - 1
        chi2_dof = chi2 / dof if dof > 0 else 0
        
        # P-value for chi-squared test
        p_value = 1 - stats.chi2.cdf(chi2, dof) if dof > 0 else 1.0
        
        # Interpretation
        if p_value < 0.05:
            universality = "Rejected - slopes differ significantly"
        else:
            universality = "Not rejected - slopes are consistent"
        
        return {
            'weighted_average_slope': weighted_avg,
            'weighted_average_error': weighted_avg_err,
            'individual_slopes': slopes,
            'slope_errors': slope_errors,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'p_value': p_value,
            'universality': universality
        }
    
    def generate_theoretical_report(self, alphap: float, alphap_err: float, 
                                  particle_type: str = 'baryon') -> str:
        """
        Generate comprehensive theoretical context report.
        
        Parameters:
        -----------
        alphap : float
            Fitted slope parameter
        alphap_err : float
            Uncertainty in slope parameter
        particle_type : str
            Type of particle
            
        Returns:
        --------
        str
            Formatted theoretical report
        """
        report = []
        report.append("=" * 60)
        report.append("THEORETICAL CONTEXT ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # 1. Chew-Frautschi expectations
        report.append("1. CHEW-FRAUTSCHI EXPECTATIONS")
        report.append("-" * 30)
        
        cf_results = self.chew_frautschi_expectations(alphap, alphap_err, particle_type)
        
        report.append(f"Particle type: {cf_results['particle_type']}")
        report.append(f"Literature range: {cf_results['literature_range'][0]:.1f} - {cf_results['literature_range'][1]:.1f} GeV⁻²")
        report.append(f"Typical value: {cf_results['typical_value']:.1f} GeV⁻²")
        report.append(f"Reference: {cf_results['reference']}")
        report.append("")
        report.append(f"Fitted α' = {cf_results['fitted_alphap']:.4f} ± {cf_results['fitted_alphap_err']:.4f} GeV⁻²")
        report.append(f"Within expected range: {'Yes' if cf_results['within_range'] else 'No'}")
        report.append(f"Z-score vs typical: {cf_results['z_score_to_typical']:.2f}")
        report.append(f"Significance: {cf_results['significance']}")
        report.append("")
        
        # 2. Parity separation analysis
        report.append("2. PARITY SEPARATION ANALYSIS")
        report.append("-" * 30)
        
        parity_results = self.parity_separation_analysis()
        
        if parity_results['positive_parity']:
            pos_fit = parity_results['positive_parity']
            report.append(f"Positive parity (n={pos_fit['n_points']}):")
            report.append(f"  α' = {pos_fit['alphap']:.4f} ± {pos_fit['alphap_err']:.4f} GeV⁻²")
            report.append(f"  χ²/dof = {pos_fit['chi2_dof']:.3f}")
        
        if parity_results['negative_parity']:
            neg_fit = parity_results['negative_parity']
            report.append(f"Negative parity (n={neg_fit['n_points']}):")
            report.append(f"  α' = {neg_fit['alphap']:.4f} ± {neg_fit['alphap_err']:.4f} GeV⁻²")
            report.append(f"  χ²/dof = {neg_fit['chi2_dof']:.3f}")
        
        if parity_results['comparison']:
            comp = parity_results['comparison']
            report.append(f"Parity comparison:")
            report.append(f"  Δα' = {comp['slope_difference']:.4f} ± {comp['slope_difference_err']:.4f} GeV⁻²")
            report.append(f"  Z-score = {comp['z_score']:.2f}")
            report.append(f"  P-value = {comp['p_value']:.4f}")
            report.append(f"  Significance: {comp['significance']}")
        report.append("")
        
        # 3. Radial-orbital analysis
        report.append("3. RADIAL-ORBITAL TRAJECTORY ANALYSIS")
        report.append("-" * 30)
        
        radial_results = self.radial_orbital_analysis()
        
        if radial_results['available']:
            report.append("Radial excitation analysis:")
            for radial_n, fit_data in radial_results['radial_fits'].items():
                fit = fit_data['fit']
                report.append(f"  n={radial_n} (n={fit_data['n_particles']} particles):")
                report.append(f"    α' = {fit['alphap']:.4f} ± {fit['alphap_err']:.4f} GeV⁻²")
            
            if radial_results['universality_test']:
                ut = radial_results['universality_test']
                report.append(f"Slope universality test:")
                report.append(f"  Weighted average α' = {ut['weighted_average_slope']:.4f} ± {ut['weighted_average_error']:.4f} GeV⁻²")
                report.append(f"  χ²/dof = {ut['chi2_dof']:.3f}")
                report.append(f"  P-value = {ut['p_value']:.4f}")
                report.append(f"  Universality: {ut['universality']}")
        else:
            report.append("Radial excitation analysis not available")
        report.append("")
        
        # 4. Summary and implications
        report.append("4. THEORETICAL IMPLICATIONS")
        report.append("-" * 30)
        
        # Overall assessment
        assessments = []
        
        if cf_results['within_range']:
            assessments.append("✓ α' consistent with Chew-Frautschi expectations")
        else:
            assessments.append("✗ α' outside typical range - may indicate new physics or systematic effects")
        
        if parity_results['comparison'] and parity_results['comparison']['p_value'] > 0.05:
            assessments.append("✓ Parity trajectories consistent - no significant mixing")
        elif parity_results['comparison']:
            assessments.append("✗ Parity trajectories differ - possible mixing or systematics")
        
        if radial_results['available'] and radial_results['universality_test']:
            if radial_results['universality_test']['p_value'] > 0.05:
                assessments.append("✓ Slope universality across radial bands")
            else:
                assessments.append("✗ Slope universality rejected - radial dependence")
        
        for assessment in assessments:
            report.append(assessment)
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_theoretical_analysis(self, alphap: float, alphap_err: float,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot theoretical context analysis results.
        
        Parameters:
        -----------
        alphap : float
            Fitted slope parameter
        alphap_err : float
            Uncertainty in slope parameter
        save_path : str, optional
            Path to save plot
            
        Returns:
        --------
        plt.Figure
            The generated plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Chew-Frautschi expectations
        cf_results = self.chew_frautschi_expectations(alphap, alphap_err)
        
        # Literature ranges
        ranges = {
            'Mesons': (0.7, 1.1),
            'Baryons': (0.8, 1.2),
            'General': (0.6, 1.3)
        }
        
        y_pos = 0
        for particle_type, (min_val, max_val) in ranges.items():
            ax1.barh(y_pos, max_val - min_val, left=min_val, height=0.6, 
                    alpha=0.3, label=particle_type)
            ax1.text((min_val + max_val) / 2, y_pos, particle_type, 
                    ha='center', va='center')
            y_pos += 1
        
        # Fitted value
        ax1.errorbar(alphap, 1.5, xerr=alphap_err, fmt='o', capsize=5, 
                    capthick=2, markersize=8, color='red', label='Fitted α\'')
        
        ax1.set_xlabel("α' (GeV⁻²)")
        ax1.set_ylabel('Particle Type')
        ax1.set_title('Chew-Frautschi Expectations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parity separation
        parity_results = self.parity_separation_analysis()
        
        if parity_results['positive_parity'] and parity_results['negative_parity']:
            pos_fit = parity_results['positive_parity']
            neg_fit = parity_results['negative_parity']
            
            parities = ['Positive', 'Negative']
            slopes = [pos_fit['alphap'], neg_fit['alphap']]
            errors = [pos_fit['alphap_err'], neg_fit['alphap_err']]
            
            ax2.errorbar(parities, slopes, yerr=errors, fmt='o', capsize=5, 
                        capthick=2, markersize=8)
            ax2.set_ylabel("α' (GeV⁻²)")
            ax2.set_title('Parity Separation')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for parity separation', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parity Separation')
        
        # Plot 3: Radial-orbital analysis
        radial_results = self.radial_orbital_analysis()
        
        if radial_results['available'] and radial_results['radial_fits']:
            radial_ns = []
            slopes = []
            errors = []
            
            for radial_n, fit_data in radial_results['radial_fits'].items():
                radial_ns.append(f'n={radial_n}')
                slopes.append(fit_data['fit']['alphap'])
                errors.append(fit_data['fit']['alphap_err'])
            
            ax3.errorbar(radial_ns, slopes, yerr=errors, fmt='o', capsize=5, 
                        capthick=2, markersize=8)
            ax3.set_ylabel("α' (GeV⁻²)")
            ax3.set_title('Radial Excitation Analysis')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Radial analysis not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Radial Excitation Analysis')
        
        # Plot 4: Summary comparison
        ax4.text(0.1, 0.9, f"Fitted α' = {alphap:.4f} ± {alphap_err:.4f} GeV⁻²", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.8, f"Chew-Frautschi: {'✓' if cf_results['within_range'] else '✗'}", 
                transform=ax4.transAxes, fontsize=12)
        
        if parity_results['comparison']:
            comp = parity_results['comparison']
            ax4.text(0.1, 0.7, f"Parity separation: {'✓' if comp['p_value'] > 0.05 else '✗'}", 
                    transform=ax4.transAxes, fontsize=12)
        
        if radial_results['available'] and radial_results['universality_test']:
            ut = radial_results['universality_test']
            ax4.text(0.1, 0.6, f"Slope universality: {'✓' if ut['p_value'] > 0.05 else '✗'}", 
                    transform=ax4.transAxes, fontsize=12)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Theoretical Assessment Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Theoretical analysis plot saved to {save_path}")
        
        plt.show()
        
        return fig
