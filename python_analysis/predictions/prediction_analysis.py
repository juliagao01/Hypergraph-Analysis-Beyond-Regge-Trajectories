"""
Predictions & Cross-Checks: Prediction Analysis

Implements comprehensive prediction analysis including:
- Kappa parameter sweeps for width-systematic propagation
- Automated PDG neighborhood scanning
- Prediction stability analysis
- Cross-check validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from pathlib import Path

@dataclass
class PredictionResult:
    """Results from mass prediction."""
    j_value: float
    predicted_mass: float
    mass_uncertainty: float
    kappa: float
    confidence_level: float
    prediction_method: str

@dataclass
class CrossCheckResult:
    """Results from PDG cross-check."""
    j_value: float
    predicted_mass: float
    mass_uncertainty: float
    search_window: Tuple[float, float]
    n_sigma: float
    pdg_candidates: List[Dict[str, Any]]
    best_match: Optional[Dict[str, Any]]
    match_quality: str  # 'exact', 'within_n_sigma', 'no_match'

class PredictionAnalyzer:
    """
    Comprehensive prediction analysis for Regge trajectories.
    
    Provides:
    - Kappa parameter sweeps for systematic uncertainty propagation
    - Automated PDG neighborhood scanning
    - Prediction stability analysis
    - Cross-check validation
    """
    
    def __init__(self):
        """Initialize prediction analyzer."""
        self.predictions = {}
        self.cross_check_results = {}
        self.kappa_sweep_results = {}
        
    def predict_masses(self, 
                      j_values: np.ndarray,
                      fit_parameters: np.ndarray,
                      parameter_covariance: np.ndarray,
                      kappa: float = 0.25,
                      confidence_level: float = 0.68) -> List[PredictionResult]:
        """
        Predict masses for given J values.
        
        Parameters:
        -----------
        j_values : np.ndarray
            J values to predict masses for
        fit_parameters : np.ndarray
            Fitted parameters [α₀, α']
        parameter_covariance : np.ndarray
            Parameter covariance matrix
        kappa : float
            Width-to-uncertainty conversion factor
        confidence_level : float
            Confidence level for uncertainty calculation
            
        Returns:
        --------
        List[PredictionResult]
            Prediction results
        """
        print(f"Predicting masses for {len(j_values)} J values...")
        
        alpha0, alphap = fit_parameters
        predictions = []
        
        for j in j_values:
            # Predict mass: M² = (J - α₀) / α'
            predicted_m2 = (j - alpha0) / alphap
            
            if predicted_m2 <= 0:
                warnings.warn(f"Negative M² predicted for J={j}: {predicted_m2}")
                continue
            
            predicted_mass = np.sqrt(predicted_m2)
            
            # Propagate uncertainties
            # ∂M/∂α₀ = -1 / (2α'√M²)
            # ∂M/∂α' = -(J - α₀) / (2α'²√M²)
            
            dm_dalpha0 = -1 / (2 * alphap * predicted_mass)
            dm_dalphap = -(j - alpha0) / (2 * alphap**2 * predicted_mass)
            
            # Covariance contribution
            mass_variance = (
                dm_dalpha0**2 * parameter_covariance[0, 0] +
                dm_dalphap**2 * parameter_covariance[1, 1] +
                2 * dm_dalpha0 * dm_dalphap * parameter_covariance[0, 1]
            )
            
            mass_uncertainty = np.sqrt(mass_variance)
            
            # Add systematic uncertainty from kappa
            if kappa > 0:
                systematic_uncertainty = kappa * predicted_mass * 0.1  # Rough estimate
                total_uncertainty = np.sqrt(mass_uncertainty**2 + systematic_uncertainty**2)
            else:
                total_uncertainty = mass_uncertainty
            
            # Scale uncertainty to confidence level
            if confidence_level == 0.68:
                uncertainty_scale = 1.0
            elif confidence_level == 0.95:
                uncertainty_scale = 1.96
            elif confidence_level == 0.99:
                uncertainty_scale = 2.58
            else:
                # Use normal distribution quantile
                uncertainty_scale = stats.norm.ppf((1 + confidence_level) / 2)
            
            scaled_uncertainty = total_uncertainty * uncertainty_scale
            
            prediction = PredictionResult(
                j_value=j,
                predicted_mass=predicted_mass,
                mass_uncertainty=scaled_uncertainty,
                kappa=kappa,
                confidence_level=confidence_level,
                prediction_method='linear_regression'
            )
            
            predictions.append(prediction)
        
        print(f"Generated {len(predictions)} mass predictions")
        return predictions
    
    def kappa_parameter_sweep(self, 
                            j_values: np.ndarray,
                            fit_parameters: np.ndarray,
                            parameter_covariance: np.ndarray,
                            kappa_values: List[float] = None,
                            confidence_level: float = 0.68) -> Dict[str, Any]:
        """
        Perform kappa parameter sweep for systematic uncertainty analysis.
        
        Parameters:
        -----------
        j_values : np.ndarray
            J values to predict masses for
        fit_parameters : np.ndarray
            Fitted parameters [α₀, α']
        parameter_covariance : np.ndarray
            Parameter covariance matrix
        kappa_values : List[float]
            Kappa values to test
        confidence_level : float
            Confidence level for uncertainty calculation
            
        Returns:
        --------
        Dict[str, Any]
            Results from kappa sweep
        """
        print("Performing kappa parameter sweep...")
        
        if kappa_values is None:
            kappa_values = [0.0, 0.15, 0.25, 0.4]
        
        sweep_results = {}
        prediction_tables = {}
        
        for kappa in kappa_values:
            predictions = self.predict_masses(
                j_values, fit_parameters, parameter_covariance, 
                kappa, confidence_level
            )
            
            sweep_results[f'kappa_{kappa}'] = predictions
            
            # Create prediction table
            table_data = []
            for pred in predictions:
                table_data.append({
                    'J': pred.j_value,
                    'Predicted_Mass_GeV': pred.predicted_mass,
                    'Mass_Uncertainty_GeV': pred.mass_uncertainty,
                    'M2_GeV2': pred.predicted_mass**2,
                    'M2_Uncertainty_GeV2': 2 * pred.predicted_mass * pred.mass_uncertainty
                })
            
            prediction_tables[f'kappa_{kappa}'] = pd.DataFrame(table_data)
        
        # Analyze stability
        stability_analysis = self._analyze_kappa_stability(sweep_results)
        
        results = {
            'kappa_values': kappa_values,
            'predictions': sweep_results,
            'prediction_tables': prediction_tables,
            'stability_analysis': stability_analysis
        }
        
        self.kappa_sweep_results = results
        return results
    
    def _analyze_kappa_stability(self, sweep_results: Dict[str, List[PredictionResult]]) -> Dict[str, Any]:
        """
        Analyze stability of predictions across kappa values.
        
        Parameters:
        -----------
        sweep_results : Dict[str, List[PredictionResult]]
            Results from kappa sweep
            
        Returns:
        --------
        Dict[str, Any]
            Stability analysis results
        """
        kappa_values = []
        max_shifts = []
        mean_shifts = []
        stability_flags = []
        
        # Get reference predictions (kappa = 0)
        if 'kappa_0.0' in sweep_results:
            ref_predictions = {pred.j_value: pred.predicted_mass 
                             for pred in sweep_results['kappa_0.0']}
        else:
            # Use first kappa as reference
            first_kappa = list(sweep_results.keys())[0]
            ref_predictions = {pred.j_value: pred.predicted_mass 
                             for pred in sweep_results[first_kappa]}
        
        for kappa_key, predictions in sweep_results.items():
            kappa = float(kappa_key.split('_')[1])
            kappa_values.append(kappa)
            
            # Calculate shifts relative to reference
            shifts = []
            for pred in predictions:
                if pred.j_value in ref_predictions:
                    shift = abs(pred.predicted_mass - ref_predictions[pred.j_value])
                    shifts.append(shift)
            
            if shifts:
                max_shift = max(shifts)
                mean_shift = np.mean(shifts)
                max_shifts.append(max_shift)
                mean_shifts.append(mean_shift)
                
                # Flag as stable if max shift < 0.1 GeV
                stability_flags.append(max_shift < 0.1)
            else:
                max_shifts.append(0.0)
                mean_shifts.append(0.0)
                stability_flags.append(True)
        
        return {
            'kappa_values': kappa_values,
            'max_shifts': max_shifts,
            'mean_shifts': mean_shifts,
            'stability_flags': stability_flags,
            'overall_stable': all(stability_flags),
            'max_shift_overall': max(max_shifts) if max_shifts else 0.0
        }
    
    def automated_pdg_neighborhood_scan(self, 
                                      predictions: List[PredictionResult],
                                      pdg_data: pd.DataFrame,
                                      n_sigma_range: Tuple[float, float] = (1.5, 2.0),
                                      n_sigma_step: float = 0.1) -> Dict[str, List[CrossCheckResult]]:
        """
        Perform automated PDG neighborhood scanning.
        
        Parameters:
        -----------
        predictions : List[PredictionResult]
            Mass predictions
        pdg_data : pd.DataFrame
            PDG particle data
        n_sigma_range : Tuple[float, float]
            Range of n_sigma values to test
        n_sigma_step : float
            Step size for n_sigma values
            
        Returns:
        --------
        Dict[str, List[CrossCheckResult]]
            Cross-check results for different n_sigma values
        """
        print("Performing automated PDG neighborhood scan...")
        
        n_sigma_values = np.arange(n_sigma_range[0], n_sigma_range[1] + n_sigma_step, n_sigma_step)
        scan_results = {}
        
        for n_sigma in n_sigma_values:
            cross_check_results = []
            
            for pred in predictions:
                # Define search window
                mass_min = pred.predicted_mass - n_sigma * pred.mass_uncertainty
                mass_max = pred.predicted_mass + n_sigma * pred.mass_uncertainty
                
                # Find PDG candidates within window
                candidates = pdg_data[
                    (pdg_data['MassGeV'] >= mass_min) & 
                    (pdg_data['MassGeV'] <= mass_max)
                ].copy()
                
                # Calculate match quality for each candidate
                candidate_matches = []
                for _, candidate in candidates.iterrows():
                    mass_diff = abs(candidate['MassGeV'] - pred.predicted_mass)
                    sigma_diff = mass_diff / pred.mass_uncertainty
                    
                    candidate_matches.append({
                        'pdg_id': candidate.get('PDG_ID', 0),
                        'name': candidate.get('Name', ''),
                        'mass_gev': candidate['MassGeV'],
                        'mass_uncertainty': candidate.get('MassSigmaGeV', 0),
                        'j': candidate.get('J', 0),
                        'status': candidate.get('Status', ''),
                        'family': candidate.get('Family', ''),
                        'mass_difference': mass_diff,
                        'sigma_difference': sigma_diff,
                        'match_quality': self._assess_match_quality(sigma_diff, candidate.get('Status', ''))
                    })
                
                # Sort candidates by match quality
                candidate_matches.sort(key=lambda x: x['sigma_difference'])
                
                # Determine overall match quality
                if not candidate_matches:
                    match_quality = 'no_match'
                    best_match = None
                elif candidate_matches[0]['sigma_difference'] <= 1.0:
                    match_quality = 'exact'
                    best_match = candidate_matches[0]
                else:
                    match_quality = 'within_n_sigma'
                    best_match = candidate_matches[0]
                
                cross_check_result = CrossCheckResult(
                    j_value=pred.j_value,
                    predicted_mass=pred.predicted_mass,
                    mass_uncertainty=pred.mass_uncertainty,
                    search_window=(mass_min, mass_max),
                    n_sigma=n_sigma,
                    pdg_candidates=candidate_matches,
                    best_match=best_match,
                    match_quality=match_quality
                )
                
                cross_check_results.append(cross_check_result)
            
            scan_results[f'n_sigma_{n_sigma:.1f}'] = cross_check_results
        
        self.cross_check_results = scan_results
        return scan_results
    
    def _assess_match_quality(self, sigma_diff: float, status: str) -> str:
        """
        Assess quality of PDG match.
        
        Parameters:
        -----------
        sigma_diff : float
            Difference in standard deviations
        status : str
            PDG status
            
        Returns:
        --------
        str
            Match quality assessment
        """
        if sigma_diff <= 0.5:
            quality = 'excellent'
        elif sigma_diff <= 1.0:
            quality = 'good'
        elif sigma_diff <= 2.0:
            quality = 'fair'
        else:
            quality = 'poor'
        
        # Adjust for PDG status
        if status in ['***', '**']:
            quality += '_established'
        elif status in ['*']:
            quality += '_tentative'
        else:
            quality += '_uncertain'
        
        return quality
    
    def create_prediction_summary_table(self, 
                                      predictions: List[PredictionResult],
                                      cross_check_results: Optional[List[CrossCheckResult]] = None) -> pd.DataFrame:
        """
        Create summary table of predictions and cross-checks.
        
        Parameters:
        -----------
        predictions : List[PredictionResult]
            Mass predictions
        cross_check_results : Optional[List[CrossCheckResult]]
            Cross-check results
            
        Returns:
        --------
        pd.DataFrame
            Summary table
        """
        table_data = []
        
        for i, pred in enumerate(predictions):
            row = {
                'J': pred.j_value,
                'Predicted_Mass_GeV': pred.predicted_mass,
                'Mass_Uncertainty_GeV': pred.mass_uncertainty,
                'M2_GeV2': pred.predicted_mass**2,
                'M2_Uncertainty_GeV2': 2 * pred.predicted_mass * pred.mass_uncertainty,
                'Kappa': pred.kappa,
                'Confidence_Level': pred.confidence_level
            }
            
            # Add cross-check information if available
            if cross_check_results and i < len(cross_check_results):
                cross_check = cross_check_results[i]
                row.update({
                    'Search_Window_Min_GeV': cross_check.search_window[0],
                    'Search_Window_Max_GeV': cross_check.search_window[1],
                    'N_Sigma': cross_check.n_sigma,
                    'PDG_Candidates_Count': len(cross_check.pdg_candidates),
                    'Match_Quality': cross_check.match_quality,
                    'Best_Match_Name': cross_check.best_match['name'] if cross_check.best_match else 'None',
                    'Best_Match_Mass_GeV': cross_check.best_match['mass_gev'] if cross_check.best_match else None,
                    'Best_Match_Sigma_Diff': cross_check.best_match['sigma_difference'] if cross_check.best_match else None
                })
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def analyze_prediction_stability(self, 
                                   kappa_sweep_results: Dict[str, Any],
                                   cross_check_results: Dict[str, List[CrossCheckResult]]) -> Dict[str, Any]:
        """
        Analyze stability of predictions across different parameters.
        
        Parameters:
        -----------
        kappa_sweep_results : Dict[str, Any]
            Results from kappa parameter sweep
        cross_check_results : Dict[str, List[CrossCheckResult]]
            Results from PDG cross-checks
            
        Returns:
        --------
        Dict[str, Any]
            Stability analysis results
        """
        print("Analyzing prediction stability...")
        
        stability_analysis = {
            'kappa_stability': kappa_sweep_results.get('stability_analysis', {}),
            'cross_check_stability': {},
            'overall_assessment': {}
        }
        
        # Analyze cross-check stability across n_sigma values
        if cross_check_results:
            n_sigma_values = []
            match_rates = []
            exact_match_rates = []
            
            for n_sigma_key, results in cross_check_results.items():
                n_sigma = float(n_sigma_key.split('_')[1])
                n_sigma_values.append(n_sigma)
                
                total_predictions = len(results)
                matches = sum(1 for r in results if r.match_quality != 'no_match')
                exact_matches = sum(1 for r in results if r.match_quality == 'exact')
                
                match_rate = matches / total_predictions if total_predictions > 0 else 0
                exact_match_rate = exact_matches / total_predictions if total_predictions > 0 else 0
                
                match_rates.append(match_rate)
                exact_match_rates.append(exact_match_rate)
            
            stability_analysis['cross_check_stability'] = {
                'n_sigma_values': n_sigma_values,
                'match_rates': match_rates,
                'exact_match_rates': exact_match_rates,
                'optimal_n_sigma': n_sigma_values[np.argmax(match_rates)] if match_rates else None
            }
        
        # Overall assessment
        kappa_stable = stability_analysis['kappa_stability'].get('overall_stable', False)
        max_kappa_shift = stability_analysis['kappa_stability'].get('max_shift_overall', 0.0)
        
        stability_analysis['overall_assessment'] = {
            'kappa_stable': kappa_stable,
            'max_kappa_shift_gev': max_kappa_shift,
            'predictions_reliable': kappa_stable and max_kappa_shift < 0.1,
            'recommendations': self._generate_stability_recommendations(
                kappa_stable, max_kappa_shift, stability_analysis.get('cross_check_stability', {})
            )
        }
        
        return stability_analysis
    
    def _generate_stability_recommendations(self, 
                                          kappa_stable: bool,
                                          max_kappa_shift: float,
                                          cross_check_stability: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on stability analysis.
        
        Parameters:
        -----------
        kappa_stable : bool
            Whether predictions are stable across kappa values
        max_kappa_shift : float
            Maximum shift in predictions across kappa values
        cross_check_stability : Dict[str, Any]
            Cross-check stability analysis
            
        Returns:
        --------
        List[str]
            List of recommendations
        """
        recommendations = []
        
        if not kappa_stable:
            recommendations.append("Monitor kappa sensitivity - predictions show significant variation")
        
        if max_kappa_shift > 0.05:
            recommendations.append("Consider systematic uncertainty in width-to-mass conversion")
        
        if cross_check_stability:
            optimal_n_sigma = cross_check_stability.get('optimal_n_sigma')
            if optimal_n_sigma:
                recommendations.append(f"Use n_sigma = {optimal_n_sigma:.1f} for optimal PDG matching")
        
        if not recommendations:
            recommendations.append("Predictions are stable and reliable")
        
        return recommendations
    
    def create_prediction_visualizations(self, 
                                       predictions: List[PredictionResult],
                                       cross_check_results: Optional[List[CrossCheckResult]] = None,
                                       output_dir: str = "prediction_plots") -> Dict[str, str]:
        """
        Create visualizations for predictions and cross-checks.
        
        Parameters:
        -----------
        predictions : List[PredictionResult]
            Mass predictions
        cross_check_results : Optional[List[CrossCheckResult]]
            Cross-check results
        output_dir : str
            Directory for output plots
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping plot type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plot_files = {}
        
        # 1. Prediction plot with uncertainties
        fig, ax = plt.subplots(figsize=(10, 6))
        
        j_values = [pred.j_value for pred in predictions]
        masses = [pred.predicted_mass for pred in predictions]
        uncertainties = [pred.mass_uncertainty for pred in predictions]
        
        ax.errorbar(j_values, masses, yerr=uncertainties, fmt='o-', capsize=5, 
                   label='Predictions', alpha=0.7)
        
        # Add cross-check points if available
        if cross_check_results:
            cross_check_masses = []
            cross_check_j_values = []
            for result in cross_check_results:
                if result.best_match:
                    cross_check_masses.append(result.best_match['mass_gev'])
                    cross_check_j_values.append(result.j_value)
            
            if cross_check_masses:
                ax.scatter(cross_check_j_values, cross_check_masses, 
                          color='red', s=100, marker='s', label='PDG Matches', zorder=5)
        
        ax.set_xlabel('J')
        ax.set_ylabel('Mass (GeV)')
        ax.set_title('Regge Trajectory Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_files['predictions'] = str(output_path / 'mass_predictions.png')
        plt.savefig(plot_files['predictions'], dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Kappa stability plot (if available)
        if self.kappa_sweep_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            stability = self.kappa_sweep_results['stability_analysis']
            kappa_values = stability['kappa_values']
            max_shifts = stability['max_shifts']
            
            ax.plot(kappa_values, max_shifts, 'o-', label='Max Mass Shift')
            ax.axhline(y=0.1, color='red', linestyle='--', label='Stability Threshold (0.1 GeV)')
            
            ax.set_xlabel('Kappa')
            ax.set_ylabel('Max Mass Shift (GeV)')
            ax.set_title('Prediction Stability vs Kappa')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_files['kappa_stability'] = str(output_path / 'kappa_stability.png')
            plt.savefig(plot_files['kappa_stability'], dpi=300, bbox_inches='tight')
            plt.close()
        
        return plot_files

if __name__ == "__main__":
    print("Prediction Analysis")
    print("Use this module for mass predictions and PDG cross-checks")
