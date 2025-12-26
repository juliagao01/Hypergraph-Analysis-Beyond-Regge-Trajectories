"""
Uncertainty Propagation for Regge Trajectory Predictions

Handles uncertainty propagation from fit parameters to predictions
of missing J states and their mass uncertainties.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats

class UncertaintyPropagator:
    """
    Propagates uncertainties from Regge fit parameters to predictions.
    
    Implements error propagation for the inverse relation:
    M² = (J - α₀) / α'
    """
    
    def __init__(self, fit_result: Dict[str, Any]):
        """
        Initialize with fit results.
        
        Parameters:
        -----------
        fit_result : Dict[str, Any]
            Results from ReggeFitter containing parameters and covariance
        """
        self.fit_result = fit_result
        self.alpha0 = fit_result['alpha0']
        self.alphap = fit_result['alphap']
        self.alpha0_err = fit_result['alpha0_err']
        self.alphap_err = fit_result['alphap_err']
        self.cov_matrix = fit_result['cov_matrix']
        
    def predict_mass_squared(self, J: float) -> Tuple[float, float]:
        """
        Predict M² for a given J value with uncertainty.
        
        Uses the inverse relation: M² = (J - α₀) / α'
        
        Parameters:
        -----------
        J : float
            Spin value to predict for
            
        Returns:
        --------
        Tuple[float, float]
            (M², σ_M²) - predicted mass squared and its uncertainty
        """
        # Predicted M²
        M2_pred = (J - self.alpha0) / self.alphap
        
        # Partial derivatives for error propagation
        dM2_dalpha0 = -1.0 / self.alphap
        dM2_dalphap = -(J - self.alpha0) / (self.alphap ** 2)
        
        # Gradient vector
        grad = np.array([dM2_dalpha0, dM2_dalphap])
        
        # Variance propagation: var(M²) = grad^T * cov * grad
        var_M2 = grad.T @ self.cov_matrix @ grad
        
        # Ensure variance is positive
        var_M2 = max(var_M2, 0.0)
        
        return M2_pred, np.sqrt(var_M2)
    
    def predict_mass(self, J: float) -> Tuple[float, float]:
        """
        Predict mass M for a given J value with uncertainty.
        
        Parameters:
        -----------
        J : float
            Spin value to predict for
            
        Returns:
        --------
        Tuple[float, float]
            (M, σ_M) - predicted mass and its uncertainty
        """
        M2_pred, sigma_M2 = self.predict_mass_squared(J)
        
        if M2_pred <= 0:
            return np.nan, np.nan
        
        M_pred = np.sqrt(M2_pred)
        
        # Error propagation for M = sqrt(M²)
        # σ_M = σ_M² / (2 * M)
        sigma_M = sigma_M2 / (2 * M_pred)
        
        return M_pred, sigma_M
    
    def predict_missing_states(self, existing_J: List[float], 
                             J_range: Tuple[float, float] = (0.5, 10.0),
                             J_step: float = 0.5) -> pd.DataFrame:
        """
        Predict masses for missing J states.
        
        Parameters:
        -----------
        existing_J : List[float]
            List of existing J values
        J_range : Tuple[float, float]
            Range of J values to consider (min, max)
        J_step : float
            Step size for J values
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions for missing J states
        """
        # Generate all J values in range
        all_J = np.arange(J_range[0], J_range[1] + J_step, J_step)
        
        # Find missing J values
        existing_J_set = set(existing_J)
        missing_J = [J for J in all_J if J not in existing_J_set]
        
        # Make predictions
        predictions = []
        for J in missing_J:
            M_pred, sigma_M = self.predict_mass(J)
            
            if not np.isnan(M_pred):
                predictions.append({
                    'J': J,
                    'M_GeV': M_pred,
                    'M_sigma_GeV': sigma_M,
                    'M2_GeV2': M_pred ** 2,
                    'M2_sigma_GeV2': 2 * M_pred * sigma_M
                })
        
        return pd.DataFrame(predictions)
    
    def confidence_band(self, J_values: np.ndarray, confidence: float = 0.68) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence band for the Regge trajectory.
        
        Parameters:
        -----------
        J_values : np.ndarray
            J values to calculate confidence band for
        confidence : float
            Confidence level (default: 0.68 for 1σ)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (lower_bound, upper_bound) - confidence band bounds
        """
        # Calculate predicted M² values and uncertainties
        M2_pred = []
        M2_uncertainties = []
        
        for J in J_values:
            M2, sigma_M2 = self.predict_mass_squared(J)
            M2_pred.append(M2)
            M2_uncertainties.append(sigma_M2)
        
        M2_pred = np.array(M2_pred)
        M2_uncertainties = np.array(M2_uncertainties)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower_bound = M2_pred - z_score * M2_uncertainties
        upper_bound = M2_pred + z_score * M2_uncertainties
        
        return lower_bound, upper_bound
    
    def plot_predictions(self, data: pd.DataFrame, predictions: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data points and predictions with confidence bands.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Original data points
        predictions : pd.DataFrame
            Predicted missing states
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The generated plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: M² vs J
        # Original data
        ax1.errorbar(data['J'], data['M2_GeV2'], 
                    xerr=None, yerr=data['M2_sigma_GeV2'],
                    fmt='o', capsize=3, capthick=1, 
                    alpha=0.7, label='Data', color='blue')
        
        # Predictions
        ax1.errorbar(predictions['J'], predictions['M2_GeV2'],
                    xerr=None, yerr=predictions['M2_sigma_GeV2'],
                    fmt='s', capsize=3, capthick=1,
                    alpha=0.7, label='Predictions', color='red')
        
        # Confidence band
        J_range = np.linspace(min(data['J'].min(), predictions['J'].min()),
                             max(data['J'].max(), predictions['J'].max()), 100)
        lower_bound, upper_bound = self.confidence_band(J_range)
        
        ax1.fill_between(J_range, lower_bound, upper_bound, 
                        alpha=0.3, color='gray', label=f'{68}% Confidence Band')
        
        ax1.set_xlabel('J (Spin)')
        ax1.set_ylabel('M² (GeV²)')
        ax1.set_title('Regge Trajectory: M² vs J')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: M vs J
        # Original data
        data_M = np.sqrt(data['M2_GeV2'])
        data_M_err = data['M2_sigma_GeV2'] / (2 * data_M)
        
        ax2.errorbar(data['J'], data_M, 
                    xerr=None, yerr=data_M_err,
                    fmt='o', capsize=3, capthick=1, 
                    alpha=0.7, label='Data', color='blue')
        
        # Predictions
        ax2.errorbar(predictions['J'], predictions['M_GeV'],
                    xerr=None, yerr=predictions['M_sigma_GeV'],
                    fmt='s', capsize=3, capthick=1,
                    alpha=0.7, label='Predictions', color='red')
        
        # Confidence band for M
        M_range = np.sqrt(np.maximum(0, J_range * self.alphap + self.alpha0))
        M_lower = np.sqrt(np.maximum(0, lower_bound))
        M_upper = np.sqrt(np.maximum(0, upper_bound))
        
        ax2.fill_between(J_range, M_lower, M_upper, 
                        alpha=0.3, color='gray', label=f'{68}% Confidence Band')
        
        ax2.set_xlabel('J (Spin)')
        ax2.set_ylabel('M (GeV)')
        ax2.set_title('Regge Trajectory: M vs J')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_with_literature(self, literature_values: Dict[str, float]) -> pd.DataFrame:
        """
        Compare predictions with literature values.
        
        Parameters:
        -----------
        literature_values : Dict[str, float]
            Dictionary mapping J values to literature mass values
            
        Returns:
        --------
        pd.DataFrame
            Comparison table with predictions vs literature
        """
        comparisons = []
        
        for J, lit_mass in literature_values.items():
            pred_mass, pred_uncertainty = self.predict_mass(J)
            
            if not np.isnan(pred_mass):
                difference = pred_mass - lit_mass
                z_score = difference / pred_uncertainty if pred_uncertainty > 0 else np.nan
                
                comparisons.append({
                    'J': J,
                    'Literature_M_GeV': lit_mass,
                    'Predicted_M_GeV': pred_mass,
                    'Predicted_Uncertainty_GeV': pred_uncertainty,
                    'Difference_GeV': difference,
                    'Z_Score': z_score,
                    'Within_1sigma': abs(z_score) <= 1 if not np.isnan(z_score) else False
                })
        
        return pd.DataFrame(comparisons)
    
    def sensitivity_analysis(self, J_test: float = 3.5, 
                           param_variations: float = 0.1) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on predictions.
        
        Parameters:
        -----------
        J_test : float
            J value to test sensitivity for
        param_variations : float
            Fractional variation in parameters to test
            
        Returns:
        --------
        Dict[str, Any]
            Sensitivity analysis results
        """
        # Base prediction
        base_mass, base_uncertainty = self.predict_mass(J_test)
        
        # Vary α₀
        alpha0_variations = np.linspace(
            self.alpha0 * (1 - param_variations),
            self.alpha0 * (1 + param_variations),
            10
        )
        
        mass_variations_alpha0 = []
        for alpha0_var in alpha0_variations:
            # Create temporary fit result with varied parameter
            temp_result = self.fit_result.copy()
            temp_result['alpha0'] = alpha0_var
            temp_result['cov_matrix'] = self.cov_matrix  # Keep original covariance
            
            temp_propagator = UncertaintyPropagator(temp_result)
            mass_var, _ = temp_propagator.predict_mass(J_test)
            mass_variations_alpha0.append(mass_var)
        
        # Vary α'
        alphap_variations = np.linspace(
            self.alphap * (1 - param_variations),
            self.alphap * (1 + param_variations),
            10
        )
        
        mass_variations_alphap = []
        for alphap_var in alphap_variations:
            temp_result = self.fit_result.copy()
            temp_result['alphap'] = alphap_var
            temp_result['cov_matrix'] = self.cov_matrix
            
            temp_propagator = UncertaintyPropagator(temp_result)
            mass_var, _ = temp_propagator.predict_mass(J_test)
            mass_variations_alphap.append(mass_var)
        
        return {
            'J_test': J_test,
            'base_mass': base_mass,
            'base_uncertainty': base_uncertainty,
            'alpha0_variations': alpha0_variations,
            'mass_variations_alpha0': mass_variations_alpha0,
            'alphap_variations': alphap_variations,
            'mass_variations_alphap': mass_variations_alphap,
            'alpha0_sensitivity': np.std(mass_variations_alpha0),
            'alphap_sensitivity': np.std(mass_variations_alphap)
        }
