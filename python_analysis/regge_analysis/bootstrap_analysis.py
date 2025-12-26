"""
Bootstrap Analysis for Regge Trajectory Fits

Implements bootstrap resampling and leave-one-out validation
for robust uncertainty estimation and model validation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

class BootstrapAnalyzer:
    """
    Performs bootstrap analysis on Regge trajectory fits.
    
    Implements:
    - Bootstrap resampling with uncertainty propagation
    - Leave-one-out cross-validation
    - Robust parameter estimation
    """
    
    def __init__(self, data: pd.DataFrame, x_col: str = 'M2_GeV2', 
                 y_col: str = 'J', x_err_col: str = 'M2_sigma_GeV2',
                 width_col: Optional[str] = 'width_GeV', kappa: float = 0.25):
        """
        Initialize bootstrap analyzer.
        
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
        width_col : str, optional
            Column name for resonance widths
        kappa : float
            Factor for width-to-uncertainty conversion
        """
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.x_err_col = x_err_col
        self.width_col = width_col
        self.kappa = kappa
        
        # Clean data
        self._clean_data()
        
    def _clean_data(self):
        """Remove rows with missing data."""
        mask = (
            self.data[self.x_col].notna() & 
            self.data[self.y_col].notna() &
            (self.data[self.x_col] > 0) &
            (self.data[self.y_col] >= 0)
        )
        
        if self.x_err_col in self.data.columns:
            mask &= self.data[self.x_err_col].notna()
            
        self.data = self.data[mask].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError("No valid data points after cleaning")
    
    def _add_width_uncertainty(self) -> np.ndarray:
        """
        Add width-based systematic uncertainty to mass uncertainties.
        
        Returns:
        --------
        np.ndarray
            Combined uncertainties
        """
        base_uncertainties = self.data[self.x_err_col].values
        
        if self.width_col and self.width_col in self.data.columns:
            widths = self.data[self.width_col].values
            width_uncertainties = self.kappa * np.where(
                np.isnan(widths), 0.0, widths
            )
            
            # Combine uncertainties in quadrature
            combined_uncertainties = np.sqrt(
                base_uncertainties**2 + width_uncertainties**2
            )
        else:
            combined_uncertainties = base_uncertainties
            
        return combined_uncertainties
    
    def bootstrap_sample(self, n_bootstrap: int = 1000, 
                        sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate bootstrap samples and fit parameters.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap iterations
        sample_size : int, optional
            Size of each bootstrap sample (default: same as data)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of fit results from bootstrap samples
        """
        if sample_size is None:
            sample_size = len(self.data)
            
        bootstrap_results = []
        
        # Get combined uncertainties
        uncertainties = self._add_width_uncertainty()
        
        print(f"Generating {n_bootstrap} bootstrap samples...")
        
        for i in tqdm(range(n_bootstrap)):
            # Generate bootstrap sample indices
            indices = np.random.choice(
                len(self.data), 
                size=sample_size, 
                replace=True
            )
            
            # Create bootstrap sample
            bootstrap_data = self.data.iloc[indices].copy()
            
            # Add noise based on uncertainties
            x_noise = np.random.normal(0, uncertainties[indices])
            bootstrap_data[self.x_col] += x_noise
            
            # Fit linear model to bootstrap sample
            try:
                # Simple linear fit
                x_boot = bootstrap_data[self.x_col].values
                y_boot = bootstrap_data[self.y_col].values
                
                # Add constant term for intercept
                X = np.column_stack([np.ones_like(x_boot), x_boot])
                
                # Weighted least squares
                weights = 1.0 / (uncertainties[indices] ** 2)
                W = np.diag(weights)
                
                # Solve weighted normal equations
                beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y_boot
                
                # Calculate covariance matrix
                residuals = y_boot - X @ beta
                sigma2 = np.sum(weights * residuals**2) / (len(x_boot) - 2)
                cov_matrix = sigma2 * np.linalg.inv(X.T @ W @ X)
                
                bootstrap_results.append({
                    'alpha0': beta[0],
                    'alphap': beta[1],
                    'alpha0_err': np.sqrt(cov_matrix[0, 0]),
                    'alphap_err': np.sqrt(cov_matrix[1, 1]),
                    'cov_matrix': cov_matrix,
                    'chi2': np.sum(weights * residuals**2),
                    'dof': len(x_boot) - 2,
                    'sample_indices': indices
                })
                
            except np.linalg.LinAlgError:
                # Skip singular matrices
                continue
        
        print(f"Successfully generated {len(bootstrap_results)} bootstrap fits")
        return bootstrap_results
    
    def analyze_bootstrap_results(self, bootstrap_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze bootstrap results to get robust parameter estimates.
        
        Parameters:
        -----------
        bootstrap_results : List[Dict[str, Any]]
            Results from bootstrap sampling
            
        Returns:
        --------
        Dict[str, Any]
            Bootstrap analysis summary
        """
        if not bootstrap_results:
            raise ValueError("No bootstrap results to analyze")
        
        # Extract parameters
        alpha0_values = [r['alpha0'] for r in bootstrap_results]
        alphap_values = [r['alphap'] for r in bootstrap_results]
        chi2_values = [r['chi2'] for r in bootstrap_results]
        
        # Calculate statistics
        alpha0_mean = np.mean(alpha0_values)
        alpha0_std = np.std(alpha0_values)
        alpha0_ci = np.percentile(alpha0_values, [2.5, 97.5])
        
        alphap_mean = np.mean(alphap_values)
        alphap_std = np.std(alphap_values)
        alphap_ci = np.percentile(alphap_values, [2.5, 97.5])
        
        chi2_mean = np.mean(chi2_values)
        chi2_std = np.std(chi2_values)
        
        # Calculate correlation
        correlation = np.corrcoef(alpha0_values, alphap_values)[0, 1]
        
        # Bootstrap covariance matrix
        bootstrap_cov = np.cov(alpha0_values, alphap_values)
        
        return {
            'alpha0': {
                'mean': alpha0_mean,
                'std': alpha0_std,
                'ci_95': alpha0_ci,
                'bias': alpha0_mean - alpha0_values[0]  # Bias relative to first fit
            },
            'alphap': {
                'mean': alphap_mean,
                'std': alphap_std,
                'ci_95': alphap_ci,
                'bias': alphap_mean - alphap_values[0]
            },
            'chi2': {
                'mean': chi2_mean,
                'std': chi2_std
            },
            'correlation': correlation,
            'bootstrap_covariance': bootstrap_cov,
            'n_bootstrap': len(bootstrap_results)
        }
    
    def leave_one_out_validation(self) -> Dict[str, Any]:
        """
        Perform leave-one-out cross-validation.
        
        Returns:
        --------
        Dict[str, Any]
            LOO validation results
        """
        loo = LeaveOneOut()
        predictions = []
        residuals = []
        
        x_data = self.data[self.x_col].values
        y_data = self.data[self.y_col].values
        uncertainties = self._add_width_uncertainty()
        
        print("Performing leave-one-out validation...")
        
        for train_idx, test_idx in tqdm(loo.split(x_data)):
            # Split data
            x_train, x_test = x_data[train_idx], x_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]
            w_train = 1.0 / (uncertainties[train_idx] ** 2)
            
            # Fit model on training data
            X_train = np.column_stack([np.ones_like(x_train), x_train])
            W_train = np.diag(w_train)
            
            try:
                beta = np.linalg.inv(X_train.T @ W_train @ X_train) @ X_train.T @ W_train @ y_train
                
                # Predict on test point
                X_test = np.array([[1, x_test[0]]])
                y_pred = X_test @ beta
                
                predictions.append(y_pred[0])
                residuals.append(y_test[0] - y_pred[0])
                
            except np.linalg.LinAlgError:
                predictions.append(np.nan)
                residuals.append(np.nan)
        
        # Calculate validation metrics
        valid_mask = ~np.isnan(predictions)
        predictions = np.array(predictions)[valid_mask]
        residuals = np.array(residuals)[valid_mask]
        y_valid = y_data[valid_mask]
        
        # Mean squared error
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        
        # Mean absolute error
        mae = np.mean(np.abs(residuals))
        
        # R² score
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'predictions': predictions,
            'residuals': residuals,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_valid': len(predictions)
        }
    
    def plot_bootstrap_distributions(self, bootstrap_results: List[Dict[str, Any]], 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot bootstrap parameter distributions.
        
        Parameters:
        -----------
        bootstrap_results : List[Dict[str, Any]]
            Bootstrap results
        save_path : str, optional
            Path to save plot
            
        Returns:
        --------
        plt.Figure
            The generated plot
        """
        alpha0_values = [r['alpha0'] for r in bootstrap_results]
        alphap_values = [r['alphap'] for r in bootstrap_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # α₀ distribution
        ax1.hist(alpha0_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(alpha0_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(alpha0_values):.4f}')
        ax1.set_xlabel('α₀ (Intercept)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Bootstrap Distribution: α₀')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # α' distribution
        ax2.hist(alphap_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(alphap_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(alphap_values):.4f}')
        ax2.set_xlabel("α' (Slope)")
        ax2.set_ylabel('Frequency')
        ax2.set_title("Bootstrap Distribution: α'")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Joint distribution
        ax3.scatter(alpha0_values, alphap_values, alpha=0.6, s=20)
        ax3.set_xlabel('α₀')
        ax3.set_ylabel("α'")
        ax3.set_title('Joint Bootstrap Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Residuals from LOO validation
        loo_results = self.leave_one_out_validation()
        residuals = loo_results['residuals']
        
        ax4.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', label='Zero')
        ax4.set_xlabel('LOO Residuals')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Leave-One-Out Residuals\nRMSE: {loo_results["rmse"]:.4f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bootstrap distributions plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_methods(self, standard_fit: Dict[str, Any], 
                       bootstrap_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare standard fit with bootstrap results.
        
        Parameters:
        -----------
        standard_fit : Dict[str, Any]
            Results from standard fitting method
        bootstrap_results : List[Dict[str, Any]]
            Bootstrap results
            
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        bootstrap_analysis = self.analyze_bootstrap_results(bootstrap_results)
        
        comparison = {
            'Method': ['Standard Fit', 'Bootstrap Mean', 'Bootstrap Std'],
            'α₀': [
                standard_fit['alpha0'],
                bootstrap_analysis['alpha0']['mean'],
                bootstrap_analysis['alpha0']['std']
            ],
            'α₀_Error': [
                standard_fit['alpha0_err'],
                bootstrap_analysis['alpha0']['std'],
                None
            ],
            "α'": [
                standard_fit['alphap'],
                bootstrap_analysis['alphap']['mean'],
                bootstrap_analysis['alphap']['std']
            ],
            "α'_Error": [
                standard_fit['alphap_err'],
                bootstrap_analysis['alphap']['std'],
                None
            ],
            'χ²/dof': [
                standard_fit['chi2_dof'],
                bootstrap_analysis['chi2']['mean'] / standard_fit['dof'],
                None
            ]
        }
        
        return pd.DataFrame(comparison)
