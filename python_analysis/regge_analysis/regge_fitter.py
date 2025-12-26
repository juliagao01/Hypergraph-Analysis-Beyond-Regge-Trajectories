"""
Regge Trajectory Fitter

Implements weighted linear regression and orthogonal distance regression (ODR)
for Regge trajectory analysis: J = α₀ + α' M²
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS
import warnings
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class ReggeFitter:
    """
    Fits Regge trajectories using multiple methods with uncertainty handling.
    
    Implements:
    - Weighted linear regression (WLS)
    - Orthogonal distance regression (ODR)
    - Standard least squares for comparison
    """
    
    def __init__(self, data: pd.DataFrame, x_col: str = 'M2_GeV2', 
                 y_col: str = 'J', x_err_col: str = 'M2_sigma_GeV2',
                 y_err_col: Optional[str] = None):
        """
        Initialize the Regge fitter.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing particle data
        x_col : str
            Column name for M² values
        y_col : str
            Column name for J (spin) values  
        x_err_col : str
            Column name for M² uncertainties
        y_err_col : str, optional
            Column name for J uncertainties (if available)
        """
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.x_err_col = x_err_col
        self.y_err_col = y_err_col
        
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
            
        if self.y_err_col and self.y_err_col in self.data.columns:
            mask &= self.data[self.y_err_col].notna()
            
        self.data = self.data[mask].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError("No valid data points after cleaning")
            
        print(f"Using {len(self.data)} data points for fitting")
        
    def linear_model(self, x: np.ndarray, alpha0: float, alphap: float) -> np.ndarray:
        """
        Linear Regge model: J = α₀ + α' M²
        
        Parameters:
        -----------
        x : np.ndarray
            M² values
        alpha0 : float
            Intercept parameter
        alphap : float
            Slope parameter (α')
            
        Returns:
        --------
        np.ndarray
            Predicted J values
        """
        return alpha0 + alphap * x
    
    def fit_wls(self) -> Dict[str, Any]:
        """
        Weighted least squares fit with uncertainties.
        
        Returns:
        --------
        Dict containing fit results
        """
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        
        # Use x uncertainties as weights (1/σ²)
        weights = 1.0 / (self.data[self.x_err_col].values ** 2)
        
        # Add constant term for intercept
        X = sm.add_constant(x)
        
        # Fit weighted least squares
        model = WLS(y, X, weights=weights)
        results = model.fit()
        
        # Extract parameters
        alpha0 = results.params[0]  # intercept
        alphap = results.params[1]  # slope
        
        # Parameter uncertainties
        alpha0_err = results.bse[0]
        alphap_err = results.bse[1]
        
        # Covariance matrix
        cov_matrix = results.cov_params()
        
        # Goodness of fit
        chi2 = results.ssr
        dof = len(x) - 2
        chi2_dof = chi2 / dof
        r_squared = results.rsquared
        
        # Store results
        self.results['wls'] = {
            'method': 'Weighted Least Squares',
            'alpha0': alpha0,
            'alpha0_err': alpha0_err,
            'alphap': alphap,
            'alphap_err': alphap_err,
            'cov_matrix': cov_matrix,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'aic': results.aic,
            'bic': results.bic,
            'model': results
        }
        
        return self.results['wls']
    
    def fit_odr(self) -> Dict[str, Any]:
        """
        Orthogonal distance regression (ODR) fit.
        
        Handles uncertainties in both x and y variables.
        
        Returns:
        --------
        Dict containing fit results
        """
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        x_err = self.data[self.x_err_col].values
        
        # Use y uncertainties if available, otherwise estimate from residuals
        if self.y_err_col and self.y_err_col in self.data.columns:
            y_err = self.data[self.y_err_col].values
        else:
            # Estimate y uncertainties from data spread
            y_err = np.std(y) * np.ones_like(y)
        
        # Define the model function for ODR
        def linear_func(params, x):
            alpha0, alphap = params
            return alpha0 + alphap * x
        
        # Create ODR model
        model = Model(linear_func)
        
        # Create data object with uncertainties
        data = RealData(x, y, sx=x_err, sy=y_err)
        
        # Initial parameter estimates from simple linear fit
        initial_params = np.polyfit(x, y, 1)
        alpha0_init = initial_params[1]
        alphap_init = initial_params[0]
        
        # Create ODR object
        odr_obj = ODR(data, model, beta0=[alpha0_init, alphap_init])
        
        # Run the fit
        result = odr_obj.run()
        
        # Extract results
        alpha0 = result.beta[0]
        alphap = result.beta[1]
        alpha0_err = result.sd_beta[0]
        alphap_err = result.sd_beta[1]
        
        # Covariance matrix
        cov_matrix = result.cov_beta
        
        # Goodness of fit
        chi2 = result.sum_square
        dof = len(x) - 2
        chi2_dof = chi2 / dof
        
        # Calculate R²
        y_pred = linear_func([alpha0, alphap], x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Store results
        self.results['odr'] = {
            'method': 'Orthogonal Distance Regression',
            'alpha0': alpha0,
            'alpha0_err': alpha0_err,
            'alphap': alphap,
            'alphap_err': alphap_err,
            'cov_matrix': cov_matrix,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'result': result
        }
        
        return self.results['odr']
    
    def fit_standard_ls(self) -> Dict[str, Any]:
        """
        Standard least squares fit for comparison.
        
        Returns:
        --------
        Dict containing fit results
        """
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        
        # Add constant term for intercept
        X = sm.add_constant(x)
        
        # Fit standard least squares
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract parameters
        alpha0 = results.params[0]
        alphap = results.params[1]
        alpha0_err = results.bse[0]
        alphap_err = results.bse[1]
        
        # Covariance matrix
        cov_matrix = results.cov_params()
        
        # Goodness of fit
        chi2 = results.ssr
        dof = len(x) - 2
        chi2_dof = chi2 / dof
        r_squared = results.rsquared
        
        # Store results
        self.results['ols'] = {
            'method': 'Ordinary Least Squares',
            'alpha0': alpha0,
            'alpha0_err': alpha0_err,
            'alphap': alphap,
            'alphap_err': alphap_err,
            'cov_matrix': cov_matrix,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'aic': results.aic,
            'bic': results.bic,
            'model': results
        }
        
        return self.results['ols']
    
    def fit_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Fit using all available methods.
        
        Returns:
        --------
        Dict containing results from all methods
        """
        print("Fitting Regge trajectory using multiple methods...")
        
        # Fit with all methods
        self.fit_standard_ls()
        self.fit_wls()
        self.fit_odr()
        
        # Compare results
        self._compare_methods()
        
        return self.results
    
    def _compare_methods(self):
        """Compare results from different fitting methods."""
        print("\n" + "="*60)
        print("REGGЕ TRAJECTORY FIT COMPARISON")
        print("="*60)
        
        for method, result in self.results.items():
            print(f"\n{result['method']}:")
            print(f"  α₀ = {result['alpha0']:.4f} ± {result['alpha0_err']:.4f}")
            print(f"  α' = {result['alphap']:.4f} ± {result['alphap_err']:.4f}")
            print(f"  χ²/dof = {result['chi2_dof']:.3f}")
            print(f"  R² = {result['r_squared']:.4f}")
    
    def get_best_fit(self, method: str = 'wls') -> Dict[str, Any]:
        """
        Get the best fit results (default: weighted least squares).
        
        Parameters:
        -----------
        method : str
            Method to use ('wls', 'odr', 'ols')
            
        Returns:
        --------
        Dict containing fit results
        """
        if method not in self.results:
            raise ValueError(f"Method '{method}' not found. Available: {list(self.results.keys())}")
        
        return self.results[method]
    
    def predict(self, x: np.ndarray, method: str = 'wls') -> np.ndarray:
        """
        Predict J values for given M² values.
        
        Parameters:
        -----------
        x : np.ndarray
            M² values to predict for
        method : str
            Method to use for prediction
            
        Returns:
        --------
        np.ndarray
            Predicted J values
        """
        result = self.get_best_fit(method)
        return self.linear_model(x, result['alpha0'], result['alphap'])
    
    def plot_fit(self, method: str = 'wls', save_path: Optional[str] = None):
        """
        Plot the data and fitted trajectory.
        
        Parameters:
        -----------
        method : str
            Method to use for plotting
        save_path : str, optional
            Path to save the plot
        """
        result = self.get_best_fit(method)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Main fit plot
        x_data = self.data[self.x_col].values
        y_data = self.data[self.y_col].values
        x_err = self.data[self.x_err_col].values
        
        # Plot data with error bars
        ax1.errorbar(x_data, y_data, xerr=x_err, fmt='o', 
                    capsize=3, capthick=1, alpha=0.7, label='Data')
        
        # Plot fit line
        x_fit = np.linspace(x_data.min() * 0.9, x_data.max() * 1.1, 100)
        y_fit = self.predict(x_fit, method)
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f"Fit: J = {result['alpha0']:.3f} + {result['alphap']:.3f} M²")
        
        ax1.set_xlabel('M² (GeV²)')
        ax1.set_ylabel('J (Spin)')
        ax1.set_title(f'Regge Trajectory Fit: {result["method"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        y_pred = self.predict(x_data, method)
        residuals = y_data - y_pred
        
        ax2.scatter(x_data, residuals, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('M² (GeV²)')
        ax2.set_ylabel('Residuals (J - J_pred)')
        ax2.set_title('Residuals vs M²')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig, (ax1, ax2)
