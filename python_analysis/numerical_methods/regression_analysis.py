"""
Numerical Methods & Statistics: Regression Analysis

Implements comprehensive regression analysis including:
- Orthogonal Distance Regression (ODR)
- Heteroskedasticity checks (Breusch-Pagan)
- Robust regression fallbacks
- Segmented (piecewise) models
- AIC/BIC model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.odr import ODR, Model, Data
import warnings
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

@dataclass
class RegressionResults:
    """Results from regression analysis."""
    method: str
    parameters: np.ndarray
    parameter_uncertainties: np.ndarray
    covariance: np.ndarray
    chi2: float
    chi2_dof: float
    r_squared: float
    dof: int
    residuals: np.ndarray
    fitted_values: np.ndarray
    x_data: np.ndarray
    y_data: np.ndarray
    y_errors: np.ndarray
    model_info: Dict[str, Any]

class RegressionAnalyzer:
    """
    Comprehensive regression analysis for Regge trajectories.
    
    Provides:
    - Orthogonal Distance Regression (ODR)
    - Heteroskedasticity detection and robust regression
    - Segmented (piecewise) models
    - Model comparison with AIC/BIC
    """
    
    def __init__(self):
        """Initialize regression analyzer."""
        self.results = {}
        self.model_comparison = {}
        
    def fit_regge_trajectory(self, 
                           x_data: np.ndarray,
                           y_data: np.ndarray,
                           y_errors: np.ndarray,
                           x_errors: Optional[np.ndarray] = None,
                           use_odr: bool = False,
                           robust_fallback: bool = True,
                           check_heteroskedasticity: bool = True) -> RegressionResults:
        """
        Fit Regge trajectory with multiple methods.
        
        Parameters:
        -----------
        x_data : np.ndarray
            M² data (GeV²)
        y_data : np.ndarray
            J data
        y_errors : np.ndarray
            J uncertainties
        x_errors : Optional[np.ndarray]
            M² uncertainties (for ODR)
        use_odr : bool
            Whether to use Orthogonal Distance Regression
        robust_fallback : bool
            Whether to use robust regression as fallback
        check_heteroskedasticity : bool
            Whether to check for heteroskedasticity
            
        Returns:
        --------
        RegressionResults
            Results from regression analysis
        """
        print("Fitting Regge trajectory...")
        
        # Remove invalid data points
        valid_mask = (y_errors > 0) & np.isfinite(y_errors) & np.isfinite(x_data) & np.isfinite(y_data)
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        y_errors_valid = y_errors[valid_mask]
        
        if x_errors is not None:
            x_errors_valid = x_errors[valid_mask]
        else:
            x_errors_valid = None
        
        if len(x_valid) < 2:
            raise ValueError("Insufficient valid data points for fitting")
        
        # Check for heteroskedasticity
        heteroskedastic = False
        if check_heteroskedasticity:
            heteroskedastic = self._check_heteroskedasticity(x_valid, y_valid, y_errors_valid)
            if heteroskedastic:
                print("Heteroskedasticity detected - using robust regression")
                robust_fallback = True
        
        # Perform fitting
        if use_odr and x_errors_valid is not None:
            results = self._fit_odr(x_valid, y_valid, x_errors_valid, y_errors_valid)
        else:
            results = self._fit_wls(x_valid, y_valid, y_errors_valid)
        
        # Robust fallback if needed
        if robust_fallback and (heteroskedastic or self._check_outliers(results)):
            robust_results = self._fit_robust(x_valid, y_valid, y_errors_valid)
            
            # Compare results
            param_diff = np.abs(robust_results.parameters - results.parameters)
            param_uncertainty = np.sqrt(np.diag(results.covariance))
            
            if np.any(param_diff > 2 * param_uncertainty):
                warnings.warn(f"Robust vs WLS α' differs by >2σ: {param_diff[1]:.4f} vs {param_uncertainty[1]:.4f}")
                print("Using robust regression results due to significant parameter differences")
                results = robust_results
        
        self.results['linear'] = results
        return results
    
    def _fit_wls(self, x_data: np.ndarray, y_data: np.ndarray, 
                y_errors: np.ndarray) -> RegressionResults:
        """Fit using Weighted Least Squares."""
        # Define linear model: J = α₀ + α'M²
        def linear_model(x, alpha0, alphap):
            return alpha0 + alphap * x
        
        # Perform weighted least squares fit
        popt, pcov = curve_fit(linear_model, x_data, y_data, 
                             sigma=y_errors, absolute_sigma=True)
        
        # Compute fit statistics
        y_pred = linear_model(x_data, *popt)
        residuals = y_data - y_pred
        chi2 = np.sum((residuals / y_errors)**2)
        dof = len(x_data) - len(popt)
        chi2_dof = chi2 / dof if dof > 0 else np.inf
        
        # Compute R-squared
        r_squared = r2_score(y_data, y_pred)
        
        # Parameter uncertainties
        param_uncertainties = np.sqrt(np.diag(pcov))
        
        return RegressionResults(
            method='WLS',
            parameters=popt,
            parameter_uncertainties=param_uncertainties,
            covariance=pcov,
            chi2=chi2,
            chi2_dof=chi2_dof,
            r_squared=r_squared,
            dof=dof,
            residuals=residuals,
            fitted_values=y_pred,
            x_data=x_data,
            y_data=y_data,
            y_errors=y_errors,
            model_info={'model_type': 'linear', 'formula': 'J = α₀ + α\'M²'}
        )
    
    def _fit_odr(self, x_data: np.ndarray, y_data: np.ndarray,
                x_errors: np.ndarray, y_errors: np.ndarray) -> RegressionResults:
        """Fit using Orthogonal Distance Regression."""
        # Define ODR model
        class LinearModel(Model):
            def __init__(self):
                Model.__init__(self, self.fcn)
            
            def fcn(self, beta, x):
                alpha0, alphap = beta
                return alpha0 + alphap * x
        
        # Create ODR data and model
        model = LinearModel()
        data = Data(x_data, y_data, wd=1/x_errors**2, we=1/y_errors**2)
        odr = ODR(data, model, beta0=[0.5, 0.9])  # Initial guess
        
        # Run ODR
        result = odr.run()
        
        # Extract results
        popt = result.beta
        pcov = result.cov_beta
        param_uncertainties = result.sd_beta
        
        # Compute fit statistics
        y_pred = model.fcn(popt, x_data)
        residuals = y_data - y_pred
        chi2 = np.sum((residuals / y_errors)**2)
        dof = len(x_data) - len(popt)
        chi2_dof = chi2 / dof if dof > 0 else np.inf
        
        # Compute R-squared
        r_squared = r2_score(y_data, y_pred)
        
        return RegressionResults(
            method='ODR',
            parameters=popt,
            parameter_uncertainties=param_uncertainties,
            covariance=pcov,
            chi2=chi2,
            chi2_dof=chi2_dof,
            r_squared=r_squared,
            dof=dof,
            residuals=residuals,
            fitted_values=y_pred,
            x_data=x_data,
            y_data=y_data,
            y_errors=y_errors,
            model_info={'model_type': 'linear_odr', 'formula': 'J = α₀ + α\'M² (ODR)'}
        )
    
    def _fit_robust(self, x_data: np.ndarray, y_data: np.ndarray,
                   y_errors: np.ndarray) -> RegressionResults:
        """Fit using robust regression methods."""
        # Use Huber regression
        X = x_data.reshape(-1, 1)
        huber = HuberRegressor(epsilon=1.35, max_iter=100)
        huber.fit(X, y_data, sample_weight=1/y_errors**2)
        
        # Extract parameters
        popt = np.array([huber.intercept_, huber.coef_[0]])
        
        # Compute uncertainties (approximate)
        y_pred = huber.predict(X)
        residuals = y_data - y_pred
        mse = np.mean(residuals**2)
        
        # Approximate covariance matrix
        X_design = np.column_stack([np.ones_like(x_data), x_data])
        cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)
        param_uncertainties = np.sqrt(np.diag(cov_matrix))
        
        # Compute fit statistics
        chi2 = np.sum((residuals / y_errors)**2)
        dof = len(x_data) - len(popt)
        chi2_dof = chi2 / dof if dof > 0 else np.inf
        r_squared = r2_score(y_data, y_pred)
        
        return RegressionResults(
            method='Robust (Huber)',
            parameters=popt,
            parameter_uncertainties=param_uncertainties,
            covariance=cov_matrix,
            chi2=chi2,
            chi2_dof=chi2_dof,
            r_squared=r_squared,
            dof=dof,
            residuals=residuals,
            fitted_values=y_pred,
            x_data=x_data,
            y_data=y_data,
            y_errors=y_errors,
            model_info={'model_type': 'robust_huber', 'formula': 'J = α₀ + α\'M² (Robust)'}
        )
    
    def _check_heteroskedasticity(self, x_data: np.ndarray, y_data: np.ndarray,
                                 y_errors: np.ndarray) -> bool:
        """
        Check for heteroskedasticity using Breusch-Pagan test.
        
        Parameters:
        -----------
        x_data : np.ndarray
            Independent variable
        y_data : np.ndarray
            Dependent variable
        y_errors : np.ndarray
            Uncertainties
            
        Returns:
        --------
        bool
            True if heteroskedasticity is detected
        """
        try:
            # Fit initial model
            X = sm.add_constant(x_data)
            model = sm.WLS(y_data, X, weights=1/y_errors**2)
            results = model.fit()
            
            # Perform Breusch-Pagan test
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(results.resid, X)
            
            # Consider heteroskedastic if p-value < 0.05
            is_heteroskedastic = bp_pvalue < 0.05
            
            print(f"Breusch-Pagan test: statistic={bp_stat:.4f}, p-value={bp_pvalue:.4f}")
            print(f"Heteroskedasticity detected: {is_heteroskedastic}")
            
            return is_heteroskedastic
            
        except Exception as e:
            warnings.warn(f"Heteroskedasticity check failed: {e}")
            return False
    
    def _check_outliers(self, results: RegressionResults) -> bool:
        """
        Check for outliers in residuals.
        
        Parameters:
        -----------
        results : RegressionResults
            Regression results
            
        Returns:
        --------
        bool
            True if outliers are detected
        """
        # Standardized residuals
        standardized_residuals = results.residuals / results.y_errors
        
        # Check for outliers (>3σ)
        outliers = np.abs(standardized_residuals) > 3
        outlier_fraction = np.mean(outliers)
        
        print(f"Outlier fraction: {outlier_fraction:.3f}")
        
        return outlier_fraction > 0.1  # More than 10% outliers
    
    def fit_segmented_model(self, x_data: np.ndarray, y_data: np.ndarray,
                          y_errors: np.ndarray, breakpoint_guess: Optional[float] = None) -> RegressionResults:
        """
        Fit segmented (piecewise) model with one breakpoint.
        
        Parameters:
        -----------
        x_data : np.ndarray
            M² data (GeV²)
        y_data : np.ndarray
            J data
        y_errors : np.ndarray
            J uncertainties
        breakpoint_guess : Optional[float]
            Initial guess for breakpoint
            
        Returns:
        --------
        RegressionResults
            Results from segmented regression
        """
        print("Fitting segmented model...")
        
        # Remove invalid data points
        valid_mask = (y_errors > 0) & np.isfinite(y_errors) & np.isfinite(x_data) & np.isfinite(y_data)
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        y_errors_valid = y_errors[valid_mask]
        
        if len(x_valid) < 4:
            raise ValueError("Insufficient data points for segmented model")
        
        # Initial guess for breakpoint
        if breakpoint_guess is None:
            breakpoint_guess = np.median(x_valid)
        
        # Define segmented model: J = α₀ + α'M² for M² < breakpoint, β₀ + β'M² for M² ≥ breakpoint
        def segmented_model(x, alpha0, alphap, beta0, betap, breakpoint):
            result = np.zeros_like(x)
            mask_low = x < breakpoint
            mask_high = x >= breakpoint
            
            result[mask_low] = alpha0 + alphap * x[mask_low]
            result[mask_high] = beta0 + betap * x[mask_high]
            
            return result
        
        # Initial parameter guess
        p0 = [0.5, 0.9, 0.5, 0.9, breakpoint_guess]
        
        # Parameter bounds
        bounds = (
            [-np.inf, 0, -np.inf, 0, np.min(x_valid)],  # Lower bounds
            [np.inf, np.inf, np.inf, np.inf, np.max(x_valid)]  # Upper bounds
        )
        
        try:
            # Fit segmented model
            popt, pcov = curve_fit(segmented_model, x_valid, y_valid, 
                                 sigma=y_errors_valid, absolute_sigma=True,
                                 p0=p0, bounds=bounds, maxfev=5000)
            
            # Compute fit statistics
            y_pred = segmented_model(x_valid, *popt)
            residuals = y_valid - y_pred
            chi2 = np.sum((residuals / y_errors_valid)**2)
            dof = len(x_valid) - len(popt)
            chi2_dof = chi2 / dof if dof > 0 else np.inf
            
            # Compute R-squared
            r_squared = r2_score(y_valid, y_pred)
            
            # Parameter uncertainties
            param_uncertainties = np.sqrt(np.diag(pcov))
            
            results = RegressionResults(
                method='Segmented',
                parameters=popt,
                parameter_uncertainties=param_uncertainties,
                covariance=pcov,
                chi2=chi2,
                chi2_dof=chi2_dof,
                r_squared=r_squared,
                dof=dof,
                residuals=residuals,
                fitted_values=y_pred,
                x_data=x_valid,
                y_data=y_valid,
                y_errors=y_errors_valid,
                model_info={
                    'model_type': 'segmented',
                    'formula': 'J = α₀ + α\'M² (M² < breakpoint), β₀ + β\'M² (M² ≥ breakpoint)',
                    'breakpoint': popt[4],
                    'breakpoint_uncertainty': param_uncertainties[4]
                }
            )
            
            self.results['segmented'] = results
            return results
            
        except Exception as e:
            warnings.warn(f"Segmented model fitting failed: {e}")
            return None
    
    def compare_models(self, linear_results: RegressionResults,
                      segmented_results: Optional[RegressionResults] = None) -> Dict[str, Any]:
        """
        Compare linear and segmented models using AIC/BIC.
        
        Parameters:
        -----------
        linear_results : RegressionResults
            Results from linear model
        segmented_results : Optional[RegressionResults]
            Results from segmented model
            
        Returns:
        --------
        Dict[str, Any]
            Model comparison results
        """
        print("Comparing models...")
        
        # Calculate AIC and BIC for linear model
        n = len(linear_results.x_data)
        k_linear = len(linear_results.parameters)
        
        aic_linear = n * np.log(linear_results.chi2 / n) + 2 * k_linear
        bic_linear = n * np.log(linear_results.chi2 / n) + k_linear * np.log(n)
        
        comparison = {
            'linear': {
                'aic': aic_linear,
                'bic': bic_linear,
                'chi2_dof': linear_results.chi2_dof,
                'r_squared': linear_results.r_squared,
                'parameters': linear_results.parameters.tolist(),
                'parameter_uncertainties': linear_results.parameter_uncertainties.tolist()
            }
        }
        
        if segmented_results is not None:
            # Calculate AIC and BIC for segmented model
            k_segmented = len(segmented_results.parameters)
            
            aic_segmented = n * np.log(segmented_results.chi2 / n) + 2 * k_segmented
            bic_segmented = n * np.log(segmented_results.chi2 / n) + k_segmented * np.log(n)
            
            comparison['segmented'] = {
                'aic': aic_segmented,
                'bic': bic_segmented,
                'chi2_dof': segmented_results.chi2_dof,
                'r_squared': segmented_results.r_squared,
                'parameters': segmented_results.parameters.tolist(),
                'parameter_uncertainties': segmented_results.parameter_uncertainties.tolist(),
                'breakpoint': segmented_results.model_info.get('breakpoint'),
                'breakpoint_uncertainty': segmented_results.model_info.get('breakpoint_uncertainty')
            }
            
            # Calculate differences
            delta_aic = aic_segmented - aic_linear
            delta_bic = bic_segmented - bic_linear
            
            comparison['model_selection'] = {
                'delta_aic': delta_aic,
                'delta_bic': delta_bic,
                'prefer_segmented_aic': delta_aic <= -4,
                'prefer_segmented_bic': delta_bic <= -4,
                'recommendation': 'segmented' if delta_aic <= -4 and delta_bic <= -4 else 'linear'
            }
            
            print(f"ΔAIC = {delta_aic:.2f}, ΔBIC = {delta_bic:.2f}")
            print(f"Recommendation: {comparison['model_selection']['recommendation']}")
        
        self.model_comparison = comparison
        return comparison
    
    def create_regression_diagnostics(self, results: RegressionResults) -> Dict[str, Any]:
        """
        Create comprehensive regression diagnostics.
        
        Parameters:
        -----------
        results : RegressionResults
            Regression results
            
        Returns:
        --------
        Dict[str, Any]
            Diagnostic information
        """
        diagnostics = {
            'fit_quality': {
                'chi2': results.chi2,
                'chi2_dof': results.chi2_dof,
                'r_squared': results.r_squared,
                'dof': results.dof
            },
            'parameters': {
                'alpha0': results.parameters[0],
                'alphap': results.parameters[1],
                'alpha0_uncertainty': results.parameter_uncertainties[0],
                'alphap_uncertainty': results.parameter_uncertainties[1]
            },
            'residual_analysis': {
                'mean_residual': np.mean(results.residuals),
                'std_residual': np.std(results.residuals),
                'max_residual': np.max(np.abs(results.residuals)),
                'outlier_count': np.sum(np.abs(results.residuals / results.y_errors) > 3)
            },
            'heteroskedasticity': {
                'detected': self._check_heteroskedasticity(results.x_data, results.y_data, results.y_errors)
            }
        }
        
        # Add segmented model info if available
        if results.method == 'Segmented':
            diagnostics['segmented_info'] = {
                'breakpoint': results.model_info.get('breakpoint'),
                'breakpoint_uncertainty': results.model_info.get('breakpoint_uncertainty'),
                'beta0': results.parameters[2],
                'betap': results.parameters[3],
                'beta0_uncertainty': results.parameter_uncertainties[2],
                'betap_uncertainty': results.parameter_uncertainties[3]
            }
        
        return diagnostics

if __name__ == "__main__":
    print("Regression Analysis")
    print("Use this module for advanced regression methods and model comparison")
