"""
Statistical Significance Analysis for Regge Trajectories

Implements comprehensive statistical analysis to ensure robust results:
- Weighted and orthogonal distance regression comparison
- Bootstrap and leave-one-out robustness tests
- Model comparison (linear vs broken-line)
- Multiple testing control across families
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import warnings
from tqdm import tqdm

class StatisticalSignificanceAnalyzer:
    """
    Comprehensive statistical analysis for Regge trajectory robustness.
    
    Implements multiple statistical tests to ensure results are not artifacts
    and are statistically significant.
    """
    
    def __init__(self, data: pd.DataFrame, x_col: str = 'M2_GeV2', 
                 y_col: str = 'J', x_err_col: str = 'M2_sigma_GeV2',
                 y_err_col: Optional[str] = None, width_col: Optional[str] = 'width_GeV',
                 kappa: float = 0.25):
        """
        Initialize statistical significance analyzer.
        
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
        y_err_col : str, optional
            Column name for J uncertainties
        width_col : str, optional
            Column name for resonance widths
        kappa : float
            Factor for width-to-uncertainty conversion
        """
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.x_err_col = x_err_col
        self.y_err_col = y_err_col
        self.width_col = width_col
        self.kappa = kappa
        
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
            
        print(f"Using {len(self.data)} data points for statistical analysis")
    
    def _add_width_uncertainty(self) -> np.ndarray:
        """Add width-based systematic uncertainty."""
        base_uncertainties = self.data[self.x_err_col].values
        
        if self.width_col and self.width_col in self.data.columns:
            widths = self.data[self.width_col].values
            width_uncertainties = self.kappa * np.where(
                np.isnan(widths), 0.0, widths
            )
            
            combined_uncertainties = np.sqrt(
                base_uncertainties**2 + width_uncertainties**2
            )
        else:
            combined_uncertainties = base_uncertainties
            
        return combined_uncertainties
    
    def fit_weighted_regression(self) -> Dict[str, Any]:
        """
        Fit weighted least squares regression.
        
        Returns:
        --------
        Dict containing WLS fit results
        """
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        uncertainties = self._add_width_uncertainty()
        
        # Weights = 1/σ²
        weights = 1.0 / (uncertainties ** 2)
        
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
        
        # Confidence intervals (95%)
        t_value = stats.t.ppf(0.975, dof)
        alpha0_ci = (beta[0] - t_value * alpha0_err, beta[0] + t_value * alpha0_err)
        alphap_ci = (beta[1] - t_value * alphap_err, beta[1] + t_value * alphap_err)
        
        return {
            'method': 'Weighted Least Squares',
            'alpha0': beta[0],
            'alphap': beta[1],
            'alpha0_err': alpha0_err,
            'alphap_err': alphap_err,
            'alpha0_ci': alpha0_ci,
            'alphap_ci': alphap_ci,
            'cov_matrix': cov_matrix,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'residuals': residuals,
            'weights': weights
        }
    
    def fit_orthogonal_distance_regression(self) -> Dict[str, Any]:
        """
        Fit orthogonal distance regression (ODR).
        
        Returns:
        --------
        Dict containing ODR fit results
        """
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        x_err = self._add_width_uncertainty()
        
        # Use y uncertainties if available, otherwise estimate
        if self.y_err_col and self.y_err_col in self.data.columns:
            y_err = self.data[self.y_err_col].values
        else:
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
        
        # Confidence intervals (95%)
        t_value = stats.t.ppf(0.975, dof)
        alpha0_ci = (alpha0 - t_value * alpha0_err, alpha0 + t_value * alpha0_err)
        alphap_ci = (alphap - t_value * alphap_err, alphap + t_value * alphap_err)
        
        return {
            'method': 'Orthogonal Distance Regression',
            'alpha0': alpha0,
            'alphap': alphap,
            'alpha0_err': alpha0_err,
            'alphap_err': alphap_err,
            'alpha0_ci': alpha0_ci,
            'alphap_ci': alphap_ci,
            'cov_matrix': cov_matrix,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'r_squared': r_squared,
            'residuals': y - y_pred,
            'result': result
        }
    
    def bootstrap_analysis(self, n_bootstrap: int = 10000) -> Dict[str, Any]:
        """
        Perform parametric bootstrap analysis.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap iterations
            
        Returns:
        --------
        Dict containing bootstrap results
        """
        print(f"Running {n_bootstrap} bootstrap iterations...")
        
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        uncertainties = self._add_width_uncertainty()
        
        bootstrap_results = []
        
        for _ in tqdm(range(n_bootstrap)):
            # Resample with noise based on uncertainties
            x_boot = x + np.random.normal(0, uncertainties)
            y_boot = y + np.random.normal(0, np.std(y) * 0.1)  # Small y noise
            
            # Fit to bootstrap sample
            try:
                # Simple linear fit
                X = np.column_stack([np.ones_like(x_boot), x_boot])
                weights = 1.0 / (uncertainties ** 2)
                W = np.diag(weights)
                
                beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y_boot
                
                bootstrap_results.append({
                    'alpha0': beta[0],
                    'alphap': beta[1]
                })
                
            except np.linalg.LinAlgError:
                continue
        
        if not bootstrap_results:
            raise ValueError("No successful bootstrap fits")
        
        # Analyze bootstrap results
        alpha0_values = [r['alpha0'] for r in bootstrap_results]
        alphap_values = [r['alphap'] for r in bootstrap_results]
        
        # Calculate statistics
        alpha0_mean = np.mean(alpha0_values)
        alpha0_std = np.std(alpha0_values)
        alpha0_ci = np.percentile(alpha0_values, [2.5, 97.5])
        
        alphap_mean = np.mean(alphap_values)
        alphap_std = np.std(alphap_values)
        alphap_ci = np.percentile(alphap_values, [2.5, 97.5])
        
        # Correlation
        correlation = np.corrcoef(alpha0_values, alphap_values)[0, 1]
        
        return {
            'n_bootstrap': len(bootstrap_results),
            'alpha0': {
                'mean': alpha0_mean,
                'std': alpha0_std,
                'ci_95': alpha0_ci,
                'bias': alpha0_mean - alpha0_values[0]
            },
            'alphap': {
                'mean': alphap_mean,
                'std': alphap_std,
                'ci_95': alphap_ci,
                'bias': alphap_mean - alphap_values[0]
            },
            'correlation': correlation,
            'bootstrap_samples': bootstrap_results
        }
    
    def leave_one_out_analysis(self) -> Dict[str, Any]:
        """
        Perform leave-one-out cross-validation.
        
        Returns:
        --------
        Dict containing LOO results
        """
        print("Running leave-one-out analysis...")
        
        loo = LeaveOneOut()
        x_data = self.data[self.x_col].values
        y_data = self.data[self.y_col].values
        uncertainties = self._add_width_uncertainty()
        
        loo_results = []
        
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
                
                loo_results.append({
                    'excluded_point': test_idx[0],
                    'alpha0': beta[0],
                    'alphap': beta[1],
                    'y_pred': y_pred[0],
                    'y_true': y_test[0],
                    'residual': y_test[0] - y_pred[0]
                })
                
            except np.linalg.LinAlgError:
                continue
        
        if not loo_results:
            raise ValueError("No successful LOO fits")
        
        # Analyze LOO results
        alpha0_values = [r['alpha0'] for r in loo_results]
        alphap_values = [r['alphap'] for r in loo_results]
        residuals = [r['residual'] for r in loo_results]
        
        # Calculate statistics
        alpha0_mean = np.mean(alpha0_values)
        alpha0_std = np.std(alpha0_values)
        alpha0_max_dev = max(abs(a - alpha0_mean) for a in alpha0_values)
        
        alphap_mean = np.mean(alphap_values)
        alphap_std = np.std(alphap_values)
        alphap_max_dev = max(abs(a - alphap_mean) for a in alphap_values)
        
        # Find most influential point
        alphap_deviations = [abs(a - alphap_mean) for a in alphap_values]
        max_influence_idx = np.argmax(alphap_deviations)
        most_influential_point = loo_results[max_influence_idx]['excluded_point']
        
        return {
            'n_successful': len(loo_results),
            'alpha0': {
                'mean': alpha0_mean,
                'std': alpha0_std,
                'max_deviation': alpha0_max_dev
            },
            'alphap': {
                'mean': alphap_mean,
                'std': alphap_std,
                'max_deviation': alphap_max_dev
            },
            'residuals': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'rmse': np.sqrt(mean_squared_error([r['y_true'] for r in loo_results], 
                                                 [r['y_pred'] for r in loo_results]))
            },
            'most_influential_point': most_influential_point,
            'loo_results': loo_results
        }
    
    def broken_line_model(self, x: np.ndarray, params: List[float]) -> np.ndarray:
        """
        Broken line model: two linear segments.
        
        Parameters:
        -----------
        x : np.ndarray
            M² values
        params : List[float]
            [alpha0, alphap1, alphap2, breakpoint]
            
        Returns:
        --------
        np.ndarray
            Predicted J values
        """
        alpha0, alphap1, alphap2, breakpoint = params
        
        y = np.zeros_like(x)
        mask1 = x <= breakpoint
        mask2 = x > breakpoint
        
        y[mask1] = alpha0 + alphap1 * x[mask1]
        y[mask2] = alpha0 + alphap1 * breakpoint + alphap2 * (x[mask2] - breakpoint)
        
        return y
    
    def fit_broken_line_model(self) -> Dict[str, Any]:
        """
        Fit broken line model (two linear segments).
        
        Returns:
        --------
        Dict containing broken line fit results
        """
        x = self.data[self.x_col].values
        y = self.data[self.y_col].values
        uncertainties = self._add_width_uncertainty()
        
        # Initial parameter estimates
        # Use linear fit for first segment, estimate breakpoint
        initial_params = np.polyfit(x, y, 1)
        alpha0_init = initial_params[1]
        alphap1_init = initial_params[0]
        alphap2_init = alphap1_init * 0.8  # Slightly different slope
        breakpoint_init = np.median(x)  # Middle of data range
        
        initial_guess = [alpha0_init, alphap1_init, alphap2_init, breakpoint_init]
        
        # Fit broken line model
        try:
            popt, pcov = curve_fit(
                self.broken_line_model, x, y, 
                p0=initial_guess,
                sigma=uncertainties,
                absolute_sigma=True,
                bounds=([-np.inf, -np.inf, -np.inf, x.min()], 
                       [np.inf, np.inf, np.inf, x.max()])
            )
            
            # Calculate uncertainties
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate goodness of fit
            y_pred = self.broken_line_model(x, popt)
            residuals = y - y_pred
            chi2 = np.sum((residuals / uncertainties) ** 2)
            dof = len(x) - 4  # 4 parameters
            chi2_dof = chi2 / dof
            
            # R²
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # AIC
            aic = 2 * 4 + chi2  # 4 parameters
            
            return {
                'method': 'Broken Line Model',
                'alpha0': popt[0],
                'alphap1': popt[1],
                'alphap2': popt[2],
                'breakpoint': popt[3],
                'alpha0_err': perr[0],
                'alphap1_err': perr[1],
                'alphap2_err': perr[2],
                'breakpoint_err': perr[3],
                'chi2': chi2,
                'dof': dof,
                'chi2_dof': chi2_dof,
                'r_squared': r_squared,
                'aic': aic,
                'residuals': residuals,
                'y_pred': y_pred
            }
            
        except (RuntimeError, ValueError) as e:
            print(f"Broken line fit failed: {e}")
            return None
    
    def model_comparison(self) -> Dict[str, Any]:
        """
        Compare linear vs broken line models.
        
        Returns:
        --------
        Dict containing model comparison results
        """
        # Fit both models
        wls_results = self.fit_weighted_regression()
        broken_line_results = self.fit_broken_line_model()
        
        if broken_line_results is None:
            print("Broken line model failed to converge")
            return {'linear_only': True, 'wls_results': wls_results}
        
        # Calculate AIC for linear model
        wls_aic = 2 * 2 + wls_results['chi2']  # 2 parameters
        
        # Calculate ΔAIC
        delta_aic = broken_line_results['aic'] - wls_aic
        
        # Model selection interpretation
        if delta_aic < -2:
            model_preference = "Broken line strongly preferred"
        elif delta_aic < 0:
            model_preference = "Broken line weakly preferred"
        elif delta_aic < 2:
            model_preference = "No strong preference"
        elif delta_aic < 7:
            model_preference = "Linear model weakly preferred"
        else:
            model_preference = "Linear model strongly preferred"
        
        return {
            'linear_only': False,
            'wls_results': wls_results,
            'broken_line_results': broken_line_results,
            'wls_aic': wls_aic,
            'broken_line_aic': broken_line_results['aic'],
            'delta_aic': delta_aic,
            'model_preference': model_preference,
            'comparison_table': pd.DataFrame({
                'Model': ['Linear (WLS)', 'Broken Line'],
                'Parameters': [2, 4],
                'χ²/dof': [wls_results['chi2_dof'], broken_line_results['chi2_dof']],
                'R²': [wls_results['r_squared'], broken_line_results['r_squared']],
                'AIC': [wls_aic, broken_line_results['aic']],
                'ΔAIC': [0, delta_aic]
            })
        }
    
    def multiple_testing_correction(self, p_values: List[float], 
                                  method: str = 'fdr_bh') -> Dict[str, Any]:
        """
        Apply multiple testing correction.
        
        Parameters:
        -----------
        p_values : List[float]
            List of p-values from multiple tests
        method : str
            Correction method ('fdr_bh', 'bonferroni', 'holm')
            
        Returns:
        --------
        Dict containing corrected p-values
        """
        from statsmodels.stats.multitest import multipletests
        
        # Apply correction
        rejected, p_corrected, _, _ = multipletests(
            p_values, alpha=0.05, method=method
        )
        
        return {
            'original_p_values': p_values,
            'corrected_p_values': p_corrected,
            'rejected': rejected,
            'method': method,
            'n_tests': len(p_values),
            'n_significant': np.sum(rejected),
            'significant_indices': np.where(rejected)[0]
        }
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete statistical significance analysis.
        
        Returns:
        --------
        Dict containing all analysis results
        """
        print("="*60)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*60)
        
        # 1. Fit comparison (WLS vs ODR)
        print("\n1. FITTING METHOD COMPARISON")
        print("-" * 30)
        
        wls_results = self.fit_weighted_regression()
        odr_results = self.fit_orthogonal_distance_regression()
        
        print(f"WLS:  α' = {wls_results['alphap']:.4f} ± {wls_results['alphap_err']:.4f}")
        print(f"ODR:  α' = {odr_results['alphap']:.4f} ± {odr_results['alphap_err']:.4f}")
        
        # Check if results are consistent
        alphap_diff = abs(wls_results['alphap'] - odr_results['alphap'])
        alphap_combined_err = np.sqrt(wls_results['alphap_err']**2 + odr_results['alphap_err']**2)
        consistency = alphap_diff / alphap_combined_err
        
        print(f"Consistency check: Δα' = {alphap_diff:.4f} ({consistency:.2f}σ)")
        
        # 2. Bootstrap analysis
        print("\n2. BOOTSTRAP ROBUSTNESS")
        print("-" * 30)
        
        bootstrap_results = self.bootstrap_analysis(n_bootstrap=5000)  # Reduced for speed
        
        print(f"Bootstrap α' = {bootstrap_results['alphap']['mean']:.4f} ± {bootstrap_results['alphap']['std']:.4f}")
        print(f"Bootstrap bias = {bootstrap_results['alphap']['bias']:.6f}")
        
        # 3. Leave-one-out analysis
        print("\n3. LEAVE-ONE-OUT ROBUSTNESS")
        print("-" * 30)
        
        loo_results = self.leave_one_out_analysis()
        
        print(f"LOO α' = {loo_results['alphap']['mean']:.4f} ± {loo_results['alphap']['std']:.4f}")
        print(f"LOO max deviation = {loo_results['alphap']['max_deviation']:.4f}")
        print(f"Most influential point: {loo_results['most_influential_point']}")
        
        # 4. Model comparison
        print("\n4. MODEL COMPARISON")
        print("-" * 30)
        
        model_comparison = self.model_comparison()
        
        if not model_comparison['linear_only']:
            print(f"Linear AIC: {model_comparison['wls_aic']:.2f}")
            print(f"Broken line AIC: {model_comparison['broken_line_aic']:.2f}")
            print(f"ΔAIC: {model_comparison['delta_aic']:.2f}")
            print(f"Model preference: {model_comparison['model_preference']}")
        else:
            print("Only linear model available")
        
        # 5. Summary and conclusions
        print("\n5. STATISTICAL SIGNIFICANCE SUMMARY")
        print("-" * 30)
        
        # Check robustness criteria
        robustness_checks = {
            'fitting_methods_consistent': consistency < 2.0,
            'bootstrap_stable': bootstrap_results['alphap']['std'] < 0.1,
            'loo_stable': loo_results['alphap']['max_deviation'] < 0.05,
            'linear_model_adequate': model_comparison.get('delta_aic', 0) > -2
        }
        
        n_robust = sum(robustness_checks.values())
        total_checks = len(robustness_checks)
        
        print(f"Robustness checks passed: {n_robust}/{total_checks}")
        for check, passed in robustness_checks.items():
            status = "✓" if passed else "✗"
            print(f"  {check}: {status}")
        
        # Overall conclusion
        if n_robust >= 3:
            conclusion = "Results are statistically robust"
        elif n_robust >= 2:
            conclusion = "Results show moderate robustness"
        else:
            conclusion = "Results require caution - limited robustness"
        
        print(f"\nOverall conclusion: {conclusion}")
        
        # Store all results
        self.results = {
            'wls_results': wls_results,
            'odr_results': odr_results,
            'bootstrap_results': bootstrap_results,
            'loo_results': loo_results,
            'model_comparison': model_comparison,
            'robustness_checks': robustness_checks,
            'conclusion': conclusion,
            'consistency_check': consistency
        }
        
        return self.results
    
    def plot_robustness_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot robustness analysis results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save plot
            
        Returns:
        --------
        plt.Figure
            The generated plot
        """
        if not self.results:
            raise ValueError("Run complete analysis first")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Fitting method comparison
        methods = ['WLS', 'ODR']
        alphap_values = [
            self.results['wls_results']['alphap'],
            self.results['odr_results']['alphap']
        ]
        alphap_errors = [
            self.results['wls_results']['alphap_err'],
            self.results['odr_results']['alphap_err']
        ]
        
        ax1.errorbar(methods, alphap_values, yerr=alphap_errors, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax1.set_ylabel("α' (GeV⁻²)")
        ax1.set_title('Fitting Method Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bootstrap distribution
        bootstrap_samples = self.results['bootstrap_results']['bootstrap_samples']
        alphap_bootstrap = [r['alphap'] for r in bootstrap_samples]
        
        ax2.hist(alphap_bootstrap, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(self.results['bootstrap_results']['alphap']['mean'], 
                   color='red', linestyle='--', label='Mean')
        ax2.set_xlabel("α' (GeV⁻²)")
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bootstrap Distribution of α\'')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Leave-one-out analysis
        loo_samples = self.results['loo_results']['loo_results']
        alphap_loo = [r['alphap'] for r in loo_samples]
        
        ax3.scatter(range(len(alphap_loo)), alphap_loo, alpha=0.7)
        ax3.axhline(self.results['loo_results']['alphap']['mean'], 
                   color='red', linestyle='--', label='Mean')
        ax3.set_xlabel('Excluded Point Index')
        ax3.set_ylabel("α' (GeV⁻²)")
        ax3.set_title('Leave-One-Out α\' Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model comparison
        if not self.results['model_comparison']['linear_only']:
            models = ['Linear', 'Broken Line']
            aic_values = [
                self.results['model_comparison']['wls_aic'],
                self.results['model_comparison']['broken_line_aic']
            ]
            
            bars = ax4.bar(models, aic_values, alpha=0.7)
            ax4.set_ylabel('AIC')
            ax4.set_title('Model Comparison (AIC)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, aic_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Only linear model available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Robustness analysis plot saved to {save_path}")
        
        plt.show()
        
        return fig
