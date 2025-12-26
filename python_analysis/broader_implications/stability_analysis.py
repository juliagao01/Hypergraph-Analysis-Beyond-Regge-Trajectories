"""
Broader Implications: Stability Analysis

Implements comprehensive stability analysis including:
- Stability under PDG data revisions
- What-if reclassification scenarios
- Robustness testing and validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
from scipy import stats
from scipy.optimize import curve_fit

@dataclass
class StabilityMetrics:
    """Metrics for stability analysis."""
    parameter_shift: float
    fit_quality_change: float
    prediction_shift: float
    gap_count_change: int
    community_structure_change: float
    overall_stability_score: float

class StabilityAnalyzer:
    """
    Analyzes stability of results under data revisions and reclassifications.
    
    Provides:
    - PDG update stability analysis
    - Reclassification impact assessment
    - Robustness testing
    - Stability reporting
    """
    
    def __init__(self, output_dir: str = "stability_analysis"):
        """
        Initialize stability analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stability_results = {}
        
    def analyze_pdg_update_stability(self, 
                                   original_data: pd.DataFrame,
                                   updated_data: pd.DataFrame,
                                   original_results: Dict[str, Any],
                                   fit_function: callable,
                                   fit_params: List[str]) -> Dict[str, Any]:
        """
        Analyze stability under PDG data updates.
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original PDG data
        updated_data : pd.DataFrame
            Updated PDG data
        original_results : Dict[str, Any]
            Results from original analysis
        fit_function : callable
            Function used for fitting
        fit_params : List[str]
            Names of fit parameters
            
        Returns:
        --------
        Dict containing stability analysis results
        """
        print("Analyzing stability under PDG updates...")
        
        # 1. Identify changes between datasets
        changes = self._identify_data_changes(original_data, updated_data)
        
        # 2. Re-run analysis on updated data
        updated_results = self._run_analysis_on_updated_data(
            updated_data, fit_function, fit_params
        )
        
        # 3. Compare results
        stability_metrics = self._compare_results(original_results, updated_results)
        
        # 4. Generate stability report
        stability_report = self._generate_stability_report(changes, stability_metrics)
        
        results = {
            'data_changes': changes,
            'updated_results': updated_results,
            'stability_metrics': stability_metrics,
            'stability_report': stability_report
        }
        
        return results
    
    def _identify_data_changes(self, original_data: pd.DataFrame, 
                             updated_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify changes between original and updated datasets."""
        changes = {
            'added_particles': [],
            'removed_particles': [],
            'modified_particles': [],
            'mass_changes': [],
            'width_changes': [],
            'status_changes': []
        }
        
        # Create sets for comparison
        original_particles = set(original_data['Name'].values)
        updated_particles = set(updated_data['Name'].values)
        
        # Find added and removed particles
        added = updated_particles - original_particles
        removed = original_particles - updated_particles
        
        changes['added_particles'] = list(added)
        changes['removed_particles'] = list(removed)
        
        # Find modified particles
        common_particles = original_particles & updated_particles
        
        for particle in common_particles:
            orig_row = original_data[original_data['Name'] == particle].iloc[0]
            updated_row = updated_data[updated_data['Name'] == particle].iloc[0]
            
            modifications = []
            
            # Check mass changes
            if abs(orig_row['MassGeV'] - updated_row['MassGeV']) > 0.001:
                mass_change = {
                    'particle': particle,
                    'original': orig_row['MassGeV'],
                    'updated': updated_row['MassGeV'],
                    'difference': updated_row['MassGeV'] - orig_row['MassGeV']
                }
                changes['mass_changes'].append(mass_change)
                modifications.append('mass')
            
            # Check width changes
            if 'ResonanceWidthGeV' in orig_row and 'ResonanceWidthGeV' in updated_row:
                if abs(orig_row['ResonanceWidthGeV'] - updated_row['ResonanceWidthGeV']) > 0.001:
                    width_change = {
                        'particle': particle,
                        'original': orig_row['ResonanceWidthGeV'],
                        'updated': updated_row['ResonanceWidthGeV'],
                        'difference': updated_row['ResonanceWidthGeV'] - orig_row['ResonanceWidthGeV']
                    }
                    changes['width_changes'].append(width_change)
                    modifications.append('width')
            
            # Check status changes
            if orig_row.get('Status') != updated_row.get('Status'):
                status_change = {
                    'particle': particle,
                    'original': orig_row.get('Status'),
                    'updated': updated_row.get('Status')
                }
                changes['status_changes'].append(status_change)
                modifications.append('status')
            
            if modifications:
                changes['modified_particles'].append({
                    'particle': particle,
                    'modifications': modifications
                })
        
        return changes
    
    def _run_analysis_on_updated_data(self, updated_data: pd.DataFrame,
                                    fit_function: callable,
                                    fit_params: List[str]) -> Dict[str, Any]:
        """Run analysis on updated data."""
        # Prepare data for fitting
        x_data = updated_data['M2GeV2'].values
        y_data = updated_data['J'].values
        y_errors = updated_data.get('M2SigmaGeV2', np.ones_like(x_data) * 0.01).values
        
        # Remove invalid data points
        valid_mask = (y_errors > 0) & np.isfinite(y_errors) & np.isfinite(x_data) & np.isfinite(y_data)
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        y_errors_valid = y_errors[valid_mask]
        
        if len(x_valid) < 2:
            raise ValueError("Insufficient valid data points for fitting")
        
        # Perform fit
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
            'residuals': residuals
        }
    
    def _compare_results(self, original_results: Dict[str, Any], 
                        updated_results: Dict[str, Any]) -> StabilityMetrics:
        """Compare original and updated results."""
        # Parameter shifts
        orig_params = original_results['parameters']
        updated_params = updated_results['parameters']
        
        param_shifts = []
        for i, param_name in enumerate(['alpha0', 'alphap']):
            shift = abs(updated_params[i] - orig_params[i])
            param_shifts.append(shift)
        
        avg_parameter_shift = np.mean(param_shifts)
        
        # Fit quality changes
        fit_quality_change = abs(updated_results['chi2_dof'] - original_results['chi2_dof'])
        
        # Prediction shifts (simplified)
        prediction_shift = avg_parameter_shift * 0.1  # Rough estimate
        
        # Gap count changes (if available)
        gap_count_change = 0
        if 'cross_check_results' in original_results and 'cross_check_results' in updated_results:
            orig_gaps = sum(1 for result in original_results['cross_check_results'] 
                          if result.get('IsSignificantGap', False))
            updated_gaps = sum(1 for result in updated_results['cross_check_results'] 
                             if result.get('IsSignificantGap', False))
            gap_count_change = updated_gaps - orig_gaps
        
        # Community structure changes (if available)
        community_structure_change = 0.0
        if 'community_analysis' in original_results and 'community_analysis' in updated_results:
            # Simplified community structure comparison
            orig_modularity = original_results['community_analysis'].get('modularity', 0.0)
            updated_modularity = updated_results['community_analysis'].get('modularity', 0.0)
            community_structure_change = abs(updated_modularity - orig_modularity)
        
        # Overall stability score (0-1, higher is more stable)
        stability_score = 1.0 - min(1.0, (
            avg_parameter_shift * 10 +  # Parameter stability
            fit_quality_change * 5 +    # Fit quality stability
            abs(gap_count_change) * 0.1 +  # Gap stability
            community_structure_change * 2  # Community stability
        ))
        
        return StabilityMetrics(
            parameter_shift=avg_parameter_shift,
            fit_quality_change=fit_quality_change,
            prediction_shift=prediction_shift,
            gap_count_change=gap_count_change,
            community_structure_change=community_structure_change,
            overall_stability_score=stability_score
        )
    
    def _generate_stability_report(self, changes: Dict[str, Any], 
                                 metrics: StabilityMetrics) -> str:
        """Generate stability report."""
        report = []
        report.append("=" * 80)
        report.append("PDG UPDATE STABILITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Data changes summary
        report.append("1. DATA CHANGES SUMMARY")
        report.append("-" * 50)
        report.append(f"Added particles: {len(changes['added_particles'])}")
        report.append(f"Removed particles: {len(changes['removed_particles'])}")
        report.append(f"Modified particles: {len(changes['modified_particles'])}")
        report.append(f"Mass changes: {len(changes['mass_changes'])}")
        report.append(f"Width changes: {len(changes['width_changes'])}")
        report.append(f"Status changes: {len(changes['status_changes'])}")
        report.append("")
        
        # Stability metrics
        report.append("2. STABILITY METRICS")
        report.append("-" * 50)
        report.append(f"Average parameter shift: {metrics.parameter_shift:.4f}")
        report.append(f"Fit quality change (Δχ²/dof): {metrics.fit_quality_change:.4f}")
        report.append(f"Prediction shift: {metrics.prediction_shift:.4f}")
        report.append(f"Gap count change: {metrics.gap_count_change:+d}")
        report.append(f"Community structure change: {metrics.community_structure_change:.4f}")
        report.append(f"Overall stability score: {metrics.overall_stability_score:.3f}")
        report.append("")
        
        # Stability assessment
        report.append("3. STABILITY ASSESSMENT")
        report.append("-" * 50)
        
        if metrics.overall_stability_score > 0.8:
            stability_level = "HIGH"
            assessment = "Results are stable under PDG updates"
        elif metrics.overall_stability_score > 0.6:
            stability_level = "MEDIUM"
            assessment = "Results show moderate stability under PDG updates"
        else:
            stability_level = "LOW"
            assessment = "Results show significant changes under PDG updates"
        
        report.append(f"Stability level: {stability_level}")
        report.append(f"Assessment: {assessment}")
        report.append("")
        
        # Recommendations
        report.append("4. RECOMMENDATIONS")
        report.append("-" * 50)
        
        if metrics.parameter_shift > 0.01:
            report.append("- Monitor parameter shifts closely")
        if metrics.fit_quality_change > 0.1:
            report.append("- Reassess fit quality and model assumptions")
        if abs(metrics.gap_count_change) > 2:
            report.append("- Review gap analysis and prediction reliability")
        if metrics.community_structure_change > 0.05:
            report.append("- Re-evaluate community structure interpretations")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def analyze_reclassification_scenarios(self, 
                                         original_data: pd.DataFrame,
                                         original_results: Dict[str, Any],
                                         reclassification_rules: List[Dict[str, Any]],
                                         fit_function: callable,
                                         fit_params: List[str]) -> Dict[str, Any]:
        """
        Analyze what-if reclassification scenarios.
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original data
        original_results : Dict[str, Any]
            Original analysis results
        reclassification_rules : List[Dict[str, Any]]
            Rules for reclassifying particles
        fit_function : callable
            Function used for fitting
        fit_params : List[str]
            Names of fit parameters
            
        Returns:
        --------
        Dict containing reclassification analysis results
        """
        print("Analyzing reclassification scenarios...")
        
        results = {
            'scenarios': [],
            'improvements': [],
            'before_after_comparison': []
        }
        
        for i, rule in enumerate(reclassification_rules):
            print(f"  Scenario {i+1}: {rule.get('description', 'Reclassification')}")
            
            # Apply reclassification rule
            reclassified_data = self._apply_reclassification_rule(original_data, rule)
            
            # Re-run analysis
            reclassified_results = self._run_analysis_on_updated_data(
                reclassified_data, fit_function, fit_params
            )
            
            # Compare with original
            improvement = self._assess_reclassification_improvement(
                original_results, reclassified_results, rule
            )
            
            # Store results
            scenario_result = {
                'scenario_id': i + 1,
                'description': rule.get('description', f'Scenario {i+1}'),
                'rule': rule,
                'reclassified_data': reclassified_data,
                'reclassified_results': reclassified_results,
                'improvement': improvement
            }
            
            results['scenarios'].append(scenario_result)
            results['improvements'].append(improvement)
            
            # Create before/after comparison
            comparison = self._create_before_after_comparison(
                original_results, reclassified_results, rule
            )
            results['before_after_comparison'].append(comparison)
        
        return results
    
    def _apply_reclassification_rule(self, data: pd.DataFrame, 
                                   rule: Dict[str, Any]) -> pd.DataFrame:
        """Apply reclassification rule to data."""
        reclassified_data = data.copy()
        
        rule_type = rule.get('type', 'reassign')
        
        if rule_type == 'reassign':
            # Reassign particles based on criteria
            criteria = rule.get('criteria', {})
            reassignments = rule.get('reassignments', {})
            
            for particle, new_assignment in reassignments.items():
                mask = reclassified_data['Name'] == particle
                if mask.any():
                    for field, value in new_assignment.items():
                        if field in reclassified_data.columns:
                            reclassified_data.loc[mask, field] = value
        
        elif rule_type == 'filter':
            # Filter particles based on criteria
            criteria = rule.get('criteria', {})
            for field, condition in criteria.items():
                if field in reclassified_data.columns:
                    if isinstance(condition, dict):
                        if 'min' in condition:
                            reclassified_data = reclassified_data[
                                reclassified_data[field] >= condition['min']
                            ]
                        if 'max' in condition:
                            reclassified_data = reclassified_data[
                                reclassified_data[field] <= condition['max']
                            ]
        
        return reclassified_data
    
    def _assess_reclassification_improvement(self, original_results: Dict[str, Any],
                                           reclassified_results: Dict[str, Any],
                                           rule: Dict[str, Any]) -> Dict[str, Any]:
        """Assess improvement from reclassification."""
        # Fit quality improvement
        chi2_improvement = original_results['chi2_dof'] - reclassified_results['chi2_dof']
        r2_improvement = reclassified_results['r_squared'] - original_results['r_squared']
        
        # Parameter stability
        orig_params = original_results['parameters']
        reclass_params = reclassified_results['parameters']
        param_stability = 1.0 - np.mean(np.abs(reclass_params - orig_params))
        
        # Overall improvement score
        improvement_score = (
            max(0, chi2_improvement) * 0.4 +  # χ² improvement
            max(0, r2_improvement) * 0.3 +     # R² improvement
            param_stability * 0.3              # Parameter stability
        )
        
        return {
            'chi2_improvement': chi2_improvement,
            'r2_improvement': r2_improvement,
            'param_stability': param_stability,
            'improvement_score': improvement_score,
            'is_beneficial': improvement_score > 0.1
        }
    
    def _create_before_after_comparison(self, original_results: Dict[str, Any],
                                      reclassified_results: Dict[str, Any],
                                      rule: Dict[str, Any]) -> Dict[str, Any]:
        """Create before/after comparison table."""
        return {
            'scenario': rule.get('description', 'Reclassification'),
            'before': {
                'alpha0': original_results['parameters'][0],
                'alphap': original_results['parameters'][1],
                'chi2_dof': original_results['chi2_dof'],
                'r_squared': original_results['r_squared'],
                'param_uncertainties': original_results['parameter_uncertainties'].tolist()
            },
            'after': {
                'alpha0': reclassified_results['parameters'][0],
                'alphap': reclassified_results['parameters'][1],
                'chi2_dof': reclassified_results['chi2_dof'],
                'r_squared': reclassified_results['r_squared'],
                'param_uncertainties': reclassified_results['parameter_uncertainties'].tolist()
            },
            'changes': {
                'alpha0_change': reclassified_results['parameters'][0] - original_results['parameters'][0],
                'alphap_change': reclassified_results['parameters'][1] - original_results['parameters'][1],
                'chi2_improvement': original_results['chi2_dof'] - reclassified_results['chi2_dof'],
                'r2_improvement': reclassified_results['r_squared'] - original_results['r_squared']
            }
        }
    
    def create_stability_visualizations(self, stability_results: Dict[str, Any]) -> None:
        """
        Create stability analysis visualizations.
        
        Parameters:
        -----------
        stability_results : Dict[str, Any]
            Results from stability analysis
        """
        # 1. Parameter stability plot
        if 'stability_metrics' in stability_results:
            metrics = stability_results['stability_metrics']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            stability_components = [
                'Parameter\nStability',
                'Fit Quality\nStability', 
                'Prediction\nStability',
                'Gap\nStability',
                'Community\nStability',
                'Overall\nStability'
            ]
            
            stability_values = [
                1.0 - min(1.0, metrics.parameter_shift * 10),
                1.0 - min(1.0, metrics.fit_quality_change * 5),
                1.0 - min(1.0, metrics.prediction_shift * 10),
                1.0 - min(1.0, abs(metrics.gap_count_change) * 0.1),
                1.0 - min(1.0, metrics.community_structure_change * 2),
                metrics.overall_stability_score
            ]
            
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in stability_values]
            
            bars = ax.bar(stability_components, stability_values, color=colors, alpha=0.7)
            ax.set_ylabel('Stability Score (0-1)')
            ax.set_title('Stability Analysis Under PDG Updates')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, stability_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Reclassification improvement plot
        if 'reclassification_results' in stability_results:
            reclass_results = stability_results['reclassification_results']
            
            if reclass_results['improvements']:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                scenarios = [f"Scenario {i+1}" for i in range(len(reclass_results['improvements']))]
                improvement_scores = [imp['improvement_score'] for imp in reclass_results['improvements']]
                chi2_improvements = [imp['chi2_improvement'] for imp in reclass_results['improvements']]
                r2_improvements = [imp['r2_improvement'] for imp in reclass_results['improvements']]
                
                x = np.arange(len(scenarios))
                width = 0.25
                
                bars1 = ax.bar(x - width, improvement_scores, width, label='Overall Improvement', alpha=0.8)
                bars2 = ax.bar(x, chi2_improvements, width, label='χ² Improvement', alpha=0.8)
                bars3 = ax.bar(x + width, r2_improvements, width, label='R² Improvement', alpha=0.8)
                
                ax.set_xlabel('Reclassification Scenarios')
                ax.set_ylabel('Improvement Score')
                ax.set_title('Reclassification Impact Analysis')
                ax.set_xticks(x)
                ax.set_xticklabels(scenarios, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
            plt.savefig(self.output_dir / 'reclassification_improvements.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_stability_results(self, stability_results: Dict[str, Any]) -> None:
        """
        Save stability analysis results.
        
        Parameters:
        -----------
        stability_results : Dict[str, Any]
            Results from stability analysis
        """
        # Save detailed results
        with open(self.output_dir / 'stability_analysis_results.json', 'w') as f:
            json.dump(stability_results, f, indent=2, default=str)
        
        # Save stability report
        if 'stability_report' in stability_results:
            with open(self.output_dir / 'stability_report.txt', 'w') as f:
                f.write(stability_results['stability_report'])
        
        # Save reclassification comparison table
        if 'reclassification_results' in stability_results:
            reclass_results = stability_results['reclassification_results']
            if reclass_results['before_after_comparison']:
                comparison_df = pd.DataFrame(reclass_results['before_after_comparison'])
                comparison_df.to_csv(self.output_dir / 'reclassification_comparison.csv', index=False)
        
        self.stability_results = stability_results

if __name__ == "__main__":
    print("Stability Analysis")
    print("Use this module to analyze stability under data revisions and reclassifications")
