"""
PDG Cross-Check for Regge Trajectory Predictions

Searches for existing PDG entries near predicted masses to validate
predictions and identify true gaps in the particle spectrum.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from particle import Particle
from particle.particle import enums
import warnings

class PDGCrossChecker:
    """
    Cross-checks Regge trajectory predictions with PDG data.
    
    Searches for existing particles with the same quantum numbers
    near predicted masses to validate predictions and identify gaps.
    """
    
    def __init__(self, predictions: pd.DataFrame, pdg_data: pd.DataFrame):
        """
        Initialize PDG cross-checker.
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions (J, M_GeV, M_sigma_GeV)
        pdg_data : pd.DataFrame
            DataFrame with PDG particle data
        """
        self.predictions = predictions.copy()
        self.pdg_data = pdg_data.copy()
        
    def find_nearby_candidates(self, window_GeV: float = 0.15, 
                             same_J: bool = True) -> pd.DataFrame:
        """
        Find PDG candidates near predicted masses.
        
        Parameters:
        -----------
        window_GeV : float
            Mass window for searching (in GeV)
        same_J : bool
            Whether to require same J value
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with nearby candidates for each prediction
        """
        results = []
        
        for _, pred in self.predictions.iterrows():
            J_pred = pred['J']
            M_pred = pred['M_GeV']
            M_sigma = pred['M_sigma_GeV']
            
            # Search window
            M_min = M_pred - window_GeV
            M_max = M_pred + window_GeV
            
            # Filter PDG data
            mask = (
                (self.pdg_data['mass_GeV'] >= M_min) &
                (self.pdg_data['mass_GeV'] <= M_max)
            )
            
            if same_J:
                mask &= (self.pdg_data['J'] == J_pred)
            
            candidates = self.pdg_data[mask].copy()
            
            if len(candidates) > 0:
                # Calculate distances and add metadata
                candidates['distance_GeV'] = np.abs(candidates['mass_GeV'] - M_pred)
                candidates['z_score'] = (candidates['mass_GeV'] - M_pred) / M_sigma
                candidates['predicted_J'] = J_pred
                candidates['predicted_M_GeV'] = M_pred
                candidates['predicted_M_sigma_GeV'] = M_sigma
                
                # Sort by distance
                candidates = candidates.sort_values('distance_GeV')
                
                results.append({
                    'predicted_J': J_pred,
                    'predicted_M_GeV': M_pred,
                    'predicted_M_sigma_GeV': M_sigma,
                    'n_candidates': len(candidates),
                    'closest_candidate': candidates.iloc[0]['name'] if len(candidates) > 0 else None,
                    'closest_distance_GeV': candidates.iloc[0]['distance_GeV'] if len(candidates) > 0 else None,
                    'closest_z_score': candidates.iloc[0]['z_score'] if len(candidates) > 0 else None,
                    'candidates': candidates
                })
            else:
                results.append({
                    'predicted_J': J_pred,
                    'predicted_M_GeV': M_pred,
                    'predicted_M_sigma_GeV': M_sigma,
                    'n_candidates': 0,
                    'closest_candidate': None,
                    'closest_distance_GeV': None,
                    'closest_z_score': None,
                    'candidates': pd.DataFrame()
                })
        
        return pd.DataFrame(results)
    
    def analyze_gaps(self, cross_check_results: pd.DataFrame, 
                    z_threshold: float = 2.0) -> Dict[str, Any]:
        """
        Analyze gaps in the particle spectrum.
        
        Parameters:
        -----------
        cross_check_results : pd.DataFrame
            Results from find_nearby_candidates
        z_threshold : float
            Z-score threshold for considering a gap significant
            
        Returns:
        --------
        Dict[str, Any]
            Gap analysis results
        """
        # Identify true gaps (no nearby candidates)
        true_gaps = cross_check_results[cross_check_results['n_candidates'] == 0]
        
        # Identify predictions with distant candidates (potential gaps)
        distant_candidates = cross_check_results[
            (cross_check_results['n_candidates'] > 0) &
            (cross_check_results['closest_z_score'].abs() > z_threshold)
        ]
        
        # Identify well-matched predictions
        well_matched = cross_check_results[
            (cross_check_results['n_candidates'] > 0) &
            (cross_check_results['closest_z_score'].abs() <= z_threshold)
        ]
        
        return {
            'total_predictions': len(cross_check_results),
            'true_gaps': len(true_gaps),
            'potential_gaps': len(distant_candidates),
            'well_matched': len(well_matched),
            'gap_fraction': len(true_gaps) / len(cross_check_results),
            'true_gap_predictions': true_gaps,
            'potential_gap_predictions': distant_candidates,
            'well_matched_predictions': well_matched
        }
    
    def plot_cross_check_results(self, cross_check_results: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cross-check results showing predictions vs nearby candidates.
        
        Parameters:
        -----------
        cross_check_results : pd.DataFrame
            Results from find_nearby_candidates
        save_path : str, optional
            Path to save plot
            
        Returns:
        --------
        plt.Figure
            The generated plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Predictions vs candidates
        predictions = cross_check_results['predicted_M_GeV']
        J_values = cross_check_results['predicted_J']
        
        # Color code by number of candidates
        colors = []
        for n_cand in cross_check_results['n_candidates']:
            if n_cand == 0:
                colors.append('red')  # True gaps
            elif n_cand == 1:
                colors.append('orange')  # Single candidate
            else:
                colors.append('blue')  # Multiple candidates
        
        scatter = ax1.scatter(J_values, predictions, c=colors, alpha=0.7, s=50)
        
        # Add candidate points
        for _, row in cross_check_results.iterrows():
            if row['n_candidates'] > 0:
                candidates = row['candidates']
                for _, candidate in candidates.iterrows():
                    ax1.scatter(candidate['J'], candidate['mass_GeV'], 
                              color='gray', alpha=0.5, s=20, marker='x')
        
        ax1.set_xlabel('J (Spin)')
        ax1.set_ylabel('M (GeV)')
        ax1.set_title('Predictions vs PDG Candidates')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='True Gaps'),
            Patch(facecolor='orange', alpha=0.7, label='Single Candidate'),
            Patch(facecolor='blue', alpha=0.7, label='Multiple Candidates'),
            plt.Line2D([0], [0], marker='x', color='gray', linestyle='', 
                      markersize=8, label='PDG Candidates')
        ]
        ax1.legend(handles=legend_elements)
        
        # Plot 2: Z-scores distribution
        z_scores = cross_check_results['closest_z_score'].dropna()
        
        if len(z_scores) > 0:
            ax2.hist(z_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', label='Perfect Match')
            ax2.axvline(1, color='orange', linestyle='--', label='1σ')
            ax2.axvline(-1, color='orange', linestyle='--')
            ax2.axvline(2, color='red', linestyle=':', label='2σ')
            ax2.axvline(-2, color='red', linestyle=':')
            
            ax2.set_xlabel('Z-Score (Distance in σ)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Prediction-Candidate Distances')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No candidates found', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('No Candidates Found')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross-check results plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_gap_report(self, gap_analysis: Dict[str, Any]) -> str:
        """
        Generate a text report summarizing gap analysis.
        
        Parameters:
        -----------
        gap_analysis : Dict[str, Any]
            Results from analyze_gaps
            
        Returns:
        --------
        str
            Formatted gap report
        """
        report = []
        report.append("=" * 60)
        report.append("REGGЕ TRAJECTORY GAP ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"  Total predictions: {gap_analysis['total_predictions']}")
        report.append(f"  True gaps (no candidates): {gap_analysis['true_gaps']}")
        report.append(f"  Potential gaps (distant candidates): {gap_analysis['potential_gaps']}")
        report.append(f"  Well-matched predictions: {gap_analysis['well_matched']}")
        report.append(f"  Gap fraction: {gap_analysis['gap_fraction']:.2%}")
        report.append("")
        
        # True gaps
        if len(gap_analysis['true_gap_predictions']) > 0:
            report.append("TRUE GAPS (No nearby candidates):")
            for _, gap in gap_analysis['true_gap_predictions'].iterrows():
                report.append(f"  J = {gap['predicted_J']:.1f}: "
                            f"M = {gap['predicted_M_GeV']:.3f} ± {gap['predicted_M_sigma_GeV']:.3f} GeV")
            report.append("")
        
        # Potential gaps
        if len(gap_analysis['potential_gap_predictions']) > 0:
            report.append("POTENTIAL GAPS (Distant candidates):")
            for _, gap in gap_analysis['potential_gap_predictions'].iterrows():
                report.append(f"  J = {gap['predicted_J']:.1f}: "
                            f"M = {gap['predicted_M_GeV']:.3f} ± {gap['predicted_M_sigma_GeV']:.3f} GeV")
                report.append(f"    Closest candidate: {gap['closest_candidate']} "
                            f"(z = {gap['closest_z_score']:.2f})")
            report.append("")
        
        # Well-matched predictions
        if len(gap_analysis['well_matched_predictions']) > 0:
            report.append("WELL-MATCHED PREDICTIONS:")
            for _, match in gap_analysis['well_matched_predictions'].iterrows():
                report.append(f"  J = {match['predicted_J']:.1f}: "
                            f"M = {match['predicted_M_GeV']:.3f} ± {match['predicted_M_sigma_GeV']:.3f} GeV")
                report.append(f"    Matches: {match['closest_candidate']} "
                            f"(z = {match['closest_z_score']:.2f})")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_gap_table(self, gap_analysis: Dict[str, Any], 
                        output_path: str) -> None:
        """
        Export gap analysis results to CSV.
        
        Parameters:
        -----------
        gap_analysis : Dict[str, Any]
            Results from analyze_gaps
        output_path : str
            Path to save CSV file
        """
        # Combine all predictions with gap classification
        all_predictions = []
        
        # True gaps
        for _, gap in gap_analysis['true_gap_predictions'].iterrows():
            all_predictions.append({
                'J': gap['predicted_J'],
                'M_GeV': gap['predicted_M_GeV'],
                'M_sigma_GeV': gap['predicted_M_sigma_GeV'],
                'gap_type': 'true_gap',
                'n_candidates': 0,
                'closest_candidate': None,
                'closest_distance_GeV': None,
                'z_score': None
            })
        
        # Potential gaps
        for _, gap in gap_analysis['potential_gap_predictions'].iterrows():
            all_predictions.append({
                'J': gap['predicted_J'],
                'M_GeV': gap['predicted_M_GeV'],
                'M_sigma_GeV': gap['predicted_M_sigma_GeV'],
                'gap_type': 'potential_gap',
                'n_candidates': gap['n_candidates'],
                'closest_candidate': gap['closest_candidate'],
                'closest_distance_GeV': gap['closest_distance_GeV'],
                'z_score': gap['closest_z_score']
            })
        
        # Well-matched
        for _, match in gap_analysis['well_matched_predictions'].iterrows():
            all_predictions.append({
                'J': match['predicted_J'],
                'M_GeV': match['predicted_M_GeV'],
                'M_sigma_GeV': match['predicted_M_sigma_GeV'],
                'gap_type': 'well_matched',
                'n_candidates': match['n_candidates'],
                'closest_candidate': match['closest_candidate'],
                'closest_distance_GeV': match['closest_distance_GeV'],
                'z_score': match['closest_z_score']
            })
        
        # Create DataFrame and save
        gap_table = pd.DataFrame(all_predictions)
        gap_table = gap_table.sort_values(['gap_type', 'J'])
        gap_table.to_csv(output_path, index=False)
        
        print(f"Gap analysis table saved to {output_path}")
    
    def compare_with_literature_ranges(self, alphap: float, alphap_err: float) -> Dict[str, Any]:
        """
        Compare fitted α' with literature ranges.
        
        Parameters:
        -----------
        alphap : float
            Fitted slope parameter
        alphap_err : float
            Uncertainty in slope parameter
            
        Returns:
        --------
        Dict[str, Any]
            Comparison with literature
        """
        # Literature ranges (GeV⁻²)
        literature_ranges = {
            'mesons': (0.7, 1.1),
            'baryons': (0.8, 1.2),
            'general': (0.6, 1.3)
        }
        
        comparisons = {}
        
        for particle_type, (min_val, max_val) in literature_ranges.items():
            # Check if within range
            within_range = min_val <= alphap <= max_val
            
            # Calculate z-scores to range boundaries
            z_min = (alphap - min_val) / alphap_err
            z_max = (alphap - max_val) / alphap_err
            
            # Distance to range center
            range_center = (min_val + max_val) / 2
            z_center = (alphap - range_center) / alphap_err
            
            comparisons[particle_type] = {
                'range': (min_val, max_val),
                'within_range': within_range,
                'z_score_to_center': z_center,
                'z_score_to_min': z_min,
                'z_score_to_max': z_max,
                'distance_to_center_GeV2': alphap - range_center
            }
        
        return comparisons
