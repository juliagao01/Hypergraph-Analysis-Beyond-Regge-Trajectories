"""
Parameterized Regge Analysis Runner

Runs the complete Regge trajectory analysis pipeline with configurable parameters
for reproducible, headless execution.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add the parent directory to the path to import analysis modules
sys.path.append(str(Path(__file__).parent.parent))

from regge_analysis.main import run_regge_analysis
from reproducibility.data_assembly import DeterministicDataAssembler

class ParameterizedReggeAnalyzer:
    """
    Parameterized Regge trajectory analysis runner.
    
    Provides configurable, reproducible analysis with:
    - Configurable parameters via settings file
    - Deterministic data assembly
    - Headless execution
    - Comprehensive output generation
    """
    
    def __init__(self, settings_file: Optional[str] = None):
        """
        Initialize parameterized analyzer.
        
        Parameters:
        -----------
        settings_file : str, optional
            Path to settings JSON file
        """
        self.settings = self._load_settings(settings_file)
        self.data_assembler = DeterministicDataAssembler(
            output_dir=self.settings['data']['output_dir']
        )
    
    def _load_settings(self, settings_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load analysis settings.
        
        Parameters:
        -----------
        settings_file : str, optional
            Path to settings file
            
        Returns:
        --------
        Dict[str, Any]
            Analysis settings
        """
        if settings_file and Path(settings_file).exists():
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        else:
            # Default settings
            settings = {
                'data': {
                    'particle_family': 'Delta',
                    'include_width_systematic': True,
                    'kappa': 0.25,
                    'output_dir': 'data_snapshots',
                    'use_frozen_snapshot': False,
                    'snapshot_path': None
                },
                'analysis': {
                    'bootstrap_n': 1000,
                    'window_factor': 2.0,
                    'canonical_alpha0': 0.0,
                    'canonical_alphap': 0.9,
                    'J_range': [0.5, 9.5],
                    'J_step': 0.5
                },
                'output': {
                    'output_dir': 'analysis_results',
                    'save_plots': True,
                    'save_reports': True,
                    'plot_format': 'png',
                    'dpi': 300
                },
                'validation': {
                    'run_tests': True,
                    'tolerance': 1e-6,
                    'sanity_bounds': {
                        'alphap_min': 0.1,
                        'alphap_max': 2.0,
                        'chi2_dof_max': 10.0
                    }
                }
            }
        
        return settings
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the complete Regge analysis pipeline.
        
        Returns:
        --------
        Dict[str, Any]
            Analysis results and metadata
        """
        print("=" * 60)
        print("PARAMETERIZED REGGE ANALYSIS")
        print("=" * 60)
        print(f"Particle Family: {self.settings['data']['particle_family']}")
        print(f"Bootstrap N: {self.settings['analysis']['bootstrap_n']}")
        print(f"Output Directory: {self.settings['output']['output_dir']}")
        print("=" * 60)
        
        # Step 1: Data Assembly
        print("\n1. DATA ASSEMBLY")
        print("-" * 30)
        
        if self.settings['data']['use_frozen_snapshot'] and self.settings['data']['snapshot_path']:
            # Load frozen snapshot
            data, metadata = self.data_assembler.load_frozen_snapshot(
                self.settings['data']['snapshot_path']
            )
            print(f"Loaded frozen snapshot: {self.settings['data']['snapshot_path']}")
        else:
            # Create new snapshot
            snapshot_path = self.data_assembler.create_frozen_snapshot(
                particle_family=self.settings['data']['particle_family'],
                include_width_systematic=self.settings['data']['include_width_systematic'],
                kappa=self.settings['data']['kappa']
            )
            data, metadata = self.data_assembler.load_frozen_snapshot(snapshot_path)
        
        print(f"Data points: {len(data)}")
        print(f"Mass range: {data['mass_GeV'].min():.3f} - {data['mass_GeV'].max():.3f} GeV")
        
        # Step 2: Run Analysis
        print("\n2. REGGE ANALYSIS")
        print("-" * 30)
        
        # Prepare data for analysis
        analysis_data = data[['M2_GeV2', 'J', 'total_M2_sigma_GeV2']].copy()
        analysis_data.columns = ['M2_GeV2', 'J', 'M2_sigma_GeV2']
        
        # Run analysis
        results = run_regge_analysis(
            data=analysis_data,
            output_dir=self.settings['output']['output_dir']
        )
        
        # Step 3: Validation
        print("\n3. VALIDATION")
        print("-" * 30)
        
        validation_results = self._validate_results(results)
        
        # Step 4: Generate Reports
        print("\n4. REPORT GENERATION")
        print("-" * 30)
        
        self._generate_reports(results, validation_results, metadata)
        
        # Step 5: Save Settings and Results
        print("\n5. SAVE RESULTS")
        print("-" * 30)
        
        self._save_results(results, validation_results, metadata)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return {
            'settings': self.settings,
            'results': results,
            'validation': validation_results,
            'metadata': metadata
        }
    
    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results against sanity bounds.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        validation = {
            'passed': True,
            'checks': {},
            'warnings': []
        }
        
        # Extract key results
        best_fit = results.get('results', {})
        if not best_fit:
            validation['passed'] = False
            validation['warnings'].append("No fit results found")
            return validation
        
        # Check α' bounds
        alphap = best_fit.get('alphap', 0)
        alphap_min = self.settings['validation']['sanity_bounds']['alphap_min']
        alphap_max = self.settings['validation']['sanity_bounds']['alphap_max']
        
        if alphap < alphap_min or alphap > alphap_max:
            validation['checks']['alphap_bounds'] = False
            validation['warnings'].append(f"α' = {alphap:.4f} outside bounds [{alphap_min}, {alphap_max}]")
        else:
            validation['checks']['alphap_bounds'] = True
        
        # Check χ²/dof
        chi2_dof = best_fit.get('chi2_dof', float('inf'))
        chi2_max = self.settings['validation']['sanity_bounds']['chi2_dof_max']
        
        if chi2_dof > chi2_max:
            validation['checks']['chi2_dof'] = False
            validation['warnings'].append(f"χ²/dof = {chi2_dof:.3f} > {chi2_max}")
        else:
            validation['checks']['chi2_dof'] = True
        
        # Check bootstrap results
        bootstrap_results = results.get('bootstrap_analysis', {})
        if bootstrap_results:
            bootstrap_std = bootstrap_results.get('alphap', {}).get('std', 0)
            if bootstrap_std <= 0:
                validation['checks']['bootstrap_variance'] = False
                validation['warnings'].append("Bootstrap variance is zero")
            else:
                validation['checks']['bootstrap_variance'] = True
        
        # Check predictions
        predictions = results.get('predictions', pd.DataFrame())
        if not predictions.empty:
            # Check if predictions are sorted by J
            if not predictions['J'].is_monotonic_increasing:
                validation['checks']['predictions_sorted'] = False
                validation['warnings'].append("Predictions not sorted by J")
            else:
                validation['checks']['predictions_sorted'] = True
        
        # Overall validation
        validation['passed'] = all(validation['checks'].values())
        
        # Print validation results
        print("Validation Results:")
        for check, passed in validation['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {check}: {status}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        return validation
    
    def _generate_reports(self, results: Dict[str, Any], 
                         validation_results: Dict[str, Any],
                         metadata: Dict[str, Any]) -> None:
        """
        Generate comprehensive analysis reports.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results
        validation_results : Dict[str, Any]
            Validation results
        metadata : Dict[str, Any]
            Data metadata
        """
        output_dir = Path(self.settings['output']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Generate reproducibility report
        reproducibility_report = self._generate_reproducibility_report(results, metadata)
        with open(output_dir / 'reproducibility_report.txt', 'w') as f:
            f.write(reproducibility_report)
        
        # Generate parameter report
        parameter_report = self._generate_parameter_report(results)
        with open(output_dir / 'parameter_report.txt', 'w') as f:
            f.write(parameter_report)
        
        # Generate validation report
        validation_report = self._generate_validation_report(validation_results)
        with open(output_dir / 'validation_report.txt', 'w') as f:
            f.write(validation_report)
        
        print(f"Reports saved to {output_dir}")
    
    def _generate_reproducibility_report(self, results: Dict[str, Any], 
                                       metadata: Dict[str, Any]) -> str:
        """Generate reproducibility report."""
        report = []
        report.append("=" * 60)
        report.append("REPRODUCIBILITY REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("Analysis Parameters:")
        report.append(f"  Particle Family: {self.settings['data']['particle_family']}")
        report.append(f"  PDG Version: {metadata.get('pdg_version', 'Unknown')}")
        report.append(f"  Data Timestamp: {metadata.get('data_timestamp', 'Unknown')}")
        report.append(f"  Bootstrap N: {self.settings['analysis']['bootstrap_n']}")
        report.append(f"  Width Systematic: {self.settings['data']['include_width_systematic']}")
        report.append(f"  Kappa: {self.settings['data']['kappa']}")
        report.append("")
        report.append("Key Results:")
        if 'results' in results:
            best_fit = results['results']
            report.append(f"  α₀ = {best_fit.get('alpha0', 'N/A'):.4f} ± {best_fit.get('alpha0_err', 'N/A'):.4f}")
            report.append(f"  α' = {best_fit.get('alphap', 'N/A'):.4f} ± {best_fit.get('alphap_err', 'N/A'):.4f}")
            report.append(f"  χ²/dof = {best_fit.get('chi2_dof', 'N/A'):.3f}")
        report.append("")
        report.append("Reproduction Command:")
        report.append(f"  python run_regge_analysis.py --settings settings.json")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _generate_parameter_report(self, results: Dict[str, Any]) -> str:
        """Generate parameter report."""
        report = []
        report.append("=" * 60)
        report.append("PARAMETER REPORT")
        report.append("=" * 60)
        report.append("")
        
        if 'results' in results:
            best_fit = results['results']
            report.append("Fitted Parameters:")
            report.append(f"  α₀ = {best_fit.get('alpha0', 'N/A'):.6f} ± {best_fit.get('alpha0_err', 'N/A'):.6f}")
            report.append(f"  α' = {best_fit.get('alphap', 'N/A'):.6f} ± {best_fit.get('alphap_err', 'N/A'):.6f}")
            report.append("")
            report.append("Goodness of Fit:")
            report.append(f"  χ² = {best_fit.get('chi2', 'N/A'):.3f}")
            report.append(f"  dof = {best_fit.get('dof', 'N/A')}")
            report.append(f"  χ²/dof = {best_fit.get('chi2_dof', 'N/A'):.3f}")
            report.append(f"  R² = {best_fit.get('r_squared', 'N/A'):.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate validation report."""
        report = []
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("Validation Status:")
        status = "PASSED" if validation_results['passed'] else "FAILED"
        report.append(f"  Overall: {status}")
        report.append("")
        
        report.append("Individual Checks:")
        for check, passed in validation_results['checks'].items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report.append(f"  {check}: {status}")
        
        if validation_results['warnings']:
            report.append("")
            report.append("Warnings:")
            for warning in validation_results['warnings']:
                report.append(f"  - {warning}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _save_results(self, results: Dict[str, Any], 
                     validation_results: Dict[str, Any],
                     metadata: Dict[str, Any]) -> None:
        """
        Save analysis results and settings.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results
        validation_results : Dict[str, Any]
            Validation results
        metadata : Dict[str, Any]
            Data metadata
        """
        output_dir = Path(self.settings['output']['output_dir'])
        
        # Save settings
        with open(output_dir / 'analysis_settings.json', 'w') as f:
            json.dump(self.settings, f, indent=2)
        
        # Save results summary
        summary = {
            'settings': self.settings,
            'metadata': metadata,
            'validation': validation_results,
            'key_results': {}
        }
        
        if 'results' in results:
            best_fit = results['results']
            summary['key_results'] = {
                'alpha0': best_fit.get('alpha0'),
                'alpha0_err': best_fit.get('alpha0_err'),
                'alphap': best_fit.get('alphap'),
                'alphap_err': best_fit.get('alphap_err'),
                'chi2_dof': best_fit.get('chi2_dof'),
                'r_squared': best_fit.get('r_squared')
            }
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}")

def create_settings_template(output_file: str = "settings_template.json") -> None:
    """
    Create a settings template file.
    
    Parameters:
    -----------
    output_file : str
        Output file path
    """
    template = {
        "data": {
            "particle_family": "Delta",
            "include_width_systematic": True,
            "kappa": 0.25,
            "output_dir": "data_snapshots",
            "use_frozen_snapshot": False,
            "snapshot_path": null
        },
        "analysis": {
            "bootstrap_n": 1000,
            "window_factor": 2.0,
            "canonical_alpha0": 0.0,
            "canonical_alphap": 0.9,
            "J_range": [0.5, 9.5],
            "J_step": 0.5
        },
        "output": {
            "output_dir": "analysis_results",
            "save_plots": True,
            "save_reports": True,
            "plot_format": "png",
            "dpi": 300
        },
        "validation": {
            "run_tests": True,
            "tolerance": 1e-6,
            "sanity_bounds": {
                "alphap_min": 0.1,
                "alphap_max": 2.0,
                "chi2_dof_max": 10.0
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Settings template created: {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Parameterized Regge Analysis")
    parser.add_argument("--settings", "-s", type=str, help="Settings JSON file")
    parser.add_argument("--particle-family", "-p", type=str, default="Delta", 
                       help="Particle family (Delta, Nstar)")
    parser.add_argument("--bootstrap-n", "-b", type=int, default=1000,
                       help="Number of bootstrap iterations")
    parser.add_argument("--output-dir", "-o", type=str, default="analysis_results",
                       help="Output directory")
    parser.add_argument("--create-template", action="store_true",
                       help="Create settings template and exit")
    
    args = parser.parse_args()
    
    if args.create_template:
        create_settings_template()
        return
    
    # Create analyzer
    analyzer = ParameterizedReggeAnalyzer(args.settings)
    
    # Override settings with command line arguments
    if args.particle_family:
        analyzer.settings['data']['particle_family'] = args.particle_family
    if args.bootstrap_n:
        analyzer.settings['analysis']['bootstrap_n'] = args.bootstrap_n
    if args.output_dir:
        analyzer.settings['output']['output_dir'] = args.output_dir
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Exit with error code if validation failed
    if not results['validation']['passed']:
        print("\nValidation failed! Check warnings above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
