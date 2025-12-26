"""
Paper Hygiene: Export Utilities

Implements comprehensive paper hygiene including:
- Centralized figure/table exporters
- Provenance stamps and metadata
- Reproducible research tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from pathlib import Path
import datetime
import platform
import sys
import os
from dataclasses import dataclass, asdict
import hashlib
import warnings

@dataclass
class ProvenanceStamp:
    """Provenance information for reproducible research."""
    analysis_date: str
    pdg_snapshot_date: str
    software_versions: Dict[str, str]
    analysis_settings: Dict[str, Any]
    data_hash: str
    code_hash: str
    platform_info: Dict[str, str]

class ExportUtilities:
    """
    Comprehensive export utilities for paper hygiene.
    
    Provides:
    - Vector figure exports (PDF/SVG)
    - CSV table exports with metadata
    - Provenance stamps and tracking
    - Reproducible research tools
    """
    
    def __init__(self, output_dir: str = "paper_exports", 
                 provenance_file: str = "provenance.json"):
        """
        Initialize export utilities.
        
        Parameters:
        -----------
        output_dir : str
            Directory for exports
        provenance_file : str
            File to store provenance information
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.provenance_file = self.output_dir / provenance_file
        self.export_log = []
        
        # Initialize provenance
        self.provenance = self._initialize_provenance()
        
    def _initialize_provenance(self) -> ProvenanceStamp:
        """Initialize provenance information."""
        return ProvenanceStamp(
            analysis_date=datetime.datetime.now().isoformat(),
            pdg_snapshot_date="2024-01-01",  # Update with actual PDG date
            software_versions={
                'python': sys.version,
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'matplotlib': plt.matplotlib.__version__,
                'scipy': self._get_package_version('scipy'),
                'scikit-learn': self._get_package_version('sklearn'),
                'networkx': self._get_package_version('networkx')
            },
            analysis_settings={
                'kappa': 0.25,
                'bootstrap_n': 2000,
                'confidence_level': 0.68,
                'n_sigma_threshold': 2.0
            },
            data_hash="",  # Will be set when data is loaded
            code_hash="",  # Will be set when code is hashed
            platform_info={
                'platform': platform.platform(),
                'python_implementation': platform.python_implementation(),
                'python_version': platform.python_version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
        )
    
    def _get_package_version(self, package_name: str) -> str:
        """Get package version safely."""
        try:
            if package_name == 'scipy':
                import scipy
                return scipy.__version__
            elif package_name == 'sklearn':
                import sklearn
                return sklearn.__version__
            elif package_name == 'networkx':
                import networkx
                return networkx.__version__
            else:
                return "unknown"
        except ImportError:
            return "not_installed"
    
    def update_provenance(self, **kwargs) -> None:
        """
        Update provenance information.
        
        Parameters:
        -----------
        **kwargs : Any
            Key-value pairs to update in provenance
        """
        for key, value in kwargs.items():
            if hasattr(self.provenance, key):
                setattr(self.provenance, key, value)
            else:
                warnings.warn(f"Unknown provenance field: {key}")
    
    def compute_data_hash(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> str:
        """
        Compute hash of data for provenance tracking.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, Dict[str, Any]]
            Data to hash
            
        Returns:
        --------
        str
            SHA-256 hash of data
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to string representation
            data_str = data.to_string()
        elif isinstance(data, dict):
            # Convert dict to sorted string representation
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def export_figure(self, fig: plt.Figure, filename: str, 
                     formats: List[str] = ['pdf', 'svg', 'png'],
                     dpi: int = 300,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Export figure in multiple formats with metadata.
        
        Parameters:
        -----------
        fig : plt.Figure
            Matplotlib figure to export
        filename : str
            Base filename (without extension)
        formats : List[str]
            List of formats to export
        dpi : int
            DPI for raster formats
        metadata : Optional[Dict[str, Any]]
            Additional metadata to include
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping format to file path
        """
        exported_files = {}
        
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            
            if fmt in ['pdf', 'svg']:
                # Vector formats
                fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                           metadata=self._create_figure_metadata(metadata))
            else:
                # Raster formats
                fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                           metadata=self._create_figure_metadata(metadata))
            
            exported_files[fmt] = str(filepath)
            
            # Log export
            self.export_log.append({
                'type': 'figure',
                'filename': filename,
                'format': fmt,
                'filepath': str(filepath),
                'timestamp': datetime.datetime.now().isoformat(),
                'metadata': metadata
            })
        
        return exported_files
    
    def _create_figure_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Create metadata for figure exports."""
        base_metadata = {
            'Creator': 'Regge Trajectory Analysis',
            'CreationDate': datetime.datetime.now().isoformat(),
            'Software': f'Python {sys.version}',
            'AnalysisDate': self.provenance.analysis_date,
            'PDGSnapshotDate': self.provenance.pdg_snapshot_date
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        return base_metadata
    
    def export_table(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                    filename: str,
                    format: str = 'csv',
                    include_metadata: bool = True,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Export table with metadata.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, Dict[str, Any]]
            Data to export
        filename : str
            Base filename (without extension)
        format : str
            Export format ('csv', 'json', 'xlsx')
        include_metadata : bool
            Whether to include provenance metadata
        metadata : Optional[Dict[str, Any]]
            Additional metadata
            
        Returns:
        --------
        str
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.{format}"
        
        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)
        
        elif format == 'json':
            if isinstance(data, pd.DataFrame):
                data_dict = data.to_dict('records')
            else:
                data_dict = data
            
            export_data = {
                'data': data_dict,
                'metadata': self._create_table_metadata(metadata) if include_metadata else None,
                'provenance': asdict(self.provenance) if include_metadata else None
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'xlsx':
            if isinstance(data, pd.DataFrame):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='Data', index=False)
                    
                    if include_metadata:
                        metadata_df = pd.DataFrame([self._create_table_metadata(metadata)])
                        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        provenance_df = pd.DataFrame([asdict(self.provenance)])
                        provenance_df.to_excel(writer, sheet_name='Provenance', index=False)
            else:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    pd.DataFrame(data).to_excel(writer, sheet_name='Data', index=False)
        
        # Log export
        self.export_log.append({
            'type': 'table',
            'filename': filename,
            'format': format,
            'filepath': str(filepath),
            'timestamp': datetime.datetime.now().isoformat(),
            'metadata': metadata
        })
        
        return str(filepath)
    
    def _create_table_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create metadata for table exports."""
        base_metadata = {
            'export_date': datetime.datetime.now().isoformat(),
            'analysis_date': self.provenance.analysis_date,
            'pdg_snapshot_date': self.provenance.pdg_snapshot_date,
            'software_versions': self.provenance.software_versions,
            'analysis_settings': self.provenance.analysis_settings
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        return base_metadata
    
    def export_regge_analysis_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Export complete Regge analysis results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Complete analysis results
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping result type to file path
        """
        exported_files = {}
        
        # Export fit parameters
        if 'fit_results' in results:
            fit_data = {
                'parameters': results['fit_results']['parameters'].tolist(),
                'parameter_uncertainties': results['fit_results']['parameter_uncertainties'].tolist(),
                'chi2': results['fit_results']['chi2'],
                'chi2_dof': results['fit_results']['chi2_dof'],
                'r_squared': results['fit_results']['r_squared'],
                'dof': results['fit_results']['dof']
            }
            exported_files['fit_parameters'] = self.export_table(
                fit_data, 'regge_fit_parameters', 'json'
            )
        
        # Export predictions
        if 'predictions' in results:
            exported_files['predictions'] = self.export_table(
                results['predictions'], 'mass_predictions', 'csv'
            )
        
        # Export PDG cross-check results
        if 'cross_check_results' in results:
            exported_files['pdg_cross_check'] = self.export_table(
                results['cross_check_results'], 'pdg_cross_check_results', 'csv'
            )
        
        # Export diagnostics
        if 'diagnostics' in results:
            exported_files['diagnostics'] = self.export_table(
                results['diagnostics'], 'regge_diagnostics', 'json'
            )
        
        # Export hypergraph metrics
        if 'hypergraph_metrics' in results:
            exported_files['hypergraph_metrics'] = self.export_table(
                results['hypergraph_metrics'], 'hypergraph_metrics', 'json'
            )
        
        # Export motif analysis
        if 'motif_analysis' in results:
            exported_files['motif_analysis'] = self.export_table(
                results['motif_analysis'], 'motif_analysis', 'json'
            )
        
        return exported_files
    
    def export_comparison_results(self, comparison_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Export peer comparison results.
        
        Parameters:
        -----------
        comparison_results : Dict[str, Any]
            Comparison analysis results
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping result type to file path
        """
        exported_files = {}
        
        # Export comparison table
        if 'comparison_table' in comparison_results:
            exported_files['comparison_table'] = self.export_table(
                comparison_results['comparison_table'], 'method_comparison', 'csv'
            )
        
        # Export performance metrics
        if 'performance_metrics' in comparison_results:
            exported_files['performance_metrics'] = self.export_table(
                comparison_results['performance_metrics'], 'performance_metrics', 'json'
            )
        
        # Export baseline analysis
        if 'baseline_analysis' in comparison_results:
            exported_files['baseline_analysis'] = self.export_table(
                comparison_results['baseline_analysis'], 'baseline_analysis', 'json'
            )
        
        return exported_files
    
    def export_stability_results(self, stability_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Export stability analysis results.
        
        Parameters:
        -----------
        stability_results : Dict[str, Any]
            Stability analysis results
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping result type to file path
        """
        exported_files = {}
        
        # Export stability metrics
        if 'stability_metrics' in stability_results:
            exported_files['stability_metrics'] = self.export_table(
                asdict(stability_results['stability_metrics']), 'stability_metrics', 'json'
            )
        
        # Export data changes
        if 'data_changes' in stability_results:
            exported_files['data_changes'] = self.export_table(
                stability_results['data_changes'], 'data_changes', 'json'
            )
        
        # Export reclassification results
        if 'reclassification_results' in stability_results:
            exported_files['reclassification_results'] = self.export_table(
                stability_results['reclassification_results'], 'reclassification_results', 'json'
            )
        
        return exported_files
    
    def create_export_summary(self) -> pd.DataFrame:
        """
        Create summary of all exports.
        
        Returns:
        --------
        pd.DataFrame
            Summary table of all exports
        """
        if not self.export_log:
            return pd.DataFrame()
        
        summary_data = []
        for entry in self.export_log:
            summary_data.append({
                'Type': entry['type'],
                'Filename': entry['filename'],
                'Format': entry['format'],
                'Filepath': entry['filepath'],
                'Timestamp': entry['timestamp'],
                'HasMetadata': entry['metadata'] is not None
            })
        
        return pd.DataFrame(summary_data)
    
    def export_provenance_report(self) -> str:
        """
        Export comprehensive provenance report.
        
        Returns:
        --------
        str
            Path to provenance report
        """
        report_path = self.output_dir / 'provenance_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROVENANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Analysis Information:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Analysis Date: {self.provenance.analysis_date}\n")
            f.write(f"PDG Snapshot Date: {self.provenance.pdg_snapshot_date}\n")
            f.write(f"Data Hash: {self.provenance.data_hash}\n")
            f.write(f"Code Hash: {self.provenance.code_hash}\n\n")
            
            f.write("Software Versions:\n")
            f.write("-" * 30 + "\n")
            for package, version in self.provenance.software_versions.items():
                f.write(f"{package}: {version}\n")
            f.write("\n")
            
            f.write("Analysis Settings:\n")
            f.write("-" * 30 + "\n")
            for setting, value in self.provenance.analysis_settings.items():
                f.write(f"{setting}: {value}\n")
            f.write("\n")
            
            f.write("Platform Information:\n")
            f.write("-" * 30 + "\n")
            for info, value in self.provenance.platform_info.items():
                f.write(f"{info}: {value}\n")
            f.write("\n")
            
            f.write("Export Summary:\n")
            f.write("-" * 30 + "\n")
            summary_df = self.create_export_summary()
            if not summary_df.empty:
                f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        return str(report_path)
    
    def save_provenance(self) -> None:
        """Save provenance information to file."""
        with open(self.provenance_file, 'w') as f:
            json.dump(asdict(self.provenance), f, indent=2, default=str)
    
    def load_provenance(self) -> None:
        """Load provenance information from file."""
        if self.provenance_file.exists():
            with open(self.provenance_file, 'r') as f:
                provenance_dict = json.load(f)
                for key, value in provenance_dict.items():
                    if hasattr(self.provenance, key):
                        setattr(self.provenance, key, value)
    
    def create_reproducibility_package(self, 
                                     results: Dict[str, Any],
                                     comparison_results: Optional[Dict[str, Any]] = None,
                                     stability_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Create complete reproducibility package.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Main analysis results
        comparison_results : Optional[Dict[str, Any]]
            Peer comparison results
        stability_results : Optional[Dict[str, Any]]
            Stability analysis results
            
        Returns:
        --------
        str
            Path to reproducibility package
        """
        print("Creating reproducibility package...")
        
        # Export all results
        exported_files = {}
        
        # Main analysis results
        exported_files.update(self.export_regge_analysis_results(results))
        
        # Comparison results
        if comparison_results:
            exported_files.update(self.export_comparison_results(comparison_results))
        
        # Stability results
        if stability_results:
            exported_files.update(self.export_stability_results(stability_results))
        
        # Export summary
        summary_df = self.create_export_summary()
        exported_files['export_summary'] = self.export_table(
            summary_df, 'export_summary', 'csv'
        )
        
        # Export provenance report
        exported_files['provenance_report'] = self.export_provenance_report()
        
        # Save provenance
        self.save_provenance()
        
        # Create README for reproducibility package
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write("# Regge Trajectory Analysis - Reproducibility Package\n\n")
            f.write(f"**Analysis Date:** {self.provenance.analysis_date}\n")
            f.write(f"**PDG Snapshot Date:** {self.provenance.pdg_snapshot_date}\n\n")
            
            f.write("## Contents\n\n")
            f.write("This package contains all results, figures, and metadata needed to reproduce the analysis.\n\n")
            
            f.write("### Data Files\n")
            f.write("- `regge_fit_parameters.json`: Fitted Regge trajectory parameters\n")
            f.write("- `mass_predictions.csv`: Mass predictions with uncertainties\n")
            f.write("- `pdg_cross_check_results.csv`: PDG cross-check results\n")
            f.write("- `regge_diagnostics.json`: Fit diagnostics and residuals\n")
            f.write("- `hypergraph_metrics.json`: Hypergraph analysis metrics\n")
            f.write("- `motif_analysis.json`: Motif and cycle analysis results\n\n")
            
            if comparison_results:
                f.write("### Comparison Results\n")
                f.write("- `method_comparison.csv`: Hypergraph vs baseline comparison\n")
                f.write("- `performance_metrics.json`: Performance benchmarking results\n")
                f.write("- `baseline_analysis.json`: Traditional analysis results\n\n")
            
            if stability_results:
                f.write("### Stability Analysis\n")
                f.write("- `stability_metrics.json`: Stability under PDG updates\n")
                f.write("- `data_changes.json`: Changes between PDG versions\n")
                f.write("- `reclassification_results.json`: Reclassification scenarios\n\n")
            
            f.write("### Metadata\n")
            f.write("- `provenance_report.txt`: Complete provenance information\n")
            f.write("- `provenance.json`: Machine-readable provenance data\n")
            f.write("- `export_summary.csv`: Summary of all exported files\n\n")
            
            f.write("## Reproducibility\n\n")
            f.write("To reproduce this analysis:\n\n")
            f.write("1. Install required software versions (see provenance.json)\n")
            f.write("2. Use the same analysis settings (see provenance.json)\n")
            f.write("3. Run the analysis pipeline with the provided data\n")
            f.write("4. Compare results with the exported files\n\n")
            
            f.write("## Software Requirements\n\n")
            for package, version in self.provenance.software_versions.items():
                f.write(f"- {package}: {version}\n")
        
        exported_files['readme'] = str(readme_path)
        
        print(f"Reproducibility package created in: {self.output_dir}")
        return str(self.output_dir)

if __name__ == "__main__":
    print("Export Utilities")
    print("Use this module for paper hygiene and reproducible research exports")
