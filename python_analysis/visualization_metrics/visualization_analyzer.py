"""
Visualization Analysis and Metrics Integration

Integrates hypergraph metrics with visualization analysis to provide quantitative
insights into particle decay network visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from pathlib import Path
import json

from .hypergraph_metrics import HypergraphMetricsAnalyzer

class VisualizationAnalyzer:
    """
    Comprehensive visualization analysis for particle decay networks.
    
    Integrates hypergraph metrics with visual complexity analysis to provide
    quantitative insights into network visualizations.
    """
    
    def __init__(self, hypergraph_data: pd.DataFrame, output_dir: str = "visualization_analysis"):
        """
        Initialize visualization analyzer.
        
        Parameters:
        -----------
        hypergraph_data : pd.DataFrame
            Hypergraph data with decay information
        output_dir : str
            Directory for saving analysis outputs
        """
        self.hypergraph_data = hypergraph_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metrics analyzer
        self.metrics_analyzer = HypergraphMetricsAnalyzer(hypergraph_data)
        
        # Analysis results storage
        self.analysis_results = {}
        
    def run_comprehensive_analysis(self, particle_families: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive visualization analysis.
        
        Parameters:
        -----------
        particle_families : List[str], optional
            List of particle families to analyze
            
        Returns:
        --------
        Dict containing comprehensive analysis results
        """
        if particle_families is None:
            particle_families = ['Delta', 'Nstar']
        
        print("=" * 60)
        print("COMPREHENSIVE VISUALIZATION ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Incidence Graph Metrics
        print("\n1. Computing Incidence Graph Metrics...")
        results['incidence_metrics'] = self._analyze_incidence_graphs(particle_families)
        
        # 2. Projection Metrics
        print("2. Computing Product Projection Metrics...")
        results['projection_metrics'] = self._analyze_projections(particle_families)
        
        # 3. Motif and Cycle Analysis
        print("3. Analyzing Motifs and Cycles...")
        results['motif_analysis'] = self._analyze_motifs_and_cycles(particle_families)
        
        # 4. Community Detection
        print("4. Detecting Communities...")
        results['community_analysis'] = self._analyze_communities(particle_families)
        
        # 5. Visual Complexity Analysis
        print("5. Analyzing Visual Complexity...")
        results['visual_complexity'] = self._analyze_visual_complexity()
        
        # 6. Generate Reports and Visualizations
        print("6. Generating Reports and Visualizations...")
        self._generate_reports_and_plots(results, particle_families)
        
        self.analysis_results = results
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return results
    
    def _analyze_incidence_graphs(self, particle_families: List[str]) -> Dict[str, Any]:
        """Analyze incidence graph metrics for all families."""
        results = {}
        
        # Global metrics
        results['global'] = self.metrics_analyzer.compute_incidence_graph_metrics()
        
        # Per-family metrics
        for family in particle_families:
            results[family] = self.metrics_analyzer.compute_incidence_graph_metrics(family)
        
        return results
    
    def _analyze_projections(self, particle_families: List[str]) -> Dict[str, Any]:
        """Analyze product projection metrics for all families."""
        results = {}
        
        # Global metrics
        results['global'] = self.metrics_analyzer.compute_projection_metrics()
        
        # Per-family metrics
        for family in particle_families:
            results[family] = self.metrics_analyzer.compute_projection_metrics(family)
        
        return results
    
    def _analyze_motifs_and_cycles(self, particle_families: List[str]) -> Dict[str, Any]:
        """Analyze motifs and cycles for all families."""
        results = {}
        
        # Global analysis
        results['global'] = self.metrics_analyzer.analyze_motifs_and_cycles()
        
        # Per-family analysis
        for family in particle_families:
            results[family] = self.metrics_analyzer.analyze_motifs_and_cycles(family)
        
        return results
    
    def _analyze_communities(self, particle_families: List[str]) -> Dict[str, Any]:
        """Analyze community detection for all families."""
        results = {}
        
        # Global analysis
        results['global'] = self.metrics_analyzer.detect_communities()
        
        # Per-family analysis
        for family in particle_families:
            results[family] = self.metrics_analyzer.detect_communities(family)
        
        return results
    
    def _analyze_visual_complexity(self) -> Dict[str, Any]:
        """Analyze visual complexity metrics."""
        # Create sample layout positions for analysis
        # In practice, these would come from your actual visualization layout
        sample_layout = self._create_sample_layout()
        
        # Compute visual complexity metrics
        complexity_metrics = self.metrics_analyzer.compute_visual_complexity_metrics(sample_layout)
        
        # Compare to baseline
        baseline_comparison = self.metrics_analyzer.compare_to_baseline_layout(sample_layout)
        
        return {
            'complexity_metrics': complexity_metrics,
            'baseline_comparison': baseline_comparison,
            'sample_layout': sample_layout
        }
    
    def _create_sample_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create sample layout positions for analysis."""
        # Get projection graph
        if 'global_projection' not in self.metrics_analyzer.projection_graphs:
            self.metrics_analyzer.compute_projection_metrics()
        
        G = self.metrics_analyzer.projection_graphs['global_projection']
        
        # Create spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        return pos
    
    def _generate_reports_and_plots(self, results: Dict[str, Any], particle_families: List[str]):
        """Generate comprehensive reports and visualizations."""
        
        # 1. Generate metrics report
        report = self._generate_metrics_report(results, particle_families)
        with open(self.output_dir / "visualization_metrics_report.txt", 'w') as f:
            f.write(report)
        
        # 2. Create metrics comparison table
        metrics_table = self._create_metrics_comparison_table(results, particle_families)
        metrics_table.to_csv(self.output_dir / "metrics_comparison.csv", index=False)
        
        # 3. Generate visualizations
        self._create_metrics_visualizations(results, particle_families)
        
        # 4. Generate summary statistics
        summary_stats = self._generate_summary_statistics(results, particle_families)
        with open(self.output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    def _generate_metrics_report(self, results: Dict[str, Any], particle_families: List[str]) -> str:
        """Generate comprehensive metrics report."""
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE VISUALIZATION METRICS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Analysis Date: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        report.append("Particle Families Analyzed: " + ", ".join(particle_families))
        report.append("")
        
        # 1. Incidence Graph Metrics
        report.append("1. INCIDENCE GRAPH METRICS")
        report.append("=" * 50)
        report.append("")
        
        for family in ['global'] + particle_families:
            metrics = results['incidence_metrics'][family]
            report.append(f"{family.upper()} Family:")
            report.append(f"  Nodes: {metrics['n_nodes']}")
            report.append(f"  Edges: {metrics['n_edges']}")
            report.append(f"  Density: {metrics['density']:.4f}")
            report.append(f"  Modularity: {metrics['modularity']:.4f}")
            report.append(f"  Average Clustering: {metrics['average_clustering']:.4f}")
            report.append(f"  Average Path Length: {metrics['average_path_length']:.3f}")
            report.append(f"  Degree Assortativity: {metrics['degree_assortativity']:.4f}")
            report.append("")
        
        # 2. Projection Metrics
        report.append("2. PRODUCT PROJECTION METRICS")
        report.append("=" * 50)
        report.append("")
        
        for family in ['global'] + particle_families:
            metrics = results['projection_metrics'][family]
            report.append(f"{family.upper()} Family:")
            report.append(f"  Nodes: {metrics['n_nodes']}")
            report.append(f"  Edges: {metrics['n_edges']}")
            report.append(f"  Density: {metrics['density']:.4f}")
            report.append(f"  Modularity: {metrics['modularity']:.4f}")
            report.append(f"  Average Clustering: {metrics['average_clustering']:.4f}")
            report.append(f"  Average Weight: {metrics.get('average_weight', 0):.3f}")
            report.append("")
        
        # 3. Motif Analysis
        report.append("3. MOTIF AND CYCLE ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        for family in ['global'] + particle_families:
            motif_data = results['motif_analysis'][family]
            report.append(f"{family.upper()} Family:")
            report.append("  Motif Z-Scores (vs random baseline):")
            for motif, z_score in motif_data['motif_z_scores'].items():
                significance = "***" if abs(z_score) > 3 else "**" if abs(z_score) > 2 else "*" if abs(z_score) > 1 else ""
                report.append(f"    {motif}: {z_score:.2f} {significance}")
            report.append("  Cycle Z-Scores:")
            for length, z_score in motif_data['cycle_z_scores'].items():
                significance = "***" if abs(z_score) > 3 else "**" if abs(z_score) > 2 else "*" if abs(z_score) > 1 else ""
                report.append(f"    {length}-cycles: {z_score:.2f} {significance}")
            report.append("")
        
        # 4. Community Detection
        report.append("4. COMMUNITY DETECTION")
        report.append("=" * 50)
        report.append("")
        
        for family in ['global'] + particle_families:
            comm_data = results['community_analysis'][family]
            report.append(f"{family.upper()} Family:")
            for method, analysis in comm_data['analysis'].items():
                report.append(f"  {method.upper()} Method:")
                report.append(f"    Communities: {analysis['n_communities']}")
                report.append(f"    Modularity: {analysis['modularity']:.4f}")
                if 'nmi' in analysis:
                    report.append(f"    NMI: {analysis['nmi']:.4f}")
                if 'ari' in analysis:
                    report.append(f"    ARI: {analysis['ari']:.4f}")
            report.append("")
        
        # 5. Visual Complexity
        report.append("5. VISUAL COMPLEXITY METRICS")
        report.append("=" * 50)
        report.append("")
        
        complexity = results['visual_complexity']
        current_metrics = complexity['baseline_comparison']['current_metrics']
        baseline_metrics = complexity['baseline_comparison']['baseline_metrics']
        improvements = complexity['baseline_comparison']['improvements']
        
        report.append("Current Layout Metrics:")
        report.append(f"  Edge Crossings: {current_metrics['edge_crossings']}")
        report.append(f"  Node Overlaps: {current_metrics['node_overlaps']}")
        report.append(f"  Average Edge Length: {current_metrics['average_edge_length']:.3f}")
        report.append(f"  Layout Area: {current_metrics['layout_area']:.3f}")
        report.append("")
        
        report.append("Baseline Layout Metrics:")
        report.append(f"  Edge Crossings: {baseline_metrics['edge_crossings']}")
        report.append(f"  Node Overlaps: {baseline_metrics['node_overlaps']}")
        report.append(f"  Average Edge Length: {baseline_metrics['average_edge_length']:.3f}")
        report.append(f"  Layout Area: {baseline_metrics['layout_area']:.3f}")
        report.append("")
        
        report.append("Improvements vs Baseline:")
        for metric, improvement in improvements.items():
            report.append(f"  {metric}: {improvement:.1f}%")
        report.append("")
        
        # 6. Key Findings
        report.append("6. KEY FINDINGS AND INSIGHTS")
        report.append("=" * 50)
        report.append("")
        
        key_findings = self._extract_key_findings(results, particle_families)
        for i, finding in enumerate(key_findings, 1):
            report.append(f"{i}. {finding}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _create_metrics_comparison_table(self, results: Dict[str, Any], particle_families: List[str]) -> pd.DataFrame:
        """Create comparison table of metrics across families."""
        table_data = []
        
        for family in ['global'] + particle_families:
            incidence = results['incidence_metrics'][family]
            projection = results['projection_metrics'][family]
            
            row = {
                'Family': family,
                'Incidence_Nodes': incidence['n_nodes'],
                'Incidence_Edges': incidence['n_edges'],
                'Incidence_Density': incidence['density'],
                'Incidence_Modularity': incidence['modularity'],
                'Incidence_Clustering': incidence['average_clustering'],
                'Incidence_Path_Length': incidence['average_path_length'],
                'Incidence_Assortativity': incidence['degree_assortativity'],
                'Projection_Nodes': projection['n_nodes'],
                'Projection_Edges': projection['n_edges'],
                'Projection_Density': projection['density'],
                'Projection_Modularity': projection['modularity'],
                'Projection_Clustering': projection['average_clustering'],
                'Projection_Average_Weight': projection.get('average_weight', 0)
            }
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def _create_metrics_visualizations(self, results: Dict[str, Any], particle_families: List[str]):
        """Create visualizations of the metrics."""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantitative Visualization Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Modularity Comparison
        ax1 = axes[0, 0]
        families = ['global'] + particle_families
        incidence_modularity = [results['incidence_metrics'][f]['modularity'] for f in families]
        projection_modularity = [results['projection_metrics'][f]['modularity'] for f in families]
        
        x = np.arange(len(families))
        width = 0.35
        
        ax1.bar(x - width/2, incidence_modularity, width, label='Incidence Graph', alpha=0.8)
        ax1.bar(x + width/2, projection_modularity, width, label='Product Projection', alpha=0.8)
        ax1.set_xlabel('Particle Family')
        ax1.set_ylabel('Modularity')
        ax1.set_title('Modularity Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(families, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Clustering Coefficient Comparison
        ax2 = axes[0, 1]
        incidence_clustering = [results['incidence_metrics'][f]['average_clustering'] for f in families]
        projection_clustering = [results['projection_metrics'][f]['average_clustering'] for f in families]
        
        ax2.bar(x - width/2, incidence_clustering, width, label='Incidence Graph', alpha=0.8)
        ax2.bar(x + width/2, projection_clustering, width, label='Product Projection', alpha=0.8)
        ax2.set_xlabel('Particle Family')
        ax2.set_ylabel('Average Clustering')
        ax2.set_title('Clustering Coefficient Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(families, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Motif Z-Scores
        ax3 = axes[0, 2]
        motif_types = ['triangles', 'squares', 'stars_3', 'stars_4']
        global_motif_scores = [results['motif_analysis']['global']['motif_z_scores'][m] for m in motif_types]
        
        colors = ['red' if abs(score) > 2 else 'blue' for score in global_motif_scores]
        bars = ax3.bar(motif_types, global_motif_scores, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Significance threshold')
        ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Motif Type')
        ax3.set_ylabel('Z-Score')
        ax3.set_title('Global Motif Z-Scores')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Community Detection Results
        ax4 = axes[1, 0]
        methods = ['louvain', 'label_propagation', 'spectral']
        global_comm_modularity = [results['community_analysis']['global']['analysis'][m]['modularity'] for m in methods]
        
        ax4.bar(methods, global_comm_modularity, alpha=0.8, color='green')
        ax4.set_xlabel('Community Detection Method')
        ax4.set_ylabel('Modularity')
        ax4.set_title('Community Detection Performance')
        ax4.grid(True, alpha=0.3)
        
        # 5. Visual Complexity Metrics
        ax5 = axes[1, 1]
        complexity = results['visual_complexity']
        current_metrics = complexity['baseline_comparison']['current_metrics']
        baseline_metrics = complexity['baseline_comparison']['baseline_metrics']
        
        metrics_names = ['Edge Crossings', 'Node Overlaps', 'Avg Edge Length']
        current_values = [current_metrics['edge_crossings'], current_metrics['node_overlaps'], 
                         current_metrics['average_edge_length']]
        baseline_values = [baseline_metrics['edge_crossings'], baseline_metrics['node_overlaps'], 
                          baseline_metrics['average_edge_length']]
        
        x_pos = np.arange(len(metrics_names))
        ax5.bar(x_pos - width/2, current_values, width, label='Current Layout', alpha=0.8)
        ax5.bar(x_pos + width/2, baseline_values, width, label='Baseline Layout', alpha=0.8)
        ax5.set_xlabel('Metric')
        ax5.set_ylabel('Value')
        ax5.set_title('Visual Complexity Comparison')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(metrics_names, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Network Size Comparison
        ax6 = axes[1, 2]
        incidence_nodes = [results['incidence_metrics'][f]['n_nodes'] for f in families]
        projection_nodes = [results['projection_metrics'][f]['n_nodes'] for f in families]
        
        ax6.bar(x - width/2, incidence_nodes, width, label='Incidence Graph', alpha=0.8)
        ax6.bar(x + width/2, projection_nodes, width, label='Product Projection', alpha=0.8)
        ax6.set_xlabel('Particle Family')
        ax6.set_ylabel('Number of Nodes')
        ax6.set_title('Network Size Comparison')
        ax6.set_xticks(x)
        ax6.set_xticklabels(families, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional specialized plots
        self._create_motif_analysis_plot(results)
        self._create_community_analysis_plot(results, particle_families)
    
    def _create_motif_analysis_plot(self, results: Dict[str, Any]):
        """Create detailed motif analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Motif and Cycle Analysis', fontsize=16, fontweight='bold')
        
        # Motif Z-scores heatmap
        ax1 = axes[0, 0]
        families = ['global', 'Delta', 'Nstar']
        motif_types = ['triangles', 'squares', 'stars_3', 'stars_4']
        
        motif_matrix = []
        for family in families:
            row = [results['motif_analysis'][family]['motif_z_scores'][m] for m in motif_types]
            motif_matrix.append(row)
        
        im1 = ax1.imshow(motif_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_xticks(range(len(motif_types)))
        ax1.set_yticks(range(len(families)))
        ax1.set_xticklabels(motif_types, rotation=45)
        ax1.set_yticklabels(families)
        ax1.set_title('Motif Z-Scores Heatmap')
        plt.colorbar(im1, ax=ax1)
        
        # Cycle Z-scores heatmap
        ax2 = axes[0, 1]
        cycle_lengths = [3, 4, 5, 6]
        
        cycle_matrix = []
        for family in families:
            row = [results['motif_analysis'][family]['cycle_z_scores'].get(l, 0) for l in cycle_lengths]
            cycle_matrix.append(row)
        
        im2 = ax2.imshow(cycle_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_xticks(range(len(cycle_lengths)))
        ax2.set_yticks(range(len(families)))
        ax2.set_xticklabels([f'{l}-cycles' for l in cycle_lengths], rotation=45)
        ax2.set_yticklabels(families)
        ax2.set_title('Cycle Z-Scores Heatmap')
        plt.colorbar(im2, ax=ax2)
        
        # Motif counts comparison
        ax3 = axes[1, 0]
        global_motif_counts = results['motif_analysis']['global']['motif_counts']
        motif_names = list(global_motif_counts.keys())
        motif_values = list(global_motif_counts.values())
        
        bars = ax3.bar(motif_names, motif_values, alpha=0.7, color='skyblue')
        ax3.set_xlabel('Motif Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Global Motif Counts')
        ax3.grid(True, alpha=0.3)
        
        # Cycle counts comparison
        ax4 = axes[1, 1]
        global_cycle_counts = results['motif_analysis']['global']['cycle_counts']
        cycle_names = [f'{l}-cycles' for l in global_cycle_counts.keys()]
        cycle_values = list(global_cycle_counts.values())
        
        bars = ax4.bar(cycle_names, cycle_values, alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Cycle Length')
        ax4.set_ylabel('Count')
        ax4.set_title('Global Cycle Counts')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "motif_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_community_analysis_plot(self, results: Dict[str, Any], particle_families: List[str]):
        """Create community analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Community Detection Analysis', fontsize=16, fontweight='bold')
        
        families = ['global'] + particle_families
        methods = ['louvain', 'label_propagation', 'spectral']
        
        # Modularity comparison
        ax1 = axes[0, 0]
        modularity_data = []
        for method in methods:
            row = [results['community_analysis'][family]['analysis'][method]['modularity'] 
                   for family in families]
            modularity_data.append(row)
        
        x = np.arange(len(families))
        width = 0.25
        
        for i, method in enumerate(methods):
            ax1.bar(x + i*width, modularity_data[i], width, label=method.capitalize(), alpha=0.8)
        
        ax1.set_xlabel('Particle Family')
        ax1.set_ylabel('Modularity')
        ax1.set_title('Community Detection Modularity')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(families, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Number of communities
        ax2 = axes[0, 1]
        n_communities_data = []
        for method in methods:
            row = [results['community_analysis'][family]['analysis'][method]['n_communities'] 
                   for family in families]
            n_communities_data.append(row)
        
        for i, method in enumerate(methods):
            ax2.bar(x + i*width, n_communities_data[i], width, label=method.capitalize(), alpha=0.8)
        
        ax2.set_xlabel('Particle Family')
        ax2.set_ylabel('Number of Communities')
        ax2.set_title('Number of Detected Communities')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(families, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # NMI scores (if available)
        ax3 = axes[1, 0]
        nmi_data = []
        for method in methods:
            row = []
            for family in families:
                analysis = results['community_analysis'][family]['analysis'][method]
                nmi = analysis.get('nmi', 0)
                row.append(nmi)
            nmi_data.append(row)
        
        for i, method in enumerate(methods):
            ax3.bar(x + i*width, nmi_data[i], width, label=method.capitalize(), alpha=0.8)
        
        ax3.set_xlabel('Particle Family')
        ax3.set_ylabel('Normalized Mutual Information')
        ax3.set_title('Community Quality (NMI)')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(families, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Community size distribution
        ax4 = axes[1, 1]
        global_louvain = results['community_analysis']['global']['analysis']['louvain']
        community_sizes = global_louvain['community_sizes']
        
        ax4.hist(community_sizes, bins=10, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Community Size')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Global Community Size Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "community_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _extract_key_findings(self, results: Dict[str, Any], particle_families: List[str]) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        # Modularity findings
        global_incidence_mod = results['incidence_metrics']['global']['modularity']
        global_proj_mod = results['projection_metrics']['global']['modularity']
        
        if global_incidence_mod > 0.3:
            findings.append(f"Strong community structure detected in incidence graph (modularity: {global_incidence_mod:.3f})")
        else:
            findings.append(f"Weak community structure in incidence graph (modularity: {global_incidence_mod:.3f})")
        
        if global_proj_mod > global_incidence_mod:
            findings.append(f"Product projection shows stronger community structure than incidence graph")
        
        # Motif findings
        global_motif_scores = results['motif_analysis']['global']['motif_z_scores']
        significant_motifs = [motif for motif, score in global_motif_scores.items() if abs(score) > 2]
        
        if significant_motifs:
            findings.append(f"Significant motif enrichment detected: {', '.join(significant_motifs)}")
        
        # Family comparison findings
        family_modularities = {}
        for family in particle_families:
            family_modularities[family] = results['incidence_metrics'][family]['modularity']
        
        best_family = max(family_modularities, key=family_modularities.get)
        findings.append(f"{best_family} family shows strongest community structure among analyzed families")
        
        # Visual complexity findings
        complexity = results['visual_complexity']
        improvements = complexity['baseline_comparison']['improvements']
        
        if 'edge_crossings_improvement_pct' in improvements and improvements['edge_crossings_improvement_pct'] > 0:
            findings.append(f"Current layout reduces edge crossings by {improvements['edge_crossings_improvement_pct']:.1f}% compared to baseline")
        
        # Community detection findings
        global_comm = results['community_analysis']['global']
        best_method = global_comm['best_method']
        best_modularity = global_comm['analysis'][best_method]['modularity']
        findings.append(f"{best_method.capitalize()} method provides best community detection (modularity: {best_modularity:.3f})")
        
        return findings
    
    def _generate_summary_statistics(self, results: Dict[str, Any], particle_families: List[str]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        summary = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'particle_families': particle_families,
            'total_incidence_nodes': results['incidence_metrics']['global']['n_nodes'],
            'total_incidence_edges': results['incidence_metrics']['global']['n_edges'],
            'total_projection_nodes': results['projection_metrics']['global']['n_nodes'],
            'total_projection_edges': results['projection_metrics']['global']['n_edges'],
            'global_incidence_modularity': results['incidence_metrics']['global']['modularity'],
            'global_projection_modularity': results['projection_metrics']['global']['modularity'],
            'significant_motifs': [motif for motif, score in results['motif_analysis']['global']['motif_z_scores'].items() 
                                 if abs(score) > 2],
            'best_community_method': results['community_analysis']['global']['best_method'],
            'visual_complexity_improvements': results['visual_complexity']['baseline_comparison']['improvements']
        }
        
        return summary

if __name__ == "__main__":
    print("Visualization Analyzer")
    print("Use this module to analyze particle decay network visualizations")
