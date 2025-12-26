"""
Demo: Quantitative Visualization Metrics Analysis

This script demonstrates how to use the visualization metrics framework
to analyze particle decay network visualizations and extract quantitative insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from visualization_metrics.visualization_analyzer import VisualizationAnalyzer

def create_sample_hypergraph_data():
    """
    Create sample hypergraph data for demonstration.
    
    Returns:
    --------
    pd.DataFrame
        Sample hypergraph data with decay information
    """
    # Sample particle decay data
    data = {
        'parent_particle': [
            'Delta(1232)', 'Delta(1600)', 'Delta(1620)', 'Delta(1700)',
            'Delta(1900)', 'Delta(1905)', 'Delta(1910)', 'Delta(1920)',
            'N(1440)', 'N(1520)', 'N(1535)', 'N(1650)', 'N(1675)',
            'N(1680)', 'N(1700)', 'N(1710)', 'N(1720)', 'N(1900)'
        ],
        'family': [
            'Delta', 'Delta', 'Delta', 'Delta', 'Delta', 'Delta', 'Delta', 'Delta',
            'Nstar', 'Nstar', 'Nstar', 'Nstar', 'Nstar', 'Nstar', 'Nstar', 'Nstar', 'Nstar', 'Nstar'
        ],
        'decay_products': [
            ['pion+', 'proton'],  # Delta(1232)
            ['pion+', 'proton'],  # Delta(1600)
            ['pion-', 'proton'],  # Delta(1620)
            ['pion+', 'proton'],  # Delta(1700)
            ['pion+', 'proton'],  # Delta(1900)
            ['pion+', 'proton'],  # Delta(1905)
            ['pion+', 'proton'],  # Delta(1910)
            ['pion+', 'proton'],  # Delta(1920)
            ['pion+', 'neutron'], # N(1440)
            ['pion+', 'neutron'], # N(1520)
            ['pion+', 'neutron'], # N(1535)
            ['pion+', 'neutron'], # N(1650)
            ['pion+', 'neutron'], # N(1675)
            ['pion+', 'neutron'], # N(1680)
            ['pion+', 'neutron'], # N(1700)
            ['pion+', 'neutron'], # N(1710)
            ['pion+', 'neutron'], # N(1720)
            ['pion+', 'neutron']  # N(1900)
        ],
        'particle_type': [
            'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon',
            'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon', 'baryon'
        ]
    }
    
    return pd.DataFrame(data)

def run_demo_analysis():
    """
    Run comprehensive demonstration of visualization metrics analysis.
    """
    print("=" * 80)
    print("QUANTITATIVE VISUALIZATION METRICS DEMO")
    print("=" * 80)
    print()
    
    # Create sample data
    print("1. Creating sample hypergraph data...")
    hypergraph_data = create_sample_hypergraph_data()
    print(f"   Created {len(hypergraph_data)} decay channels")
    print(f"   Particle families: {hypergraph_data['family'].unique()}")
    print()
    
    # Initialize analyzer
    print("2. Initializing visualization analyzer...")
    analyzer = VisualizationAnalyzer(hypergraph_data, output_dir="demo_visualization_analysis")
    print("   Analyzer initialized successfully")
    print()
    
    # Run comprehensive analysis
    print("3. Running comprehensive analysis...")
    particle_families = ['Delta', 'Nstar']
    results = analyzer.run_comprehensive_analysis(particle_families)
    print("   Analysis completed successfully")
    print()
    
    # Display key results
    print("4. Key Analysis Results:")
    print("-" * 40)
    
    # Global metrics
    global_incidence = results['incidence_metrics']['global']
    global_projection = results['projection_metrics']['global']
    
    print(f"Global Incidence Graph:")
    print(f"  Nodes: {global_incidence['n_nodes']}")
    print(f"  Edges: {global_incidence['n_edges']}")
    print(f"  Modularity: {global_incidence['modularity']:.4f}")
    print(f"  Clustering: {global_incidence['average_clustering']:.4f}")
    print()
    
    print(f"Global Product Projection:")
    print(f"  Nodes: {global_projection['n_nodes']}")
    print(f"  Edges: {global_projection['n_edges']}")
    print(f"  Modularity: {global_projection['modularity']:.4f}")
    print(f"  Clustering: {global_projection['average_clustering']:.4f}")
    print()
    
    # Per-family comparison
    print("Per-Family Comparison:")
    for family in particle_families:
        incidence = results['incidence_metrics'][family]
        projection = results['projection_metrics'][family]
        
        print(f"{family} Family:")
        print(f"  Incidence - Nodes: {incidence['n_nodes']}, Modularity: {incidence['modularity']:.4f}")
        print(f"  Projection - Nodes: {projection['n_nodes']}, Modularity: {projection['modularity']:.4f}")
    print()
    
    # Motif analysis
    print("Motif Analysis (Global):")
    global_motifs = results['motif_analysis']['global']
    for motif, z_score in global_motifs['motif_z_scores'].items():
        significance = "***" if abs(z_score) > 3 else "**" if abs(z_score) > 2 else "*" if abs(z_score) > 1 else ""
        print(f"  {motif}: z = {z_score:.2f} {significance}")
    print()
    
    # Community detection
    print("Community Detection (Global):")
    global_comm = results['community_analysis']['global']
    for method, analysis in global_comm['analysis'].items():
        print(f"  {method.capitalize()}: {analysis['n_communities']} communities, modularity: {analysis['modularity']:.4f}")
    print()
    
    # Visual complexity
    print("Visual Complexity Analysis:")
    complexity = results['visual_complexity']
    current_metrics = complexity['baseline_comparison']['current_metrics']
    improvements = complexity['baseline_comparison']['improvements']
    
    print(f"  Edge Crossings: {current_metrics['edge_crossings']}")
    print(f"  Node Overlaps: {current_metrics['node_overlaps']}")
    print(f"  Average Edge Length: {current_metrics['average_edge_length']:.3f}")
    
    if 'edge_crossings_improvement_pct' in improvements:
        print(f"  Edge Crossing Improvement: {improvements['edge_crossings_improvement_pct']:.1f}%")
    print()
    
    # Output files
    print("5. Generated Output Files:")
    print("-" * 40)
    output_files = [
        "visualization_metrics_report.txt",
        "metrics_comparison.csv",
        "summary_statistics.json",
        "metrics_visualization.png",
        "motif_analysis.png",
        "community_analysis.png"
    ]
    
    for file in output_files:
        file_path = analyzer.output_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
    print()
    
    # Key findings
    print("6. Key Findings:")
    print("-" * 40)
    key_findings = analyzer._extract_key_findings(results, particle_families)
    for i, finding in enumerate(key_findings, 1):
        print(f"  {i}. {finding}")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the generated reports in 'demo_visualization_analysis/'")
    print("2. Examine the visualization plots for insights")
    print("3. Use the metrics comparison table for quantitative analysis")
    print("4. Integrate these metrics into your paper's visualization section")
    
    return results

def demonstrate_individual_components():
    """
    Demonstrate individual components of the visualization metrics framework.
    """
    print("\n" + "=" * 80)
    print("INDIVIDUAL COMPONENTS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    hypergraph_data = create_sample_hypergraph_data()
    
    # Initialize metrics analyzer directly
    from visualization_metrics.hypergraph_metrics import HypergraphMetricsAnalyzer
    metrics_analyzer = HypergraphMetricsAnalyzer(hypergraph_data)
    
    print("\n1. Incidence Graph Metrics:")
    print("-" * 30)
    incidence_metrics = metrics_analyzer.compute_incidence_graph_metrics()
    print(f"Nodes: {incidence_metrics['n_nodes']}")
    print(f"Edges: {incidence_metrics['n_edges']}")
    print(f"Modularity: {incidence_metrics['modularity']:.4f}")
    print(f"Clustering: {incidence_metrics['average_clustering']:.4f}")
    
    print("\n2. Product Projection Metrics:")
    print("-" * 30)
    projection_metrics = metrics_analyzer.compute_projection_metrics()
    print(f"Nodes: {projection_metrics['n_nodes']}")
    print(f"Edges: {projection_metrics['n_edges']}")
    print(f"Modularity: {projection_metrics['modularity']:.4f}")
    print(f"Average Weight: {projection_metrics.get('average_weight', 0):.3f}")
    
    print("\n3. Motif Analysis:")
    print("-" * 30)
    motif_analysis = metrics_analyzer.analyze_motifs_and_cycles()
    print("Motif Z-Scores:")
    for motif, z_score in motif_analysis['motif_z_scores'].items():
        print(f"  {motif}: {z_score:.2f}")
    
    print("\n4. Community Detection:")
    print("-" * 30)
    community_analysis = metrics_analyzer.detect_communities()
    for method, analysis in community_analysis['analysis'].items():
        print(f"{method.capitalize()}: {analysis['n_communities']} communities, modularity: {analysis['modularity']:.4f}")
    
    print("\n5. Visual Complexity Metrics:")
    print("-" * 30)
    # Create sample layout
    sample_layout = {'node1': (0, 0), 'node2': (1, 1), 'node3': (2, 0)}
    complexity_metrics = metrics_analyzer.compute_visual_complexity_metrics(sample_layout)
    print(f"Edge Crossings: {complexity_metrics['edge_crossings']}")
    print(f"Node Overlaps: {complexity_metrics['node_overlaps']}")
    print(f"Average Edge Length: {complexity_metrics['average_edge_length']:.3f}")

def create_metrics_table_demo():
    """
    Demonstrate creating metrics comparison table.
    """
    print("\n" + "=" * 80)
    print("METRICS TABLE DEMONSTRATION")
    print("=" * 80)
    
    hypergraph_data = create_sample_hypergraph_data()
    metrics_analyzer = HypergraphMetricsAnalyzer(hypergraph_data)
    
    # Create metrics table
    particle_families = ['Delta', 'Nstar']
    metrics_table = metrics_analyzer.create_metrics_table(particle_families)
    
    print("\nMetrics Comparison Table:")
    print("-" * 50)
    print(metrics_table.to_string(index=False))
    
    # Save table
    metrics_table.to_csv("demo_metrics_table.csv", index=False)
    print(f"\nTable saved to 'demo_metrics_table.csv'")

if __name__ == "__main__":
    # Run main demo
    results = run_demo_analysis()
    
    # Demonstrate individual components
    demonstrate_individual_components()
    
    # Create metrics table demo
    create_metrics_table_demo()
    
    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 80)
    print("\nThe visualization metrics framework provides:")
    print("✓ Quantitative analysis of hypergraph structure")
    print("✓ Motif and cycle analysis with statistical significance")
    print("✓ Community detection and quality assessment")
    print("✓ Visual complexity metrics and layout comparison")
    print("✓ Comprehensive reporting and visualization generation")
    print("\nThese metrics can be integrated into your paper to provide")
    print("quantitative support for your visualization findings.")
