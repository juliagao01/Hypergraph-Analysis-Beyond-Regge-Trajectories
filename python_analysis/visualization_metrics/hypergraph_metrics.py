"""
Quantitative Hypergraph Metrics

Computes comprehensive metrics from particle decay hypergraphs including:
- Incidence graph metrics (modularity, clustering, assortativity, path length)
- Product co-occurrence projections
- Motif and cycle analysis
- Community detection and subgroup discovery
- Visual complexity and readability metrics
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings
from collections import defaultdict, Counter

class HypergraphMetricsAnalyzer:
    """
    Analyzes hypergraphs to extract quantitative metrics and insights.
    
    Implements:
    - Incidence graph metrics
    - Product co-occurrence projections
    - Motif and cycle analysis
    - Community detection
    - Visual complexity metrics
    """
    
    def __init__(self, hypergraph_data: pd.DataFrame):
        """
        Initialize hypergraph metrics analyzer.
        
        Parameters:
        -----------
        hypergraph_data : pd.DataFrame
            Hypergraph data with columns for decay information
        """
        self.hypergraph_data = hypergraph_data.copy()
        self.incidence_graphs = {}
        self.projection_graphs = {}
        self.metrics = {}
        
    def compute_incidence_graph_metrics(self, particle_family: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert hypergraph to incidence graph and compute metrics.
        
        Parameters:
        -----------
        particle_family : str, optional
            Filter for specific particle family
            
        Returns:
        --------
        Dict containing incidence graph metrics
        """
        # Filter data if family specified
        if particle_family:
            data = self.hypergraph_data[self.hypergraph_data['family'] == particle_family]
        else:
            data = self.hypergraph_data
        
        # Create incidence graph
        G = nx.Graph()
        
        # Add nodes for particles and decay channels
        for _, row in data.iterrows():
            parent = row['parent_particle']
            decay_products = row['decay_products']
            
            # Add parent node
            G.add_node(parent, type='particle')
            
            # Add decay channel node
            decay_id = f"decay_{parent}_{hash(tuple(sorted(decay_products)))}"
            G.add_node(decay_id, type='decay', products=decay_products)
            
            # Connect parent to decay channel
            G.add_edge(parent, decay_id)
            
            # Connect decay channel to products
            for product in decay_products:
                G.add_node(product, type='particle')
                G.add_edge(decay_id, product)
        
        # Store graph
        key = particle_family if particle_family else 'global'
        self.incidence_graphs[key] = G
        
        # Compute metrics
        metrics = {}
        
        # Basic graph metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Connectivity metrics
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
        else:
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
            metrics['diameter'] = nx.diameter(subgraph)
            metrics['n_components'] = nx.number_connected_components(G)
        
        # Clustering metrics
        metrics['average_clustering'] = nx.average_clustering(G)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Degree metrics
        degrees = [d for n, d in G.degree()]
        metrics['average_degree'] = np.mean(degrees)
        metrics['degree_std'] = np.std(degrees)
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        
        # Centrality metrics
        metrics['average_betweenness'] = np.mean(list(nx.betweenness_centrality(G).values()))
        metrics['average_closeness'] = np.mean(list(nx.closeness_centrality(G).values()))
        
        # Modularity (community structure)
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            modularity = nx.community.modularity(G, communities)
            metrics['modularity'] = modularity
            metrics['n_communities'] = len(communities)
        except:
            metrics['modularity'] = 0.0
            metrics['n_communities'] = 1
        
        # Particle type analysis
        particle_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'particle']
        decay_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'decay']
        
        metrics['n_particles'] = len(particle_nodes)
        metrics['n_decay_channels'] = len(decay_nodes)
        metrics['particle_decay_ratio'] = len(particle_nodes) / len(decay_nodes) if len(decay_nodes) > 0 else 0
        
        return metrics
    
    def compute_projection_metrics(self, particle_family: Optional[str] = None) -> Dict[str, Any]:
        """
        Create product co-occurrence projection and compute metrics.
        
        Parameters:
        -----------
        particle_family : str, optional
            Filter for specific particle family
            
        Returns:
        --------
        Dict containing projection metrics
        """
        # Filter data if family specified
        if particle_family:
            data = self.hypergraph_data[self.hypergraph_data['family'] == particle_family]
        else:
            data = self.hypergraph_data
        
        # Create product co-occurrence graph
        G_proj = nx.Graph()
        
        # Count co-occurrences
        co_occurrences = defaultdict(int)
        
        for _, row in data.iterrows():
            products = row['decay_products']
            
            # Add edges between all pairs of products
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    pair = tuple(sorted([products[i], products[j]]))
                    co_occurrences[pair] += 1
        
        # Add nodes and edges to projection
        for (prod1, prod2), weight in co_occurrences.items():
            G_proj.add_node(prod1)
            G_proj.add_node(prod2)
            G_proj.add_edge(prod1, prod2, weight=weight)
        
        # Store graph
        key = f"{particle_family}_projection" if particle_family else 'global_projection'
        self.projection_graphs[key] = G_proj
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = G_proj.number_of_nodes()
        metrics['n_edges'] = G_proj.number_of_edges()
        metrics['density'] = nx.density(G_proj)
        
        # Connectivity metrics
        if nx.is_connected(G_proj):
            metrics['average_path_length'] = nx.average_shortest_path_length(G_proj)
            metrics['diameter'] = nx.diameter(G_proj)
        else:
            largest_cc = max(nx.connected_components(G_proj), key=len)
            subgraph = G_proj.subgraph(largest_cc)
            metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
            metrics['diameter'] = nx.diameter(subgraph)
            metrics['n_components'] = nx.number_connected_components(G_proj)
        
        # Clustering metrics
        metrics['average_clustering'] = nx.average_clustering(G_proj)
        metrics['transitivity'] = nx.transitivity(G_proj)
        
        # Degree metrics
        degrees = [d for n, d in G_proj.degree()]
        metrics['average_degree'] = np.mean(degrees)
        metrics['degree_std'] = np.std(degrees)
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G_proj)
        
        # Weight metrics
        weights = [attr['weight'] for _, _, attr in G_proj.edges(data=True)]
        metrics['average_weight'] = np.mean(weights)
        metrics['weight_std'] = np.std(weights)
        
        # Modularity
        try:
            communities = list(nx.community.greedy_modularity_communities(G_proj))
            modularity = nx.community.modularity(G_proj, communities)
            metrics['modularity'] = modularity
            metrics['n_communities'] = len(communities)
        except:
            metrics['modularity'] = 0.0
            metrics['n_communities'] = 1
        
        return metrics
    
    def analyze_motifs_and_cycles(self, particle_family: Optional[str] = None, 
                                n_randomizations: int = 100) -> Dict[str, Any]:
        """
        Analyze motifs and cycles in the projection graph.
        
        Parameters:
        -----------
        particle_family : str, optional
            Filter for specific particle family
        n_randomizations : int
            Number of randomizations for baseline comparison
            
        Returns:
        --------
        Dict containing motif and cycle analysis
        """
        # Get projection graph
        key = f"{particle_family}_projection" if particle_family else 'global_projection'
        if key not in self.projection_graphs:
            self.compute_projection_metrics(particle_family)
        
        G = self.projection_graphs[key]
        
        # Compute motif counts
        motif_counts = self._count_motifs(G)
        
        # Compute cycle counts
        cycle_counts = self._count_cycles(G)
        
        # Generate random baseline
        random_motifs = []
        random_cycles = []
        
        for _ in range(n_randomizations):
            # Create degree-preserving randomization
            G_random = nx.double_edge_swap(G.copy(), nswap=G.number_of_edges() * 2)
            
            # Count motifs in random graph
            random_motif_counts = self._count_motifs(G_random)
            random_motifs.append(random_motif_counts)
            
            # Count cycles in random graph
            random_cycle_counts = self._count_cycles(G_random)
            random_cycles.append(random_cycle_counts)
        
        # Compute z-scores
        motif_z_scores = {}
        for motif_type in motif_counts:
            random_values = [counts[motif_type] for counts in random_motifs]
            mean_random = np.mean(random_values)
            std_random = np.std(random_values)
            
            if std_random > 0:
                z_score = (motif_counts[motif_type] - mean_random) / std_random
            else:
                z_score = 0.0
            
            motif_z_scores[motif_type] = z_score
        
        cycle_z_scores = {}
        for cycle_length in cycle_counts:
            random_values = [counts[cycle_length] for counts in random_cycles]
            mean_random = np.mean(random_values)
            std_random = np.std(random_values)
            
            if std_random > 0:
                z_score = (cycle_counts[cycle_length] - mean_random) / std_random
            else:
                z_score = 0.0
            
            cycle_z_scores[cycle_length] = z_score
        
        return {
            'motif_counts': motif_counts,
            'cycle_counts': cycle_counts,
            'motif_z_scores': motif_z_scores,
            'cycle_z_scores': cycle_z_scores,
            'random_baseline': {
                'motif_means': {motif: np.mean([counts[motif] for counts in random_motifs]) 
                               for motif in motif_counts},
                'cycle_means': {length: np.mean([counts[length] for counts in random_cycles]) 
                               for length in cycle_counts}
            }
        }
    
    def _count_motifs(self, G: nx.Graph) -> Dict[str, int]:
        """Count small motifs in the graph."""
        motifs = {
            'triangles': 0,
            'squares': 0,
            'stars_3': 0,  # 3-leaf stars
            'stars_4': 0,  # 4-leaf stars
        }
        
        # Count triangles
        motifs['triangles'] = sum(1 for _ in nx.triangles(G).values()) // 3
        
        # Count squares (4-cycles)
        squares = 0
        for node in G.nodes():
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor > node:  # Avoid double counting
                    common_neighbors = neighbors & set(G.neighbors(neighbor))
                    squares += len(common_neighbors)
        motifs['squares'] = squares // 4
        
        # Count stars
        for node in G.nodes():
            degree = G.degree(node)
            if degree >= 3:
                motifs['stars_3'] += 1
            if degree >= 4:
                motifs['stars_4'] += 1
        
        return motifs
    
    def _count_cycles(self, G: nx.Graph, max_length: int = 6) -> Dict[int, int]:
        """Count cycles of different lengths."""
        cycle_counts = {}
        
        for length in range(3, max_length + 1):
            cycles = 0
            for node in G.nodes():
                # Find simple cycles starting from this node
                cycles += len(list(nx.simple_cycles(G.subgraph([node] + list(G.neighbors(node))), length)))
            cycle_counts[length] = cycles // (2 * length)  # Avoid double counting
        
        return cycle_counts
    
    def detect_communities(self, particle_family: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect communities in the projection graph and analyze alignment with known taxonomies.
        
        Parameters:
        -----------
        particle_family : str, optional
            Filter for specific particle family
            
        Returns:
        --------
        Dict containing community analysis results
        """
        # Get projection graph
        key = f"{particle_family}_projection" if particle_family else 'global_projection'
        if key not in self.projection_graphs:
            self.compute_projection_metrics(particle_family)
        
        G = self.projection_graphs[key]
        
        # Detect communities using multiple algorithms
        communities = {}
        
        # Louvain method
        try:
            louvain_communities = list(nx.community.louvain_communities(G))
            communities['louvain'] = louvain_communities
        except:
            communities['louvain'] = [set(G.nodes())]
        
        # Label propagation
        try:
            label_communities = list(nx.community.label_propagation_communities(G))
            communities['label_propagation'] = label_communities
        except:
            communities['label_propagation'] = [set(G.nodes())]
        
        # Spectral clustering
        try:
            spectral_communities = list(nx.community.spectral_clustering(G, k=min(5, len(G.nodes()))))
            communities['spectral'] = spectral_communities
        except:
            communities['spectral'] = [set(G.nodes())]
        
        # Analyze community properties
        community_analysis = {}
        
        for method, comms in communities.items():
            analysis = {
                'n_communities': len(comms),
                'community_sizes': [len(comm) for comm in comms],
                'modularity': nx.community.modularity(G, comms),
                'coverage': sum(len(comm) for comm in comms) / len(G.nodes())
            }
            
            # Compute NMI and ARI if we have ground truth labels
            if 'particle_type' in self.hypergraph_data.columns:
                ground_truth = self._get_ground_truth_labels(G.nodes())
                predicted_labels = self._communities_to_labels(comms, G.nodes())
                
                analysis['nmi'] = normalized_mutual_info_score(ground_truth, predicted_labels)
                analysis['ari'] = adjusted_rand_score(ground_truth, predicted_labels)
            
            community_analysis[method] = analysis
        
        return {
            'communities': communities,
            'analysis': community_analysis,
            'best_method': max(community_analysis.keys(), 
                             key=lambda x: community_analysis[x]['modularity'])
        }
    
    def _get_ground_truth_labels(self, nodes: List[str]) -> List[int]:
        """Extract ground truth labels for nodes."""
        labels = []
        for node in nodes:
            # Extract particle type from node name or data
            if 'pion' in node.lower():
                labels.append(0)
            elif 'kaon' in node.lower():
                labels.append(1)
            elif 'proton' in node.lower() or 'neutron' in node.lower():
                labels.append(2)
            else:
                labels.append(3)  # Other
        return labels
    
    def _communities_to_labels(self, communities: List[set], nodes: List[str]) -> List[int]:
        """Convert community assignments to label array."""
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
        
        return [node_to_community.get(node, 0) for node in nodes]
    
    def compute_visual_complexity_metrics(self, layout_positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Compute visual complexity and readability metrics.
        
        Parameters:
        -----------
        layout_positions : Dict[str, Tuple[float, float]]
            Node positions from layout algorithm
            
        Returns:
        --------
        Dict containing visual complexity metrics
        """
        # Get projection graph for edge analysis
        if 'global_projection' in self.projection_graphs:
            G = self.projection_graphs['global_projection']
        else:
            self.compute_projection_metrics()
            G = self.projection_graphs['global_projection']
        
        metrics = {}
        
        # Edge crossing analysis
        edge_crossings = self._count_edge_crossings(G, layout_positions)
        metrics['edge_crossings'] = edge_crossings
        metrics['crossing_density'] = edge_crossings / G.number_of_edges() if G.number_of_edges() > 0 else 0
        
        # Edge length analysis
        edge_lengths = []
        for u, v in G.edges():
            if u in layout_positions and v in layout_positions:
                pos_u = layout_positions[u]
                pos_v = layout_positions[v]
                length = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
                edge_lengths.append(length)
        
        metrics['average_edge_length'] = np.mean(edge_lengths) if edge_lengths else 0
        metrics['edge_length_std'] = np.std(edge_lengths) if edge_lengths else 0
        
        # Node overlap analysis
        node_overlaps = self._count_node_overlaps(layout_positions)
        metrics['node_overlaps'] = node_overlaps
        metrics['overlap_density'] = node_overlaps / len(layout_positions) if layout_positions else 0
        
        # Layout area analysis
        if layout_positions:
            x_coords = [pos[0] for pos in layout_positions.values()]
            y_coords = [pos[1] for pos in layout_positions.values()]
            
            metrics['layout_width'] = max(x_coords) - min(x_coords)
            metrics['layout_height'] = max(y_coords) - min(y_coords)
            metrics['layout_area'] = metrics['layout_width'] * metrics['layout_height']
            metrics['node_density'] = len(layout_positions) / metrics['layout_area'] if metrics['layout_area'] > 0 else 0
        
        return metrics
    
    def _count_edge_crossings(self, G: nx.Graph, positions: Dict[str, Tuple[float, float]]) -> int:
        """Count edge crossings in the layout."""
        crossings = 0
        edges = list(G.edges())
        
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                u1, v1 = edges[i]
                u2, v2 = edges[j]
                
                if u1 in positions and v1 in positions and u2 in positions and v2 in positions:
                    if self._edges_intersect(positions[u1], positions[v1], positions[u2], positions[v2]):
                        crossings += 1
        
        return crossings
    
    def _edges_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _count_node_overlaps(self, positions: Dict[str, Tuple[float, float]], 
                           threshold: float = 0.1) -> int:
        """Count overlapping nodes in the layout."""
        overlaps = 0
        nodes = list(positions.keys())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pos1 = positions[nodes[i]]
                pos2 = positions[nodes[j]]
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance < threshold:
                    overlaps += 1
        
        return overlaps
    
    def compare_to_baseline_layout(self, layout_positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Compare current layout to baseline (simple DAG/forest layout).
        
        Parameters:
        -----------
        layout_positions : Dict[str, Tuple[float, float]]
            Current layout positions
            
        Returns:
        --------
        Dict containing comparison metrics
        """
        # Get current metrics
        current_metrics = self.compute_visual_complexity_metrics(layout_positions)
        
        # Create baseline layout (simple hierarchical)
        baseline_positions = self._create_baseline_layout()
        baseline_metrics = self.compute_visual_complexity_metrics(baseline_positions)
        
        # Compute improvements
        improvements = {}
        
        for metric in current_metrics:
            if metric in baseline_metrics and baseline_metrics[metric] > 0:
                improvement = (baseline_metrics[metric] - current_metrics[metric]) / baseline_metrics[metric] * 100
                improvements[f"{metric}_improvement_pct"] = improvement
        
        return {
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvements
        }
    
    def _create_baseline_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create a simple baseline layout."""
        if 'global_projection' in self.projection_graphs:
            G = self.projection_graphs['global_projection']
        else:
            self.compute_projection_metrics()
            G = self.projection_graphs['global_projection']
        
        # Simple hierarchical layout
        positions = {}
        nodes = list(G.nodes())
        
        for i, node in enumerate(nodes):
            x = i % 10  # Simple grid
            y = i // 10
            positions[node] = (x, y)
        
        return positions
    
    def generate_comprehensive_report(self, particle_families: List[str] = None) -> str:
        """
        Generate comprehensive metrics report.
        
        Parameters:
        -----------
        particle_families : List[str], optional
            List of particle families to analyze
            
        Returns:
        --------
        str
            Formatted report
        """
        if particle_families is None:
            particle_families = ['Delta', 'Nstar']
        
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE HYPERGRAPH METRICS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Global metrics
        report.append("GLOBAL METRICS")
        report.append("-" * 40)
        
        global_incidence = self.compute_incidence_graph_metrics()
        global_projection = self.compute_projection_metrics()
        
        report.append("Incidence Graph:")
        report.append(f"  Nodes: {global_incidence['n_nodes']}")
        report.append(f"  Edges: {global_incidence['n_edges']}")
        report.append(f"  Density: {global_incidence['density']:.4f}")
        report.append(f"  Modularity: {global_incidence['modularity']:.4f}")
        report.append(f"  Average Clustering: {global_incidence['average_clustering']:.4f}")
        report.append("")
        
        report.append("Product Projection:")
        report.append(f"  Nodes: {global_projection['n_nodes']}")
        report.append(f"  Edges: {global_projection['n_edges']}")
        report.append(f"  Density: {global_projection['density']:.4f}")
        report.append(f"  Modularity: {global_projection['modularity']:.4f}")
        report.append("")
        
        # Per-family metrics
        for family in particle_families:
            report.append(f"{family.upper()} FAMILY METRICS")
            report.append("-" * 40)
            
            incidence = self.compute_incidence_graph_metrics(family)
            projection = self.compute_projection_metrics(family)
            
            report.append("Incidence Graph:")
            report.append(f"  Nodes: {incidence['n_nodes']}")
            report.append(f"  Edges: {incidence['n_edges']}")
            report.append(f"  Modularity: {incidence['modularity']:.4f}")
            report.append(f"  Clustering: {incidence['average_clustering']:.4f}")
            report.append(f"  Path Length: {incidence['average_path_length']:.3f}")
            report.append("")
            
            report.append("Product Projection:")
            report.append(f"  Nodes: {projection['n_nodes']}")
            report.append(f"  Edges: {projection['n_edges']}")
            report.append(f"  Modularity: {projection['modularity']:.4f}")
            report.append(f"  Clustering: {projection['average_clustering']:.4f}")
            report.append("")
        
        # Motif analysis
        report.append("MOTIF AND CYCLE ANALYSIS")
        report.append("-" * 40)
        
        motif_analysis = self.analyze_motifs_and_cycles()
        
        report.append("Motif Z-Scores (vs random baseline):")
        for motif, z_score in motif_analysis['motif_z_scores'].items():
            significance = "***" if abs(z_score) > 3 else "**" if abs(z_score) > 2 else "*" if abs(z_score) > 1 else ""
            report.append(f"  {motif}: {z_score:.2f} {significance}")
        
        report.append("")
        report.append("Cycle Z-Scores:")
        for length, z_score in motif_analysis['cycle_z_scores'].items():
            significance = "***" if abs(z_score) > 3 else "**" if abs(z_score) > 2 else "*" if abs(z_score) > 1 else ""
            report.append(f"  {length}-cycles: {z_score:.2f} {significance}")
        
        report.append("")
        
        # Community detection
        report.append("COMMUNITY DETECTION")
        report.append("-" * 40)
        
        community_analysis = self.detect_communities()
        
        for method, analysis in community_analysis['analysis'].items():
            report.append(f"{method.upper()} Method:")
            report.append(f"  Communities: {analysis['n_communities']}")
            report.append(f"  Modularity: {analysis['modularity']:.4f}")
            if 'nmi' in analysis:
                report.append(f"  NMI: {analysis['nmi']:.4f}")
            if 'ari' in analysis:
                report.append(f"  ARI: {analysis['ari']:.4f}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def create_metrics_table(self, particle_families: List[str] = None) -> pd.DataFrame:
        """
        Create a summary table of metrics for all families.
        
        Parameters:
        -----------
        particle_families : List[str], optional
            List of particle families to include
            
        Returns:
        --------
        pd.DataFrame
            Summary metrics table
        """
        if particle_families is None:
            particle_families = ['Delta', 'Nstar']
        
        table_data = []
        
        for family in particle_families:
            # Compute metrics
            incidence = self.compute_incidence_graph_metrics(family)
            projection = self.compute_projection_metrics(family)
            
            # Create row
            row = {
                'Family': family,
                'Incidence_Nodes': incidence['n_nodes'],
                'Incidence_Edges': incidence['n_edges'],
                'Incidence_Modularity': incidence['modularity'],
                'Incidence_Clustering': incidence['average_clustering'],
                'Projection_Nodes': projection['n_nodes'],
                'Projection_Edges': projection['n_edges'],
                'Projection_Modularity': projection['modularity'],
                'Projection_Clustering': projection['average_clustering'],
                'Average_Path_Length': incidence['average_path_length'],
                'Degree_Assortativity': incidence['degree_assortativity']
            }
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)

if __name__ == "__main__":
    # Example usage
    print("Hypergraph Metrics Analyzer")
    print("Use this module to analyze particle decay hypergraphs")
