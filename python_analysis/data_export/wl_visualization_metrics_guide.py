"""
Wolfram Language Visualization Metrics Guide

This module provides code snippets to be added to the Wolfram Language notebook
for implementing quantitative visualization metrics including hypergraph analysis,
motif counting, community detection, and visual complexity metrics.
"""

# =============================================================================
# WOLFRAM LANGUAGE VISUALIZATION METRICS CODE (Add to your notebook)
# =============================================================================

VISUALIZATION_METRICS_CODE = '''
(* ===== Quantitative Visualization Metrics ===== *)

(* 1. Hypergraph → Incidence + Projections *)

(* Convert hypergraph to incidence graph *)
createIncidenceGraph[hypergraphData_] := Module[
  {G, parent, products, decayId},
  
  G = Graph[];
  
  Do[
    parent = row["parent_particle"];
    products = row["decay_products"];
    
    (* Add parent node *)
    G = AddVertex[G, parent, VertexLabels -> parent];
    
    (* Add decay channel node *)
    decayId = StringJoin["decay_", parent, "_", ToString[Hash[Sort[products]]]];
    G = AddVertex[G, decayId, VertexLabels -> decayId];
    
    (* Connect parent to decay channel *)
    G = AddEdge[G, parent -> decayId];
    
    (* Connect decay channel to products *)
    Do[
      G = AddVertex[G, product, VertexLabels -> product];
      G = AddEdge[G, decayId -> product],
      {product, products}
    ],
    {row, hypergraphData}
  ];
  
  G
];

(* Create product co-occurrence projection *)
createProductProjection[hypergraphData_] := Module[
  {G, coOccurrences, products, pairs},
  
  G = Graph[];
  coOccurrences = Association[];
  
  (* Count co-occurrences *)
  Do[
    products = row["decay_products"];
    
    (* Add edges between all pairs of products *)
    Do[
      Do[
        pairs = Sort[{products[[i]], products[[j]]}];
        If[KeyExistsQ[coOccurrences, pairs],
          coOccurrences[pairs] += 1,
          coOccurrences[pairs] = 1
        ],
        {j, i + 1, Length[products]}
      ],
      {i, 1, Length[products]}
    ],
    {row, hypergraphData}
  ];
  
  (* Add nodes and edges to projection *)
  Do[
    G = AddVertex[G, pair[[1]], VertexLabels -> pair[[1]]];
    G = AddVertex[G, pair[[2]], VertexLabels -> pair[[2]]];
    G = AddEdge[G, pair[[1]] -> pair[[2]], EdgeWeight -> weight],
    {pair, Keys[coOccurrences]},
    {weight, {coOccurrences[pair]}}
  ];
  
  G
];

(* Compute graph-level metrics *)
computeGraphMetrics[G_] := Module[
  {metrics},
  
  metrics = <|
    "n_nodes" -> VertexCount[G],
    "n_edges" -> EdgeCount[G],
    "density" -> EdgeCount[G] / (VertexCount[G] * (VertexCount[G] - 1) / 2),
    "average_degree" -> Mean[VertexDegree[G]],
    "degree_std" -> StandardDeviation[VertexDegree[G]],
    "average_clustering" -> Mean[LocalClusteringCoefficient[G]],
    "transitivity" -> GlobalClusteringCoefficient[G],
    "average_path_length" -> Mean[GraphDistanceMatrix[G]],
    "diameter" -> GraphDiameter[G],
    "degree_assortativity" -> DegreeAssortativity[G],
    "average_betweenness" -> Mean[BetweennessCentrality[G]],
    "average_closeness" -> Mean[ClosenessCentrality[G]]
  |>;
  
  (* Add modularity if communities can be detected *)
  Try[
    communities = FindGraphCommunities[G];
    modularity = Modularity[G, communities];
    metrics["modularity"] = modularity;
    metrics["n_communities"] = Length[communities],
    metrics["modularity"] = 0.0;
    metrics["n_communities"] = 1
  ];
  
  metrics
];

(* 2. Motif and Cycle Counts *)

(* Count small motifs *)
countMotifs[G_] := Module[
  {motifs = <||>},
  
  (* Count triangles *)
  motifs["triangles"] = Count[FindCycle[G, 3, All], _];
  
  (* Count squares (4-cycles) *)
  motifs["squares"] = Count[FindCycle[G, 4, All], _];
  
  (* Count stars *)
  motifs["stars_3"] = Count[VertexDegree[G], d_ /; d >= 3];
  motifs["stars_4"] = Count[VertexDegree[G], d_ /; d >= 4];
  
  motifs
];

(* Count cycles of different lengths *)
countCycles[G_, maxLength_: 6] := Module[
  {cycleCounts = <||>},
  
  Do[
    cycleCounts[length] = Count[FindCycle[G, length, All], _],
    {length, 3, maxLength}
  ];
  
  cycleCounts
];

(* Generate random baseline for comparison *)
generateRandomBaseline[G_, nRandomizations_: 100] := Module[
  {randomMotifs, randomCycles, baseline},
  
  randomMotifs = Table[
    GRandom = RandomGraph[VertexCount[G], EdgeCount[G]];
    countMotifs[GRandom],
    {nRandomizations}
  ];
  
  randomCycles = Table[
    GRandom = RandomGraph[VertexCount[G], EdgeCount[G]];
    countCycles[GRandom],
    {nRandomizations}
  ];
  
  baseline = <|
    "motif_means" -> <|
      "triangles" -> Mean[randomMotifs[[All, "triangles"]]],
      "squares" -> Mean[randomMotifs[[All, "squares"]]],
      "stars_3" -> Mean[randomMotifs[[All, "stars_3"]]],
      "stars_4" -> Mean[randomMotifs[[All, "stars_4"]]]
    |>,
    "cycle_means" -> <|
      "3" -> Mean[randomCycles[[All, 3]]],
      "4" -> Mean[randomCycles[[All, 4]]],
      "5" -> Mean[randomCycles[[All, 5]]],
      "6" -> Mean[randomCycles[[All, 6]]]
    |>
  |>;
  
  baseline
];

(* Compute z-scores *)
computeZScores[observed_, baseline_] := Module[
  {zScores = <||>},
  
  Do[
    mean = baseline["motif_means"][motif];
    std = StandardDeviation[Table[baseline["motif_means"][motif], {100}]];
    If[std > 0,
      zScores[motif] = (observed[motif] - mean) / std,
      zScores[motif] = 0.0
    ],
    {motif, Keys[baseline["motif_means"]]}
  ];
  
  zScores
];

(* 3. Subgroup Discovery *)

(* Community detection with multiple algorithms *)
detectCommunities[G_] := Module[
  {communities = <||>},
  
  (* Louvain method *)
  Try[
    communities["louvain"] = FindGraphCommunities[G, Method -> "Louvain"],
    communities["louvain"] = {VertexList[G]}
  ];
  
  (* Label propagation *)
  Try[
    communities["label_propagation"] = FindGraphCommunities[G, Method -> "LabelPropagation"],
    communities["label_propagation"] = {VertexList[G]}
  ];
  
  (* Spectral clustering *)
  Try[
    communities["spectral"] = FindGraphCommunities[G, Method -> "Spectral"],
    communities["spectral"] = {VertexList[G]}
  ];
  
  communities
];

(* Analyze community properties *)
analyzeCommunities[G_, communities_] := Module[
  {analysis = <||>},
  
  Do[
    analysis[method] = <|
      "n_communities" -> Length[communities[method]],
      "community_sizes" -> Length /@ communities[method],
      "modularity" -> Modularity[G, communities[method]],
      "coverage" -> Total[Length /@ communities[method]] / VertexCount[G]
    |>,
    {method, Keys[communities]}
  ];
  
  analysis
];

(* Compare communities to known taxonomies *)
compareToTaxonomy[communities_, taxonomyLabels_] := Module[
  {comparison = <||>},
  
  Do[
    (* Convert communities to labels *)
    predictedLabels = Flatten[Table[
      Table[i, {Length[comm]}],
      {i, Length[communities[method]]},
      {comm, communities[method]}
    ]];
    
    (* Compute NMI and ARI *)
    comparison[method] = <|
      "nmi" -> NormalizedMutualInformation[taxonomyLabels, predictedLabels],
      "ari" -> AdjustedRandIndex[taxonomyLabels, predictedLabels]
    |>,
    {method, Keys[communities]}
  ];
  
  comparison
];

(* 4. Visual Complexity & Readability Metrics *)

(* Compute layout quality metrics *)
computeLayoutMetrics[G_, positions_] := Module[
  {metrics = <||>},
  
  (* Edge crossings *)
  metrics["edge_crossings"] = countEdgeCrossings[G, positions];
  metrics["crossing_density"] = metrics["edge_crossings"] / EdgeCount[G];
  
  (* Edge lengths *)
  edgeLengths = Table[
    {u, v} = edge;
    posU = positions[u];
    posV = positions[v];
    Norm[posU - posV],
    {edge, EdgeList[G]}
  ];
  metrics["average_edge_length"] = Mean[edgeLengths];
  metrics["edge_length_std"] = StandardDeviation[edgeLengths];
  
  (* Node overlaps *)
  metrics["node_overlaps"] = countNodeOverlaps[positions];
  metrics["overlap_density"] = metrics["node_overlaps"] / VertexCount[G];
  
  (* Layout area *)
  xCoords = positions[[All, 1]];
  yCoords = positions[[All, 2]];
  metrics["layout_width"] = Max[xCoords] - Min[xCoords];
  metrics["layout_height"] = Max[yCoords] - Min[yCoords];
  metrics["layout_area"] = metrics["layout_width"] * metrics["layout_height"];
  metrics["node_density"] = VertexCount[G] / metrics["layout_area"];
  
  metrics
];

(* Count edge crossings *)
countEdgeCrossings[G_, positions_] := Module[
  {crossings = 0, edges = EdgeList[G]},
  
  Do[
    Do[
      {u1, v1} = edges[[i]];
      {u2, v2} = edges[[j]];
      
      If[edgesIntersect[positions[u1], positions[v1], positions[u2], positions[v2]],
        crossings += 1
      ],
      {j, i + 1, Length[edges]}
    ],
    {i, 1, Length[edges]}
  ];
  
  crossings
];

(* Check if two line segments intersect *)
edgesIntersect[p1_, p2_, p3_, p4_] := Module[
  {ccw},
  
  ccw[A_, B_, C_] := (C[[2]] - A[[2]]) * (B[[1]] - A[[1]]) > (B[[2]] - A[[2]]) * (C[[1]] - A[[1]]);
  
  ccw[p1, p3, p4] != ccw[p2, p3, p4] && ccw[p1, p2, p3] != ccw[p1, p2, p4]
];

(* Count node overlaps *)
countNodeOverlaps[positions_, threshold_: 0.1] := Module[
  {overlaps = 0, nodes = Keys[positions]},
  
  Do[
    Do[
      pos1 = positions[nodes[[i]]];
      pos2 = positions[nodes[[j]]];
      
      If[Norm[pos1 - pos2] < threshold,
        overlaps += 1
      ],
      {j, i + 1, Length[nodes]}
    ],
    {i, 1, Length[nodes]}
  ];
  
  overlaps
];

(* Compare to baseline layout *)
compareToBaseline[G_, currentPositions_] := Module[
  {baselinePositions, currentMetrics, baselineMetrics, improvements},
  
  (* Create baseline layout *)
  baselinePositions = createBaselineLayout[G];
  
  (* Compute metrics *)
  currentMetrics = computeLayoutMetrics[G, currentPositions];
  baselineMetrics = computeLayoutMetrics[G, baselinePositions];
  
  (* Compute improvements *)
  improvements = <||>;
  Do[
    If[baselineMetrics[metric] > 0,
      improvements[StringJoin[metric, "_improvement_pct"]] = 
        (baselineMetrics[metric] - currentMetrics[metric]) / baselineMetrics[metric] * 100
    ],
    {metric, Keys[currentMetrics]}
  ];
  
  <|
    "current_metrics" -> currentMetrics,
    "baseline_metrics" -> baselineMetrics,
    "improvements" -> improvements
  |>
];

(* Create simple baseline layout *)
createBaselineLayout[G_] := Module[
  {positions = <||>, nodes = VertexList[G]},
  
  Do[
    positions[nodes[[i]]] = {Mod[i, 10], Quotient[i, 10]},
    {i, 1, Length[nodes]}
  ];
  
  positions
];

(* 5. Main Analysis Function *)

(* Run comprehensive visualization metrics analysis *)
analyzeVisualizationMetrics[hypergraphData_, particleFamilies_: {"Delta", "Nstar"}] := Module[
  {results = <||>},
  
  Print["=" * 60];
  Print["QUANTITATIVE VISUALIZATION METRICS ANALYSIS"];
  Print["=" * 60];
  
  (* 1. Incidence Graph Analysis *)
  Print["1. Computing Incidence Graph Metrics..."];
  incidenceGraph = createIncidenceGraph[hypergraphData];
  results["incidence_metrics"] = computeGraphMetrics[incidenceGraph];
  
  (* 2. Product Projection Analysis *)
  Print["2. Computing Product Projection Metrics..."];
  projectionGraph = createProductProjection[hypergraphData];
  results["projection_metrics"] = computeGraphMetrics[projectionGraph];
  
  (* 3. Motif Analysis *)
  Print["3. Analyzing Motifs and Cycles..."];
  motifCounts = countMotifs[projectionGraph];
  cycleCounts = countCycles[projectionGraph];
  randomBaseline = generateRandomBaseline[projectionGraph];
  motifZScores = computeZScores[motifCounts, randomBaseline];
  
  results["motif_analysis"] = <|
    "motif_counts" -> motifCounts,
    "cycle_counts" -> cycleCounts,
    "motif_z_scores" -> motifZScores,
    "random_baseline" -> randomBaseline
  |>;
  
  (* 4. Community Detection *)
  Print["4. Detecting Communities..."];
  communities = detectCommunities[projectionGraph];
  communityAnalysis = analyzeCommunities[projectionGraph, communities];
  
  results["community_analysis"] = <|
    "communities" -> communities,
    "analysis" -> communityAnalysis
  |>;
  
  (* 5. Visual Complexity Analysis *)
  Print["5. Analyzing Visual Complexity..."];
  currentLayout = GraphLayout[projectionGraph];
  visualComplexity = compareToBaseline[projectionGraph, currentLayout];
  
  results["visual_complexity"] = visualComplexity;
  
  (* 6. Generate Report *)
  Print["6. Generating Report..."];
  report = generateVisualizationReport[results, particleFamilies];
  
  results["report"] = report;
  
  Print["=" * 60];
  Print["ANALYSIS COMPLETE!");
  Print["=" * 60];
  
  results
];

(* Generate comprehensive report *)
generateVisualizationReport[results_, particleFamilies_] := Module[
  {report = ""},
  
  report = StringJoin[
    "=" * 80, "\n",
    "QUANTITATIVE VISUALIZATION METRICS REPORT", "\n",
    "=" * 80, "\n\n",
    "Analysis Date: ", DateString[], "\n",
    "Particle Families: ", StringRiffle[particleFamilies, ", "], "\n\n",
    
    "1. INCIDENCE GRAPH METRICS", "\n",
    "-" * 50, "\n",
    "Nodes: ", ToString[results["incidence_metrics"]["n_nodes"]], "\n",
    "Edges: ", ToString[results["incidence_metrics"]["n_edges"]], "\n",
    "Density: ", ToString[NumberForm[results["incidence_metrics"]["density"], {4, 4}]], "\n",
    "Modularity: ", ToString[NumberForm[results["incidence_metrics"]["modularity"], {4, 4}]], "\n",
    "Average Clustering: ", ToString[NumberForm[results["incidence_metrics"]["average_clustering"], {4, 4}]], "\n\n",
    
    "2. PRODUCT PROJECTION METRICS", "\n",
    "-" * 50, "\n",
    "Nodes: ", ToString[results["projection_metrics"]["n_nodes"]], "\n",
    "Edges: ", ToString[results["projection_metrics"]["n_edges"]], "\n",
    "Density: ", ToString[NumberForm[results["projection_metrics"]["density"], {4, 4}]], "\n",
    "Modularity: ", ToString[NumberForm[results["projection_metrics"]["modularity"], {4, 4}]], "\n",
    "Average Clustering: ", ToString[NumberForm[results["projection_metrics"]["average_clustering"], {4, 4}]], "\n\n",
    
    "3. MOTIF AND CYCLE ANALYSIS", "\n",
    "-" * 50, "\n",
    "Motif Z-Scores (vs random baseline):", "\n"
  ];
  
  Do[
    zScore = results["motif_analysis"]["motif_z_scores"][motif];
    significance = If[Abs[zScore] > 3, "***", If[Abs[zScore] > 2, "**", If[Abs[zScore] > 1, "*", ""]]];
    report = report <> "  " <> motif <> ": " <> ToString[NumberForm[zScore, {3, 2}]] <> " " <> significance <> "\n",
    {motif, Keys[results["motif_analysis"]["motif_z_scores"]]}
  ];
  
  report = report <> "\n";
  
  (* Community detection results *)
  report = report <> "4. COMMUNITY DETECTION", "\n";
  report = report <> "-" * 50, "\n";
  
  Do[
    analysis = results["community_analysis"]["analysis"][method];
    report = report <> method <> " Method:", "\n";
    report = report <> "  Communities: " <> ToString[analysis["n_communities"]], "\n";
    report = report <> "  Modularity: " <> ToString[NumberForm[analysis["modularity"], {4, 4}]], "\n";
    report = report <> "\n",
    {method, Keys[results["community_analysis"]["analysis"]]}
  ];
  
  (* Visual complexity results *)
  report = report <> "5. VISUAL COMPLEXITY METRICS", "\n";
  report = report <> "-" * 50, "\n";
  
  currentMetrics = results["visual_complexity"]["current_metrics"];
  baselineMetrics = results["visual_complexity"]["baseline_metrics"];
  improvements = results["visual_complexity"]["improvements"];
  
  report = report <> "Current Layout:", "\n";
  report = report <> "  Edge Crossings: " <> ToString[currentMetrics["edge_crossings"]], "\n";
  report = report <> "  Node Overlaps: " <> ToString[currentMetrics["node_overlaps"]], "\n";
  report = report <> "  Average Edge Length: " <> ToString[NumberForm[currentMetrics["average_edge_length"], {3, 3}]], "\n\n";
  
  report = report <> "Baseline Layout:", "\n";
  report = report <> "  Edge Crossings: " <> ToString[baselineMetrics["edge_crossings"]], "\n";
  report = report <> "  Node Overlaps: " <> ToString[baselineMetrics["node_overlaps"]], "\n";
  report = report <> "  Average Edge Length: " <> ToString[NumberForm[baselineMetrics["average_edge_length"], {3, 3}]], "\n\n";
  
  report = report <> "Improvements vs Baseline:", "\n";
  Do[
    report = report <> "  " <> metric <> ": " <> ToString[NumberForm[improvement, {3, 1}]] <> "%", "\n",
    {metric, Keys[improvements]},
    {improvement, {improvements[metric]}}
  ];
  
  report = report <> "\n" <> "=" * 80;
  
  report
];

(* 6. Export Results for Python Comparison *)

(* Export visualization metrics results *)
exportVisualizationResults[results_, outputFile_: "visualization_metrics_results.json"] := Module[
  {exportData},
  
  exportData = <|
    "analysis_date" -> DateString[],
    "incidence_metrics" -> results["incidence_metrics"],
    "projection_metrics" -> results["projection_metrics"],
    "motif_analysis" -> results["motif_analysis"],
    "community_analysis" -> results["community_analysis"],
    "visual_complexity" -> results["visual_complexity"]
  |>;
  
  Export[outputFile, exportData];
  Print["Visualization metrics results exported to ", outputFile];
  
  exportData
];

(* 7. Example Usage *)

(* Run analysis on your hypergraph data *)
(* results = analyzeVisualizationMetrics[yourHypergraphData]; *)

(* Export results for Python comparison *)
(* exportVisualizationResults[results]; *)

(* Display key findings *)
(* Print[results["report"]]; *)
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_VISUALIZATION_OUTPUT = """
Expected visualization_metrics_results.json format:
{
  "analysis_date": "2024-12-01T14:30:22",
  "incidence_metrics": {
    "n_nodes": 45,
    "n_edges": 67,
    "density": 0.067,
    "modularity": 0.234,
    "average_clustering": 0.456,
    "average_path_length": 2.34,
    "degree_assortativity": 0.123
  },
  "projection_metrics": {
    "n_nodes": 12,
    "n_edges": 23,
    "density": 0.348,
    "modularity": 0.456,
    "average_clustering": 0.567,
    "average_weight": 2.34
  },
  "motif_analysis": {
    "motif_counts": {
      "triangles": 15,
      "squares": 8,
      "stars_3": 12,
      "stars_4": 5
    },
    "motif_z_scores": {
      "triangles": 2.34,
      "squares": 1.23,
      "stars_3": 3.45,
      "stars_4": 0.89
    },
    "cycle_counts": {
      "3": 15,
      "4": 8,
      "5": 3,
      "6": 1
    }
  },
  "community_analysis": {
    "analysis": {
      "louvain": {
        "n_communities": 3,
        "modularity": 0.456,
        "coverage": 1.0
      },
      "label_propagation": {
        "n_communities": 4,
        "modularity": 0.423,
        "coverage": 1.0
      }
    }
  },
  "visual_complexity": {
    "current_metrics": {
      "edge_crossings": 5,
      "node_overlaps": 2,
      "average_edge_length": 1.23
    },
    "baseline_metrics": {
      "edge_crossings": 12,
      "node_overlaps": 8,
      "average_edge_length": 2.45
    },
    "improvements": {
      "edge_crossings_improvement_pct": 58.3,
      "node_overlaps_improvement_pct": 75.0
    }
  }
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_visualization_results(json_path: str) -> bool:
    """
    Validate that exported visualization results meet expected format.
    
    Parameters:
    -----------
    json_path : str
        Path to the exported JSON file
    
    Returns:
    --------
    bool
        True if validation passes
    """
    import json
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    required_keys = ['analysis_date', 'incidence_metrics', 'projection_metrics', 
                    'motif_analysis', 'community_analysis', 'visual_complexity']
    
    for key in required_keys:
        if key not in results:
            print(f"Missing key: {key}")
            return False
    
    # Check incidence metrics
    incidence = results['incidence_metrics']
    incidence_keys = ['n_nodes', 'n_edges', 'density', 'modularity', 'average_clustering']
    
    for key in incidence_keys:
        if key not in incidence:
            print(f"Missing incidence metric: {key}")
            return False
    
    # Check projection metrics
    projection = results['projection_metrics']
    projection_keys = ['n_nodes', 'n_edges', 'density', 'modularity', 'average_clustering']
    
    for key in projection_keys:
        if key not in projection:
            print(f"Missing projection metric: {key}")
            return False
    
    # Check motif analysis
    motif = results['motif_analysis']
    if 'motif_counts' not in motif or 'motif_z_scores' not in motif:
        print("Missing motif analysis components")
        return False
    
    # Check community analysis
    community = results['community_analysis']
    if 'analysis' not in community:
        print("Missing community analysis")
        return False
    
    # Check visual complexity
    complexity = results['visual_complexity']
    if 'current_metrics' not in complexity or 'baseline_metrics' not in complexity:
        print("Missing visual complexity metrics")
        return False
    
    print("Visualization results validation passed!")
    return True

def compare_wl_python_visualization(wl_json_path: str, python_results: dict) -> dict:
    """
    Compare Wolfram Language and Python visualization metrics results.
    
    Parameters:
    -----------
    wl_json_path : str
        Path to WL visualization results JSON
    python_results : dict
        Python visualization results
    
    Returns:
    --------
    dict
        Comparison results
    """
    import json
    
    with open(wl_json_path, 'r') as f:
        wl_results = json.load(f)
    
    comparison = {}
    
    # Compare incidence metrics
    if 'incidence_metrics' in wl_results and 'incidence_metrics' in python_results:
        wl_incidence = wl_results['incidence_metrics']
        py_incidence = python_results['incidence_metrics']['global']
        
        comparison['incidence_metrics'] = {
            'wl_nodes': wl_incidence['n_nodes'],
            'python_nodes': py_incidence['n_nodes'],
            'wl_modularity': wl_incidence['modularity'],
            'python_modularity': py_incidence['modularity'],
            'node_agreement': wl_incidence['n_nodes'] == py_incidence['n_nodes'],
            'modularity_diff': abs(wl_incidence['modularity'] - py_incidence['modularity'])
        }
    
    # Compare projection metrics
    if 'projection_metrics' in wl_results and 'projection_metrics' in python_results:
        wl_projection = wl_results['projection_metrics']
        py_projection = python_results['projection_metrics']['global']
        
        comparison['projection_metrics'] = {
            'wl_nodes': wl_projection['n_nodes'],
            'python_nodes': py_projection['n_nodes'],
            'wl_modularity': wl_projection['modularity'],
            'python_modularity': py_projection['modularity'],
            'node_agreement': wl_projection['n_nodes'] == py_projection['n_nodes'],
            'modularity_diff': abs(wl_projection['modularity'] - py_projection['modularity'])
        }
    
    # Compare motif analysis
    if 'motif_analysis' in wl_results and 'motif_analysis' in python_results:
        wl_motifs = wl_results['motif_analysis']['motif_z_scores']
        py_motifs = python_results['motif_analysis']['global']['motif_z_scores']
        
        motif_comparison = {}
        for motif in wl_motifs:
            if motif in py_motifs:
                wl_score = wl_motifs[motif]
                py_score = py_motifs[motif]
                motif_comparison[motif] = {
                    'wl_z_score': wl_score,
                    'python_z_score': py_score,
                    'difference': abs(wl_score - py_score)
                }
        
        comparison['motif_analysis'] = motif_comparison
    
    # Compare visual complexity
    if 'visual_complexity' in wl_results and 'visual_complexity' in python_results:
        wl_complexity = wl_results['visual_complexity']['current_metrics']
        py_complexity = python_results['visual_complexity']['baseline_comparison']['current_metrics']
        
        comparison['visual_complexity'] = {
            'wl_edge_crossings': wl_complexity['edge_crossings'],
            'python_edge_crossings': py_complexity['edge_crossings'],
            'crossing_agreement': wl_complexity['edge_crossings'] == py_complexity['edge_crossings']
        }
    
    return comparison

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE VISUALIZATION METRICS GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(VISUALIZATION_METRICS_CODE)
    print("\nExpected output format:")
    print(EXPECTED_VISUALIZATION_OUTPUT)
    print("\nThis will provide:")
    print("1. Hypergraph to incidence graph conversion")
    print("2. Product co-occurrence projections")
    print("3. Motif and cycle analysis with statistical significance")
    print("4. Community detection and subgroup discovery")
    print("5. Visual complexity and readability metrics")
    print("6. Quantitative comparison with baseline layouts")
    print("7. Comprehensive reporting for paper integration")
