"""
Wolfram Language Reproducibility Guide

This module provides code snippets to be added to the Wolfram Language notebook
for implementing reproducibility features including deterministic data assembly,
parameterized execution, and testing.
"""

# =============================================================================
# WOLFRAM LANGUAGE REPRODUCIBILITY CODE (Add to your notebook)
# =============================================================================

REPRODUCIBILITY_CODE = '''
(* ===== Reproducibility Framework ===== *)

(* 1. Deterministic Data Assembly *)

(* Settings association for reproducible analysis *)
analysisSettings = <|
  "pdg_version" -> "2024",
  "particle_family" -> "Delta",
  "include_width_systematic" -> True,
  "kappa" -> 0.25,
  "bootstrap_n" -> 1000,
  "window_factor" -> 2.0,
  "canonical_alpha0" -> 0.0,
  "canonical_alphap" -> 0.9,
  "J_range" -> {0.5, 9.5},
  "J_step" -> 0.5,
  "output_dir" -> "analysis_results",
  "save_plots" -> True,
  "save_reports" -> True,
  "validation_bounds" -> <|
    "alphap_min" -> 0.1,
    "alphap_max" -> 2.0,
    "chi2_dof_max" -> 10.0
  |>
|>;

(* Function to create frozen data snapshot *)
createFrozenSnapshot[family_String, settings_Association] := Module[
  {data, timestamp, filename, metadata},
  
  (* Get current timestamp *)
  timestamp = DateString[{"Year", "Month", "Day", "Hour", "Minute", "Second"}];
  
  (* Assemble data using existing functions *)
  data = BuildTrajectoryData[family,
    "IncludeWidthAsSys" -> settings["include_width_systematic"],
    "Kappa" -> settings["kappa"]
  ];
  
  (* Create filename with metadata *)
  filename = StringJoin[
    family, "_baryons_pdg", settings["pdg_version"], "_", 
    StringReplace[timestamp, ":" -> ""], ".csv"
  ];
  
  (* Create metadata *)
  metadata = <|
    "creation_timestamp" -> timestamp,
    "pdg_version" -> settings["pdg_version"],
    "particle_family" -> family,
    "include_width_systematic" -> settings["include_width_systematic"],
    "kappa" -> settings["kappa"],
    "n_particles" -> Length[Normal[data]],
    "mass_range_GeV" -> {Min[data[All, "MassGeV"]], Max[data[All, "MassGeV"]]},
    "J_range" -> {Min[data[All, "J"]], Max[data[All, "J"]]},
    "csv_file" -> filename
  |>;
  
  (* Export data *)
  Export[filename, Normal[data]];
  
  (* Export metadata *)
  Export[StringReplace[filename, ".csv" -> ".json"], metadata];
  
  Print["Created frozen snapshot: ", filename];
  Print["Metadata: ", StringReplace[filename, ".csv" -> ".json"]];
  Print["Particles: ", Length[Normal[data]]];
  Print["Mass range: ", Min[data[All, "MassGeV"]], " - ", Max[data[All, "MassGeV"]], " GeV"];
  
  {filename, metadata}
];

(* 2. Parameterized Analysis Runner *)

(* Function to run complete analysis with settings *)
runReggeAnalysis[settings_Association] := Module[
  {data, results, validation},
  
  Print["=" * 60];
  Print["PARAMETERIZED REGGE ANALYSIS"];
  Print["=" * 60];
  Print["Particle Family: ", settings["particle_family"]];
  Print["Bootstrap N: ", settings["bootstrap_n"]];
  Print["Output Directory: ", settings["output_dir"]];
  Print["=" * 60];
  
  (* Step 1: Data Assembly *)
  Print["1. DATA ASSEMBLY"];
  Print["-" * 30];
  
  data = BuildTrajectoryData[settings["particle_family"],
    "IncludeWidthAsSys" -> settings["include_width_systematic"],
    "Kappa" -> settings["kappa"]
  ];
  
  Print["Data points: ", Length[Normal[data]]];
  Print["Mass range: ", Min[data[All, "MassGeV"]], " - ", Max[data[All, "MassGeV"]], " GeV"];
  
  (* Step 2: Regge Analysis *)
  Print["2. REGGE ANALYSIS"];
  Print["-" * 30];
  
  (* Use existing analysis code here *)
  (* ... (your existing fitting and analysis code) ... *)
  
  (* Step 3: Validation *)
  Print["3. VALIDATION"];
  Print["-" * 30];
  
  validation = validateResults[results, settings];
  
  (* Step 4: Generate Reports *)
  Print["4. REPORT GENERATION"];
  Print["-" * 30];
  
  generateReports[results, validation, settings];
  
  (* Step 5: Save Results *)
  Print["5. SAVE RESULTS"];
  Print["-" * 30];
  
  saveResults[results, validation, settings];
  
  Print["=" * 60];
  Print["ANALYSIS COMPLETE!");
  Print["=" * 60];
  
  <|
    "settings" -> settings,
    "results" -> results,
    "validation" -> validation,
    "data" -> data
  |>
];

(* 3. Validation Functions *)

(* Validate analysis results *)
validateResults[results_, settings_] := Module[
  {validation = <|"passed" -> True, "checks" -> <||>, "warnings" -> {}|>},
  
  (* Check α'' bounds *)
  If[results["alphap"] < settings["validation_bounds"]["alphap_min"] || 
     results["alphap"] > settings["validation_bounds"]["alphap_max"],
    validation["checks"]["alphap_bounds"] = False;
    AppendTo[validation["warnings"], 
      "α'' = " <> ToString[results["alphap"]] <> " outside bounds"],
    validation["checks"]["alphap_bounds"] = True
  ];
  
  (* Check χ²/dof *)
  If[results["chi2_dof"] > settings["validation_bounds"]["chi2_dof_max"],
    validation["checks"]["chi2_dof"] = False;
    AppendTo[validation["warnings"], 
      "χ²/dof = " <> ToString[results["chi2_dof"]] <> " > " <> 
      ToString[settings["validation_bounds"]["chi2_dof_max"]]],
    validation["checks"]["chi2_dof"] = True
  ];
  
  (* Check bootstrap variance *)
  If[results["bootstrap_std"] <= 0,
    validation["checks"]["bootstrap_variance"] = False;
    AppendTo[validation["warnings"], "Bootstrap variance is zero"],
    validation["checks"]["bootstrap_variance"] = True
  ];
  
  (* Overall validation *)
  validation["passed"] = AllTrue[Values[validation["checks"]], #&];
  
  (* Print validation results *)
  Print["Validation Results:"];
  Do[
    status = If[validation["checks"][check], "✓", "✗"];
    Print["  ", check, ": ", status],
    {check, Keys[validation["checks"]]}
  ];
  
  If[Length[validation["warnings"]] > 0,
    Print["Warnings:"];
    Do[Print["  - ", warning], {warning, validation["warnings"]}]
  ];
  
  validation
];

(* 4. Report Generation *)

(* Generate reproducibility report *)
generateReproducibilityReport[results_, settings_] := Module[
  {report},
  
  report = StringJoin[
    "=" * 60, "\n",
    "REPRODUCIBILITY REPORT", "\n",
    "=" * 60, "\n\n",
    "Analysis Parameters:", "\n",
    "  Particle Family: ", settings["particle_family"], "\n",
    "  PDG Version: ", settings["pdg_version"], "\n",
    "  Bootstrap N: ", ToString[settings["bootstrap_n"]], "\n",
    "  Width Systematic: ", ToString[settings["include_width_systematic"]], "\n",
    "  Kappa: ", ToString[settings["kappa"]], "\n\n",
    "Key Results:", "\n",
    "  α₀ = ", ToString[results["alpha0"]], " ± ", ToString[results["alpha0_err"]], "\n",
    "  α'' = ", ToString[results["alphap"]], " ± ", ToString[results["alphap_err"]], "\n",
    "  χ²/dof = ", ToString[results["chi2_dof"]], "\n\n",
    "Reproduction Command:", "\n",
    "  runReggeAnalysis[analysisSettings]", "\n\n",
    "=" * 60
  ];
  
  report
];

(* Generate parameter report *)
generateParameterReport[results_] := Module[
  {report},
  
  report = StringJoin[
    "=" * 60, "\n",
    "PARAMETER REPORT", "\n",
    "=" * 60, "\n\n",
    "Fitted Parameters:", "\n",
    "  α₀ = ", ToString[results["alpha0"]], " ± ", ToString[results["alpha0_err"]], "\n",
    "  α'' = ", ToString[results["alphap"]], " ± ", ToString[results["alphap_err"]], "\n\n",
    "Goodness of Fit:", "\n",
    "  χ² = ", ToString[results["chi2"]], "\n",
    "  dof = ", ToString[results["dof"]], "\n",
    "  χ²/dof = ", ToString[results["chi2_dof"]], "\n",
    "  R² = ", ToString[results["r_squared"]], "\n\n",
    "=" * 60
  ];
  
  report
];

(* Generate validation report *)
generateValidationReport[validation_] := Module[
  {report},
  
  report = StringJoin[
    "=" * 60, "\n",
    "VALIDATION REPORT", "\n",
    "=" * 60, "\n\n",
    "Validation Status:", "\n",
    "  Overall: ", If[validation["passed"], "PASSED", "FAILED"], "\n\n",
    "Individual Checks:", "\n"
  ];
  
  Do[
    status = If[validation["checks"][check], "✓ PASS", "✗ FAIL"];
    report = report <> "  " <> check <> ": " <> status <> "\n",
    {check, Keys[validation["checks"]]}
  ];
  
  If[Length[validation["warnings"]] > 0,
    report = report <> "\nWarnings:\n";
    Do[report = report <> "  - " <> warning <> "\n", {warning, validation["warnings"]}]
  ];
  
  report = report <> "\n" <> "=" * 60;
  
  report
];

(* 5. Save Results *)

(* Save analysis results *)
saveResults[results_, validation_, settings_] := Module[
  {outputDir, summary},
  
  outputDir = settings["output_dir"];
  
  (* Create output directory *)
  If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];
  
  (* Save settings *)
  Export[FileNameJoin[{outputDir, "analysis_settings.json"}], settings];
  
  (* Save results summary *)
  summary = <|
    "settings" -> settings,
    "validation" -> validation,
    "key_results" -> <|
      "alpha0" -> results["alpha0"],
      "alpha0_err" -> results["alpha0_err"],
      "alphap" -> results["alphap"],
      "alphap_err" -> results["alphap_err"],
      "chi2_dof" -> results["chi2_dof"],
      "r_squared" -> results["r_squared"]
    |>
  |>;
  
  Export[FileNameJoin[{outputDir, "analysis_summary.json"}], summary];
  
  (* Save reports *)
  Export[FileNameJoin[{outputDir, "reproducibility_report.txt"}], 
         generateReproducibilityReport[results, settings]];
  Export[FileNameJoin[{outputDir, "parameter_report.txt"}], 
         generateParameterReport[results]];
  Export[FileNameJoin[{outputDir, "validation_report.txt"}], 
         generateValidationReport[validation]];
  
  Print["Results saved to ", outputDir];
];

(* 6. Testing Framework *)

(* Unit tests for reproducibility *)
runReproducibilityTests[] := Module[
  {testResults = <|"passed" -> True, "tests" -> <||>|>},
  
  Print["Running reproducibility tests..."];
  
  (* Test 1: Data assembly *)
  testResults["tests"]["data_assembly"] = testDataAssembly[];
  
  (* Test 2: Parameter reproducibility *)
  testResults["tests"]["parameter_reproducibility"] = testParameterReproducibility[];
  
  (* Test 3: Validation bounds *)
  testResults["tests"]["validation_bounds"] = testValidationBounds[];
  
  (* Overall result *)
  testResults["passed"] = AllTrue[Values[testResults["tests"]], #&];
  
  (* Print results *)
  Print["Test Results:"];
  Do[
    status = If[testResults["tests"][test], "✓ PASS", "✗ FAIL"];
    Print["  ", test, ": ", status],
    {test, Keys[testResults["tests"]]}
  ];
  
  If[testResults["passed"],
    Print["✅ All tests passed!"],
    Print["❌ Some tests failed!"]
  ];
  
  testResults
];

(* Test data assembly *)
testDataAssembly[] := Module[
  {data, result = True},
  
  (* Test Delta baryon assembly *)
  data = BuildTrajectoryData["Delta", "IncludeWidthAsSys" -> True, "Kappa" -> 0.25];
  
  (* Assertions *)
  If[Length[Normal[data]] <= 0, result = False];
  If[!MemberQ[Keys[data[1]], "M2GeV2"], result = False];
  If[!MemberQ[Keys[data[1]], "J"], result = False];
  
  (* Check mass range *)
  If[Min[data[All, "MassGeV"]] <= 0.5, result = False];
  If[Max[data[All, "MassGeV"]] >= 5.0, result = False];
  
  (* Check J values *)
  If[!AllTrue[data[All, "J"], # >= 0&], result = False];
  
  result
];

(* Test parameter reproducibility *)
testParameterReproducibility[] := Module[
  {results1, results2, tolerance = 1*^-6, result = True},
  
  (* Run analysis twice *)
  results1 = runReggeAnalysis[analysisSettings];
  results2 = runReggeAnalysis[analysisSettings];
  
  (* Compare results *)
  If[Abs[results1["results"]["alphap"] - results2["results"]["alphap"]] > tolerance,
    result = False];
  If[Abs[results1["results"]["alpha0"] - results2["results"]["alpha0"]] > tolerance,
    result = False];
  
  result
];

(* Test validation bounds *)
testValidationBounds[] := Module[
  {results, result = True},
  
  (* Run analysis *)
  results = runReggeAnalysis[analysisSettings];
  
  (* Check bounds *)
  If[results["results"]["alphap"] < 0.1 || results["results"]["alphap"] > 2.0,
    result = False];
  If[results["results"]["chi2_dof"] > 10.0,
    result = False];
  
  result
];

(* 7. Main Execution *)

(* Create frozen snapshot *)
Print["Creating frozen snapshot..."];
{snapshotFile, snapshotMetadata} = createFrozenSnapshot["Delta", analysisSettings];

(* Run parameterized analysis *)
Print["Running parameterized analysis..."];
analysisResults = runReggeAnalysis[analysisSettings];

(* Run tests *)
Print["Running reproducibility tests..."];
testResults = runReproducibilityTests[];

(* Export results for Python comparison *)
reproducibilityResults = <|
  "snapshot_file" -> snapshotFile,
  "snapshot_metadata" -> snapshotMetadata,
  "analysis_results" -> analysisResults,
  "test_results" -> testResults,
  "settings" -> analysisSettings
|>;

Export["python_analysis/data_export/reproducibility_results.json", reproducibilityResults];
Print["Reproducibility results exported to reproducibility_results.json"]
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_REPRODUCIBILITY_OUTPUT = """
Expected reproducibility_results.json format:
{
  "snapshot_file": "delta_baryons_pdg2024_20241201_143022.csv",
  "snapshot_metadata": {
    "creation_timestamp": "2024-12-01T14:30:22",
    "pdg_version": "2024",
    "particle_family": "Delta",
    "include_width_systematic": true,
    "kappa": 0.25,
    "n_particles": 15,
    "mass_range_GeV": [1.232, 2.400],
    "J_range": [0.5, 2.5],
    "csv_file": "delta_baryons_pdg2024_20241201_143022.csv"
  },
  "analysis_results": {
    "settings": {...},
    "results": {
      "alpha0": 0.123,
      "alphap": 0.891,
      "alpha0_err": 0.045,
      "alphap_err": 0.023,
      "chi2_dof": 1.234
    },
    "validation": {
      "passed": true,
      "checks": {
        "alphap_bounds": true,
        "chi2_dof": true,
        "bootstrap_variance": true
      }
    }
  },
  "test_results": {
    "passed": true,
    "tests": {
      "data_assembly": true,
      "parameter_reproducibility": true,
      "validation_bounds": true
    }
  }
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_reproducibility_results(json_path: str) -> bool:
    """
    Validate that exported reproducibility results meet expected format.
    
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
    
    required_keys = ['snapshot_file', 'snapshot_metadata', 'analysis_results', 'test_results']
    
    for key in required_keys:
        if key not in results:
            print(f"Missing key: {key}")
            return False
    
    # Check snapshot metadata
    metadata = results['snapshot_metadata']
    metadata_keys = ['creation_timestamp', 'pdg_version', 'particle_family', 'n_particles']
    
    for key in metadata_keys:
        if key not in metadata:
            print(f"Missing metadata key: {key}")
            return False
    
    # Check analysis results
    analysis = results['analysis_results']
    if 'results' not in analysis:
        print("Missing analysis results")
        return False
    
    # Check test results
    tests = results['test_results']
    if 'passed' not in tests or 'tests' not in tests:
        print("Missing test results")
        return False
    
    print("Reproducibility results validation passed!")
    return True

def compare_wl_python_reproducibility(wl_json_path: str, python_results: dict) -> dict:
    """
    Compare Wolfram Language and Python reproducibility results.
    
    Parameters:
    -----------
    wl_json_path : str
        Path to WL reproducibility results JSON
    python_results : dict
        Python reproducibility results
    
    Returns:
    --------
    dict
        Comparison results
    """
    import json
    
    with open(wl_json_path, 'r') as f:
        wl_results = json.load(f)
    
    comparison = {}
    
    # Compare snapshot metadata
    if 'snapshot_metadata' in wl_results and 'metadata' in python_results:
        wl_metadata = wl_results['snapshot_metadata']
        py_metadata = python_results['metadata']
        
        wl_particles = wl_metadata.get('n_particles', 0)
        py_particles = len(python_results.get('data', []))
        
        comparison['snapshot_metadata'] = {
            'wl_particles': wl_particles,
            'python_particles': py_particles,
            'agreement': wl_particles == py_particles
        }
    
    # Compare analysis results
    if 'analysis_results' in wl_results and 'results' in python_results:
        wl_analysis = wl_results['analysis_results'].get('results', {})
        py_analysis = python_results['results']
        
        if 'alphap' in wl_analysis and 'alphap' in py_analysis:
            wl_alphap = wl_analysis['alphap']
            py_alphap = py_analysis['alphap']
            
            diff = abs(wl_alphap - py_alphap)
            rel_diff = diff / wl_alphap if wl_alphap != 0 else float('inf')
            
            comparison['analysis_results'] = {
                'wl_alphap': wl_alphap,
                'python_alphap': py_alphap,
                'absolute_difference': diff,
                'relative_difference': rel_diff,
                'agreement': rel_diff < 0.1  # 10% tolerance
            }
    
    # Compare test results
    if 'test_results' in wl_results and 'validation' in python_results:
        wl_tests = wl_results['test_results']
        py_validation = python_results['validation']
        
        wl_passed = wl_tests.get('passed', False)
        py_passed = py_validation.get('passed', False)
        
        comparison['test_results'] = {
            'wl_passed': wl_passed,
            'python_passed': py_passed,
            'agreement': wl_passed == py_passed
        }
    
    return comparison

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE REPRODUCIBILITY GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(REPRODUCIBILITY_CODE)
    print("\nExpected output format:")
    print(EXPECTED_REPRODUCIBILITY_OUTPUT)
    print("\nThis will provide:")
    print("1. Deterministic data assembly with frozen snapshots")
    print("2. Parameterized analysis runner")
    print("3. Comprehensive validation framework")
    print("4. Automated testing and CI integration")
    print("5. Reproducibility reports for paper submission")
