"""
Wolfram Language Data Export Guide

This module provides code snippets to be added to the Wolfram Language notebook
for exporting particle data to Python-compatible formats.

Add these cells to your WL notebook after the data processing sections.
"""

# =============================================================================
# WOLFRAM LANGUAGE EXPORT CODE (Add to your notebook)
# =============================================================================

EXPORT_CODE = '''
(* ===== Data Export for Python Analysis ===== *)

(* Export Δ baryon data with uncertainties *)
deltaExportData = Normal @ Select[deltaPlus, 
  NumberQ[#["M2GeV2"]] && NumberQ[#["J"]] &];

(* Create export table with all necessary fields *)
exportTable = Table[
  <|
    "name" -> deltaExportData[[i, "Name"]],
    "J" -> deltaExportData[[i, "J"]],
    "parity" -> deltaExportData[[i, "Parity"]],
    "mass_GeV" -> deltaExportData[[i, "MassGeV"]],
    "mass_sigma_GeV" -> deltaExportData[[i, "MassSigmaGeV"]],
    "width_GeV" -> If[NumberQ[deltaExportData[[i, "ResonanceWidth"]]], 
      deltaExportData[[i, "ResonanceWidth"]], Missing["NotAvailable"]],
    "M2_GeV2" -> deltaExportData[[i, "M2GeV2"]],
    "M2_sigma_GeV2" -> deltaExportData[[i, "M2SigmaGeV2"]],
    "pdg_status" -> deltaExportData[[i, "Status"]]
  |>,
  {i, Length[deltaExportData]}
];

(* Export to CSV with metadata *)
metadata = <|
  "export_date" -> DateString[],
  "pdg_version" -> "2024",
  "kappa_width_to_sigma" -> 0.25,
  "wolfram_version" -> $Version,
  "particle_family" -> "Delta",
  "total_particles" -> Length[exportTable]
|>;

(* Save data and metadata *)
Export["python_analysis/data_export/delta_baryons.csv", exportTable];
Export["python_analysis/data_export/metadata.json", metadata];

(* Export all particle data for hypergraph analysis *)
allParticleData = Normal @ Select[particleList, 
  QuantityQ[#["Mass"]] && NumericQ[#["Spin"]] &];

allExportTable = Table[
  <|
    "name" -> allParticleData[[i, "Name"]],
    "mass_GeV" -> QuantityMagnitude @ UnitConvert[allParticleData[[i, "Mass"]], "GeV"],
    "spin" -> allParticleData[[i, "Spin"]],
    "quark_content" -> allParticleData[[i, "QuarkContent"]],
    "particle_type" -> allParticleData[[i, "ParticleType"]]
  |>,
  {i, Length[allParticleData]}
];

Export["python_analysis/data_export/all_particles.csv", allExportTable];

Print["Data export complete. Files saved to python_analysis/data_export/"]
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_CSV_FORMAT = """
Expected CSV format for delta_baryons.csv:
name,J,parity,mass_GeV,mass_sigma_GeV,width_GeV,M2_GeV2,M2_sigma_GeV2,pdg_status
"Delta(1232)++",3/2,1,1.232,0.002,0.118,1.518,0.005,"****"
"Delta(1600)++",3/2,1,1.600,0.010,0.350,2.560,0.032,"***"
...
"""

METADATA_FORMAT = """
Expected metadata.json format:
{
  "export_date": "2024-01-15T10:30:00",
  "pdg_version": "2024",
  "kappa_width_to_sigma": 0.25,
  "wolfram_version": "14.2.0",
  "particle_family": "Delta",
  "total_particles": 15
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_export_data(csv_path, metadata_path):
    """
    Validate that exported data meets expected format.
    
    Parameters:
    -----------
    csv_path : str
        Path to the exported CSV file
    metadata_path : str
        Path to the metadata JSON file
    
    Returns:
    --------
    bool
        True if validation passes
    """
    import pandas as pd
    import json
    
    # Check CSV format
    df = pd.read_csv(csv_path)
    required_columns = [
        'name', 'J', 'parity', 'mass_GeV', 'mass_sigma_GeV',
        'width_GeV', 'M2_GeV2', 'M2_sigma_GeV2', 'pdg_status'
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return False
    
    # Check metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    required_metadata = [
        'export_date', 'pdg_version', 'kappa_width_to_sigma',
        'wolfram_version', 'particle_family', 'total_particles'
    ]
    
    missing_meta = set(required_metadata) - set(metadata.keys())
    if missing_meta:
        print(f"Missing metadata: {missing_meta}")
        return False
    
    print("Data export validation passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE DATA EXPORT GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(EXPORT_CODE)
    print("\nExpected output formats:")
    print(EXPECTED_CSV_FORMAT)
    print(METADATA_FORMAT)
