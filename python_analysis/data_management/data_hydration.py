"""
Data & Provenance: Data Hydration

Implements comprehensive data management including:
- Frozen PDG snapshots with metadata
- Strict unit checks and validation
- Deterministic filters for reproducible research
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import datetime
import warnings
from scipy import constants
import hashlib

@dataclass
class DataMetadata:
    """Metadata for frozen PDG data."""
    pdg_date: str
    software_version: str
    kappa: float
    filters_applied: Dict[str, Any]
    data_hash: str
    extraction_timestamp: str
    unit_system: str = "GeV"
    particle_count: int = 0
    family_count: int = 0

class DataHydration:
    """
    Comprehensive data hydration with provenance tracking.
    
    Provides:
    - Frozen PDG snapshots with metadata
    - Strict unit checks and validation
    - Deterministic filters for reproducible research
    """
    
    def __init__(self, output_dir: str = "frozen_data"):
        """
        Initialize data hydration system.
        
        Parameters:
        -----------
        output_dir : str
            Directory for frozen data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = None
        self.frozen_data = None
        
    def freeze_pdg_snapshot(self, 
                          particle_data: pd.DataFrame,
                          pdg_date: str,
                          software_version: str,
                          kappa: float = 0.25,
                          filters_applied: Optional[Dict[str, Any]] = None) -> str:
        """
        Freeze PDG snapshot with metadata.
        
        Parameters:
        -----------
        particle_data : pd.DataFrame
            Raw particle data from PDG
        pdg_date : str
            PDG snapshot date
        software_version : str
            Software version used for extraction
        kappa : float
            Width-to-uncertainty conversion factor
        filters_applied : Optional[Dict[str, Any]]
            Filters applied during extraction
            
        Returns:
        --------
        str
            Path to frozen data file
        """
        print("Freezing PDG snapshot...")
        
        # Create metadata
        data_hash = self._compute_data_hash(particle_data)
        extraction_timestamp = datetime.datetime.now().isoformat()
        
        self.metadata = DataMetadata(
            pdg_date=pdg_date,
            software_version=software_version,
            kappa=kappa,
            filters_applied=filters_applied or {},
            data_hash=data_hash,
            extraction_timestamp=extraction_timestamp,
            particle_count=len(particle_data),
            family_count=particle_data.get('Family', pd.Series()).nunique()
        )
        
        # Create frozen data structure
        frozen_data = {
            'particle_data': particle_data.to_dict('records'),
            'metadata': self.metadata.__dict__,
            'schema': self._get_data_schema(particle_data)
        }
        
        # Save frozen data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        frozen_file = self.output_dir / f"pdg_snapshot_{pdg_date}_{timestamp}.json"
        
        with open(frozen_file, 'w') as f:
            json.dump(frozen_data, f, indent=2, default=str)
        
        # Save separate metadata file
        metadata_file = self.output_dir / f"metadata_{pdg_date}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata.__dict__, f, indent=2, default=str)
        
        print(f"Frozen data saved to: {frozen_file}")
        print(f"Metadata saved to: {metadata_file}")
        
        return str(frozen_file)
    
    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of data for provenance tracking."""
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _get_data_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data schema for documentation."""
        return {
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict()
        }
    
    def load_frozen_data(self, frozen_file: str) -> Tuple[pd.DataFrame, DataMetadata]:
        """
        Load frozen PDG data.
        
        Parameters:
        -----------
        frozen_file : str
            Path to frozen data file
            
        Returns:
        --------
        Tuple[pd.DataFrame, DataMetadata]
            Particle data and metadata
        """
        print(f"Loading frozen data from: {frozen_file}")
        
        with open(frozen_file, 'r') as f:
            frozen_data = json.load(f)
        
        # Reconstruct DataFrame
        particle_data = pd.DataFrame(frozen_data['particle_data'])
        
        # Reconstruct metadata
        metadata_dict = frozen_data['metadata']
        metadata = DataMetadata(**metadata_dict)
        
        self.frozen_data = particle_data
        self.metadata = metadata
        
        print(f"Loaded {len(particle_data)} particles from PDG snapshot {metadata.pdg_date}")
        
        return particle_data, metadata
    
    def extract_mass_width_data(self, 
                              particle_data: pd.DataFrame,
                              strict_units: bool = True) -> pd.DataFrame:
        """
        Extract mass and width data with strict unit checks.
        
        Parameters:
        -----------
        particle_data : pd.DataFrame
            Raw particle data
        strict_units : bool
            Whether to enforce strict unit checks
            
        Returns:
        --------
        pd.DataFrame
            Processed mass/width data
        """
        print("Extracting mass and width data...")
        
        # Initialize processed data
        processed_data = []
        
        for _, row in particle_data.iterrows():
            try:
                # Extract mass with unit conversion
                mass_gev = self._convert_to_gev(row.get('Mass', 0), 
                                              row.get('MassUnit', 'GeV'),
                                              strict_units)
                
                # Extract width with unit conversion
                width_gev = self._convert_to_gev(row.get('ResonanceWidth', 0),
                                               row.get('WidthUnit', 'GeV'),
                                               strict_units)
                
                # Extract uncertainties
                mass_uncertainty = self._convert_to_gev(row.get('MassUncertainty', 0),
                                                      row.get('MassUnit', 'GeV'),
                                                      strict_units)
                
                width_uncertainty = self._convert_to_gev(row.get('WidthUncertainty', 0),
                                                       row.get('WidthUnit', 'GeV'),
                                                       strict_units)
                
                # Validate data
                if strict_units:
                    self._validate_units(mass_gev, width_gev, mass_uncertainty, width_uncertainty)
                
                # Create processed row
                processed_row = {
                    'Name': row.get('Name', ''),
                    'PDG_ID': row.get('PDG_ID', 0),
                    'MassGeV': mass_gev,
                    'ResonanceWidthGeV': width_gev,
                    'MassSigmaGeV': mass_uncertainty,
                    'WidthSigmaGeV': width_uncertainty,
                    'J': row.get('J', 0),
                    'P': row.get('P', 0),
                    'C': row.get('C', 0),
                    'G': row.get('G', 0),
                    'I': row.get('I', 0),
                    'S': row.get('S', 0),
                    'B': row.get('B', 0),
                    'Q': row.get('Q', 0),
                    'Status': row.get('Status', ''),
                    'Family': row.get('Family', ''),
                    'ParticleType': row.get('ParticleType', ''),
                    'M2GeV2': mass_gev**2,
                    'M2SigmaGeV2': 2 * mass_gev * mass_uncertainty  # Error propagation
                }
                
                processed_data.append(processed_row)
                
            except Exception as e:
                if strict_units:
                    raise ValueError(f"Unit conversion failed for particle {row.get('Name', 'Unknown')}: {e}")
                else:
                    warnings.warn(f"Skipping particle {row.get('Name', 'Unknown')} due to unit conversion error: {e}")
        
        result_df = pd.DataFrame(processed_data)
        
        print(f"Successfully processed {len(result_df)} particles")
        return result_df
    
    def _convert_to_gev(self, value: float, unit: str, strict: bool = True) -> float:
        """
        Convert value to GeV with strict unit checking.
        
        Parameters:
        -----------
        value : float
            Value to convert
        unit : str
            Source unit
        strict : bool
            Whether to enforce strict unit checking
            
        Returns:
        --------
        float
            Value in GeV
        """
        if not isinstance(value, (int, float)) or np.isnan(value):
            if strict:
                raise ValueError(f"Invalid value: {value}")
            return 0.0
        
        unit_lower = unit.lower()
        
        # Conversion factors to GeV
        conversion_factors = {
            'gev': 1.0,
            'mev': 1e-3,
            'kev': 1e-6,
            'ev': 1e-9,
            'tev': 1e3,
            'pev': 1e6
        }
        
        if unit_lower not in conversion_factors:
            if strict:
                raise ValueError(f"Unknown unit: {unit}")
            else:
                warnings.warn(f"Unknown unit {unit}, assuming GeV")
                return value
        
        return value * conversion_factors[unit_lower]
    
    def _validate_units(self, mass: float, width: float, 
                       mass_uncertainty: float, width_uncertainty: float) -> None:
        """
        Validate unit consistency.
        
        Parameters:
        -----------
        mass : float
            Mass in GeV
        width : float
            Width in GeV
        mass_uncertainty : float
            Mass uncertainty in GeV
        width_uncertainty : float
            Width uncertainty in GeV
        """
        # Check for reasonable mass values (0.1 MeV to 10 TeV)
        if not (1e-4 <= mass <= 1e4):
            raise ValueError(f"Mass {mass} GeV outside reasonable range")
        
        # Check for reasonable width values
        if width < 0:
            raise ValueError(f"Negative width: {width} GeV")
        
        # Check for reasonable uncertainties
        if mass_uncertainty < 0:
            raise ValueError(f"Negative mass uncertainty: {mass_uncertainty} GeV")
        
        if width_uncertainty < 0:
            raise ValueError(f"Negative width uncertainty: {width_uncertainty} GeV")
        
        # Check that uncertainties are reasonable compared to values
        if mass > 0 and mass_uncertainty > mass:
            warnings.warn(f"Mass uncertainty {mass_uncertainty} GeV larger than mass {mass} GeV")
        
        if width > 0 and width_uncertainty > width:
            warnings.warn(f"Width uncertainty {width_uncertainty} GeV larger than width {width} GeV")

class DeterministicFilters:
    """
    Centralized deterministic filters for reproducible research.
    """
    
    def __init__(self):
        """Initialize deterministic filters."""
        self.filter_rules = self._initialize_filter_rules()
    
    def _initialize_filter_rules(self) -> Dict[str, Any]:
        """Initialize filter rules as a single Association."""
        return {
            'status_filters': {
                'include_stars': ['***', '**', '*'],  # Established states
                'exclude_stars': ['#', '~'],  # Tentative/omitted states
                'min_status': '**'  # Minimum status requirement
            },
            'isobar_filters': {
                'include_isobars': True,
                'exclude_isobars': False,
                'isobar_types': ['Delta', 'N*', 'Lambda*', 'Sigma*', 'Xi*']
            },
            'parity_filters': {
                'include_positive_parity': True,
                'include_negative_parity': True,
                'parity_values': [1, -1]
            },
            'quantum_number_filters': {
                'min_j': 0.5,
                'max_j': 15.5,
                'min_mass': 0.5,  # GeV
                'max_mass': 5.0,  # GeV
                'exclude_glueballs': True,
                'exclude_hybrids': True
            },
            'family_filters': {
                'include_families': ['Delta', 'N*', 'Lambda*', 'Sigma*', 'Xi*'],
                'exclude_families': ['glueball', 'hybrid', 'exotic']
            },
            'quality_filters': {
                'min_mass_uncertainty': 0.001,  # GeV
                'max_mass_uncertainty': 0.5,    # GeV
                'min_width_uncertainty': 0.001,  # GeV
                'max_width_uncertainty': 1.0    # GeV
            }
        }
    
    def apply_filters(self, data: pd.DataFrame, 
                     filter_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply deterministic filters to data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to filter
        filter_config : Optional[Dict[str, Any]]
            Custom filter configuration
            
        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        print("Applying deterministic filters...")
        
        if filter_config:
            # Merge custom config with default rules
            rules = self.filter_rules.copy()
            for key, value in filter_config.items():
                if key in rules:
                    rules[key].update(value)
                else:
                    rules[key] = value
        else:
            rules = self.filter_rules
        
        filtered_data = data.copy()
        initial_count = len(filtered_data)
        
        # Apply status filters
        if 'status_filters' in rules:
            filtered_data = self._apply_status_filters(filtered_data, rules['status_filters'])
        
        # Apply isobar filters
        if 'isobar_filters' in rules:
            filtered_data = self._apply_isobar_filters(filtered_data, rules['isobar_filters'])
        
        # Apply parity filters
        if 'parity_filters' in rules:
            filtered_data = self._apply_parity_filters(filtered_data, rules['parity_filters'])
        
        # Apply quantum number filters
        if 'quantum_number_filters' in rules:
            filtered_data = self._apply_quantum_number_filters(filtered_data, rules['quantum_number_filters'])
        
        # Apply family filters
        if 'family_filters' in rules:
            filtered_data = self._apply_family_filters(filtered_data, rules['family_filters'])
        
        # Apply quality filters
        if 'quality_filters' in rules:
            filtered_data = self._apply_quality_filters(filtered_data, rules['quality_filters'])
        
        final_count = len(filtered_data)
        print(f"Filtered {initial_count} → {final_count} particles ({initial_count - final_count} removed)")
        
        return filtered_data
    
    def _apply_status_filters(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply status-based filters."""
        if 'include_stars' in rules:
            data = data[data['Status'].isin(rules['include_stars'])]
        
        if 'exclude_stars' in rules:
            data = data[~data['Status'].isin(rules['exclude_stars'])]
        
        return data
    
    def _apply_isobar_filters(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply isobar-based filters."""
        if not rules.get('include_isobars', True):
            data = data[~data['Family'].isin(rules.get('isobar_types', []))]
        
        return data
    
    def _apply_parity_filters(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply parity-based filters."""
        if not rules.get('include_positive_parity', True):
            data = data[data['P'] != 1]
        
        if not rules.get('include_negative_parity', True):
            data = data[data['P'] != -1]
        
        return data
    
    def _apply_quantum_number_filters(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply quantum number filters."""
        if 'min_j' in rules:
            data = data[data['J'] >= rules['min_j']]
        
        if 'max_j' in rules:
            data = data[data['J'] <= rules['max_j']]
        
        if 'min_mass' in rules:
            data = data[data['MassGeV'] >= rules['min_mass']]
        
        if 'max_mass' in rules:
            data = data[data['MassGeV'] <= rules['max_mass']]
        
        return data
    
    def _apply_family_filters(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply family-based filters."""
        if 'include_families' in rules:
            data = data[data['Family'].isin(rules['include_families'])]
        
        if 'exclude_families' in rules:
            data = data[~data['Family'].isin(rules['exclude_families'])]
        
        return data
    
    def _apply_quality_filters(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply quality-based filters."""
        if 'min_mass_uncertainty' in rules:
            data = data[data['MassSigmaGeV'] >= rules['min_mass_uncertainty']]
        
        if 'max_mass_uncertainty' in rules:
            data = data[data['MassSigmaGeV'] <= rules['max_mass_uncertainty']]
        
        if 'min_width_uncertainty' in rules:
            data = data[data['WidthSigmaGeV'] >= rules['min_width_uncertainty']]
        
        if 'max_width_uncertainty' in rules:
            data = data[data['WidthSigmaGeV'] <= rules['max_width_uncertainty']]
        
        return data
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of applied filters for paper documentation."""
        return {
            'filter_rules': self.filter_rules,
            'filter_description': self._generate_filter_description()
        }
    
    def _generate_filter_description(self) -> str:
        """Generate human-readable filter description."""
        rules = self.filter_rules
        
        description = []
        description.append("Applied filters:")
        
        if 'status_filters' in rules:
            status_rules = rules['status_filters']
            description.append(f"- Status: Include {status_rules.get('include_stars', [])}, "
                             f"exclude {status_rules.get('exclude_stars', [])}")
        
        if 'family_filters' in rules:
            family_rules = rules['family_filters']
            description.append(f"- Families: Include {family_rules.get('include_families', [])}")
        
        if 'quantum_number_filters' in rules:
            qn_rules = rules['quantum_number_filters']
            description.append(f"- J range: {qn_rules.get('min_j', 0)} to {qn_rules.get('max_j', 15.5)}")
            description.append(f"- Mass range: {qn_rules.get('min_mass', 0)} to {qn_rules.get('max_mass', 5.0)} GeV")
        
        return "\n".join(description)

if __name__ == "__main__":
    print("Data Hydration")
    print("Use this module for frozen PDG snapshots, unit checks, and deterministic filters")
