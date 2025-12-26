import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns

# Use Unicode for Greek characters
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set style for publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_regge_trajectory():
    """Create complete Regge trajectory plot for Delta baryons."""
    
    # Load the unified states data
    data = pd.read_csv('python_analysis/unified_analysis_results/unified_states.csv')
    
    # Filter for Delta family and valid data points
    delta_data = data[data['family'] == 'Delta'].copy()
    valid_mask = (delta_data['m2_gev2'] > 0) & (delta_data['j'] > 0)
    delta_data = delta_data[valid_mask]
    
    # Sort by mass squared for proper trajectory
    delta_data = delta_data.sort_values('m2_gev2')
    
    # Extract data for fitting
    x_data = delta_data['m2_gev2'].values
    y_data = delta_data['j'].values
    y_errors = delta_data['m2_sigma_gev2'].values * 2 * delta_data['mass_gev'].values  # Convert to J errors
    
    # Fit linear Regge trajectory: J = α₀ + α'M²
    def regge_model(x, alpha0, alphap):
        return alpha0 + alphap * x
    
    # Perform weighted least squares fit
    popt, pcov = curve_fit(regge_model, x_data, y_data, 
                          sigma=y_errors, absolute_sigma=True)
    
    alpha0, alphap = popt
    alpha0_err, alphap_err = np.sqrt(np.diag(pcov))
    
    # Calculate fit statistics
    y_pred = regge_model(x_data, alpha0, alphap)
    residuals = y_data - y_pred
    chi2 = np.sum((residuals / y_errors)**2)
    dof = len(x_data) - 2
    chi2_dof = chi2 / dof
    
    # Calculate R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points with error bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(delta_data)))
    
    for i, (_, row) in enumerate(delta_data.iterrows()):
        ax.errorbar(row['m2_gev2'], row['j'], 
                   yerr=row['m2_sigma_gev2'] * 2 * row['mass_gev'],
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color=colors[i], alpha=0.8, linewidth=2)
        
        # Add particle labels
        ax.annotate(row['name'], 
                   (row['m2_gev2'], row['j']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    # Plot fitted trajectory
    x_fit = np.linspace(x_data.min() * 0.9, x_data.max() * 1.1, 100)
    y_fit = regge_model(x_fit, alpha0, alphap)
    
    ax.plot(x_fit, y_fit, 'r-', linewidth=3, alpha=0.8, 
            label=f'Linear Fit: J = {alpha0:.3f} ± {alpha0_err:.3f} + ({alphap:.3f} ± {alphap_err:.3f}) M²')
    
    # Add confidence band
    y_fit_upper = regge_model(x_fit, alpha0 + alpha0_err, alphap + alphap_err)
    y_fit_lower = regge_model(x_fit, alpha0 - alpha0_err, alphap - alphap_err)
    ax.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.2, color='red', 
                   label='1σ Confidence Band')
    
    # Customize the plot
    ax.set_xlabel(r'$M^2$ (GeV²)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$J$ (Spin)', fontsize=14, fontweight='bold')
    ax.set_title('Regge Trajectory: Δ Baryon Family', fontsize=16, fontweight='bold')
    
    # Add fit statistics text box
    stats_text = f'Fit Statistics:\n' \
                 f'α0 = {alpha0:.3f} ± {alpha0_err:.3f}\n' \
                 f'α\' = {alphap:.3f} ± {alphap_err:.3f}\n' \
                 f'χ²/dof = {chi2_dof:.3f}\n' \
                 f'R² = {r_squared:.3f}\n' \
                 f'N = {len(delta_data)} states'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    # Set axis limits with some padding
    ax.set_xlim(x_data.min() * 0.9, x_data.max() * 1.1)
    ax.set_ylim(y_data.min() - 0.5, y_data.max() + 0.5)
    
    # Add theoretical context
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('delta_regge_trajectory.png', dpi=300, bbox_inches='tight')
    plt.savefig('delta_regge_trajectory.pdf', bbox_inches='tight')
    
    print("Regge trajectory plot saved as 'delta_regge_trajectory.png' and 'delta_regge_trajectory.pdf'")
    
    # Print fit results
    print(f"\nRegge Trajectory Fit Results:")
    print(f"α₀ = {alpha0:.4f} ± {alpha0_err:.4f}")
    print(f"α' = {alphap:.4f} ± {alphap_err:.4f}")
    print(f"χ²/dof = {chi2_dof:.3f}")
    print(f"R² = {r_squared:.4f}")
    print(f"Number of states: {len(delta_data)}")
    
    return fig, ax, (alpha0, alphap, alpha0_err, alphap_err, chi2_dof, r_squared)

if __name__ == "__main__":
    create_regge_trajectory()
