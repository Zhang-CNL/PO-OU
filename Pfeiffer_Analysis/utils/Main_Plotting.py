from utils.Plotting.load_data_graphs import plot_loaded_data, plot_spike_position_integration
from utils.Plotting.Data_Aligning_Plots import  plot_theta_phase_assignment, plot_cell_phase_polar
from utils.Plotting.PF_Plotting import plot_place_fields, decoding_error_plots
#from utils.Plotting.Decoding_Plotting import decoding_error_plots
from utils.Plotting.theta_modality_plots import plot_modality_classification, plot_phase_locked, plot_place_field_comparison, plot_unimodal_bimodal_summary, plot_individual_cell_histograms
from utils.PostProcessing.postdecoding_processing_uni_bi import plot_decoded_comparison

def plot_data_overview(data, preprocessed, max_cells=None):
    """Plot loaded data overview."""
    
    figs = plot_loaded_data(
        data['position'], data['spikes'],
        data['excitatory'], data['inhibitory'],
        data['lfp'],
        max_cells=max_cells
    )
    
    integrate_plot = plot_spike_position_integration(
        preprocessed['spike_info_original'],
        data['position'],
        data['excitatory'],
        n_cells=10,
        cell_selection='random'
    )
    
    figs['spike_position_integration'] = integrate_plot
    
    return figs


def plot_place_field_results(initial_variables, place_field_results, 
                            preprocessed, data):
    """Plot place field analysis results."""
    
    pf_plots = plot_place_fields(
        initial_variables,
        place_field_results['pf_results'],
        preprocessed['spike_info_original'],
        data['position'],
        excitatory_neurons=data['excitatory'],
    )
    
    error_plots = decoding_error_plots(
        place_field_results['decoding_error'],
        time_bin_size=0.25
    )
    
    return {
        'place_fields': pf_plots,
        'decoding_error': error_plots
    }


def plot_theta_results(theta_results, preprocessed, n_cells=12):
    """Plot theta phase analysis results."""
    
    figures = {}
    
    # Phase assignment verification
    figures['phase_assignment'] = plot_theta_phase_assignment(
        theta_results['spike_info_with_phase'],
        theta_results['lfp_with_cycles'],
        n_cells=10,
        time_window=10
    )
    
    # Phase-locked firing
    figures['phase_locked'] = plot_phase_locked(
        theta_results['firing_rate_per_phase'],
        theta_results['phase_centers'],
        n_cells=n_cells
    )
    
    # Modality classification
    figures['modality'] = plot_modality_classification(
        theta_results['firing_rate_per_phase'],
        theta_results['modality_results'],
        theta_results['population_stats'],
        theta_results['phase_centers']
    )
    
    return figures


def plot_modality_results(theta_results, modality_analysis, data):
    """Plot unimodal vs bimodal comparison results."""
    
    figures = {}
    
    # Summary plots
    fig1, fig2, fig3, fig4 = plot_unimodal_bimodal_summary(
        theta_results['firing_rate_per_phase'],
        theta_results['modality_results'],
        theta_results['population_stats'],
        theta_results['phase_centers'],
        modality_analysis['unimodal_props'],
        modality_analysis['bimodal_props'],
        data['excitatory']
    )
    figures['summary'] = [fig1, fig2, fig3, fig4]
    
    # Individual cell histograms
    fig_uni, fig_bi = plot_individual_cell_histograms(
        theta_results['firing_rate_per_phase'],
        theta_results['modality_results'],
        theta_results['phase_centers'],
        n_per_category=6
    )
    figures['individual_cells'] = {'unimodal': fig_uni, 'bimodal': fig_bi}
    
    # Place field comparison
    figures['pf_comparison'] = plot_place_field_comparison(
        modality_analysis['unimodal_props'],
        modality_analysis['bimodal_props']
    )
    
    # Decoded comparison
    figures['decoded_comparison'] = plot_decoded_comparison(
        modality_analysis['decoding_results'],
        theta_results['phase_centers']
    )
    
    return figures