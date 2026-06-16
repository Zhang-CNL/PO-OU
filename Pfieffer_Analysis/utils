
import pynapple as nap
import numpy as np
import pathlib as Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import os
import sys
import pickle


from utils.init_conditions import init_conditions
from utils.Data_Loading import load_all_data
from utils.PreProcessing import preprocess_data
from utils.Place_Fields.PF_analysis import analyze_place_fields
from utils.PreProcessing.data_aligning import assign_theta_phase_to_spikes
from utils.Plotting.Main_Plotting import plot_data_overview, plot_place_field_results
from utils.Plotting.load_data_graphs import  plot_spike_position_integration
from utils.Plotting.Data_Aligning_Plots import plot_theta_phase_assignment
from utils.Theta_Phase.analyze_theta import detect_theta_cycles
from utils.Theta_Phase.phase_locking import calculate_phase_locked
from utils.Uni_Bi.Uni_Bi_cells import classify_theta_modality, calculate_place_field_uni_bimodal
from utils.Plotting.theta_modality_plots import plot_phase_locked, plot_place_field_comparison, plot_all_phase_locked, plot_modality_all_cells, plot_modality_classification, plot_modality_pie, plot_modality_overlay
from utils.Uni_Bi.decoding_uni_bi import decode_by_modality
from utils.Theta_Phase.theta_oscillation_properties import pre_decoding_check

def save_all_figures(figures_dict, save_dir, filename_tag):
    """Save all figures to organized PDFs."""
    
    from datetime import datetime
    
    date_folder = datetime.now().strftime('%Y-%m-%d')
    top_folder = os.path.join(save_dir, date_folder, filename_tag)
    os.makedirs(top_folder, exist_ok=True)
    
    for category, figs in figures_dict.items():
        pdf_path = os.path.join(top_folder, f"{category}.pdf")
        
        with PdfPages(pdf_path) as pdf:
            if isinstance(figs, dict):
                for name, fig in figs.items():
                    if isinstance(fig, list):
                        for f in fig:
                            pdf.savefig(f, bbox_inches='tight')
                            plt.close(f)
                    else:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            elif isinstance(figs, list):
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            else:
                pdf.savefig(figs, bbox_inches='tight')
                plt.close(figs)
        
        print(f"Saved {category} to {pdf_path}")
    
    return top_folder

def full_run(ani, csc, experiments, basepath_position, base_path_lfp, 
            save_figs, save_pickle):
    
    # ani = 'Janni'
    # csc = 'CSC9.ncs'
    EXPERIMENTS = experiments
    SAVE_DIR = f"/project/bioinformatics/WZhang_lab/s437598/_Individual_Animal_Runs/{ani}_Results"
    OUTPUT_DIR = Path(
        f'/project/bioinformatics/WZhang_lab/s437598/'
        f'_Individual_Animal_Runs/2026_Final_{ani}'
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    initial_variables, timepoints_to_remove_default, bimodal_windows = init_conditions()
    initial_variables['bimodal_windows'] = bimodal_windows
        
    #initialize data gathered
    data = {}
    preprocessed = {}
    place_fields = {}
    lfp_cycles_all = {}
    spike_phase = {}
    phase_locked = {}
    modality = {}
    place_field_props_all = {}
    predecode_results = {}
    decoded_results    = {}
    decoded_predecodes = {}

    #initialize figures saved
    all_figs = {
        "Position_Spike_Data":        {},
        "Session_Stitching":          {},
        "Spike_Position_Integration": {}}
    all_pf_figs = {
        "Linear_Heatmap":  {},
        "Linear_Examples": {},
        "Raster_Plot":     {},
        "2D_Fields":       {},
        "Decoding_Error":  {}}
    all_theta_phase_figs = {"Theta_Phase_Assignment": {}}
    all_lfp_cycle_figs = {
        "LFP_Troughs":  {},
        "Cycle_Durations": {}}
    all_spike_phase_figs = {"Spike_Position_Phase": {}}
    all_phase_locked_figs = {"Phase_Locking": {}}
    all_modality_figs = {"Modality_Summary": {}}
    # Pool storage across all sessions
    pooled_unimodal = None
    pooled_bimodal  = None

    
    for expt in EXPERIMENTS:
        
        ### Load Data
        print(f"=== Load Data For {ani} Exp: {expt} ===")
        data[expt] = load_all_data(ani, expt, basepath_position, base_path_lfp, csc)
        print()
        
        ### Preprocessing
        print(f"=== Preprocessing Data for {ani} Exp: {expt} ===")
        preprocessed[expt] = preprocess_data(
            data[expt], ani, expt, timepoints_to_remove_default
        )
        print()
        
        ### Get Place Fields
        print(f"=== Place Fields for {ani} Exp: {expt} ===")
        place_fields[expt] = analyze_place_fields(
            initial_variables, data[expt], preprocessed[expt],
            bin_size=2, velocity_cutoff=10, firing_rate_cutoff=1
        )
        print()
        
        ### Find theta cycles
        print(f"=== Detect Theta Cycles for {ani} Exp: {expt} ===")
        preprocessedlft = preprocessed[expt]
        lfp_cycles, lfp_cycle_figs = detect_theta_cycles(
            preprocessedlft['lfp'],initial_variables,
            plot=True, segment_seconds=6.0, random_segment=False
        )
        lfp_cycles_all[expt] = lfp_cycles
        print()

        
        ### Align spikes to theta phase
        print(f"=== Assigning Theta Phase to Spikes for {ani} Exp: {expt} ===")
        spike_phase[expt] = assign_theta_phase_to_spikes(
            preprocessed[expt]['spike_info'],
            lfp_cycles_all[expt],
        )

        ### Phase Locking 
        print(f"=== Calculating phase locking for cells in {ani}: {expt} ===")
        firing_rate_per_phase, phase_centers = calculate_phase_locked(
            spike_phase[expt],
            total_duration=preprocessed[expt]['total_duration'],
            speed_cutoff=initial_variables['velocity_cutoff'],
            phase_bin=initial_variables['phase_bin'],
            gaussian_sigma=initial_variables['gaussian_smoothing_sigma'],
            theta_length_min_max=initial_variables['theta_length_min_max'],
            minimum_spike_count=initial_variables['minimum_spike_count'],
            limit_analysis_by_theta_length=bool(initial_variables['limit_analysis_by_theta_length']),
        )
        phase_locked[expt] = {
            "firing_rate_per_phase": firing_rate_per_phase,
            "phase_centers": phase_centers}
        
        print(f"=== Classifying modality for {ani}: {expt} ===")
        modality_results, population_stats = classify_theta_modality(
            phase_locked[expt]["firing_rate_per_phase"],
            phase_locked[expt]["phase_centers"],
            spike_phase[expt],
            excitatory_neurons=preprocessed[expt].get('excitatory_neurons', None),
            phase_bin=initial_variables['phase_bin'],
            rayleigh_p_cutoff=initial_variables['rayleigh_test_p_value_cutoff'])
        modality[expt] = {
            "results":     modality_results,
            "population":  population_stats}
        
        ### Get Modality Specific Place Fields
        print(f"=== Place field uni/bimodal for {ani}: {expt} ===")
        excitatory_set      = set(data[expt]['excitatory'])
        true_excit_in_phase = sorted(set(spike_phase[expt].keys()) & excitatory_set)
        place_field_props, unimodal, bimodal = calculate_place_field_uni_bimodal(
            field_results       = place_fields[expt]['pf_results'],
            spike_info          = spike_phase[expt],
            modality_results    = modality[expt]['results'],
            excitatory_neurons  = true_excit_in_phase,
            velocity_cutoff     = 10,   # matches find_PFs
            min_field_fraction  = initial_variables['minimum_place_field_firing_rate_fraction'],
            min_contiguous_bins = initial_variables['minimum_contiguous_place_field_bins'])
        place_field_props_all[expt] = {
            'place_field_props': place_field_props,
            'unimodal':          unimodal,
            'bimodal':           bimodal}
        # Concatenate into pooled arrays
        if pooled_unimodal is None:
            pooled_unimodal = {k: list(v) for k, v in unimodal.items()}
            pooled_bimodal  = {k: list(v) for k, v in bimodal.items()}
        else:
            for k in pooled_unimodal:
                pooled_unimodal[k].extend(unimodal[k])
                pooled_bimodal[k].extend(bimodal[k])
        print()

        ### Predecoding
        print(f"\n=== Pre-decoding check for {expt} ===")
        predecode_results[expt] = pre_decoding_check(
            position_data      = preprocessed[expt]['position'],
            spike_data         = preprocessed[expt]['spike_info_moving'],
            lfp_data           = preprocessed[expt]['lfp_moving'],
            excitatory_neurons = data[expt]['excitatory'],
            bimodal_windows    = bimodal_windows,
            initial_variables  = initial_variables,
            vel_cut            = 10,
        )
        
        print(f"Decoding {expt}")
        results, predecodes = decode_by_modality(
            predecode         = predecode_results[expt],
            field_results     = place_fields[expt]['pf_results'],
            modality_results  = modality[expt]['results'],
            initial_variables = initial_variables,
        )

        decoded_results[expt]    = results
        decoded_predecodes[expt] = predecodes
        
        #Package figures 
        
        ##Raw data
        print(f"=== Plotting overview for {expt} ===")
        figs = plot_data_overview(data[expt], preprocessed[expt])

        all_figs["Position_Spike_Data"][expt]        = figs["Track Position vs Speed"]
        all_figs["Spike_Position_Integration"][expt] = figs["spike_position_integration"]
        if "Session Stitching" in figs:
            all_figs["Session_Stitching"][expt] = figs["Session Stitching"]
            
        ##Place Fields
        print(f"=== Plotting place field results for {expt} ===")
        figs = plot_place_field_results(
            initial_variables,
            place_fields[expt],
            preprocessed[expt],
            data[expt])
        pf_plots = figs['place_fields'] or {}
        lin = pf_plots.get('linear_figs', {})
        all_pf_figs["Linear_Heatmap"][expt]  = lin.get('linear_heatmap')
        all_pf_figs["Linear_Examples"][expt] = lin.get('linear_examples')
        all_pf_figs["Raster_Plot"][expt]     = lin.get('raster_plot')
        all_pf_figs["2D_Fields"][expt]       = pf_plots.get('fig_2d')
        all_pf_figs["Decoding_Error"][expt]  = figs.get('decoding_error')
        
        #Theta phase assignment
        print(f"=== Plot Theta Phase Assignment for {expt} ===")
        fig = plot_theta_phase_assignment(
            spike_phase[expt],
            lfp_cycles_all[expt],
            excitatory_neurons=data[expt]['excitatory'],
            n_cells=6, time_window=10,
            skip_top_n=3,
            random_seed=93)
        all_theta_phase_figs["Theta_Phase_Assignment"][expt] = fig
        print()
        
        ### LFP Cycles
        all_lfp_cycle_figs["LFP_Troughs"][expt]  = lfp_cycle_figs.get('lfp_phase')
        all_lfp_cycle_figs["Cycle_Durations"][expt] = lfp_cycle_figs.get('cycle_durations')
        
        ### Theta Spike Alignment
        fig_spk_int = plot_spike_position_integration(
            spike_phase[expt], data[expt]['position'],data[expt]['excitatory'],
            n_cells=5, cell_selection='most_active',skip_top_n=5)
        all_spike_phase_figs["Spike_Position_Phase"][expt] = fig_spk_int
        
        #Phase locking
        fig_pl = plot_phase_locked(
            firing_rate_per_phase,
            phase_centers,
            n_cells=20)
        all_phase_locked_figs["Phase_Locking"][expt] = fig_pl
        
        figs_pla = plot_all_phase_locked(
            phase_locked[expt]["firing_rate_per_phase"],
            phase_locked[expt]["phase_centers"],
            cols=4,
            rows_per_page=5)
        pdf_dir_pla = os.path.join(SAVE_DIR, f"2026_Final_{ani}_phase_locking_all_cells")
        os.makedirs(pdf_dir_pla, exist_ok=True)
        pdf_path_pla = os.path.join(pdf_dir_pla, f"2026_Final_{ani}_{expt}_phase_locking_all_cells.pdf")
        if save_figs:
            with PdfPages(pdf_path_pla) as pdf:
                for fig in figs_pla:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
        ### Modality Classification
        summary_fig = plot_modality_classification(
            phase_locked[expt]["firing_rate_per_phase"],
            modality_results, population_stats,
            phase_locked[expt]["phase_centers"], n_examples=3)
        all_modality_figs["Modality_Summary"][expt] = summary_fig
        #Full per-session PDF: every cell, grouped by modality
        figs = plot_modality_all_cells(
            phase_locked[expt]["firing_rate_per_phase"],
            modality_results,
            phase_locked[expt]["phase_centers"],
            cols=4, rows_per_page=5)
        pdf_dir_md  = os.path.join(SAVE_DIR, f"2026_Final_{ani}_modality_all_cells")
        os.makedirs(pdf_dir_md, exist_ok=True)
        pdf_path = os.path.join(pdf_dir_md, f"2026_Final_{ani}_{expt}_modality_all_cellsLin_1and2_only.pdf")
        if save_figs:
            with PdfPages(pdf_path) as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        overlay_fig = plot_modality_overlay(
            population_stats,
            phase_locked[expt]['phase_centers'])
        all_modality_figs.setdefault('Modality_Overlay', {})[expt] = overlay_fig
        pie_fig = plot_modality_pie(modality_results)
        all_modality_figs.setdefault('Modality_Pie', {})[expt] = pie_fig
                    
    #Putting figures in dictionary
    
    # Drop None values + empty categories 
    all_figs = {k: v for k, v in all_figs.items() if v}
    all_pf_figs = {
    cat: {expt: fig for expt, fig in d.items() if fig is not None}
    for cat, d in all_pf_figs.items()}
    all_pf_figs = {k: v for k, v in all_pf_figs.items() if v}
    all_theta_phase_figs = {
        cat: {expt: fig for expt, fig in d.items() if fig is not None}
        for cat, d in all_theta_phase_figs.items()}
    all_theta_phase_figs = {k: v for k, v in all_theta_phase_figs.items() if v}
    all_lfp_cycle_figs = {
        cat: {expt: fig for expt, fig in d.items() if fig is not None}
        for cat, d in all_lfp_cycle_figs.items()}
    all_lfp_cycle_figs = {k: v for k, v in all_lfp_cycle_figs.items() if v}
    all_spike_phase_figs = {
        cat: {expt: fig for expt, fig in d.items() if fig is not None}
        for cat, d in all_spike_phase_figs.items()}
    all_spike_phase_figs = {k: v for k, v in all_spike_phase_figs.items() if v}
    
    all_phase_locked_figs = {
        cat: {expt: fig for expt, fig in d.items() if fig is not None}
        for cat, d in all_phase_locked_figs.items()}
    all_phase_locked_figs = {k: v for k, v in all_phase_locked_figs.items() if v}
    all_modality_figs = {k: v for k, v in all_modality_figs.items() if v}
    
    for k in pooled_unimodal:
        pooled_unimodal[k] = np.array(pooled_unimodal[k], dtype=float)
        pooled_bimodal[k]  = np.array(pooled_bimodal[k],  dtype=float)
    fig_pfm = plot_place_field_comparison(pooled_unimodal, pooled_bimodal,
                                expt_name='All Linear Track Sessions')
    fig_pfm = {'PlaceFieldComparison': {'pooled':fig_pfm}}
    
    
    if save_figs:
        save_all_figures(all_figs, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_all_linear')
        save_all_figures(all_pf_figs, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_place_fields')
        save_all_figures(all_theta_phase_figs, save_dir=SAVE_DIR,filename_tag=f'2026_Final_{ani}_theta_phase_assignment')
        save_all_figures(all_lfp_cycle_figs, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_lfp_cycle_detection')
        save_all_figures(all_spike_phase_figs, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_spike_phase')
        save_all_figures(all_modality_figs, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_modality_summary')
        save_all_figures(fig_pfm, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_place_field_comparison_pooledy')
        save_all_figures(all_phase_locked_figs, save_dir=SAVE_DIR, filename_tag=f'2026_Final_{ani}_phase_locking')
        
    if save_pickle:
        modality_save_path = OUTPUT_DIR / f'2026_final_modality_results_{ani}.pkl'
        with open(modality_save_path, 'wb') as f:
            pickle.dump({
                'modality':    modality,           # full dict, keyed by expt
                'animal':      ani,
                'experiments': EXPERIMENTS,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved modality to {modality_save_path}")
        
        predecode_save_path = OUTPUT_DIR / f'2026_final_predecode_results_{ani}.pkl'
        with open(predecode_save_path, 'wb') as f:
            pickle.dump({
                'predecode_results': predecode_results,
                'animal':            ani,
                'experiments':       EXPERIMENTS,
                'initial_variables': initial_variables,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved predecode to {predecode_save_path}")
        
        save_path = OUTPUT_DIR / f'2026_final_decoded_theta_sequences_{ani}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'decoded_results':    decoded_results,
                'decoded_predecodes': decoded_predecodes,
                'animal':             ani,
                'experiments':        EXPERIMENTS,
                'initial_variables':  initial_variables,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nSaved to {save_path}")
        print(f"Size: {save_path.stat().st_size / 1e6:.1f} MB")
