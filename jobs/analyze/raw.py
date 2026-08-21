from typing import Any
import hippocampalseq.plotting as hsepl

def raw_analysis(results_path: str, track_type: str, raw_data: dict[str,Any]):
    place_field_data = raw_data['place_field_data']
    raw_data = raw_data['raw_data']
    hsepl.plot_place_fields(
        place_field_data,
        raw_data,
        track_type,
        results_path
    ) 
    hsepl.plot_trajectory_with_velocity(
        raw_data.raw_position[['x','y']].values,
        raw_data.raw_position['Velocity'].values,
        raw_data.environment_size if track_type == 'Open' else None,
        file_path=results_path,
        file_name="true_trajectories.pdf"
    )

    hsepl.plot_lfp_data(
        raw_data.lfp_data,
        file_path=results_path,
        file_name="lfp_data.pdf"
    )
    
    if len(raw_data.environment_size) == 1:
        hsepl.plot_session_stitching(
            raw_data.running_position,
            raw_data.running_spikes,
            file_path=results_path,
            file_name="session_stitching.pdf"
        )