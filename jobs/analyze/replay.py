from typing import Any
import hippocampalseq as hse
import hippocampalseq.plotting as hsepl

def replay_analysis(
        results_path: str, 
        track_type: str, 
        raw_data: dict[str, [hse.RawData, hse.PlaceFields]], 
        replay_data: dict[str, Any],
        parameters: dict[str, Any]
    ):
    pass