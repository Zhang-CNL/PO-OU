import os 
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .core import reset_plotting, change_font_sizes
from .placefields import plot_linear_placefields,plot_open_placefields
from .trajectories import plot_spikemat_position_aligned

import hippocampalseq as hse

def plot_place_fields(
        place_field_data: hse.PlaceFields, 
        raw_data: hse.RawData, 
        track_type: str, 
        results_path: str,
        filename: str = "place_fields.pdf"
    ):
    with PdfPages(os.path.join(results_path, filename)) as pdf:
        for i,cell_id in enumerate(place_field_data.place_cell_ids):
            change_font_sizes(14, 14, 16)
            fig,ax = plt.subplots(1,2,figsize=(20,10), dpi=300)
            ax[0].set_title(f"Place cell {i}")
            if track_type == 'Linear' and place_field_data.place_fields.shape[-1] == 1:
                plot_linear_placefields(
                    place_field_data.place_fields,
                    pfs=[i],
                    ax=ax[0]
                )
                esize= None
            else:
                plot_open_placefields(
                        place_field_data.place_fields,
                        pfs=[i],
                        show_titles=False,
                        ax=ax[0],
                    )
                esize = [(0,200),(0,200)]

            plot_spikemat_position_aligned(
                raw_data.running_spike_info,
                raw_data.raw_position,
                place_field_data.place_cell_ids, 
                environment_size=esize,
                cell_selection=[int(cell_id)],
                ax=ax[1]
            )
            ax[1].set_ylabel("")
            ax[1].set_yticklabels([])
            fig.set_tight_layout(True)
            pdf.savefig()
            plt.close(fig)

    reset_plotting()