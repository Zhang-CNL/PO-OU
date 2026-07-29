from scipy.special import logsumexp
from pathlib import Path
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
from scipy import ndimage
import math


def translate_decoding(arr, shift_y, shift_x):

    ny, nx = arr.shape
    out = np.zeros_like(arr)

    # src = source array from decoding
    # dst = destination after shifting
    # start indices
    src_y0 = max(0, -shift_y)
    dst_y0 = max(0,  shift_y)
    src_x0 = max(0, -shift_x)
    dst_x0 = max(0,  shift_x)

    # Height/width of overlapping region
    h = min(ny - src_y0, ny - dst_y0)
    w = min(nx - src_x0, nx - dst_x0)

    if h <= 0 or w <= 0:
        return out  # everything shifted out of frame

    out[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = arr[src_y0:src_y0 + h, src_x0:src_x0 + w]

    return out

def resize_rows_to_decoded_size(arr, decoded_data_size):
    n_rows, n_cols = arr.shape
    if n_rows < decoded_data_size:
        size_diff = decoded_data_size - n_rows
        if size_diff % 2 == 0:
            pre = post = size_diff // 2
        else:
            pre = math.ceil(size_diff / 2)
            post = size_diff - pre
        arr = np.pad(arr, ((pre, post), (0, 0)), mode="constant", constant_values=0.0)
    elif n_rows > decoded_data_size:
        size_diff = n_rows - decoded_data_size
        if size_diff % 2 == 0:
            start = size_diff // 2
            end = n_rows - size_diff // 2
            arr = arr[start:end, :]
        elif size_diff == 1:
            arr = arr[:-1, :]
        else:
            c = math.ceil(size_diff / 2)
            start = c - 1          # 1-based -> 0-based
            end = n_rows - c
            arr = arr[start:end, :]
    return arr


def resize_translated_to_decoded_size(arr, decoded_data_size):
    n_rows, n_cols = arr.shape

    #Fix number of rows
    if n_rows < decoded_data_size:
        size_diff = decoded_data_size - n_rows
        if size_diff % 2 == 0:  # even
            pre = post = size_diff // 2
        else:                   # odd
            pre = math.ceil(size_diff / 2)
            post = size_diff - pre
        arr = np.pad(arr, ((pre, post), (0, 0)), mode="constant", constant_values=0.0)

    elif n_rows > decoded_data_size:
        size_diff = n_rows - decoded_data_size
        if size_diff % 2 == 0:  # even
            start = size_diff // 2
            end = n_rows - size_diff // 2
            arr = arr[start:end, :]
        elif size_diff == 1:
            arr = arr[:-1, :]
        else:
            c = math.ceil(size_diff / 2)
            start = c
            end = n_rows - c
            arr = arr[start:end, :]

    # update dims after row ops
    n_rows, n_cols = arr.shape

    # Fix # columns
    if n_cols < decoded_data_size:
        size_diff = decoded_data_size - n_cols
        if size_diff % 2 == 0:  # even
            pre = post = size_diff // 2
        else:                   # odd
            pre = math.ceil(size_diff / 2)
            post = size_diff - pre
        arr = np.pad(arr, ((0, 0), (pre, post)), mode="constant", constant_values=0.0)

    elif n_cols > decoded_data_size:
        size_diff = n_cols - decoded_data_size
        if size_diff % 2 == 0:  # even
            start = size_diff // 2
            end = n_cols - size_diff // 2
            arr = arr[:, start:end]
        elif size_diff == 1:
            arr = arr[:, :-1]
        else:
            c = math.ceil(size_diff / 2)
            start = c 
            end = n_cols - c
            arr = arr[:, start:end]

    return arr


def decode_theta_sequences(predecode, field_data, initial_variables, use_max_pp=True):
        
    # Parameters
    decoding_time_window = initial_variables["decoding_time_window"]    # 0.02
    bin_size = initial_variables["bin_size"]    #2 cm
    use_max_pp = initial_variables.get("use_maximum_posterior_probability") #1, equivalent to true
    decoding_time_advance = initial_variables.get("decoding_time_advance")  #0.005
    
    
    # Get data
    decoding_time_info = np.asarray(predecode["DTI"])                 # MATLAB Decoding_Time_Info
    #print(f"decoding time info shape: {decoding_time_info.shape}")
    decoding_window_index = np.asarray(predecode["decoding_window_index"])
    #print(f"{decoding_window_index.shape[0]} windows used for decoding")
    
    # rows = spikes, columns = all possible windows based on decoding_time_info
    decoding_spike_index = predecode["decoding_spike_index"]
    #3print(f"{decoding_spike_index.shape[0]} spikes used for decoding in this window")
    #print(f"decoding spike info shape: {decoding_spike_index.shape}")
    n_all_windows = decoding_time_info.shape[0]
    n_valid_windows = len(decoding_window_index)
    
    # Get place fields
    Field_Data = field_data["Field_Data"]
    cell_ids = np.asarray(field_data["cell_ids"])
    ny, nx, n_cells = Field_Data.shape
    
    #Do this as in the decoding error calculation
    fields_modified = Field_Data.copy()
    for i in range(n_cells):
        field = fields_modified[:,:, i]
        positive = field[field>0]
        if positive.size > 0:
            minimum = positive.min()
            eps = minimum / 10 if minimum / 10 > 0 else minimum
            field[field <= 0] = eps
        else:
            field[:] = 1.0
        fields_modified[:,:, i] = field
    # sum over cells used below in the exp(-T * sum(Field_Data,3)) term
    sum_fields = Field_Data.sum(axis=2)  # shape (ny, nx), inh+exc
    #mapping from cell IDs to indices
    cell_id_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
    
    #defines the size of the canvas for translation and rotation analysis 
    #This ensures we are always in the middle of the canvas 
    # and looking at the "forward" direction for the rat
    #remember nx, ny are number of spatial bins for x and y
    #SO this finds the middle of the canvas we built with the bins
    Mid_X = int(round(nx / 2.0))
    Mid_Y = int(round(ny / 2.0))
    #print(f"Values of Mid_X and Mid_Y: {Mid_X}, {Mid_Y}")
    #Length of the 1D axis along movement direction after rotation
    decoded_data_size = (max(Mid_X, Mid_Y) * 2) + 1

    decoded_sequence = np.zeros((3, n_valid_windows)) #max position along movement direction
    untranslated_decoded_sequence = np.zeros((3, n_valid_windows))
    decoded_data = np.zeros((decoded_data_size, n_valid_windows)) # 1D posterior along mvoement direction for window i
    not_translated_decoded_data = np.zeros((decoded_data_size, n_valid_windows))
    decoded_x_data = np.zeros((nx, n_valid_windows)) # max x index
    decoded_y_data = np.zeros((ny, n_valid_windows)) #max y index
    
    decoded_x_data_untranslated = np.zeros((nx, n_valid_windows))
    decoded_y_data_untranslated = np.zeros((ny, n_valid_windows))
    
    #This is the only decoding sequence we will build, we will not look at unidirectional linear tracks
    # or modality of neurons -> for this case, we don't care about unimodal vs bimodal cells
    #     %Decoded_Sequence
    # %|                           1                         
    # %|        Location of Max Posterior Probability        
    # %|            Relative to Movement Direction           
    # %| (for both open field and bidirectional linear track)
    
    #Actual decoding loop
    # Compress spike index to only those we care about
    #which should be faster than going through all the windows
    decoding_spike_index_valid = decoding_spike_index[:, decoding_window_index]
    
    for i in range(n_valid_windows):
        #current window
        win_pos = decoding_window_index[i]
        
        #Find spikes for this window
        subset_spike_data = decoding_spike_index_valid[:, i]
        subset_spike_data = subset_spike_data[subset_spike_data > 0]
        
        #random checkpoint for my sanity
        # if i == 2:
        #     print(f"subset_spike_data (These are the cells that spike in this window): {subset_spike_data}")
        
        # if subset_spike_data.size==0:
        #     #if there are no spikes in this window ---> skip it
        #     continue
        
        #Get spike indices from subset_spike_data    
        spike_indices = [cell_id_idx[cid] for cid in subset_spike_data if cid in cell_id_idx]
        if len(spike_indices) == 0:
            continue
        
        #actual decoding lines
        #reminder: this selects are the place fields spiking in this window
        # then multiplies those firing rates together at each x,y location
        fields_for_spikes = fields_modified[:, :, spike_indices]
        # sum_fields is the summed firing rates of all cells at each x,y location
        #this is calculated above so it doesn't have to be done each time in the loop
        # poisson likelihood at each location assuming independent poisson neurons,
        # spike counts exist in this window, with window duration being decoding_time_window
        # lambda_prod = fields_for_spikes.prod(axis=2)
        # decoded_prod = lambda_prod * np.exp(-decoding_time_window * sum_fields)
        # total_prod = decoded_prod.sum() # normalization constant
        # if total_prod > 0:
        #     decoded_prod = decoded_prod / total_prod
        
        log_lambda = np.log(fields_for_spikes).sum(axis=2)        # log prod
        log_post   = log_lambda - decoding_time_window * sum_fields
        log_post  -= log_post.max()                                # numerical stability
        decoded_prod = np.exp(log_post)
        decoded_prod /= decoded_prod.sum() if decoded_prod.sum() > 0 else 1
            
        #Intuitively, the decoded matrix could be ahead or behind the rat 
        # so we shift the decoded matrix to match this 
        
        # Rat position in bin coordinates of the current window (decoding_window_index[i])
        #This is it's physical location in cm --> converted to bin index with the /bin_size
        X = decoding_time_info[win_pos, 0] / bin_size
        Y = decoding_time_info[win_pos, 1] / bin_size
        # Round to nearest bin
        X = int(np.round(X))
        Y = int(np.round(Y))
        

        
        # Now matlab uses imtranslate to shift the decoded matrix which I think is essentially this
        # Compute shifts
        # + shift_x and shift_y ==> move right and up
        # - shift_x and shift_y ==> move left and down
        #These are coordinates relative to the rat's physical location
        #This is basically where we want the rat to be after the shift
        
        #remember nx, ny are number of spatial bins for x and y
        #SO this finds the middle of the canvas and compares with rat's position
        # Mid_X = int(round(nx / 2.0))
        # Mid_Y = int(round(ny / 2.0))
        shift_x = Mid_X - X
        shift_y = Mid_Y - Y
        
        # if i == 5:
        #     print(f"win {i}, raw x,y (cm):", decoding_time_info[win_pos, 0:2])
        #     print(f"   bins X,Y:", X, Y, "shifts:", shift_x, shift_y)

        #Steps after defining the shift
        #1. Translation
        translation = translate_decoding(decoded_prod, shift_y, shift_x)
        

        #2. Rotation
        #Define angle of movement
        angle_movement = decoding_time_info[win_pos, 4]
        #rotates with head direction
        rotated_image = ndimage.rotate(translation,-(angle_movement + 180.0), #Why +180?
                                        order=1)
        rotate_minus_translation = ndimage.rotate(decoded_prod,-(angle_movement + 180.0),
                                        order=1) # no translation

        #3. Resizing rotated + translated image
        rotated_image = resize_rows_to_decoded_size(rotated_image, decoded_data_size)
        rotate_minus_translation = resize_rows_to_decoded_size(rotate_minus_translation, decoded_data_size)
        # if i == 2:
        #     print("Angle movement:", angle_movement)
        #     print("translation shape:", translation.shape)
        #     print("rotated_image shape (after resize):", rotated_image.shape)
            
        #4. Summing rotated +- translated image
        # Sum across direction perpendicular to rat's movement direction
        rotated_1d = rotated_image.sum(axis=1)
        rotate_minus_translation = rotate_minus_translation.sum(axis=1)
        
        #5. Saving
        not_translated_decoded_data[:,i] = rotate_minus_translation
        decoded_data[:, i] = rotated_1d
        #Why do we do both of these things? I am unsure,  this is probably for linear part
        #from unrotated decoding
        
        #6. Linear decoding --> no translation or rotation? 
        decoded_x_data[:, i] = decoded_prod.sum(axis=0)  # sum over rows -> X axis
        decoded_y_data[:, i] = decoded_prod.sum(axis=1)  # sum over cols -> Y axis
        
        #7. Max position --> for rotated data
        if use_max_pp:
            if rotated_1d.max() > 0:
                max_position = int(rotated_1d.argmax())
                max_position_2 = int(rotate_minus_translation.argmax())   
            else:
                max_position = 0
                max_position_2 = 0
        else:
            if rotated_1d.max() > 0:
                positions    = np.arange(len(rotated_1d))
                max_position = float((positions * rotated_1d).sum() / rotated_1d.sum())
                if rotate_minus_translation.max() > 0:
                    max_position_2 = float((positions * rotate_minus_translation).sum()
                                            / rotate_minus_translation.sum())
                else:
                    max_position_2 = 0
            else:
                max_position   = 0
                max_position_2 = 0
        
        #8. Saving
        decoded_sequence[0, i] = max_position
        untranslated_decoded_sequence[0, i] = max_position_2
        
        #For now I don't think I need this becasue it's making a reference frame of the rat relative to world view?  
        # IDK go ask Brad
        #So this recenters posterior and finds max x and y from that window individually i guess
        # I translated and copied this from brad's code but am not sure i need it
        
        # #9. Centering translated data
        # translated_image = resize_translated_to_decoded_size(translation.copy(), decoded_data_size)
        # no_translation_image = resize_translated_to_decoded_size(decoded_prod.copy(), decoded_data_size)
        # # X/Y marginals in the translated, centered window
        
        # # 10. Summing translated data without rotations? 
        # x_image = translated_image.sum(axis=0)  # sum over rows -> X axis
        # y_image = translated_image.sum(axis=1)  # sum over cols -> Y axis
        
        # ximage_notranslation = no_translation_image.sum(axis=0)  # sum over rows -> X axis
        # yimage_notranslation = no_translation_image.sum(axis=1)  # sum over cols -> Y axis

        # if use_max_pp:
        #     if translated_image.max() > 0:
        #         max_x_position = int(x_image.argmax()) 
        #         max_y_position = int(y_image.argmax())
        #     else:
        #         max_x_position = 0
        #         max_y_position = 0
        # else:
        #     if translated_image.max() > 0:
        #         xs = np.arange(len(x_image))
        #         ys = np.arange(len(y_image))
        #         max_x_position = float((xs * x_image).sum() / x_image.sum())
        #         max_y_position = float((ys * y_image).sum() / y_image.sum())
        #     else:
        #         max_x_position = 0
        #         max_y_position = 0

        # decoded_sequence[0, i] = max_position      # from rotated_1d
        # decoded_sequence[1, i] = max_x_position    # from translated window, X
        # decoded_sequence[2, i] = max_y_position    # from translated window, Y
    #What is the difference between decoded sequence and decoded data and which do we use for graphing purposes? 
        
    return {
    #"decoded_sequence": decoded_sequence,   # shape (3, n_valid_windows)
    "decoded_data": decoded_data,           # shape (decoded_data_size, n_valid_windows)
    "decoded_x_data": decoded_x_data,       # shape (nx, n_valid_windows)
    "decoded_y_data": decoded_y_data,       # shape (ny, n_valid_windows)
    "not_translated_decoded_data": not_translated_decoded_data, # shape (decoded_data_size, n_valid_windows)
    "use_max_pp": use_max_pp,
    "decoding_time_window": decoding_time_window,
    "decoding_time_advance": decoding_time_advance,
    "decoding_window_index": decoding_window_index}
    
    
# put this in theta_sequence_decoding.py alongside decode_theta_sequences

def decode_unimodal_vs_bimodal(predecode, field_results, uni_cells, bi_cells,
                                initial_variables):
    """
    Decode each valid window using unimodal cells and bimodal cells SEPARATELY,
    then jointly normalize the two posteriors so that P_uni + P_bi integrates
    to 1 over space. This preserves the relative contribution of each cell
    population to representing position in each window.

    Mirrors MATLAB IRFS_DECODE_THETA_SEQUENCES_WITH_UNIMODAL_VS_BIMODAL_CELLS.

    Key differences from decode_theta_sequences:
      - exp(-T * sum_fields) uses only the modality of interest (not all cells)
      - posteriors are JOINTLY normalized (sum of both = 1 over space)
      - returns two posterior arrays per window (uni and bi)
      - every valid window is processed (regardless of which modality spiked,
        matching MATLAB which iterates over the full Decoding_Window_Index)
    """
    decoding_time_window = initial_variables['decoding_time_window']
    bin_size             = initial_variables['bin_size']

    DTI                        = np.asarray(predecode['DTI'])
    decoding_window_index      = np.asarray(predecode['decoding_window_index'])
    decoding_spike_index       = predecode['decoding_spike_index']
    decoding_spike_index_valid = decoding_spike_index[:, decoding_window_index]
    n_valid = len(decoding_window_index)

    Field_Data = field_results['Field_Data']
    cell_ids   = np.asarray(field_results['cell_ids'])
    ny, nx, n_cells = Field_Data.shape
    cell_id_idx = {cid: idx for idx, cid in enumerate(cell_ids)}

    # zero -> min/10 (for the product term)
    fields_modified = Field_Data.copy()
    for i in range(n_cells):
        f = fields_modified[:, :, i]
        pos = f[f > 0]
        if pos.size > 0:
            m = pos.min()
            eps = m / 10 if m / 10 > 0 else m
            f[f <= 0] = eps
        else:
            f[:] = 1.0
        fields_modified[:, :, i] = f

    # ---- per-modality sum_fields (uses ORIGINAL Field_Data, not modified) ----
    uni_set = set(uni_cells)
    bi_set  = set(bi_cells)
    uni_idx_in_fields = [cell_id_idx[c] for c in uni_cells if c in cell_id_idx]
    bi_idx_in_fields  = [cell_id_idx[c] for c in bi_cells  if c in cell_id_idx]

    sum_uni_fields = (Field_Data[:, :, uni_idx_in_fields].sum(axis=2)
                      if uni_idx_in_fields else np.zeros((ny, nx)))
    sum_bi_fields  = (Field_Data[:, :, bi_idx_in_fields].sum(axis=2)
                      if bi_idx_in_fields  else np.zeros((ny, nx)))

    # Precompute the exp terms (constant across windows)
    exp_uni = np.exp(-decoding_time_window * sum_uni_fields)
    exp_bi  = np.exp(-decoding_time_window * sum_bi_fields)

    # Canvas
    Mid_X = int(round(nx / 2.0))
    Mid_Y = int(round(ny / 2.0))
    decoded_data_size = (max(Mid_X, Mid_Y) * 2) + 1

    decoded_data_uni = np.zeros((decoded_data_size, n_valid))
    decoded_data_bi  = np.zeros((decoded_data_size, n_valid))

    for i in range(n_valid):
        win_pos = decoding_window_index[i]

        # split spikes by modality
        all_spikes = decoding_spike_index_valid[:, i]
        all_spikes = all_spikes[all_spikes > 0]
        uni_spike_idx = [cell_id_idx[c] for c in all_spikes
                         if c in uni_set and c in cell_id_idx]
        bi_spike_idx  = [cell_id_idx[c] for c in all_spikes
                         if c in bi_set  and c in cell_id_idx]

        # product terms (empty subset -> prod = 1, so result = exp term alone)
        if len(uni_spike_idx) > 0:
            uni_prod = fields_modified[:, :, uni_spike_idx].prod(axis=2)
        else:
            uni_prod = np.ones((ny, nx))
        if len(bi_spike_idx) > 0:
            bi_prod = fields_modified[:, :, bi_spike_idx].prod(axis=2)
        else:
            bi_prod = np.ones((ny, nx))

        uni_decoded = uni_prod * exp_uni
        bi_decoded  = bi_prod  * exp_bi

        # JOINT normalization — preserves relative population contribution
        joint_sum = uni_decoded.sum() + bi_decoded.sum()
        if joint_sum > 0:
            uni_decoded = uni_decoded / joint_sum
            bi_decoded  = bi_decoded  / joint_sum

        # Translate to rat position
        X = int(np.round(DTI[win_pos, 0] / bin_size))
        Y = int(np.round(DTI[win_pos, 1] / bin_size))
        shift_x = Mid_X - X
        shift_y = Mid_Y - Y
        uni_translated = translate_decoding(uni_decoded, shift_y, shift_x)
        bi_translated  = translate_decoding(bi_decoded,  shift_y, shift_x)

        # Rotate to movement direction
        angle_movement = DTI[win_pos, 4]
        uni_rotated = ndimage.rotate(uni_translated, -(angle_movement + 180.0),
                                      order=1)
        bi_rotated  = ndimage.rotate(bi_translated,  -(angle_movement + 180.0),
                                      order=1)

        # Resize and collapse perpendicular to movement
        uni_rotated = resize_rows_to_decoded_size(uni_rotated, decoded_data_size)
        bi_rotated  = resize_rows_to_decoded_size(bi_rotated,  decoded_data_size)
        decoded_data_uni[:, i] = uni_rotated.sum(axis=1)
        decoded_data_bi[:,  i] = bi_rotated.sum(axis=1)

    return {
        'decoded_data_uni':      decoded_data_uni,
        'decoded_data_bi':       decoded_data_bi,
        'decoding_window_index': decoding_window_index,
        'decoding_time_window':  decoding_time_window,
        'decoding_time_advance': initial_variables.get('decoding_time_advance'),
        'n_uni_cells':           len(uni_cells),
        'n_bi_cells':            len(bi_cells),
        'uni_cell_ids':          uni_cells,
        'bi_cell_ids':           bi_cells,
    }        
        
        
        

