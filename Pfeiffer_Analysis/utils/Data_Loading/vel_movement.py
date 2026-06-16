import numpy as np
from scipy.signal import butter, filtfilt

def calculate_velocity(position_tsd):
    """
    So I use Brad's operation for calculating velocity for now instead of centered from Zhu et al for now
    Brad uses the forward differences then applies a butterworth filter to smooth
    This removes outliers by clipping large time jumps to median
    However, centered differences is better for edges.  
    For now we will stick with Brad's version
    """
    time = position_tsd.index
    dt = np.diff(time)
    dx = np.diff(position_tsd['x'].values)
    #Length N-1
    dy = np.diff(position_tsd['y'].values) 
    #Length N (repeats last time point)
    dt = np.concatenate([dt, [dt[-1]]]) 
    dx = np.concatenate([dx, [dx[-1]]])
    dy = np.concatenate([dy, [dy[-1]]])
    
    # Filter time
    # This is to make sure gaps larger than 10 seconds 
    # do not artifically induce low speeds
    median_dt = np.median(dt)
    dt[dt > 10 * median_dt] = median_dt
    #This filtering removes jittery jumps from the camera 
    # obscuring the LED sometimes
    b, a = butter(2, 0.02)
    dt_filtered = filtfilt(b, a, dt)
    dt_filtered[dt_filtered <= 0] = np.min(dt_filtered[dt_filtered > 0]) / 10
    
    # Filter position changes - same reasoning
    b, a = butter(2, 0.2)
    dx_filtered = filtfilt(b, a, dx)
    dy_filtered = filtfilt(b, a, dy)
    
    # Calculate velocity
    distance = np.sqrt(dx_filtered**2 + dy_filtered**2)
    velocity = np.abs(distance / dt_filtered)
    velocity[velocity < 0] = 0
    
    return velocity, dt_filtered, distance


def calculate_movement_direction(x_pos, y_pos):
    """
    Calculate movement direction in degrees (0-360)
    This has been added from Brad's code but left and right only matter when in open field
    - 0° = moving down
    - 90° = moving left  
    - 180° = moving up
    - 270° = moving right
    """
    n_points = len(x_pos)
    angles = np.zeros(n_points)
    
    for i in range(1, n_points):
        x_diff = x_pos[i] - x_pos[i-1]
        y_diff = y_pos[i] - y_pos[i-1]
        
        # Calculate angle based on quadrant
        # This exactly matches the MATLAB logic
        if y_diff == 0 and x_diff > 0:
            angle = 270
        elif y_diff == 0 and x_diff < 0:
            angle = 90
        elif x_diff == 0 and y_diff > 0:
            angle = 180
        elif x_diff == 0 and y_diff < 0:
            angle = 360
        elif x_diff > 0 and y_diff > 0:
            angle = 180 + np.rad2deg(np.arctan(np.abs(x_diff) / np.abs(y_diff)))
        elif x_diff > 0 and y_diff < 0:
            angle = 270 + np.abs(np.rad2deg(np.arctan(np.abs(y_diff) / np.abs(x_diff))))
        elif x_diff < 0 and y_diff > 0:
            angle = 90 + np.abs(np.rad2deg(np.arctan(np.abs(y_diff) / np.abs(x_diff))))
        elif x_diff < 0 and y_diff < 0:
            angle = np.rad2deg(np.arctan(np.abs(x_diff) / np.abs(y_diff)))
        else:
            angle = 0
        
        # Ensure angle is in [0, 360]
        if angle > 360:
            angle = angle - 360
        
        angles[i] = angle
    
    # Set first angle to match second
    angles[0] = angles[1]
    
    return angles