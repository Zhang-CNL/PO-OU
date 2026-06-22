
RAT_NAMES = ['Harpy', 'Imp', 'Janni', 'Naga']
PFEIFFER_ENV_WIDTH_CM  = 200
PFEIFFER_ENV_HEIGHT_CM = 200
PFEIFFER_PLACEFIELD_MIN_TUNE_SPIKES_PSEC = 1
PFEIFFER_RECORDING_FPS = 1 / 30

PFEIFFER_NOISY_EPOCHS = {
    'Janni': {
        "Linear2": {
            "starts": [18721, 23511],
            "ends"  : [22773, 29423]
        },
        "Linear3": {
            "starts": [11650, 16390],
            "ends"  : [15498, 20184]
        },
        "Open2": {
            "starts": [33756],
            "ends"  : [34007]
        }
    },
    "Harpy": {
        "Linear1": {
            "starts": [12850, 17880],
            "ends"  : [12956, 17929]
        },
        "Linear2": {
            "starts": [19307, 19476],
            "ends"  : [19322, 19489]
        },
        "Linear3": {
            "starts": [27025],
            "ends"  : [27035]
        },
        "Open1": {
            "starts": [27332],
            "ends"  : [27639]
        },
        "Open2": {
            "starts": [19528, 19582, 19701, 20802, 21607, 21690, 21701, 22141, 22258],
            "ends"  : [19539, 19592, 19722, 20815, 21621, 21696, 21702, 22180, 22265]
        }
    },
    "Imp": {
        "Linear1": {
            "starts": [25880, 30570, 33920],
            "ends"  : [29735, 33885, 33962]
        },
        "Open1": {
            "starts": [25160],
            "ends"  : [25275]
        },
        "Open2": {
            "starts": [20122, 20147],
            "ends"  : [20126, 20164]
        }       
    }
}