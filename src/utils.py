import mne
import numpy as np
from matplotlib import pyplot as plt
from mne_bids import read_raw_bids

def load_subject(given_path):
    """
    Load subject data from the given path, preprocess it, and return the processed data.

    Parameters:
    given_path (str): The path to the data in BIDS format.

    Returns:
    mne.io.Raw: The preprocessed raw data object.

    """
    # Read raw BIDS data
    raw = read_raw_bids(bids_path=given_path)
    
    # Load the raw data
    raw.load_data()

    # Exclude MEG channels known to cause ICA plotting issues
    baddies = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
    raw.drop_channels(baddies)

    # Set up electrode montage to standard 10-20 system
    raw.set_montage('standard_1020', match_case=False, on_missing='warn')
    
    return raw



def pipeline(raw, tmin=-1, tmax=1, baseline=(-0.2, 0.05), reject_amp=None, perform_ica=True):
    """
    Apply a pipeline of preprocessing steps to raw EEG data.

    Parameters:
    raw (mne.io.Raw): The raw EEG data.
    tmin (float): Start time of the epoch in seconds.
    tmax (float): End time of the epoch in seconds.
    baseline (tuple): The time interval to apply baseline correction.
    reject_amp (float or None): Threshold for trial rejection based on amplitude.
    perform_ica (bool): Whether to perform Independent Component Analysis (ICA).

    Returns:
    mne.io.Raw: The preprocessed raw data.
    mne.Epochs: The epoched data.

    """
    # EEG trace re-referenced to scalp average
    raw = raw.set_eeg_reference(ref_channels="average")

    # Set low/high-pass filters (l_freq = high-pass / h_freq = low-pass)
    raw = raw.filter(l_freq=1, h_freq=25, fir_design='firwin')

    # Downsample
    raw = raw.resample(sfreq=128)

    # Set events for epoching
    events = mne.find_events(raw, initial_event=True)
    event_dict = {'regular': 1, 'random': 3}

    # Epoching, prepare for getting evoked responses
    epochs = mne.Epochs(
        raw,
        events, event_dict,
        tmin=tmin, 
        tmax=tmax,
        baseline=baseline
    )

    # Number of ICA components based on raw data rank
    rank = mne.compute_rank(raw)['eeg']
    
    # Initialize and fit ICA
    # DOC: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
    if perform_ica: 
        ica = mne.preprocessing.ICA(method="picard", n_components=rank)
        ica.fit(raw, verbose=True)
        raw = ica.apply(raw)

    # Remove trials if amplitude exceeds a certain threshold
    if reject_amp: 
        epochs.drop_bad(reject={'eeg': reject_amp})

    return raw, epochs




def perform_tfr(epochs):
    """
    Perform time-frequency analysis on epoched data using Morlet wavelets.

    Parameters:
    epochs (mne.Epochs): The epoched data.

    Returns:
    dict: A dictionary containing time-frequency representations for each event type.

    """
    # Define frequency range
    freqs = np.logspace(np.log10(5), np.log10(20), num=20)

    # Define number of cycles per frequency
    n_cycles = freqs / 2.0  # Use a more standard number of cycles

    # Perform time-frequency analysis using Morlet wavelets
    power_regular = mne.time_frequency.tfr_multitaper(
        epochs['regular'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False, decim=3, n_jobs=-1)

    power_random = mne.time_frequency.tfr_multitaper(
        epochs['random'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False, decim=3, n_jobs=-1)
    
    return {'regular': power_regular, 'random': power_random}



