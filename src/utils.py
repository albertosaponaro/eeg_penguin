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
    
    # ICA
    if perform_ica: 
        raw = ica(raw)

    # Remove trials if amplitude exceeds a certain threshold
    if reject_amp: 
        epochs.drop_bad(reject={'eeg': reject_amp})

    return raw, epochs


def ica(raw, n_componets=40, debug_logs=False, debug_images=False):
    """
    Perform Independent Component Analysis (ICA) on raw EEG data to remove artifacts.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data.
    n_components : int, optional
        Number of components to decompose the data into. Default is 40.
    debug_logs : bool, optional
        If True, print debugging logs. Default is False.
    debug_images : bool, optional
        If True, display debug images (e.g., component plots). Default is False.

    Returns:
    --------
    raw : mne.io.Raw
        The raw EEG data after ICA artifact removal.
    """
    
    # Initialize ICA and fit it to the data
    # ICA is stochastic, here we opt for the random seed 2
    ica = mne.preprocessing.ICA(method = 'picard', n_components = n_componets, random_state = 2, verbose = debug_logs)
    ica.fit(raw, verbose = debug_logs)
    
    # Sanity check
    if debug_images: ica.plot_components()

    # Find bad components
    componnets1, scores1 = ica.find_bads_eog(raw, 'AFz', verbose = debug_logs)
    componnets2, scores2 = ica.find_bads_muscle(raw, verbose = debug_logs)
    
    # Sanity check
    if debug_images:
        ica.plot_scores(scores1, componnets1)
        ica.plot_scores(scores2, componnets2)

    # Remove bad components
    raw = ica.apply(raw, exclude = componnets1 + componnets2, verbose = debug_logs)
    

    return raw


def perform_tfr(epochs):
    """
    Perform time-frequency analysis on epoched data using Morlet wavelets.

    Parameters:
    epochs (mne.Epochs): The epoched data.

    Returns:
    dict: A dictionary containing time-frequency representations for each event type.

    """
    # Define frequency range
    freqs = np.logspace(np.log10(5), np.log10(20), num=30)

    # Define number of cycles per frequency
    n_cycles = freqs / 2.0  # Use a more standard number of cycles

    # Perform time-frequency analysis using Morlet wavelets
    power_regular = mne.time_frequency.tfr_multitaper(
        epochs['regular'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False)

    power_random = mne.time_frequency.tfr_multitaper(
        epochs['random'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False)
    
    return {'regular': power_regular, 'random': power_random}



