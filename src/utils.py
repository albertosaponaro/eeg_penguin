import mne
from matplotlib import pyplot as plt
from mne_bids import read_raw_bids

def load_subject(given_path):
    raw = read_raw_bids(bids_path = given_path)
    raw.load_data()

    # Exclude MEG channels (probably these are the ones, also result in ICA plotting issues)
    baddies = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
    raw.drop_channels(baddies)

    # Set up montage
    raw.set_montage('standard_1020', match_case = False, on_missing = 'warn')
    #raw = mne.pick_types(raw.info, meg=False, eeg=True, exclude=['bads'])
    #interpolating bads throws error?
    
    # OK Check: 1 channel and psd
    #plt.plot(raw[0,:][0].T)
    #plt.show()
    #raw.compute_psd().plot()
    #plt.show()
    
    return raw

def pipeline(raw, tmin=-1, tmax=1, baseline=(-0.2, 0.05), reject_amp=100e-6):
    
    # EEG trace re-referenced to scalp average
    raw = raw.set_eeg_reference(ref_channels = "average")

    # Set low/high-pass filters (l_freq = high-pass / h_freq = low-pass)
    raw = raw.filter(l_freq = 1, h_freq = 25, fir_design = 'firwin')

    # Downsample
    raw = raw.resample(sfreq=128)

    # Set events for epoching
    events = mne.find_events(raw, initial_event=True)
    event_dict = {'regular': 1, 'random': 3}

    # Epoching, prepare for getting evoked responses
    epochs = mne.Epochs(
        raw,
        events, event_dict,
        tmin = tmin, 
        tmax = tmax,
        baseline = baseline
    )
    
    # Initialize and fit ICA
    # DOC: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
    #ica = mne.preprocessing.ICA(method = "picard", n_components = 64)
    #ica.fit(raw, verbose = True)
    
    # Remove trials if amplitude exceeds ±μ100
    epochs.drop_bad(reject={'eeg': reject_amp})

    return raw, epochs
    
    # TODO: add code for single subj pipeline
