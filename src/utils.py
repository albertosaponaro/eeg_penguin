import mne
from matplotlib import pyplot as plt
from mne_bids import read_raw_bids

def load_subject(given_path):
    
    raw = read_raw_bids(bids_path = given_path)
    raw.load_data()

    # Exclude MEG channels (probably these are the ones, also result in ICA plotting issues)
    #raw.info['bads'] = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
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

def pipeline():
    # TODO: add code for single subj pipeline
    pass