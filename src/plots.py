import mne

def plot_ERPs(epochs, title='Event-Related Potentials (ERPs)'):
    e1 = epochs['regular'].average()
    e2 = epochs['random'].average()
    evokeds = dict(regular = e1, random = e2)
    picks = [f"PO{n}" for n in range(7, 9)]

    return mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine="mean", title=title)


def plot_spectra(spectra, mode='mean'):
    
    # spectra: {'regular': power_regular, 'random': power_random}

    # IMPORTANT: change channels between PO3+PO7 for left and PO4+PO8 for right
    for title, power in spectra.items():
        power.plot_joint(picks=["PO3","PO7"], mode=mode, title=f"{title} + Left posterior: PO3 & PO7", baseline=(-0.2,0.05), timefreqs=[(0.5, 10), (1.3, 8)])
        power.plot_joint(picks=["PO4","PO8"], mode=mode, title=f"{title} + Right posterior: PO3 & PO7", baseline=(-0.2,0.05), timefreqs=[(0.5, 10), (1.3, 8)])

        #add these to the arguments for more topographic windows (seconds, herz)
        #timefreqs=[(0.5, 10), (1.3, 8)]