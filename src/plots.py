import os
import mne
import matplotlib.pyplot as plt

def plot_ERPs(epochs, title='Event-Related Potentials (ERPs)', save_path=None):
    """
    Plot the Event-Related Potentials (ERPs) from the given epochs data.

    Parameters:
    - epochs (dict): A dictionary containing epochs data for regular and random events.
    - title (str): The title for the plot. Defaults to 'Event-Related Potentials (ERPs)'.
    - save_path (str): The path to save the plot as a PNG file. If None, the plot will not be saved. Defaults to None.
    """

    # Compute average evoked responses for regular and random events
    e1 = epochs['regular'].average()
    e2 = epochs['random'].average()

    # Create a dictionary of evoked responses
    evokeds = dict(regular=e1, random=e2)

    # Define the channels to plot (PO7 and PO8)
    picks = [f"PO{n}" for n in range(7, 9)]

    # Plot the average evoked responses
    fig = mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine="mean", title=title)[0]

    # Save the plot if save_path is provided
    if save_path:
        fig.savefig(save_path)
        print(f"Plot saved as {save_path}")



def plot_spectra(spectra, mode='mean', title="TFR", save_path=None):
    """
    Plot spectra from the given dictionary and save the plot as a PNG file.

    Parameters:
    - spectra (dict): A dictionary containing experiment type as keys and power spectra objects as values.
    - mode (str): The mode of plotting. Defaults to 'mean'.
    - save_path (str): The path to save the plot. If None, the plot will not be saved. Defaults to None.
    """
    figs = []

    # Iterate over each stimuli and power in spectra
    for i, (stimuli, power) in enumerate(spectra.items()):
        # Plot joint plot for PO3 and PO7
        figs.append(power.plot_joint(picks=["PO3", "PO7"], mode=mode,
                        title=f"{stimuli} + reft posterior: PO3 & PO7",
                        baseline=(-0.2, 0.05), timefreqs=[(0.5, 10), (1.3, 8)]))
        
        # Plot joint plot for PO4 and PO8
        figs.append(power.plot_joint(picks=["PO4", "PO8"], mode=mode,
                        title=f"{stimuli} + right posterior: PO4 & PO8",
                        baseline=(-0.2, 0.05), timefreqs=[(0.5, 10), (1.3, 8)]))

    # Create a 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Set title for the main figure
    fig.suptitle(title)

    # Iterate over each stimuli and power in spectra
    for i, (stimuli, power) in enumerate(spectra.items()):
        # Plot joint plot for PO3 and PO7
        axes[i, 0].imshow(figs[i * 2].canvas.buffer_rgba(), aspect='auto', origin='upper')
        axes[i, 0].set_title(f"{stimuli} + Left posterior: PO3 & PO7")
        axes[i, 0].axis('off')

        # Plot joint plot for PO4 and PO8
        axes[i, 1].imshow(figs[i * 2 + 1].canvas.buffer_rgba(), aspect='auto', origin='upper')
        axes[i, 1].set_title(f"{stimuli} + Right posterior: PO4 & PO8")
        axes[i, 1].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()

import matplotlib.pyplot as plt

def plot_topomaps(evoked, times=[0.52, 2.0], title='Topomaps', save_path=None):
    """
    Plot topographic maps for regular and random events.

    Parameters:
    - evoked (dict): A dictionary containing evoked objects for regular and random events.
    - times (list): Time points at which topomaps are plotted. Defaults to [0.52, 2.0].
    - title (str): Title for the main figure. Defaults to 'Topomaps'.
    - save_path (str): Path to save the plot as a PNG file. If None, the plot will not be saved. Defaults to None.
    """
    # Plot topomaps for regular and random events
    fig_regular = evoked['regular'].plot_topomap(times=times, ch_type='eeg');
    fig_random = evoked['random'].plot_topomap(times=times, ch_type='eeg');
    
    # Create a 2x1 plot
    fig, axes = plt.subplots(2, 1, figsize=(4, 5))

    # Set title for the main figure
    fig.suptitle(title)

    # Plot regular event topomap
    axes[0].imshow(fig_regular.canvas.buffer_rgba(), aspect='auto', origin='upper')
    axes[0].set_title(f"Regular Event")
    axes[0].axis('off')

    # Plot random event topomap
    axes[1].imshow(fig_random.canvas.buffer_rgba(), aspect='auto', origin='upper')
    axes[1].set_title(f"Random Event")
    axes[1].axis('off')
    
    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()
