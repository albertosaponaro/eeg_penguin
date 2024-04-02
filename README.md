# Penguin EEG Project

#### Symmetry Perception by Alberto Saponaro (3422656) and Georgi Panayotov (3667327)


## Project Structure

```
eeg_penguin
│
├── data
│   └── README.md (info about data)
│
├── notebooks 
│   ├── 01_Penguin_Sub-001_ERP.ipynb
│   ├── 02_Penguin_Sub-001_TFR.ipynb
│   ├── 03_Penguin_All-Sub_ERP.ipynb
│   ├── 04_Penguin_All-Sub_ERP.ipynb
│   └── 05_Penguin_Stat-Analysis.ipynb
│
├── plots
│
├── references
│   └── README.md (info about references)
│
├── src
│   ├── utils.py
│   └── plots.py 
│
├── .gitignore
├── README
└── requirements.txt

```

## How To Run

1. **Create a Virtual Environment**:

    Run the following command in the project folder to create a virtual environment.

    ```
    python -m venv .venv
    ```

2. **Activate the Virtual Environment**:

    Run the following command in the project folder to activate the virtual environment.

    - MacOS/Linux:
        ```
        source .venv/bin/activate
        ```
    - Windows:
        ```
        source .venv/bin/activate
        ```

3. **Install Requirements**:

    Once the virtual environment is activated, you can install the dependencies listed in the requirements.txt file using pip.

    ```
    pip install -r requirements.txt
    ```

4. **Run The Notebooks**:
    
    With the virtual environment activated and all dependencies installed, you can now run the jupyter notebooks.

    ```
    jupyter notebook
    ```

5. **Deactivate the Virtual Environment**:
    ```
    deactivate
    ```
