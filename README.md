🧪 Installation instructions 
1. Download Miniconda. See: https://www.youtube.com/watch?v=EBbcsjBSEi8
2. Open Anaconda Prompt
3. cd into the local repo location
4. Create the venv using: conda env create -f environment.yaml
5. Activate the venv: conda activate smb-env
6. Set the notebook kernel: python -m ipykernel install --user --name smb-env --display-name "SMB Workshop"
7. (a) Launch the notebook in a browser window: jupyter lab or
   (b) Launch the notebook in VS Code: code .
