
# Contributing to Machine Learning for Perovskite Materials

## Setting Up and Testing in Visual Studio Code (VS Code)

### 1. Clone the Repository
1. Open a terminal and run:
   ```bash
   git clone https://github.com/username/machine-learning-materials-science.git
   cd machine-learning-materials-science
   ```

### 2. Install Dependencies
Ensure you have Python 3.8 or higher installed. Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```
Alternatively, set up a Conda environment:
```bash
conda create -n ml-materials python=3.8
conda activate ml-materials
pip install -r requirements.txt
```

### 3. Open the Project in VS Code
1. Launch Visual Studio Code.
2. Open the cloned project directory: `File > Open Folder`.

### 4. Install the Python Extension
1. Go to the Extensions view (`Ctrl+Shift+X` on Windows/Linux or `Cmd+Shift+X` on macOS).
2. Search for "Python" and install the official Python extension.

### 5. Select the Python Interpreter
1. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS) to open the Command Palette.
2. Type `Python: Select Interpreter` and choose the environment where you installed the dependencies (e.g., `ml-materials`).

### 6. Run the Project
To run the project:
1. Open `main.py`.
2. Press `F5` to start debugging or right-click and select "Run Python File in Terminal".

### 7. Run Jupyter Notebooks
1. Open any notebook in the `notebooks/` folder.
2. Ensure the kernel is set to the Python environment you configured earlier.
3. Run the cells interactively.

### 8. Run Tests
1. Open a terminal in VS Code.
2. Run the following command to execute all tests:
   ```bash
   pytest tests/
   ```

## Contribution Guidelines
If you'd like to contribute to this project, please follow these steps:
1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear commit messages.
4. Push the changes to your forked repository.
5. Open a pull request on the main repository.

Thank you for contributing!
