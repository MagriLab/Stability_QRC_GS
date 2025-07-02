# Stability RF_QRC with Generalised Synchronisation

This repository provides a framework for studying the stability of chaotic dynamical systems using **Quantum Reservoir Computing (QRC) and Recurrence-free Quantum reservoir computing (RF-QRC)** using **Generalised Synchronisation (GS) theory**. It includes tools for computing **Lyapunov exponents** via a Qiskit-based implementation and for demonstrating the flexibility of the framework to handle various systems such as the Lorenz63 and Lorenz96 models.

## Repository Structure

- **`src/Notebook.ipynb`**  
  Contains the main workflow to evaluate Lyapunov exponents using a trained quantum reservoir computer. Inputs can be modified to extend the framework to other systems such as **Lorenz96**, and to vary quantum noise models and the number of measurement **shots**.

- **`src/QRC/qrc.py`**  
  Implements the **Qiskit-based quantum reservoir** and the methods required to compute **Lyapunov exponents**, including both conditional and autonomous versions, using quantum circuit simulations.

## Usage

To get started, simply run:
First, install the required dependencies:
```bash
pip install -r requirements.txt
```
Then run the following notebook
```bash

python Notebook.ipynb
```

## Citation
If you use this code in your research, please cite the corresponding paper:

Robust quantum reservoir computers for forecasting chaotic dynamics: generalized synchronization and stability (https://arxiv.org/abs/2506.22335)


