# Stability RF_QRC with Generalised Synchronisation

This repository provides a robust framework for studying the stability of chaotic dynamical systems using **Quantum Reservoir Computing (QRC)** under **Generalised Synchronisation (GS)**. It includes tools for computing **Lyapunov exponents** via a Qiskit-based implementation and for demonstrating the flexibility of the framework to handle various systems such as the Lorenz63 and Lorenz96 models.

## 📁 Repository Structure

- **`Notebook.py`**  
  Contains the main workflow to evaluate Lyapunov exponents using a trained quantum reservoir computer. Inputs can be modified to extend the framework to other systems such as **Lorenz96**, and to vary quantum noise models and the number of measurement **shots**.

- **`QRC.py`**  
  Implements the **Qiskit-based quantum reservoir** and the methods required to compute **Lyapunov exponents**, including both conditional and autonomous versions, using quantum circuit simulations.

## 🔧 Features

- Robust training of quantum reservoirs via generalised synchronisation.
- Computation of conditional and autonomous Lyapunov exponents.
- Flexible configuration for different dynamical systems.
- Support for noisy quantum simulations and finite shot statistics (NISQ-compatible).

## 🚀 Usage

To get started, simply run:
```bash
python Notebook.py
