# Forward And Inverse Diffraction In Phasor Fields
This repository is the official implementation of [Forward and Inverse Diffraction in Phasor Fields](https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-5-11420&id=568850).

### Abstract
Non-line-of-sight (NLOS) imaging is an inverse problem that consists of reconstructing a hidden scene out of the direct line-of-sight given the time-resolved light scattered back by the hidden scene on a relay wall.  

This research focuses on the Phasor Fields technique which uses a forward diffraction operator to reconstruct hidden scenes from time-resolved scattered light. We investigate this seemingly counterintuitive approach, using a forward operator to solve an inverse problem, by drawing parallels between Phasor Fields and inverse diffraction methods from optics. We reach the following contributions:
1. We propose novel interpretations of the relay wall's function as either a phase conjugator or a hologram recorder, framing NLOS imaging as an inverse diffraction problem. 
2. We introduce "Inverse Phasor Fields", a new algorithm to reconstruct the hidden scene posing the NLOS imaging problem as an inverse diffraction problem.
3. We present a computational metric to assess the quality and limitations of NLOS reconstruction setups, relating it to the established Rayleigh criterion.

### Results

### Citation
```bibtex
@article{garciapueyo2025forwardinverse,
    title = {Forward and inverse diffraction in phasor fields},
    author = {Jorge Garcia-Pueyo and Adolfo Mu\~{n}oz},
    journal = {Opt. Express},
    volume = {33},
    number = {5},
    pages = {11420--11441},
    publisher = {Optica Publishing Group},
    month = {Mar},
    year = {2025},
    url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-33-5-11420},
    doi = {10.1364/OE.553755},
}
```

## Replicating the results: Getting Started
The recommended way to run the scripts to replicate the results is:
1. Create a python virtual environment to install the dependencies.
2. Download the data of NLOS captures (real and simulated)
3. Run the scripts to reconstruct the hidden scenes (real and simulated) using Phasor Fields and our method **Inverse Phasor Fields**.

#### 1. Create virtual environment and install the dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Download the data of NLOS captures (real and simulated)
We test our reconstruction algorithms on:
- Real data from [Phasor Field Diffraction Based Reconstruction for Fast Non-Line-of-Sight Imaging Systems](https://biostat.wisc.edu/~compoptics/phasornlos20/fastnlos.html): it can be downladed from this [link](https://biostat.wisc.edu/~compoptics/phasornlos20/archive/tdata.zip).
- Simulated data using [*mitransient*](https://github.com/diegoroyo/mitransient) and [*y-tal*](https://github.com/diegoroyo/tal/): it can be downloaded from this [link](https://nas-graphics.unizar.es/s/ifg3iN3b3qSLNao). You can also simulate new scenes following steps in [How to simulate new scenes](#4-how-to-simulate-new-scenes).

The commands to download the data are:
1. Create a `data/` folder. and move the downloaded data there:
```bash
mkdir data/
```
2. Download the real data:
```bash
wget -O data/pfdiffraction_fastnlos.zip https://biostat.wisc.edu/~compoptics/phasornlos20/archive/tdata.zip
unzip data/pfdiffraction_fastnlos.zip -d data/pfdiffraction_fastnlos
```
3. Download the simulated data:
```bash
wget -O data/ForwardInverseDiffractionInPhasorFields_SimulatedData.zip https://nas-graphics.unizar.es/s/ifg3iN3b3qSLNao/download/ForwardInverseDiffractionInPhasorFields_SimulatedData.zip
unzip data/ForwardInverseDiffractionInPhasorFields_SimulatedData.zip -d data/simulated
```

#### 3. Run the scripts



#### 4. How to simulate new scenes