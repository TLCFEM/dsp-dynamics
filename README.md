# Spurious Response Due to Linear Interpolation of Input Load and Some Remedies

This repository contains the source code and example models of paper [0.1080/13632469.2024.2372814](https://doi.org/10.1080/13632469.2024.2372814).

To cite or reproduce figures in the paper, you can find the corresponding figure and copy the source code in your work. Please check CI/CD workflow to see how to generate figures used in the paper.

The numerical examples used in the paper are developed in `suanPan`. To perform the numerical analysis, one can download and install [`suanPan`](https://github.com/TLCFEM/suanPan). Then run the model via, for example, the following command in the corresponding
folders under the corresponding folders in `MODEL`.

```sh
suanpan -f Sine.supan
```

To generate the figures used in the paper, one can create a Python environment and then execute the scripts.

```bash
# create a virtual environment
pip install virtualenv
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install necessary packages
pip install -r requirements.txt

mkdir PIC
cd DFT

python DampingForce.py
python Deformation.py
python FrameResult.py
python FundamentalSolution.py
python InertialForce.py
python Newmark.py
python Nuttall.py
python PureSine.py
python SDOF.py
```
