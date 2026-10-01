---
layout: home
title: Home
nav_order: 1
---

# FieldTools

FieldTools calculates electric fields from MD trajectories, at selected atoms or projected onto selected bonds,
and decomposes them into the contributions of each protein residue and solvent species.

[Open the tutorial in Google Colab](https://colab.research.google.com/github/bunzela/FieldTools/blob/main/FieldTools.ipynb)
· [Source code and documentation](https://github.com/bunzela/FieldTools)

## Quick start

    pip install -r requirements.txt
    python utils/FieldTools.py -nc data/KPC.nc -parm data/KPC.parm7 \
                               -target data/field_target.dat -solvent WAT,Na+ \
                               -TIP4P True -out field.pkl

Run `python utils/FieldTools.py -h` for all options.
