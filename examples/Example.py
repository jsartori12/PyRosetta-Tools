#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:57:52 2026

@author: joao
"""

import utils_pyrosetta



pdbace2rbd = "RBD_ACE2.pdb_relax.pdb"

by_term_by_residue = utils_pyrosetta.Energy_contribution(pdb = pdbace2rbd, by_term=True)
sumofterms_by_residue = utils_pyrosetta.Energy_contribution(pdb = pdbace2rbd, by_term=False)


Interface_metrics = utils_pyrosetta.Get_Interface_descriptors(pdb = pdbace2rbd, partner1 = "A", partner2 = "D", minimize=False)


df_dms = utils_pyrosetta.Run_DMS_Parallel(
    pdb=pdbace2rbd,
    positions=[300,311,312,313],
    n_cpu=4,
    chain = "D",
    save_structures=True,          # dump PDB for every mutant
    output_dir="./DMS_results",
    fast_relax_repeats=0,          # apply 1 FastRelax round per mutant
)

utils_pyrosetta.plot_dms_heatmap(df_dms)


