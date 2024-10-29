from pyscf_util.misc.mole import get_mol
from pyscf_util.iCIPT2.iCIPT2_4C_d2h import kernel

import os

BASIS = ["unc-ccpvdz-dk", "unc-ccpvtz-dk", "unc-ccpvqz-dk", "unc-ccpv5z-dk"]
# CMIN = [1e-4, 5e-5, 3e-5, 1.5e-5, 9e-6, 5e-6]
CMIN = "1e-4 5e-5 3e-5 1.5e-5 9e-6 5e-6"
# BASIS = ["ccpvdz-dk"]  # TEST
# CMIN = [1e-3]
# CMIN = "1e-3"

# FCIDUMP_NAME = "FCIDUMP_%s"
# RELDUMP_NAME = "RELDUMP_%s"
FCIDUMP_4C_NAME = "FCIDUMP_4C_%s"


for basis in BASIS:

    # SOICI and ICIPT2 #

    Cl_atm = get_mol("Cl 0 0 0", 0, 1, basis=basis, symmetry="d2h", verbose=11)

    kernel(
        True,
        task_name="4C_Cl_small_%s" % basis,
        fcidump=FCIDUMP_4C_NAME % (basis),
        segment="5 0 4 4 %d 0" % (Cl_atm.nao - 13),
        nelec_val=7,
        # relative=1,
        cmin=CMIN,
        perturbation=1,
        Task="1 1 3 1 1 1",
        # doublegroup="d2h",
    )

    kernel(
        True,
        task_name="4C_Cl_large_%s" % basis,
        fcidump=FCIDUMP_4C_NAME % (basis),
        segment="1 4 4 4 %d 0" % (Cl_atm.nao - 13),
        nelec_val=7,
        # relative=1,
        cmin=CMIN,
        perturbation=1,
        Task="1 1 3 1 1 1",
        # doublegroup="d2h",
    )
