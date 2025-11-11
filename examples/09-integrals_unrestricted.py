from pyscf import gto, scf

# from pyscf_util.iCIPT2.iCIPT2_coov import kernel
from pyscf_util.Integrals.integral_unrestricted import (
    fcidump_unrestricted,
)
from pyscf import tools

# from pyscf_util.iCIPT2.iCIPT2 import kernel
import pyscf

mol = gto.M(
    verbose=0,
    atom="""
            C   0.000000000000       0.000000000000      -0.621265
            C   0.000000000000       0.000000000000       0.621265
            """,
    basis={"C": "cc-pvdz", "O": "cc-pvdz"},
    spin=0,
    charge=0,
    symmetry="d2h",
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()

FCIDUMP_NAME = "FCIDUMP_C2_UNRESTRICTED"

fcidump_unrestricted(mol, mf, mf.mo_coeff, _filename=FCIDUMP_NAME)

# no sym #

mol.symmetry = 'C1'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

FCIDUMP_NAME = "FCIDUMP_C2_nosym"
tools.fcidump.from_scf(mf, FCIDUMP_NAME, 1e-10)

# generate random orthgonal mat of 8x8
import numpy as np
from scipy.stats import ortho_group

U = ortho_group.rvs(8)
U2 = ortho_group.rvs(8)

print(U)
print(U2)

# rotate mo_coeff
mo_coeff_alpha = mf.mo_coeff.copy()
mo_coeff_beta = mf.mo_coeff.copy()
mo_coeff_alpha[:, 2:10] = np.dot(mo_coeff_alpha[:, 2:10], U)
mo_coeff_beta[:, 2:10] = np.dot(mo_coeff_beta[:, 2:10], U2)

FCIDUMP_NAME = "FCIDUMP_C2_UNRESTRICTED_random"
fcidump_unrestricted(mol, mf, mo_coeff_alpha, mo_coeff_beta, _filename=FCIDUMP_NAME)
