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
