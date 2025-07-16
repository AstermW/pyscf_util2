from pyscf import gto, scf
from pyscf_util.iCIPT2.iCIPT2_coov import kernel
from pyscf_util.Integrals.integral_Coov import (
    FCIDUMP_Coov,
)
from pyscf import tools
from pyscf_util.iCIPT2.iCIPT2 import kernel

mol = gto.M(
    verbose=0,
    atom="""
            C   0.000000000000       0.000000000000      -0.621265
            O   0.000000000000       0.000000000000       0.621265
            """,
    basis={"C": "cc-pvdz", "O": "cc-pvdz"},
    spin=0,
    charge=0,
    symmetry="coov",
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()

FCIDUMP_NAME = "FCIDUMP_CO_COOV"

FCIDUMP_Coov(mol, mf, FCIDUMP_NAME)

mol = gto.M(
    verbose=0,
    atom="""
            C   0.000000000000       0.000000000000      -0.621265
            O   0.000000000000       0.000000000000       0.621265
            """,
    basis={"C": "cc-pvdz", "O": "cc-pvdz"},
    spin=0,
    charge=0,
    symmetry="c2v",
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()

FCIDUMP_NAME = "FCIDUMP_CO"

tools.fcidump.from_scf(mf, FCIDUMP_NAME, 1e-10)
