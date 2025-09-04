from pyscf import gto, scf

# from pyscf_util.iCIPT2.iCIPT2_coov import kernel
from pyscf_util.Integrals.integral_Dooh import (
    FCIDUMP_Dooh,
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
    symmetry="dooh",
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()

FCIDUMP_NAME = "FCIDUMP_C2_DOOH"

FCIDUMP_Dooh(mol, mf, FCIDUMP_NAME)

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

FCIDUMP_NAME = "FCIDUMP_C2"

# tools.fcidump.from_scf(mf, FCIDUMP_NAME, 1e-10)
OrbSym = pyscf.symm.label_orb_symm(
    mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff[:, :14]
)
OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]
# tools.fcidump.from_mo(mol, FCIDUMP_NAME, mf.mo_coeff[:, :14], OrbSymID, tol=1e-10)
