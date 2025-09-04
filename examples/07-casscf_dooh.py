from pyscf import gto, scf

# from pyscf_util.iCIPT2.iCIPT2_coov import kernel
from pyscf import tools

# from pyscf_util.iCIPT2.iCIPT2 import kernel
import pyscf
from pyscf_util.Integrals.integral_MRPT2 import get_generalized_fock
from pyscf_util.MeanField.iciscf import kernel as iciscf_kernel
from pyscf_util.MeanField.mcscf import kernel as casscf_kernel
from pyscf_util.Integrals.integral_CASCI import dump_heff_casci
from pyscf_util.iCIPT2.iCIPT2 import kernel
from pyscf_util.Integrals.integral_Dooh import (
    FCIDUMP_Dooh,
)

mol = gto.M(
    verbose=4,
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

norb = 8
nelec = 8

# res1 = iciscf_kernel(mol, mf, nelec, norb, _ici_state=[[0, 0, 1]])

mol2 = gto.M(
    verbose=4,
    atom="""
            C   0.000000000000       0.000000000000      -0.621265
            C   0.000000000000       0.000000000000       0.621265
            """,
    basis={"C": "cc-pvdz", "O": "cc-pvdz"},
    spin=0,
    charge=0,
    symmetry="dooh",
)
mol2.build()
mf2 = scf.RHF(mol2)
mf2.kernel()

norb = 8
nelec = 8

# res2 = iciscf_kernel(mol2, mf2, nelec, norb, _ici_state=[[0,0,1]])
# FCIDUMP_Dooh(mol2, res2, "FCIDUMP_C2_DOOH")
# exit(1)
res2 = iciscf_kernel(mol2, mf2, nelec, norb, _ici_state=[[2, 6, 1], [2, 7, 1]])
res2 = casscf_kernel(mol2, mf2, nelec, norb, _pyscf_state=[[0, 6, 1], [0, 7, 1]])
# FCIDUMP_Dooh(mol2, res2,"FCIDUMP_C2_DOOH2")
