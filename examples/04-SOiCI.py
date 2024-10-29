from pyscf import gto, scf
from pyscf import tools

from pyscf_util._atmMinCASOrb._orb_loader import LoadAtmHFOrb
from pyscf_util.misc.mole import get_mol
from pyscf_util.MeanField.scf import kernel as scf_kernel

Cl_atm = get_mol("Cl 0 0 0", 0, 1, basis="unc-cc-pvdz-dk", symmetry="d2h", verbose=11)
Cl_scf = scf_kernel(Cl_atm, sfx1e=True, run=True)
Cl_res = LoadAtmHFOrb("Cl", 0, "unc-cc-pvdz-dk", with_sfx2c=True, rerun=True)

Cl_scf.mo_coeff = Cl_res["mo_coeff"]
Cl_scf.mo_energy = Cl_res["mo_energy"]

# dump FCIDUMP

FCIDUMP_NAME = "FCIDUMP"

tools.fcidump.from_scf(Cl_scf, FCIDUMP_NAME, 1e-10)

dm1 = Cl_scf.make_rdm1()  # TODO: check X2C !!
