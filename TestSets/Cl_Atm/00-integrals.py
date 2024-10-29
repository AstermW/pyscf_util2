from pyscf import gto, scf
from pyscf import tools

from pyscf_util._atmMinCASOrb._orb_loader import LoadAtmHFOrb
from pyscf_util.misc.mole import get_mol
from pyscf_util.MeanField.scf import kernel as scf_kernel
from pyscf_util.Relativisitc.sfX2C_soDKH import fetch_X2C_soDKH1
from pyscf_util.Relativisitc.integral_4C import (
    FCIDUMP_Rela4C_SU2,
)
import pyscf_util.File.file_sodkh13 as file_sodkh13
from pyscf import data
import numpy

BASIS = ["unc-ccpvdz-dk", "unc-ccpvtz-dk", "unc-ccpvqz-dk", "unc-ccpv5z-dk"]
# BASIS = ["ccpvdz-dk"]  # TEST

FCIDUMP_NAME = "FCIDUMP_%s"
RELDUMP_NAME = "RELDUMP_%s"
FCIDUMP_4C_NAME = "FCIDUMP_4C_%s"

for basis in BASIS:

    # SOICI and ICIPT2 #

    Cl_atm = get_mol("Cl 0 0 0", 0, 1, basis=basis, symmetry="d2h", verbose=11)
    Cl_scf = scf_kernel(Cl_atm, sfx1e=True, run=True)
    Cl_res = LoadAtmHFOrb("Cl", 0, basis, with_sfx2c=True, rerun=True)

    Cl_scf.mo_coeff = Cl_res["mo_coeff"]
    Cl_scf.mo_energy = Cl_res["mo_energy"]

    # dump FCIDUMP

    tools.fcidump.from_scf(Cl_scf, FCIDUMP_NAME % (basis), 1e-10)

    dm1 = numpy.zeros((Cl_atm.nao, Cl_atm.nao))
    nelectrons = Cl_atm.nelectron
    for i in range((nelectrons - 5) // 2):
        dm1[i, i] = 2
    for i in range((nelectrons - 5) // 2, (nelectrons - 5) // 2 + 3):
        dm1[i, i] = 5.0 / 3.0

    hsf = numpy.zeros((1, Cl_atm.nao, Cl_atm.nao))
    hso = fetch_X2C_soDKH1(
        Cl_atm, Cl_scf, Cl_res["mo_coeff"], dm1, _test=False, _get_1e=True, _get_2e=True
    )
    hso *= (data.nist.ALPHA**2) / 4.0
    hso = numpy.vstack((hso, hsf))
    file_sodkh13.Dump_Relint_iCI(RELDUMP_NAME % (basis), hso, Cl_atm.nao)

    # 4C iCIPT2 #

    mol = gto.M(
        atom="Cl 0 0 0",
        basis=basis,
        verbose=5,
        charge=-1,
        spin=0,
        symmetry="d2h",
    )
    mol.build()
    mf = scf.dhf.RDHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.with_breit = True
    mf.kernel()

    FCIDUMP_Rela4C_SU2(
        mol, mf, True, filename=FCIDUMP_4C_NAME % (basis), mode="outcore"
    )
