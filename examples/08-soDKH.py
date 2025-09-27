import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf
from pyscf_util.MeanField.iciscf import *
from pyscf_util.File import file_cmoao
from pyscf_util.File import file_sodkh13
from pyscf_util.Relativisitc.sfX2C_soDKH import *
from pyscf import data
from pyscf_util.Integrals.integral_sfX2C import fcidump_sfx2c


def build_dm1(nao, nelec_p, nelec_tot):
    ncore = (nelec_tot - nelec_p) // 2
    res = numpy.zeros((nao, nao))
    for i in range(ncore):
        res[i, i] = 2.0
    for i in range(ncore, ncore + 3):
        res[i, i] = float(nelec_p) / 3
    return res


def OrbSymInfo(Mol, mo_coeff):
    IRREP_MAP = {}
    nsym = len(Mol.irrep_name)
    for i in range(nsym):
        IRREP_MAP[Mol.irrep_name[i]] = i
    # print(IRREP_MAP)

    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb, mo_coeff)
    IrrepOrb = []
    for i in range(len(OrbSym)):
        IrrepOrb.append(symm.irrep_name2id(Mol.groupname, OrbSym[i]))
    return IrrepOrb


Mol = pyscf.gto.Mole()
Mol.atom = """
F 0.0 0.0 0.0
"""
Mol.basis = "cc-pvdz"
Mol.symmetry = True
Mol.spin = 0
Mol.charge = -1
Mol.verbose = 4
Mol.unit = "angstorm"
Mol.build()

SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
SCF.max_cycle = 32
SCF.conv_tol = 5e-12
SCF.run()

mo_coeff = SCF.mo_coeff.copy()

Mol.spin = 1
Mol.charge = 0
Mol.symmetry = "D2h"
Mol.build()

SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
SCF.max_cycle = 32
SCF.conv_tol = 5e-12
SCF.mo_coeff = mo_coeff

norb = 3
nelec = 5

iCISCF_driver = iCISCF(
    SCF,
    norb,
    nelec,
    cmin=0.0,
    tol=1e-12,
    state=[
        [1, 5, 1, [1]],
        [1, 6, 1, [1]],
        [1, 7, 1, [1]],
    ],
)
iCISCF_driver.max_cycle_macro = 8

energy, _, _, mo_coeff, mo_energy = iCISCF_driver.kernel(mo_coeff=mo_coeff)

########## construct dm ##########

dm1 = build_dm1(mo_coeff.shape[0], 5, 9)

hsf = numpy.zeros((1, Mol.nao, Mol.nao))
hso = fetch_X2C_soDKH1(Mol, SCF, mo_coeff, dm1)
hso *= (data.nist.ALPHA**2) / 4.0
hso = numpy.vstack((hso, hsf))
orbsym = OrbSymInfo(Mol, mo_coeff)
DumpFileName = "FCIDUMP_F"

fcidump_sfx2c(Mol, SCF, mo_coeff, DumpFileName, 1e-12)
file_sodkh13.Dump_Relint_iCI("RELDUMP_F", hso, Mol.nao)

#####################################################################################

Mol = pyscf.gto.Mole()
Mol.atom = """
O 0.0 0.0 0.0
"""
Mol.basis = "cc-pvdz"
Mol.symmetry = True
Mol.spin = 0
Mol.charge = -2
Mol.verbose = 4
Mol.unit = "angstorm"
Mol.build()

SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
SCF.max_cycle = 32
SCF.conv_tol = 5e-12
SCF.run()

mo_coeff = SCF.mo_coeff.copy()

Mol.spin = 2
Mol.charge = 0
Mol.symmetry = "D2h"
Mol.build()

SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
SCF.max_cycle = 32
SCF.conv_tol = 5e-12
SCF.mo_coeff = mo_coeff

norb = 3
nelec = 4

iCISCF_driver = iCISCF(
    SCF,
    norb,
    nelec,
    cmin=0.0,
    tol=1e-12,
    state=[
        [2, 1, 1, [1]],
        [2, 2, 1, [1]],
        [2, 3, 1, [1]],
    ],
)
iCISCF_driver.max_cycle_macro = 8

energy, _, _, mo_coeff, mo_energy = iCISCF_driver.kernel(mo_coeff=mo_coeff)

########## construct dm ##########

dm1 = build_dm1(mo_coeff.shape[0], 4, 8)

hsf = numpy.zeros((1, Mol.nao, Mol.nao))
hso = fetch_X2C_soDKH1(Mol, SCF, mo_coeff, dm1)
hso *= (data.nist.ALPHA**2) / 4.0
hso = numpy.vstack((hso, hsf))
orbsym = OrbSymInfo(Mol, mo_coeff)
DumpFileName = "FCIDUMP_O"

fcidump_sfx2c(Mol, SCF, mo_coeff, DumpFileName, 1e-12)
file_sodkh13.Dump_Relint_iCI("RELDUMP_O", hso, Mol.nao)
