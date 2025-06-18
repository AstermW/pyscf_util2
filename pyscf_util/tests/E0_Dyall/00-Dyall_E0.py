import pyscf
import numpy
from functools import reduce
from pyscf import tools
import tempfile
import h5py
from pyscf.ao2mo import outcore
from pyscf.mcscf.casci import get_fock

from pyscf_util.Integrals.integral_CASCI import dump_heff_casci
from pyscf_util.MeanField import iciscf
from pyscf_util.iCIPT2.iCIPT2 import kernel
import os
from pyscf.tools import fcidump
from pyscf import symm
from pyscf_util.Integrals.integral_sfX2C import fcidump_sfx2c
from pyscf_util.Integrals.integral_MRPT2 import fcidump_mrpt2, fcidump_mrpt2_outcore

from pyscf_util.File import file_rdm
from pyscf_util.File import file_cmoao
from pyscf_util.Integrals.integral_MRPT2 import get_generalized_fock


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


if __name__ == "__main__":

    ### take Cr2 as the show case ###

    Mol = pyscf.gto.Mole()
    Mol.atom = """
Cr     0.0000      0.0000   %f 
Cr     0.0000      0.0000  -%f 
""" % (
        1.68 / 2,
        1.68 / 2,
    )
    Mol.basis = "ccpvdz-dk"
    Mol.symmetry = "Dooh"
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "angstorm"
    Mol.build()

    SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
    SCF.max_cycle = 32
    SCF.conv_tol = 1e-9
    SCF.run()

    Mol.spin = 0
    Mol.build()

    norb = 12
    nelec = 12
    CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)

    cas_space_symmetry = {
        "A1u": 2,  # 5
        "A1g": 2,  # 0
        "E1ux": 1,  # 7
        "E1gy": 1,  # 3
        "E1gx": 1,  # 2
        "E1uy": 1,  # 6
        "E2gy": 1,  # 1
        "E2gx": 1,  # 0
        "E2uy": 1,  # 4
        "E2ux": 1,  # 5
    }

    ### generate init guess for CAS ###

    mo_init = pyscf.mcscf.sort_mo_by_irrep(
        CASSCF_Driver, CASSCF_Driver.mo_coeff, cas_space_symmetry
    )  # right!
    SCF.mo_coeff = mo_init
    CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
    # CASSCF_Driver = iciscf.iCISCF(SCF, norb, nelec, cmin=0.0, tol=1e-10)
    CASSCF_Driver.canonicalization = True

    ### run ###

    energy, _, _, mo_coeff, mo_energy = CASSCF_Driver.kernel(mo_coeff=mo_init)

    Mol.symmetry = "D2h"
    Mol.build()

    ### call icipt2 to generate rdm1 ###

    mo_coeff = CASSCF_Driver.mo_coeff

    dump_heff_casci(
        Mol,
        CASSCF_Driver,
        mo_coeff[:, :18],
        mo_coeff[:, 18:30],
        _filename="FCIDUMP_Cr2",
    )

    kernel(
        IsCSF=True,
        task_name="cr2_rdm1",
        fcidump="FCIDUMP_Cr2",
        segment="0 0 6 6 0 0",
        nelec_val=12,
        rotatemo=0,
        cmin=0.0,
        perturbation=0,
        dumprdm=1,
        relative=0,
        Task="0 0 1 1",
        inputocfg=0,
        etol=1e-10,
        selection=1,
        doublegroup=None,
        direct=None,
        start_with=None,
        end_with=[".csv"],
    )
    
    # exit(1)

    os.system("mv rdm1.csv cr2_rdm1.csv")

    mo_coeff = CASSCF_Driver.mo_coeff
    rdm1 = file_rdm.ReadIn_rdm1("cr2_rdm1", 12, 12)

    fock_mat = get_generalized_fock(CASSCF_Driver, mo_coeff, rdm1)

    fock_diag = numpy.diag(fock_mat)
    print("Fock matrix diagonal elements:", fock_diag)
    print("mo energy                    :", CASSCF_Driver.mo_energy)
    
    # exit(1)

    ############# prepare data to test E0 Dyall #############

    file_cmoao.Dump_Cmoao("gfock_test", fock_mat)

    # dump FCIDUMP

    fcidump_sfx2c(Mol, SCF, mo_coeff, "FCIDUMP_Cr2_Benchmark", 1e-10)

    kernel(
        IsCSF=True,
        task_name="cr2_rdm1",
        fcidump="FCIDUMP_Cr2_Benchmark",
        segment="18 0 6 6 0 %d" % (Mol.nao - 30),
        nelec_val=12,
        rotatemo=0,
        cmin=0.0,
        perturbation=0,
        dumprdm=0,
        relative=0,
        Task="0 0 1 1",
        inputocfg=0,
        etol=1e-10,
        selection=1,
        doublegroup=None,
        direct=None,
        start_with=None,
        end_with=[".csv", ".PrimeSpace"],
    )

    PrimeSpaceName = f"SpinTwo_0_Irrep_0_Cmin_{0.0:.3e}.PrimeSpace"
