import pyscf
from pyscf import scf
import numpy as np
from mokit.lib.py2fch_direct import fchk
from mokit.lib.gaussian import mo_fch2py
import os
from pyscf_util.misc._parse_bdf_chkfil import (
    read_ao2somat_from_chkfil,
    ao2somat_split_based_on_irrep,
)
from pyscf_util.misc._parse_bdf_orbfile import BDFOrbParser
from pyscf_util.misc.dump_to_bdforb import dump_to_scforb


def convert_bdf_to_pyscf(
    # geometry,
    # basis="cc-pvdz",
    # symmetry="d2h",
    Mol: pyscf.gto.Mole,
    mf: scf.RHF,
    chkfil_path="02S.chkfil",  #### 必须保证 chkfil 文件 和 Mol 和 scforb 文件分子构型时一致的，注意，bdf 做带对称性的计算时候会调整分子构型 ####
    scforb_path="bdf_test/02S.scforb",
    output_fch="test.fch",
    output_fch_new="test_new.fch",
    output_scforb="02S_nosymm.scforb",
    old_bdf_convention = False,
    is_casorb = False
    # max_scf_cycles=32,
):
    """
    Convert PySCF calculation to BDF format and perform SCF calculation.

    Parameters
    ----------
    Mol :
    mf  :
    chkfil_path : str, optional
        Path to BDF checkpoint file
    scforb_path : str, optional
        Path to BDF orbital file
    output_fch : str, optional
        Path for intermediate fch file
    output_fch_new : str, optional
        Path for final fch file
    output_scforb : str, optional
        Path for output BDF orbital file
    max_scf_cycles : int, optional
        Maximum number of SCF cycles, default is 32

    Returns
    -------
    tuple
        (mf, mo_coeffs_bdf) where mf is the PySCF mean-field object and mo_coeffs_bdf
        are the molecular orbitals in BDF format
    """
    # Check BDF2FCH environment variable
    BDF2FCH = os.getenv("BDF2FCH")
    if BDF2FCH is None:
        raise ValueError("BDF2FCH is not set")

    # Initial SCF calculation
    max_cycle_bak = mf.max_cycle
    mf.max_cycle = 1  # do not do scf just build
    mf.mo_coeff = np.zeros((Mol.nao, Mol.nao))
    mf.mo_energy = np.zeros(Mol.nao)
    # mf.kernel()

    # Read BDF symmetry orbitals
    ao2somat_bdf = read_ao2somat_from_chkfil(chkfil_path)
    ao2somat_bdf = ao2somat_split_based_on_irrep(ao2somat_bdf, Mol)

    # Read and process orbital coefficients
    parser = BDFOrbParser(scforb_path)
    parser.parse_file()
    if old_bdf_convention:
        parser.BDFold_2_new()

    # Pack orbital data
    mo_coeffs = []
    energies = []
    occupancies = []

    for irrep in range(len(Mol.irrep_name)):
        mo_coeff_tmp = ao2somat_bdf[irrep] @ parser.get_sym_data(irrep)
        mo_coeffs.append(mo_coeff_tmp)
        energies.append(parser.get_sym_energies(irrep))
        occupancies.append(parser.get_sym_occupations(irrep))

    mo_coeffs = np.hstack(mo_coeffs)
    energies = np.hstack(energies)
    occupancies = np.hstack(occupancies)

    # Dump to BDF format
    dump_to_scforb(Mol, mo_coeffs, energies, occupancies, output_scforb, is_casorb=is_casorb)

    # Convert to fch format
    fchk(mf, output_fch)
    os.system(f"{BDF2FCH} {output_scforb} {output_fch} {output_fch_new}")

    # Read back and perform final SCF
    mo_coeffs_bdf = mo_fch2py(output_fch_new)

    mf.max_cycle = max_cycle_bak

    # fch file must be removed #

    os.system(f"rm {output_fch}")
    os.system(f"rm {output_fch_new}")

    return mo_coeffs_bdf


if __name__ == "__main__":
    # Example usage with C10H8
    GEOMETRY = """
C              4.598900802567      -0.000000000000      -1.338594114377
C              4.598900802567      -0.000000000000       1.338594114377
H              6.383749016945      -0.000000000000       2.354252931327
H              6.383749016945      -0.000000000000      -2.354252931327
C              2.352149666151      -0.000000000000      -2.650354117785
C              2.352149666151      -0.000000000000       2.650354117785
C             -0.000000000000      -0.000000000000      -1.355251105302
C              0.000000000000       0.000000000000       1.355251105302
H              2.348156674849      -0.000000000000      -4.705947173482
H              2.348156674849      -0.000000000000       4.705947173482
C             -2.352149666151       0.000000000000      -2.650354117785
C             -2.352149666151       0.000000000000       2.650354117785
C             -4.598900802567       0.000000000000      -1.338594114377
C             -4.598900802567       0.000000000000       1.338594114377
H             -2.348156674849       0.000000000000      -4.705947173482
H             -2.348156674849       0.000000000000       4.705947173482
H             -6.383749016945       0.000000000000      -2.354252931327
H             -6.383749016945       0.000000000000       2.354252931327
"""
    Mol = pyscf.gto.Mole()
    Mol.atom = GEOMETRY
    Mol.basis = "cc-pvdz"
    Mol.symmetry = "d2h"
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "bohr"
    Mol.build()

    # build mf #
    mf = scf.RHF(Mol)

    # convert to bdf #
    mo_coeffs_bdf = convert_bdf_to_pyscf(Mol, mf)

    # check ortho #
    
    ovlp = Mol.intor("int1e_ovlp")
    ovlp_mo = mo_coeffs_bdf.T @ ovlp @ mo_coeffs_bdf
    
    # print diagonal elements of ovlp_mo #
    
    print(np.diag(ovlp_mo))

    # check #

    #dm_init = mf.make_rdm1(mo_coeffs_bdf)
    #mf.kernel(dm0=dm_init)  # should end within one or two cycles #
