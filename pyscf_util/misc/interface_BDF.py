import pyscf
from pyscf import scf, tools
import numpy
from functools import reduce

from mokit.lib.py2bdf import py2bdf
from mokit.lib.py2fch_direct import fchk

import os

################################################################
### interface BDF and pyscf with MOKIT
################################################################

BDF2FCH = os.getenv("BDF2FCH")
if BDF2FCH is None:
    raise ValueError("BDF2FCH is not set")

################################################################
## MOKIT is a wonnderful package !
################################################################

if __name__ == "__main__":

    from _parse_bdf_orbfile import BDFOrbParser

    #### use C10H8 as an example ####

    GEOMETRY = """
C       -2.433633500      0.708353500      0.000000000 
C       -2.433633500     -0.708353500      0.000000000 
H       -3.378134500     -1.245817000      0.000000000 
H       -3.378134500      1.245817000      0.000000000 
C       -1.244704000      1.402507000      0.000000000 
C       -1.244704000     -1.402507000      0.000000000 
C        0.000000000      0.717168000      0.000000000 
C        0.000000000     -0.717168000      0.000000000 
H       -1.242591000      2.490280000      0.000000000 
H       -1.242591000     -2.490280000      0.000000000 
C        1.244704000      1.402507000      0.000000000 
C        1.244704000     -1.402507000      0.000000000 
C        2.433633500      0.708353500      0.000000000 
C        2.433633500     -0.708353500      0.000000000 
H        1.242591000      2.490280000      0.000000000 
H        1.242591000     -2.490280000      0.000000000 
H        3.378134500      1.245817000      0.000000000 
H        3.378134500     -1.245817000      0.000000000
"""
    Mol = pyscf.gto.Mole()
    Mol.atom = GEOMETRY
    Mol.basis = "cc-pvdz"
    Mol.symmetry = "d2h"
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "angstorm"
    Mol.build()

    #### use pyscf to get the SCF energy ####

    mf = scf.RHF(Mol)
    mf.kernel()
    print(mf.e_tot)
    mf.analyze()

    CASSCF_Driver = pyscf.mcscf.CASSCF(mf, 10, 10)

    # fchk(CASSCF_Driver, "test_C10H8.fch")
    py2bdf(CASSCF_Driver, "test_C10H8.inp")

    # print(Mol.symm_orb)
    # for x in Mol.symm_orb:
    #     print(x.shape)
    # print(Mol.irrep_name)

    # 打印 the first orb

    ovlp = Mol.intor("int1e_ovlp")
    coeff = Mol.symm_orb[0].T @ ovlp @ mf.mo_coeff[:, 0]
    print(coeff)

    print(Mol.symm_orb[0].T @ ovlp @ Mol.symm_orb[0])

    # exit(1)

    # copy and backup the fch file
    # os.system("cp test_C10H8.fch test_C10H8.fch.bak")

    # run BDF2FCH
    # os.system(f"{BDF2FCH} 02S.casorb test_C10H8.fch")

    # read in bdf's casorb file

    # parser = BDFOrbParser("02S.casorb")
    parser = BDFOrbParser("02S.scforb")
    parser.parse_file(verbose=True)
    parser.BDF_convention_old2new()
    # print(parser.sym_blocks)
    # print(parser.sym_orbital_energies)
    # print(parser.sym_occupations)

    # for each irrep construct the mo_coeff

    print(parser.get_sym_data(0, "ALPHA")[:, 0])
    print(parser.get_sym_data(0, "ALPHA")[0, :])

    mo_coeffs = []
    ovlp = Mol.intor("int1e_ovlp")

    # # # for symm_orb, sym_block in zip(Mol.symm_orb, parser.sym_blocks):
    # for key in parser.sym_blocks.keys():
    #     symm_orb = Mol.symm_orb[key]
    #     print(symm_orb.shape)
    #     sym_block = parser.sym_blocks[key]
    #     so_coeff = sym_block['ALPHA']['data']
    #     print(so_coeff.shape)
    #     mo_coeff = symm_orb @ so_coeff
    #     # check ortho #
    #     mo_ovlp = mo_coeff.T @ ovlp @ mo_coeff
    #     print(mo_ovlp)
    #     mo_coeffs.append(mo_coeff)
    # mo_coeffs = numpy.hstack(mo_coeffs)
    # print(mo_coeffs.shape)

    ## NOTE: BDF 内部 symm orb 的 convention 和 pyscf 的 convention 不一样
