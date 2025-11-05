import pyscf
from pyscf import tools
import numpy as np
from pyscf.ao2mo import incore
from functools import reduce
import numpy


def fcidump_unrestricted(
    _mol, _scf, _mocoeff_alpha, _mocoeff_beta=None, _filename="FCIDUMP"
):
    norb = _mocoeff_alpha.shape[1]
    nao = _mocoeff_alpha.shape[0]
    if _mocoeff_beta is None:
        _mocoeff_beta = _mocoeff_alpha
    assert _mocoeff_alpha.shape == _mocoeff_beta.shape

    # energy_core

    energy_core = _mol.get_enuc()

    # int 1e

    h1e_A = reduce(numpy.dot, (_mocoeff_alpha.T, _scf.get_hcore(), _mocoeff_alpha))
    h1e_B = reduce(numpy.dot, (_mocoeff_beta.T, _scf.get_hcore(), _mocoeff_beta))

    h1e = numpy.zeros((2 * norb, 2 * norb))

    for i in range(norb):
        for j in range(norb):
            h1e[2 * i, 2 * j] = h1e_A[i, j]
            h1e[2 * i + 1, 2 * j + 1] = h1e_B[i, j]

    # int 2e

    int2e_full_AA = pyscf.ao2mo.full(
        eri_or_mol=_mol, mo_coeff=_mocoeff_alpha, compact=True
    )
    int2e_full_AA = pyscf.ao2mo.restore(1, int2e_full_AA.copy(), norb)
    int2e_full_BB = pyscf.ao2mo.full(
        eri_or_mol=_mol, mo_coeff=_mocoeff_beta, compact=True
    )
    int2e_full_BB = pyscf.ao2mo.restore(1, int2e_full_BB.copy(), norb)

    # restore #

    int2e_full_AB = pyscf.ao2mo.general(
        eri_or_mol=_mol,
        mo_coeffs=(_mocoeff_alpha, _mocoeff_alpha, _mocoeff_beta, _mocoeff_beta),
        compact=False,
    ).reshape(norb, norb, norb, norb)

    # get orbsym

    OrbSym_alpha = pyscf.symm.label_orb_symm(
        _mol, _mol.irrep_name, _mol.symm_orb, _mocoeff_alpha
    )
    orbSym_beta = pyscf.symm.label_orb_symm(
        _mol, _mol.irrep_name, _mol.symm_orb, _mocoeff_beta
    )
    OrbSymID_alpha = [pyscf.symm.irrep_name2id(_mol.groupname, x) for x in OrbSym_alpha]
    OrbSymID_beta = [pyscf.symm.irrep_name2id(_mol.groupname, x) for x in orbSym_beta]

    if not all([OrbSymID_alpha[i] == OrbSymID_beta[i] for i in range(norb)]):
        raise RuntimeError
    OrbSymID = OrbSymID_alpha

    # DUMP

    if _filename == None:
        return (
            (h1e_A, h1e_B),
            (int2e_full_AA, int2e_full_BB, int2e_full_AB),
            energy_core,
            _mocoeff_alpha.shape[1],
            _mol.nelectron,
            OrbSymID,
        )

    # dump #

    nmo = norb
    nelec = _mol.nelectron
    ms = 0
    tol = 1e-10
    nuc = energy_core
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT

    with open(_filename, "w") as fout:  # 4-fold symmetry
        tools.fcidump.write_head(fout, nmo, nelec, ms, OrbSymID)
        output_format = float_format + " %4d %4d %4d %4d\n"
        ## AA and BB part ##
        for i in range(nmo):
            for j in range(i + 1):
                for k in range(i + 1):
                    if i > k:
                        for l in range(k + 1):
                            if abs(int2e_full_AA[i][j][k][l]) > tol:
                                fout.write(
                                    output_format
                                    % (
                                        int2e_full_AA[i][j][k][l],
                                        2 * i + 1,
                                        2 * j + 1,
                                        2 * k + 1,
                                        2 * l + 1,
                                    )
                                )
                            if abs(int2e_full_BB[i][j][k][l]) > tol:
                                fout.write(
                                    output_format
                                    % (
                                        int2e_full_BB[i][j][k][l],
                                        2 * i + 2,
                                        2 * j + 2,
                                        2 * k + 2,
                                        2 * l + 2,
                                    )
                                )
                    else:  # i==k
                        for l in range(j + 1):
                            if abs(int2e_full_AA[i][j][k][l]) > tol:
                                fout.write(
                                    output_format
                                    % (
                                        int2e_full_AA[i][j][k][l],
                                        2 * i + 1,
                                        2 * j + 1,
                                        2 * k + 1,
                                        2 * l + 1,
                                    )
                                )
                            if abs(int2e_full_BB[i][j][k][l]) > tol:
                                fout.write(
                                    output_format
                                    % (
                                        int2e_full_BB[i][j][k][l],
                                        2 * i + 2,
                                        2 * j + 2,
                                        2 * k + 2,
                                        2 * l + 2,
                                    )
                                )
        ## AB part ##

        for i in range(nmo):
            for j in range(i + 1):
                for k in range(nmo):
                    for l in range(k + 1):
                        if abs(int2e_full_AB[i][j][k][l]) > tol:
                            fout.write(
                                output_format
                                % (
                                    int2e_full_AB[i][j][k][l],
                                    2 * i + 1,
                                    2 * j + 1,
                                    2 * k + 2,
                                    2 * l + 2,
                                )
                            )

        tools.fcidump.write_hcore(
            fout, h1e, nmo * 2, tol=tol, float_format=float_format
        )
        output_format = float_format + "  0  0  0  0\n"
        fout.write(output_format % nuc)
