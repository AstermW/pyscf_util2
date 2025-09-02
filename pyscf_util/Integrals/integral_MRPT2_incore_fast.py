import pyscf
import numpy
from functools import reduce
from pyscf import tools
import tempfile
import h5py
from pyscf.ao2mo import outcore
from pyscf.mcscf.casci import get_fock
from pyscf_util.misc.misc import _combine4, _combine2
from pyscf_util.File import file_rdm

from pyscf_util.Integrals._util import generate_eri_pprs, generate_eri_prps

############## integral MRPT2 ##############

# TODO: opt it!


def fcidump_mrpt2_incore_fast(
    mol, scf, mo_coeff, nfzc, nact, nvir, filename="FCIDUMP", tol=1e-10, full_integrals=False
):
    nmo = nfzc + nact + nvir
    assert nmo <= mol.nao
    # nmo = my_scf.mo_coeff.shape[1]
    nelec = mol.nelectron
    ms = 0
    # tol = 1e-10
    nuc = mol.get_enuc()
    float_format = tools.fcidump.DEFAULT_FLOAT_FORMAT

    h1e = reduce(numpy.dot, (mo_coeff.T, scf.get_hcore(), mo_coeff))
    h1e = h1e[:nmo, :nmo]

    # print(h1e)

    # int2e_full = pyscf.ao2mo.full(eri_or_mol=mol, mo_coeff=mo_coeff[:, :nmo], aosym="4")
    # int2e_full = pyscf.ao2mo.restore(8, int2e_full.copy(), nmo)

    OrbSym = pyscf.symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, mo_coeff[:, :nmo]
    )
    OrbSymID = [pyscf.symm.irrep_name2id(mol.groupname, x) for x in OrbSym]

    with open(filename, "w") as fout:  # 8-fold symmetry

        tools.fcidump.write_head(fout, nmo, nelec, ms, OrbSymID)
        output_format = float_format + " %4d %4d %4d %4d\n"

        # (1) space CCVV

        ### CCVV part ###

        if full_integrals:

            int2e_cvcv = pyscf.ao2mo.general(
                mol,
                (
                    mo_coeff[:, :nfzc],
                    mo_coeff[:, nfzc + nact :],
                    mo_coeff[:, :nfzc],
                    mo_coeff[:, nfzc + nact :],
                ),
                aosym="1",
                compact=False,
            ).reshape(nfzc, nvir, nfzc, nvir)

            for p in range(int2e_cvcv.shape[0]):
                for q in range(int2e_cvcv.shape[1]):
                    for r in range(p + 1):
                        for s in range(int2e_cvcv.shape[3]):
                            if abs(int2e_cvcv[p, q, r, s]) < tol:
                                continue
                            fout.write(
                                output_format
                                % (
                                    int2e_cvcv[p, q, r, s],
                                    p + 1,
                                    nfzc + nact + q + 1,
                                    r + 1,
                                    nfzc + nact + s + 1,
                                )
                            )

            del int2e_cvcv

        # (2) space ACVV

        ### CVAV part ###

        if full_integrals:

            int2e_cvav = pyscf.ao2mo.general(
                mol,
                (
                    mo_coeff[:, nfzc : nfzc + nact],
                    mo_coeff[:, nfzc + nact :],
                    mo_coeff[:, :nfzc],
                    mo_coeff[:, nfzc + nact :],
                ),
                aosym="1",
                compact=False,
            ).reshape(nact, nvir, nfzc, nvir)

            for p in range(int2e_cvav.shape[0]):
                for q in range(int2e_cvav.shape[1]):
                    for r in range(int2e_cvav.shape[2]):
                        for s in range(int2e_cvav.shape[3]):
                            if abs(int2e_cvav[p, q, r, s]) < tol:
                                continue
                            fout.write(
                                output_format
                                % (
                                    int2e_cvav[p, q, r, s],
                                    nfzc + p + 1,
                                    nfzc + nact + q + 1,
                                    r + 1,
                                    nfzc + nact + s + 1,
                                )
                            )

            del int2e_cvav

        # (3) space CCAV

        ### CVCA part ###

        if full_integrals:

            int2e_cvca = pyscf.ao2mo.general(
                mol,
                (
                    mo_coeff[:, :nfzc],
                    mo_coeff[:, nfzc : nfzc + nact],
                    mo_coeff[:, :nfzc],
                    mo_coeff[:, nfzc + nact :],
                ),
                aosym="1",
                compact=False,
            ).reshape(nfzc, nact, nfzc, nvir)

            for p in range(int2e_cvca.shape[0]):
                for q in range(int2e_cvca.shape[1]):
                    for r in range(int2e_cvca.shape[2]):
                        for s in range(int2e_cvca.shape[3]):
                            if abs(int2e_cvca[p, q, r, s]) < tol:
                                continue
                            fout.write(
                                output_format
                                % (
                                    int2e_cvca[p, q, r, s],
                                    p + 1,
                                    nfzc + q + 1,
                                    r + 1,
                                    nfzc + nact + s + 1,
                                )
                            )

            del int2e_cvca

        # (4) P space

        ### AAAA part ###

        int2e_aaaa = pyscf.ao2mo.general(
            mol,
            (
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc : nfzc + nact],
            ),
            aosym="1",
            compact=False,
        ).reshape(nact, nact, nact, nact)

        for p in range(int2e_aaaa.shape[0]):
            for q in range(int2e_aaaa.shape[1]):
                for r in range(int2e_aaaa.shape[2]):
                    for s in range(int2e_aaaa.shape[3]):
                        if abs(int2e_aaaa[p, q, r, s]) < tol:
                            continue
                        fout.write(
                            output_format
                            % (
                                int2e_aaaa[p, q, r, s],
                                nfzc + p + 1,
                                nfzc + q + 1,
                                nfzc + r + 1,
                                nfzc + s + 1,
                            )
                        )

        del int2e_aaaa

        # (5) CAAA or VAAA

        ### PAAA part ###

        int2e_paaa = pyscf.ao2mo.general(
            mol,
            (
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, :],
                mo_coeff[:, nfzc : nfzc + nact],
            ),
            aosym="1",
            compact=False,
        ).reshape(nact, nact, nfzc + nact + nvir, nact)

        for p in range(int2e_paaa.shape[0]):
            for q in range(p + 1):
                for r in range(int2e_paaa.shape[2]):
                    for s in range(int2e_paaa.shape[3]):
                        if abs(int2e_paaa[p, q, r, s]) < tol:
                            continue
                        fout.write(
                            output_format
                            % (
                                int2e_paaa[p, q, r, s],
                                nfzc + p + 1,
                                nfzc + q + 1,
                                r + 1,
                                nfzc + s + 1,
                            )
                        )

        del int2e_paaa

        # (6) VVAA

        ### VVAA part ###

        int2e_vvaa = pyscf.ao2mo.general(
            mol,
            (
                mo_coeff[:, nfzc + nact :],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc + nact :],
                mo_coeff[:, nfzc : nfzc + nact],
            ),
            aosym="1",
            compact=False,
        ).reshape(nvir, nact, nvir, nact)

        for p in range(int2e_vvaa.shape[0]):
            for q in range(int2e_vvaa.shape[1]):
                for r in range(p + 1):
                    for s in range(int2e_vvaa.shape[3]):
                        if abs(int2e_vvaa[p, q, r, s]) < tol:
                            continue
                        fout.write(
                            output_format
                            % (
                                int2e_vvaa[p, q, r, s],
                                nfzc + nact + p + 1,
                                nfzc + q + 1,
                                nfzc + nact + r + 1,
                                nfzc + s + 1,
                            )
                        )

        del int2e_vvaa

        # (7) ACAC

        ### ACAC part ###

        int2e_acac = pyscf.ao2mo.general(
            mol,
            (
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, :nfzc],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, :nfzc],
            ),
            aosym="1",
            compact=False,
        ).reshape(nact, nfzc, nact, nfzc)

        for p in range(int2e_acac.shape[0]):
            for q in range(int2e_acac.shape[1]):
                for r in range(p + 1):
                    for s in range(int2e_acac.shape[3]):
                        if abs(int2e_acac[p, q, r, s]) < tol:
                            continue
                        fout.write(
                            output_format
                            % (
                                int2e_acac[p, q, r, s],
                                nfzc + p + 1,
                                q + 1,
                                nfzc + r + 1,
                                s + 1,
                            )
                        )

        del int2e_acac

        # (7) C V

        ### CVAA part ###

        int2e_cvaa = pyscf.ao2mo.general(
            mol,
            (
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, :nfzc],
                mo_coeff[:, nfzc + nact :],
            ),
            aosym="1",
            compact=False,
        ).reshape(nact, nact, nfzc, nvir)

        # for p in range(int2e_cvaa.shape[0]):
        #     for q in range(int2e_cvaa.shape[1]):
        #         for r in range(int2e_cvaa.shape[2]):
        #             for s in range(r + 1):
        for p in range(int2e_cvaa.shape[0]):
            for q in range(p+1):
                for r in range(int2e_cvaa.shape[2]):
                    for s in range(int2e_cvaa.shape[3]):
        # for p in range(int2e_cvaa.shape[0]):
        #     for q in range(int2e_cvaa.shape[1]):
        #         for r in range(int2e_cvaa.shape[2]):
        #             for s in range(int2e_cvaa.shape[3]):
                        if abs(int2e_cvaa[p, q, r, s]) < tol:
                            continue
                        fout.write(
                            output_format
                            % (
                                int2e_cvaa[p, q, r, s],
                                nfzc + p + 1,
                                nfzc + q + 1,
                                r + 1,
                                nfzc + nact + s + 1,
                            )
                        )

        del int2e_cvaa

        ### ACAV part ###

        int2e_acav = pyscf.ao2mo.general(
            mol,
            (
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, :nfzc],
                # mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc : nfzc + nact],
                mo_coeff[:, nfzc + nact :],
            ),
            aosym="1",
            compact=False,
        # ).reshape(nfzc, nact, nact, nvir)
        ).reshape(nact, nfzc, nact, nvir)

        for p in range(int2e_acav.shape[0]):
            for q in range(int2e_acav.shape[1]):
                for r in range(int2e_acav.shape[2]):
                    for s in range(int2e_acav.shape[3]):
                        if abs(int2e_acav[p, q, r, s]) < tol:
                            continue
                        fout.write(
                            output_format
                            % (
                                int2e_acav[p, q, r, s],
                                nfzc + p + 1,
                                q + 1,
                                nfzc + r + 1,
                                nfzc + nact + s + 1,
                            )
                        )

        del int2e_acav

        ### pprs part ###

        eri_pprs = generate_eri_pprs(mol, mo_coeff)

        for p in range(eri_pprs.shape[0]):
            for r in range(eri_pprs.shape[1]):
                for s in range(eri_pprs.shape[2]):
                    if abs(eri_pprs[p, r, s]) < tol:
                        continue
                    fout.write(
                        output_format
                        % (
                            eri_pprs[p, r, s],
                            p + 1,
                            p + 1,
                            r + 1,
                            s + 1,
                        )
                    )

        del eri_pprs

        ### prps part ###

        eri_prps = generate_eri_prps(mol, mo_coeff)

        for p in range(eri_prps.shape[0]):
            for r in range(eri_prps.shape[1]):
                for s in range(eri_prps.shape[2]):
                    if abs(eri_prps[p, r, s]) < tol:
                        continue
                    fout.write(
                        output_format
                        % (
                            eri_prps[p, r, s],
                            p + 1,
                            r + 1,
                            p + 1,
                            s + 1,
                        )
                    )

        del eri_prps

        ### h1e part ###

        tools.fcidump.write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + "  0  0  0  0\n"
        fout.write(output_format % nuc)


if __name__ == "__main__":

    from pyscf_util.Integrals.integral_CASCI import dump_heff_casci
    from pyscf_util.MeanField import iciscf
    from pyscf_util.iCIPT2.iCIPT2 import kernel
    import os
    from pyscf.tools import fcidump
    from pyscf import symm
    from pyscf_util.Integrals.integral_sfX2C import fcidump_sfx2c
    from pyscf_util.Integrals.integral_MRPT2 import fcidump_mrpt2, fcidump_mrpt2_outcore

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

    ### take Cr2 as the show case ###

    Mol = pyscf.gto.Mole()
    Mol.atom = """
Cr     0.0000      0.0000   %f 
Cr     0.0000      0.0000  -%f 
""" % (
        1.68 / 2,
        1.68 / 2,
    )
    Mol.basis = "def2-svp"
    Mol.symmetry = "Dooh"
    Mol.spin = 2
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
    CASSCF_Driver = iciscf.iCISCF(SCF, norb, nelec, cmin=0.0)
    CASSCF_Driver.canonicalization = True

    ### run ###

    energy, _, _, mo_coeff, mo_energy = CASSCF_Driver.kernel(mo_coeff=mo_init)

    Mol.symmetry = "D2h"
    Mol.build()

    ### call icipt2 to generate rdm1 ###

    # mo_coeff = CASSCF_Driver.mo_coeff

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

    os.system("mv rdm1.csv cr2_rdm1.csv")

    mo_coeff = CASSCF_Driver.mo_coeff
    rdm1 = file_rdm.ReadIn_rdm1("cr2_rdm1", 12, 12)

    print(rdm1)
    # print(get_generalized_fock(CASSCF_Driver, mo_coeff, rdm1))
    print(mo_energy)

    # fcidump #

    orbsym = OrbSymInfo(Mol, CASSCF_Driver.mo_coeff)
    fcidump_mrpt2_incore_fast(
        Mol, SCF, mo_coeff, 18, 12, Mol.nao - 30, "FCIDUMP_Cr2_incore_fast", 1e-10
    )
    fcidump_sfx2c(Mol, SCF, mo_coeff, "FCIDUMP_Cr2_Benchmark", 1e-10)
    fcidump_mrpt2_outcore(
        Mol, SCF, mo_coeff, 18, 12, Mol.nao - 30, "FCIDUMP_Cr2_outcore", 1e-10
    )
    fcidump_mrpt2(Mol, SCF, mo_coeff, 18, 12, Mol.nao - 30, "FCIDUMP_Cr2_incore", 1e-10)
