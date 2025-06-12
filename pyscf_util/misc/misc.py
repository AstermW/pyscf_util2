import numpy as np
import pyscf
from pyscf import tools, symm
from pyscf_util.misc.mole import get_orbsym


def _combine2(a, b):
    if a > b:
        return a * (a + 1) // 2 + b
    else:
        return b * (b + 1) // 2 + a


def _combine4(a, b, c, d):
    return _combine2(_combine2(a, b), _combine2(c, d))


def read_mo_coeff_from_molden(mol, filename: str):
    _, mo_energy, mo_coeff, _, _, _ = tools.molden.load(filename)

    # 根据 mo_energy 的从小到大的顺序获取排序索引
    sorted_indices = np.argsort(mo_energy)

    # 根据排序索引重排 mo_energy 和 mo_coeff
    mo_energy = mo_energy[sorted_indices]
    mo_coeff = mo_coeff[:, sorted_indices]

    if mol is None:
        # do not do orbsym check #
        return mo_coeff, mo_energy
    else:
        _, orbsym_id = get_orbsym(mol, mo_coeff)
        return mo_coeff, mo_energy, orbsym_id


def get_irrep_orbid(mol, mo_coeff, begin_indx, end_indx):

    act_orb = mo_coeff[:, begin_indx:end_indx]
    # print("begindx ", begin_indx, " end_indx ", end_indx)
    irrep_ids = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, act_orb)

    irrep_orbid = {}

    for id, irrep in enumerate(irrep_ids):
        if irrep not in irrep_orbid.keys():
            irrep_orbid[irrep] = []
        irrep_orbid[irrep].append(id + begin_indx)

    return irrep_orbid


def read_mcscf_mo_coeff_from_molden(
    mol, filename: str, NFZC: dict, NACT: dict, NVIR: dict = None, debug: bool = False
):

    _, mo_energy, mo_coeff, mo_occ, _, _ = tools.molden.load(filename)

    irrep_orbid = get_irrep_orbid(mol, mo_coeff, 0, mol.nao)
    order = []

    for key in irrep_orbid.keys():
        order.extend(irrep_orbid[key][: NFZC[key]])
    nfzc = len(order)
    for key in irrep_orbid.keys():
        order.extend(irrep_orbid[key][NFZC[key] : NFZC[key] + NACT[key]])
    nact = len(order) - nfzc
    if NVIR is None:
        for key in irrep_orbid.keys():
            order.extend(irrep_orbid[key][NFZC[key] + NACT[key] :])
    else:
        for key in irrep_orbid.keys():
            order.extend(
                irrep_orbid[key][
                    NFZC[key] + NACT[key] : NFZC[key] + NACT[key] + NVIR[key]
                ]
            )
    nvir = len(order) - nact - nfzc

    def takeSecond(elem):
        return elem[1]

    fzc_frag = list(zip(order[:nfzc], mo_energy[order][:nfzc]))
    fzc_frag.sort(key=takeSecond)
    act_frag = list(zip(order[nfzc : nfzc + nact], mo_occ[order][nfzc : nfzc + nact]))
    act_frag.sort(key=takeSecond, reverse=True)
    vir_frag = list(zip(order[nfzc + nact :], mo_energy[order][nfzc + nact :]))
    vir_frag.sort(key=takeSecond)

    if debug:
        print(mo_energy[order][:nfzc])
        print(mo_energy[order][nfzc : nfzc + nact])
        print(mo_energy[order][nfzc + nact :])
        print(mo_occ[order][:nfzc])
        print(mo_occ[order][nfzc : nfzc + nact])
        print(mo_occ[order][nfzc + nact :])

        print(fzc_frag)
        print(act_frag)
        print(vir_frag)

    order = []
    for x in fzc_frag:
        order.append(x[0])
    for x in act_frag:
        order.append(x[0])
    for x in vir_frag:
        order.append(x[0])

    _, orbsym_id = get_orbsym(mol, mo_coeff[:, order])

    return (
        mo_coeff[:, order],
        mo_energy[order],
        mo_occ[order],
        orbsym_id,
        nfzc,
        nact,
        nvir,
    )
