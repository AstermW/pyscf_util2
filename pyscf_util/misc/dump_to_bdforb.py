#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PySCF 数据转换为 BDF scforb 格式的脚本

作者: 自动生成
日期: 2024
"""

import numpy as np
from typing import List, Dict, Optional
import pyscf


def dump_to_scforb(
    mol: pyscf.gto.Mole,
    mo_coeffs: np.ndarray,
    energies: np.ndarray,
    occupancies: np.ndarray,
    filename: str = "02S.scforb",
    title: str = "SCF Canonical Orbital",
    is_casorb=False,
):
    """
    将 PySCF 数据转换为 BDF scforb 格式

    参数:
        mol: PySCF 分子对象
        mo_coeffs: 分子轨道系数矩阵 (nbas, nmo)
        energies: 轨道能量数组 (nmo,)
        occupancies: 占据数数组 (nmo,)
        filename: 输出文件名
        title: 文件标题
    """

    # 检查输入数据的一致性
    nbas, nmo = mo_coeffs.shape
    assert len(energies) == nmo, f"能量数组长度 {len(energies)} 与轨道数 {nmo} 不匹配"
    assert (
        len(occupancies) == nmo
    ), f"占据数数组长度 {len(occupancies)} 与轨道数 {nmo} 不匹配"

    # # 获取对称性信息， 不考虑
    # if hasattr(mol, 'symm_orb') and mol.symm_orb is not None:
    #     # 有对称性的情况
    #     irrep_names = mol.irrep_name if hasattr(mol, 'irrep_name') else [f"SYM{i}" for i in range(len(mol.symm_orb))]
    #     sym_data = _process_symmetry_data(mol, mo_coeffs, energies, occupancies)
    # else:
    # 无对称性的情况，创建单个对称性块
    irrep_names = ["A"]
    sym_data = {
        0: {
            "name": "A",
            "norb": nmo,
            "coeffs": mo_coeffs,
            "energies": energies,
            "occupancies": occupancies,
        }
    }

    # 根据轨道能量排序 #

    sym_info = sym_data[0]

    if not is_casorb:
        coeffs = sym_info["coeffs"]
        energies = sym_info["energies"]
        occupancies = sym_info["occupancies"]
        idx = np.argsort(energies)
        # print(idx)
        sym_data[0]["coeffs"] = coeffs[:, idx]
        sym_data[0]["energies"] = energies[idx]
        sym_data[0]["occupancies"] = occupancies[idx]
        energies = energies[idx]
        occupancies = occupancies[idx]
    else:
        # 按 occupancies 从大到小排序
        idx_occ = np.argsort(-occupancies)
        coeffs = sym_info["coeffs"][:, idx_occ]
        energies = sym_info["energies"][idx_occ]
        occupancies = sym_info["occupancies"][idx_occ]

        # print(idx_occ)

        # 对于 occupancy > 2-1e-4 和 < 1e-4 的轨道按轨道能量排序
        high_occ_idx = np.where(occupancies > 2 - 1e-4)[0]
        low_occ_idx = np.where(occupancies < 1e-4)[0]
        mid_occ_idx = np.setdiff1d(
            np.arange(len(occupancies)), np.concatenate((high_occ_idx, low_occ_idx))
        )

        high_occ_sort_idx = high_occ_idx[np.argsort(energies[high_occ_idx])]
        low_occ_sort_idx = low_occ_idx[np.argsort(energies[low_occ_idx])]
        sorted_idx = np.concatenate((high_occ_sort_idx, mid_occ_idx, low_occ_sort_idx))

        # print(high_occ_sort_idx)
        # print(low_occ_sort_idx)

        coeffs = coeffs[:, sorted_idx]
        energies = energies[sorted_idx]
        occupancies = occupancies[sorted_idx]

        # 修改 occupancies 数组
        nocc = mol.nelectron // 2
        occupancies[:nocc] = 1.0
        occupancies[nocc:] = 0.0

        sym_data[0]["coeffs"] = coeffs
        sym_data[0]["energies"] = energies
        sym_data[0]["occupancies"] = occupancies

    # 写入文件
    with open(filename, "w") as f:
        # 写入标题
        f.write(f"TITLE - {title}\n")
        f.write("$MOCOEF  %d\n" % len(irrep_names))

        # 写入每个对称性块的轨道系数
        for sym_idx in sorted(sym_data.keys()):
            sym_info = sym_data[sym_idx]
            norb = sym_info["norb"]
            coeffs = sym_info["coeffs"]

            # 写入 SYM 头部
            f.write(f"SYM={sym_idx + 1:3d} NORB={norb:9d} ALPHA\n")

            # 写入轨道系数矩阵 (按列写入，每个轨道一列)
            for col in range(coeffs.shape[1]):
                orbital_coeffs = coeffs[:, col]
                _write_array_data(f, orbital_coeffs)

        # 写入轨道能量
        f.write("ORBITAL ENERGY\n")
        _write_array_data(f, energies)

        # 写入占据数
        f.write("OCCUPATION\n")
        _write_array_data(f, occupancies, "{:12.5E}", 10)

        # 写入结束标记
        f.write("$END\n")

        # write mole geometry data in bohr #

        natm = mol.natm
        coords = mol.atom_coords()
        f.write(f"$COORD{natm:9d}\n")
        for i in range(natm):
            f.write(
                f"{mol.atom_symbol(i):3s}     {coords[i, 0]:21.12f}{coords[i, 1]:21.12f}{coords[i, 2]:21.12f}\n"
            )
        f.write("$END\n")

        # dump nbas and nocc info #

        nocc = mol.nelectron // 2
        f.write(f"$NBF{nbas:9d}{nbas:9d}   1\n")
        f.write(f"NOCC{nocc:9d}{nocc:9d}\n")
        f.write("$END\n")

    print(f"数据已成功写入到 {filename}")
    print(f"总轨道数: {nmo}")
    print(f"对称性块数: {len(sym_data)}")
    for sym_idx, sym_info in sym_data.items():
        print(f"  SYM {sym_idx + 1}: {sym_info['norb']} 个轨道")


def _write_array_data(
    f, data: np.ndarray, format_str: str = "{:25.16E}", items_per_line: int = 5
):
    """
    按 BDF 格式写入数组数据

    参数:
        f: 文件对象
        data: 要写入的数据数组
        format_str: 数值格式字符串
        items_per_line: 每行的数据个数
    """
    data_flat = data.flatten()

    for i in range(0, len(data_flat), items_per_line):
        line_data = data_flat[i : i + items_per_line]
        line_str = "".join([format_str.format(val) for val in line_data])
        f.write(line_str + "\n")
