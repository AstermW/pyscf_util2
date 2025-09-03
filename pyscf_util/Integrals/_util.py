#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按壳层生成 (**|kl) 积分的脚本

作者: 自动生成
日期: 2024
"""

import numpy as np
import pyscf
from pyscf import gto, lib
from typing import Iterator, Tuple, List, Dict
import time
from pyscf.ao2mo._ao2mo import nr_e2


def get_shell_slices(mol: pyscf.gto.Mole, ao_block_size: int) -> List[Tuple[int, int]]:
    """
    根据给定的 AO 块大小，返回壳层切片，使得每个切片的 AO 块大小不大于指定的 AO 块大小。

    参数:
        mol: PySCF 分子对象
        ao_block_size: 最大 AO 块大小

    返回:
        壳层切片列表，每个切片为 (start, end) 格式
    """
    shell_slices = []
    current_ao_count = 0
    start_shell = 0

    # print(mol.aoslice_by_atom())

    current_ao_loc = 0

    for shell_id in range(mol.nbas):
        # ao_start, ao_end = mol.aoslice_by_atom()[shell_id, 2:4]
        # ao_count = ao_end - ao_start
        nctr = mol.bas_nctr(shell_id)
        ao_count = nctr * (2 * mol.bas_angular(shell_id) + 1)

        if current_ao_count + ao_count > ao_block_size:
            shell_slices.append((start_shell, shell_id))
            start_shell = shell_id
            current_ao_count = 0

        current_ao_count += ao_count

    # 添加最后一个切片
    if start_shell < mol.nbas:
        shell_slices.append((start_shell, mol.nbas))

    return shell_slices


def _get_eri_ijkl_given_kl_shell_slices(
    mol: pyscf.gto.Mole, k_shell: int, l_shell: int, shell_slices
) -> np.ndarray:
    """
    给定 k 和 l 壳层切片，生成 (**|kl) 积分
    """

    assert k_shell < len(shell_slices)
    assert l_shell < len(shell_slices)

    k_start, k_end = shell_slices[k_shell]
    l_start, l_end = shell_slices[l_shell]

    shell_slice = (0, mol.nbas, 0, mol.nbas, k_start, k_end, l_start, l_end)

    eri_slice = mol.intor("int2e", shls_slice=shell_slice)

    return eri_slice


def _get_eri_ikjl_given_kl_shell_slices(
    mol: pyscf.gto.Mole, k_shell: int, l_shell: int, shell_slices
) -> np.ndarray:
    """
    给定 k 和 l 壳层切片，生成 (*k|*l) 积分
    """

    assert k_shell < len(shell_slices)
    assert l_shell < len(shell_slices)

    k_start, k_end = shell_slices[k_shell]
    l_start, l_end = shell_slices[l_shell]

    shell_slice = (0, mol.nbas, k_start, k_end, 0, mol.nbas, l_start, l_end)

    eri_slice = mol.intor("int2e", shls_slice=shell_slice)

    return eri_slice


def generate_eri_pprs(
    mol: pyscf.gto.Mole, mo_coeff, ao_block_size: int = 64, verbose: int = 1
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    按壳层生成 (**|kl) 电子排斥积分

    参数:
        mol: PySCF 分子对象
        compact: 是否使用紧凑存储格式
        verbose: 输出详细程度

    生成:
        (k_shell, l_shell, eri_block): 壳层索引和对应的积分块
    """

    if verbose > 0:
        print("-" * 64)
        print("开始按壳层生成 转积分 ijkl,ip,jp,kr,ls --> pprs")
        print(f"分子       : {mol.atom}")
        print(f"基组       : {mol.basis}")
        print(f"总壳层数   : {mol.nbas}")
        print(f"总基函数数 : {mol.nao}")

    # 获取壳层信息

    shell_slices = get_shell_slices(mol, ao_block_size)

    # print(shell_slices)

    # total_shells = mol.nbas
    nao = mol.nao

    # 统计信息
    total_pairs = 0
    processed_pairs = 0
    start_time = time.time()

    ao_pair_2_ao = np.einsum("ip,jp->ijp", mo_coeff, mo_coeff)
    ao_pair_2_ao = ao_pair_2_ao.reshape(nao * nao, nao)

    ao_loc = mol.ao_loc_nr()

    # print(ao_loc)

    eri_ppkl = np.zeros((nao, nao, nao))

    # 遍历所有 (k,l) 壳层对
    for id_k in range(len(shell_slices)):
        for id_l in range(len(shell_slices)):

            eri_block = _get_eri_ijkl_given_kl_shell_slices(
                mol, id_k, id_l, shell_slices
            )
            _, _, shape2, shape3 = eri_block.shape
            eri_block = eri_block.reshape(
                nao * nao, eri_block.shape[2] * eri_block.shape[3]
            )

            k_start, k_end = shell_slices[id_k]
            l_start, l_end = shell_slices[id_l]

            # print(eri_block.shape)

            eri_ppkl[
                :, ao_loc[k_start] : ao_loc[k_end], ao_loc[l_start] : ao_loc[l_end]
            ] = lib.ddot(ao_pair_2_ao.T, eri_block).reshape(nao, shape2, shape3)

            # np.einsum("pq,prs->qrs", ao_pair_2_ao, eri_block)

            # eri_ppkl[
            #     :, ao_loc[l_start] : ao_loc[l_end], ao_loc[k_start] : ao_loc[k_end]
            # ] = eri_ppkl[
            #     :, ao_loc[k_start] : ao_loc[k_end], ao_loc[l_start] : ao_loc[l_end]
            # ].transpose(
            #     0, 2, 1
            # )

            processed_pairs += 1
            total_pairs += 1

            # 输出进度
            if verbose > 0 and processed_pairs % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"已处理 {processed_pairs} 个 shell slices 对，用时 {elapsed:.2f}s"
                )

    # the last two indices #

    # eri_ppkl = np.einsum("pqr,qs->psr", eri_ppkl, mo_coeff)
    # eri_ppkl = np.einsum("psr,rt->pst", eri_ppkl, mo_coeff)

    eri_ppkl_out = np.zeros(eri_ppkl.shape)
    orb_slices = (0, nao, 0, nao)

    nr_e2(eri_ppkl, mo_coeff, orb_slices, out=eri_ppkl_out)
    eri_ppkl = eri_ppkl_out

    # print the final result
    elapsed = time.time() - start_time
    if verbose > 0:
        print(f"已处理 {processed_pairs} 个 shell slices 对，用时 {elapsed:.2f}s")
        print("-" * 64)

    return eri_ppkl


def generate_eri_prps(
    mol: pyscf.gto.Mole, mo_coeff, ao_block_size: int = 64, verbose: int = 1
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    按壳层生成 (*k|*l) 电子排斥积分

    参数:
        mol: PySCF 分子对象
        compact: 是否使用紧凑存储格式
        verbose: 输出详细程度

    生成:
        (k_shell, l_shell, eri_block): 壳层索引和对应的积分块
    """

    if verbose > 0:
        print("-" * 64)
        print("开始按壳层生成 转积分 ikjl,ip,jp,kr,ls --> prps")
        print(f"分子       : {mol.atom}")
        print(f"基组       : {mol.basis}")
        print(f"总壳层数   : {mol.nbas}")
        print(f"总基函数数 : {mol.nao}")

    # 获取壳层信息

    shell_slices = get_shell_slices(mol, ao_block_size)

    # print(shell_slices)

    # total_shells = mol.nbas
    nao = mol.nao

    # 统计信息
    total_pairs = 0
    processed_pairs = 0
    start_time = time.time()

    ao_pair_2_ao = np.einsum("ip,jp->ijp", mo_coeff, mo_coeff)
    ao_pair_2_ao = ao_pair_2_ao.reshape(nao * nao, nao)

    ao_loc = mol.ao_loc_nr()

    # print(ao_loc)

    eri_pkpl = np.zeros((nao, nao, nao))

    # 遍历所有 (k,l) 壳层对
    for id_k in range(len(shell_slices)):
        for id_l in range(len(shell_slices)):

            eri_block = _get_eri_ikjl_given_kl_shell_slices(
                mol, id_k, id_l, shell_slices
            )
            eri_block = eri_block.transpose(0, 2, 1, 3)
            _, _, shape2, shape3 = eri_block.shape
            eri_block = eri_block.reshape(
                nao * nao, eri_block.shape[2] * eri_block.shape[3]
            )

            k_start, k_end = shell_slices[id_k]
            l_start, l_end = shell_slices[id_l]

            # print(eri_block.shape)

            eri_pkpl[
                :, ao_loc[k_start] : ao_loc[k_end], ao_loc[l_start] : ao_loc[l_end]
            ] = lib.ddot(ao_pair_2_ao.T, eri_block).reshape(nao, shape2, shape3)

            # np.einsum("pq,prs->qrs", ao_pair_2_ao, eri_block)

            # eri_pkpl[
            #     :, ao_loc[l_start] : ao_loc[l_end], ao_loc[k_start] : ao_loc[k_end]
            # ] = eri_pkpl[
            #     :, ao_loc[k_start] : ao_loc[k_end], ao_loc[l_start] : ao_loc[l_end]
            # ].transpose(
            #     0, 2, 1
            # )

            processed_pairs += 1
            total_pairs += 1

            # 输出进度
            if verbose > 0 and processed_pairs % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"已处理 {processed_pairs} 个 shell slices 对，用时 {elapsed:.2f}s"
                )

    # the last two indices #

    # eri_pkpl = np.einsum("pqr,qs->psr", eri_pkpl, mo_coeff)
    # eri_pkpl = np.einsum("psr,rt->pst", eri_pkpl, mo_coeff)

    eri_pkpl_out = np.zeros(eri_pkpl.shape)
    orb_slices = (0, nao, 0, nao)

    nr_e2(eri_pkpl, mo_coeff, orb_slices, out=eri_pkpl_out)
    eri_pkpl = eri_pkpl_out

    # print the final result
    elapsed = time.time() - start_time

    if verbose > 0:
        print(f"已处理 {processed_pairs} 个 shell slices 对，用时 {elapsed:.2f}s")
        print("-" * 64)

    return eri_pkpl


def _get_shell_ao_range(mol: pyscf.gto.Mole, shell_id: int) -> Tuple[int, int]:
    """
    获取指定壳层的 AO 范围

    参数:
        mol: 分子对象
        shell_id: 壳层ID

    返回:
        (start_ao, end_ao): AO 起始和结束索引
    """
    ao_loc = mol.ao_loc_nr()
    return ao_loc[shell_id], ao_loc[shell_id + 1]


def _ao_to_shell(mol: pyscf.gto.Mole, ao_id: int) -> int:
    """
    将 AO 索引转换为壳层索引

    参数:
        mol: 分子对象
        ao_id: AO 索引

    返回:
        shell_id: 壳层索引
    """
    ao_loc = mol.ao_loc_nr()
    for shell_id in range(mol.nbas):
        if ao_loc[shell_id] <= ao_id < ao_loc[shell_id + 1]:
            return shell_id
    raise ValueError(f"无效的 AO 索引: {ao_id}")


def analyze_shell_structure(mol: pyscf.gto.Mole):
    """
    分析分子的壳层结构

    参数:
        mol: PySCF 分子对象
    """
    print("=== 壳层结构分析 ===")
    print(f"总原子数:   {mol.natm}")
    print(f"总壳层数:   {mol.nbas}")
    print(f"总基函数数: {mol.nao}")

    ao_loc = mol.ao_loc_nr()

    print("\n壳层详细信息:")
    print("-" * 64)
    print(
        f"{'壳层ID':<6} {'原子':<4} {'角动量':<6} {'收缩数':<6} {'AO范围':<12} {'AO数量':<6}"
    )
    print("-" * 64)

    for shell_id in range(mol.nbas):
        atom_id = mol.bas_atom(shell_id)
        angular = mol.bas_angular(shell_id)
        nctr = mol.bas_nctr(shell_id)
        ao_start, ao_end = _get_shell_ao_range(mol, shell_id)
        nao_shell = ao_end - ao_start

        print(
            f"{shell_id:>6} {atom_id:>4} {angular:>6} {nctr:>6}       "
            f"[{ao_start:>5}:{ao_end:>6}] {nao_shell:>6}"
        )

    print(f"\n总的 (k,l) 壳层对数: {mol.nbas * (mol.nbas + 1) // 2}")
    print(f"总的积分数量估计: {mol.nao**4}")


# 示例用法和测试
def main():
    """主函数 - 示例用法"""

    # 创建测试分子 (水分子)
    mol = pyscf.gto.Mole()
    mol.atom = """
    O  0.0000  0.0000  0.0000
    H  0.7571  0.0000  0.5861
    H -0.7571  0.0000  0.5861
    """
    mol.basis = "cc-pvqz"
    mol.verbose = 1
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mo_coeff = mf.mo_coeff

    # 分析壳层结构
    analyze_shell_structure(mol)

    print("\n=== 测试壳层积分生成 ===")

    res = mol.intor("int2e", shls_slice=(0, 1, 0, 1, 1, 2, 1, 2))
    print(res.shape)

    shell_slices = get_shell_slices(mol, 4)
    print(shell_slices)

    generate_eri_pprs(mol, mo_coeff, ao_block_size=16, verbose=2)

    generate_eri_prps(mol, mo_coeff, ao_block_size=16, verbose=2)

    exit(1)

    # 测试生成几个壳层对的积分
    count = 0
    for k_shell, l_shell, eri_block in generate_eri_shell_optimized(
        mol, compact=True, verbose=2
    ):
        print(f"壳层对 ({k_shell}, {l_shell}): 积分块形状 {eri_block.shape}")
        print(f"  最大值: {np.max(np.abs(eri_block)):.8f}")
        print(f"  非零元素数: {np.count_nonzero(eri_block)}")

        count += 1
        if count >= 5:  # 只显示前5个壳层对
            break

    # 保存积分示例
    print("\n=== 保存积分示例 ===")
    save_shell_integrals(mol, output_prefix="test_eri", save_format="npz")


if __name__ == "__main__":
    main()
