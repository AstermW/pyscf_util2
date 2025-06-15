#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 .casorb 文件的脚本
识别 SYM= 行并读取接下来的轨道系数数组
同时读取 ORBITAL ENERGY 和 OCCUPATION 数据

作者: 自动生成
日期: 2024
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional


class BDFOrbParser:
    """解析 .casorb 文件的类"""

    def __init__(self, filename: str):
        """
        初始化解析器

        参数:
            filename: .casorb 文件路径
        """
        self.filename = filename
        self.sym_blocks = {}
        self.orbital_energies = None
        self.occupations = None
        self.sym_orbital_energies = {}
        self.sym_occupations = {}

    def parse_file(self, verbose=False) -> Dict[int, Dict]:
        """
        解析整个文件

        返回:
            字典，键为SYM编号，值为包含轨道信息的字典
        """
        with open(self.filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 匹配 SYM= 行的模式
            sym_match = re.match(r"SYM=\s*(\d+)\s+NORB=\s*(\d+)\s+(ALPHA|BETA)", line)

            if sym_match:
                sym_num = (
                    int(sym_match.group(1)) - 1
                )  # BDF is 1-based but python is 0-based
                norb = int(sym_match.group(2))
                spin_type = sym_match.group(3)

                if verbose:
                    print(f"找到 SYM={sym_num}, NORB={norb}, {spin_type}")

                # 读取接下来的数组数据
                orbital_data, lines_read = self._read_orbital_data(lines, i + 1, norb)

                # 存储数据
                if sym_num not in self.sym_blocks:
                    self.sym_blocks[sym_num] = {}

                self.sym_blocks[sym_num][spin_type] = {
                    "norb": norb,
                    "data": orbital_data,
                    "shape": orbital_data.shape if orbital_data is not None else None,
                }

                i += lines_read + 1

            # 匹配 ORBITAL ENERGY 行
            elif line == "ORBITAL ENERGY":
                if verbose:
                    print("找到 ORBITAL ENERGY 数据")
                self.orbital_energies, lines_read = self._read_energy_occupation_data(
                    lines, i + 1
                )
                i += lines_read + 1

            # 匹配 OCCUPATION 行
            elif line == "OCCUPATION":
                if verbose:
                    print("找到 OCCUPATION 数据")
                self.occupations, lines_read = self._read_energy_occupation_data(
                    lines, i + 1
                )
                i += lines_read + 1

            else:
                i += 1

        # 根据对称性切分轨道能和占据数
        self._split_energies_and_occupations(verbose)

        return self.sym_blocks

    def _read_orbital_data(
        self, lines: List[str], start_idx: int, norb: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        读取轨道数据数组

        参数:
            lines: 文件所有行的列表
            start_idx: 开始读取的行索引
            norb: 轨道数量

        返回:
            (数据数组, 读取的行数)
        """
        data_lines = []
        lines_read = 0

        i = start_idx
        while i < len(lines):
            line = lines[i].strip()

            # 如果遇到空行或新的SYM行，停止读取
            if (
                not line
                or line.startswith("SYM=")
                or line.startswith("$")
                or line.startswith("TITLE")
                or line == "ORBITAL ENERGY"
            ):
                break

            # 尝试解析数值
            try:
                # 分割行并转换为浮点数
                numbers = []
                parts = line.split()
                for part in parts:
                    # 处理科学计数法格式，如 0.4562980396185539E+00
                    if "E" in part or "e" in part:
                        numbers.append(float(part))
                    else:
                        # 尝试转换为浮点数
                        try:
                            numbers.append(float(part))
                        except ValueError:
                            # 如果不是数字，可能是其他内容，停止读取
                            break

                if numbers:
                    data_lines.extend(numbers)
                    lines_read += 1
                else:
                    break

            except (ValueError, IndexError):
                # 如果解析失败，停止读取
                break

            i += 1

        # 将数据转换为numpy数组
        if data_lines:
            try:
                data_array = np.array(data_lines, dtype=np.float64)
                # 重新整形为 norb x (数据长度/norb) 的矩阵
                if len(data_array) % norb == 0:
                    cols = len(data_array) // norb
                    assert cols == norb
                    data_array = data_array.reshape(norb, cols)
                return data_array.T, lines_read
            except Exception as e:
                print(f"警告：数据整形失败: {e}")
                return np.array(data_lines), lines_read

        return None, lines_read

    def _read_energy_occupation_data(
        self, lines: List[str], start_idx: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        读取轨道能或占据数数据

        参数:
            lines: 文件所有行的列表
            start_idx: 开始读取的行索引

        返回:
            (数据数组, 读取的行数)
        """
        data_lines = []
        lines_read = 0

        i = start_idx
        while i < len(lines):
            line = lines[i].strip()

            # 如果遇到空行、$END 或其他关键字，停止读取
            if (
                not line
                or line.startswith("$")
                or line == "OCCUPATION"
                or line == "ORBITAL ENERGY"
            ):
                break

            # 尝试解析数值
            try:
                # 分割行并转换为浮点数
                numbers = []
                parts = line.split()
                for part in parts:
                    # 处理科学计数法格式
                    if "E" in part or "e" in part:
                        numbers.append(float(part))
                    else:
                        # 尝试转换为浮点数
                        try:
                            numbers.append(float(part))
                        except ValueError:
                            # 如果不是数字，停止读取
                            break

                if numbers:
                    data_lines.extend(numbers)
                    lines_read += 1
                else:
                    break

            except (ValueError, IndexError):
                # 如果解析失败，停止读取
                break

            i += 1

        # 将数据转换为numpy数组
        if data_lines:
            return np.array(data_lines, dtype=np.float64), lines_read

        return None, lines_read

    def _split_energies_and_occupations(self, verbose=False):
        """
        根据每个对称性的NORB将轨道能和占据数切块
        """
        if self.orbital_energies is None or self.occupations is None:
            print("警告：轨道能或占据数数据缺失")
            return

        # 获取所有对称性的NORB信息，按SYM编号排序
        sym_norbs = []
        for sym_num in sorted(self.sym_blocks.keys()):
            for spin_type in self.sym_blocks[sym_num]:
                norb = self.sym_blocks[sym_num][spin_type]["norb"]
                sym_norbs.append((sym_num, spin_type, norb))

        # 切分轨道能和占据数
        start_idx = 0
        for sym_num, spin_type, norb in sym_norbs:
            end_idx = start_idx + norb

            if end_idx <= len(self.orbital_energies):
                # 提取对应的轨道能
                sym_energies = self.orbital_energies[start_idx:end_idx]

                # 存储到对应的对称性块中
                if sym_num not in self.sym_orbital_energies:
                    self.sym_orbital_energies[sym_num] = {}
                self.sym_orbital_energies[sym_num][spin_type] = sym_energies

                if verbose:
                    print(
                        f"SYM={sym_num} {spin_type}: 轨道能范围 [{sym_energies.min():.6f}, {sym_energies.max():.6f}]"
                    )

            if end_idx <= len(self.occupations):
                # 提取对应的占据数
                sym_occs = self.occupations[start_idx:end_idx]

                # 存储到对应的对称性块中
                if sym_num not in self.sym_occupations:
                    self.sym_occupations[sym_num] = {}
                self.sym_occupations[sym_num][spin_type] = sym_occs

                occupied_count = np.sum(sym_occs > 1e-6)  # 占据轨道数量
                if verbose:
                    print(
                        f"SYM={sym_num} {spin_type}: 占据轨道数 {occupied_count}/{norb}"
                    )

            start_idx = end_idx

    def print_summary(self):
        """打印解析结果摘要"""
        print("\n=== 解析结果摘要 ===")
        for sym_num in sorted(self.sym_blocks.keys()):
            print(f"\nSYM = {sym_num}:")
            for spin_type, data in self.sym_blocks[sym_num].items():
                print(f"  {spin_type}:")
                print(f"    NORB: {data['norb']}")
                print(f"    轨道系数形状: {data['shape']}")
                # if data["data"] is not None:
                #     print(
                #         f"    轨道系数范围: [{data['data'].min():.6f}, {data['data'].max():.6f}]"
                #     )

                # 打印轨道能信息
                if (
                    sym_num in self.sym_orbital_energies
                    and spin_type in self.sym_orbital_energies[sym_num]
                ):
                    energies = self.sym_orbital_energies[sym_num][spin_type]
                    print(
                        f"    轨道能范围: [{energies.min():.6f}, {energies.max():.6f}] Hartree"
                    )

                # 打印占据数信息
                if (
                    sym_num in self.sym_occupations
                    and spin_type in self.sym_occupations[sym_num]
                ):
                    occs = self.sym_occupations[sym_num][spin_type]
                    occupied_count = np.sum(occs > 1e-6)
                    total_electrons = np.sum(occs)
                    print(f"    占据轨道数: {occupied_count}/{len(occs)}")
                    print(f"    总电子数: {total_electrons:.6f}")

    def get_sym_data(
        self, sym_num: int, spin_type: str = "ALPHA"
    ) -> Optional[np.ndarray]:
        """
        获取指定SYM和自旋类型的轨道系数数据

        参数:
            sym_num: SYM编号
            spin_type: 'ALPHA' 或 'BETA'

        返回:
            数据数组或None
        """
        if sym_num in self.sym_blocks and spin_type in self.sym_blocks[sym_num]:
            return self.sym_blocks[sym_num][spin_type]["data"]
        return None

    def get_sym_energies(
        self, sym_num: int, spin_type: str = "ALPHA"
    ) -> Optional[np.ndarray]:
        """
        获取指定SYM和自旋类型的轨道能数据

        参数:
            sym_num: SYM编号
            spin_type: 'ALPHA' 或 'BETA'

        返回:
            轨道能数组或None
        """
        if (
            sym_num in self.sym_orbital_energies
            and spin_type in self.sym_orbital_energies[sym_num]
        ):
            return self.sym_orbital_energies[sym_num][spin_type]
        return None

    def get_sym_occupations(
        self, sym_num: int, spin_type: str = "ALPHA"
    ) -> Optional[np.ndarray]:
        """
        获取指定SYM和自旋类型的占据数数据

        参数:
            sym_num: SYM编号
            spin_type: 'ALPHA' 或 'BETA'

        返回:
            占据数数组或None
        """
        if (
            sym_num in self.sym_occupations
            and spin_type in self.sym_occupations[sym_num]
        ):
            return self.sym_occupations[sym_num][spin_type]
        return None

    def save_sym_data(
        self, sym_num: int, spin_type: str = "ALPHA", output_prefix: str = None
    ):
        """
        保存指定SYM的所有数据到文件

        参数:
            sym_num: SYM编号
            spin_type: 'ALPHA' 或 'BETA'
            output_prefix: 输出文件前缀，如果为None则自动生成
        """
        if output_prefix is None:
            output_prefix = f"sym_{sym_num}_{spin_type.lower()}"

        # 保存轨道系数
        coeff_data = self.get_sym_data(sym_num, spin_type)
        if coeff_data is not None:
            coeff_file = f"{output_prefix}_coefficients.txt"
            np.savetxt(coeff_file, coeff_data, fmt="%.12e")
            print(f"轨道系数已保存到: {coeff_file}")

        # 保存轨道能
        energy_data = self.get_sym_energies(sym_num, spin_type)
        if energy_data is not None:
            energy_file = f"{output_prefix}_energies.txt"
            np.savetxt(energy_file, energy_data, fmt="%.12e")
            print(f"轨道能已保存到: {energy_file}")

        # 保存占据数
        occ_data = self.get_sym_occupations(sym_num, spin_type)
        if occ_data is not None:
            occ_file = f"{output_prefix}_occupations.txt"
            np.savetxt(occ_file, occ_data, fmt="%.12e")
            print(f"占据数已保存到: {occ_file}")

        if coeff_data is None and energy_data is None and occ_data is None:
            print(f"未找到 SYM={sym_num}, {spin_type} 的数据")

    def print_orbital_analysis(self, sym_num: int, spin_type: str = "ALPHA"):
        """
        打印指定对称性的轨道分析

        参数:
            sym_num: SYM编号
            spin_type: 'ALPHA' 或 'BETA'
        """
        energies = self.get_sym_energies(sym_num, spin_type)
        occupations = self.get_sym_occupations(sym_num, spin_type)

        if energies is None or occupations is None:
            print(f"SYM={sym_num} {spin_type} 的轨道能或占据数数据缺失")
            return

        print(f"\n=== SYM={sym_num} {spin_type} 轨道分析 ===")
        print(
            f"{'轨道编号':<8} {'轨道能(Hartree)':<15} {'轨道能(eV)':<12} {'占据数':<10} {'状态'}"
        )
        print("-" * 60)

        for i, (energy, occ) in enumerate(zip(energies, occupations)):
            energy_ev = energy * 27.2114  # Hartree to eV
            status = "占据" if occ > 1e-6 else "虚轨道"
            if 1e-6 < occ < 1.99:
                status = "部分占据"

            print(f"{i+1:<8} {energy:<15.6f} {energy_ev:<12.3f} {occ:<10.6f} {status}")

    def BDFold_2_pyscf(self):
        if len(self.sym_blocks) >= 4:
            # switch self.sym_blocks[1] and self.sym_blocks[2]
            self.sym_blocks[1], self.sym_blocks[2] = (
                self.sym_blocks[2],
                self.sym_blocks[1],
            )
            # switch self.sym_blocks[1] and self.sym_blocks[3]
            self.sym_blocks[1], self.sym_blocks[3] = (
                self.sym_blocks[3],
                self.sym_blocks[1],
            )
            # do the same thing for self.sym_orbital_energies and self.sym_occupations
            self.sym_orbital_energies[1], self.sym_orbital_energies[2] = (
                self.sym_orbital_energies[2],
                self.sym_orbital_energies[1],
            )
            self.sym_orbital_energies[1], self.sym_orbital_energies[3] = (
                self.sym_orbital_energies[3],
                self.sym_orbital_energies[1],
            )
            self.sym_occupations[1], self.sym_occupations[2] = (
                self.sym_occupations[2],
                self.sym_occupations[1],
            )
            self.sym_occupations[1], self.sym_occupations[3] = (
                self.sym_occupations[3],
                self.sym_occupations[1],
            )

        if len(self.sym_blocks) >= 8:
            # switch self.sym_blocks[5] and self.sym_blocks[7]
            self.sym_blocks[5], self.sym_blocks[7] = (
                self.sym_blocks[7],
                self.sym_blocks[5],
            )
            # switch self.sym_blocks[6] and self.sym_blocks[7]
            self.sym_blocks[6], self.sym_blocks[7] = (
                self.sym_blocks[7],
                self.sym_blocks[6],
            )
            self.sym_orbital_energies[5], self.sym_orbital_energies[7] = (
                self.sym_orbital_energies[7],
                self.sym_orbital_energies[5],
            )
            self.sym_orbital_energies[6], self.sym_orbital_energies[7] = (
                self.sym_orbital_energies[7],
                self.sym_orbital_energies[6],
            )
            self.sym_occupations[5], self.sym_occupations[7] = (
                self.sym_occupations[7],
                self.sym_occupations[5],
            )
            self.sym_occupations[6], self.sym_occupations[7] = (
                self.sym_occupations[7],
                self.sym_occupations[6],
            )

    def BDFold_2_new(self):
        """
        2 3 互换
        6 7 互换
        """
        if len(self.sym_blocks) >= 4:
            # switch self.sym_blocks[2] and self.sym_blocks[3]
            self.sym_blocks[2], self.sym_blocks[3] = (
                self.sym_blocks[3],
                self.sym_blocks[2],
            )
        if len(self.sym_blocks) >= 8:
            # switch self.sym_blocks[6] and self.sym_blocks[7]
            self.sym_blocks[6], self.sym_blocks[7] = (
                self.sym_blocks[7],
                self.sym_blocks[6],
            )

    def BDFnew_2_pyscf(self):
        self.BDFold_2_new()
        self.BDFold_2_pyscf()


def main():
    """主函数 - 示例用法"""
    # 使用示例
    casorb_file = "./02S.casorb.old"  # 修改为你的文件路径

    try:
        # 创建解析器
        parser = BDFOrbParser(casorb_file)

        # 解析文件
        print("开始解析文件...")
        sym_blocks = parser.parse_file()

        # 打印摘要
        parser.print_summary()

        # switch convention and print again #

        parser.BDFold_2_pyscf()
        parser.print_summary()

        # 示例：获取特定SYM的数据
        print("\n=== 示例：获取SYM=2的ALPHA数据 ===")
        sym2_data = parser.get_sym_data(2, "ALPHA")
        sym2_energies = parser.get_sym_energies(2, "ALPHA")
        sym2_occs = parser.get_sym_occupations(2, "ALPHA")

        if sym2_data is not None:
            print(f"SYM=2 ALPHA 轨道系数形状: {sym2_data.shape}")
            print("前3行轨道系数:")
            print(sym2_data[:3])

        if sym2_energies is not None:
            print(f"SYM=2 ALPHA 轨道能: {sym2_energies}")

        if sym2_occs is not None:
            print(f"SYM=2 ALPHA 占据数: {sym2_occs}")

        # 保存数据到文件
        parser.save_sym_data(2, "ALPHA", "sym2_alpha_complete")

        # 轨道分析
        parser.print_orbital_analysis(2, "ALPHA")

        # 示例：遍历所有SYM块
        print("\n=== 所有SYM块的详细信息 ===")
        for sym_num in sorted(sym_blocks.keys()):
            for spin_type in sym_blocks[sym_num]:
                data = sym_blocks[sym_num][spin_type]["data"]
                norb = sym_blocks[sym_num][spin_type]["norb"]
                energies = parser.get_sym_energies(sym_num, spin_type)
                occs = parser.get_sym_occupations(sym_num, spin_type)

                print(f"SYM={sym_num} {spin_type}: NORB={norb}")
                print(f"  轨道系数数据点数: {data.size if data is not None else 0}")
                print(
                    f"  轨道能数据点数: {len(energies) if energies is not None else 0}"
                )
                print(f"  占据数数据点数: {len(occs) if occs is not None else 0}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {casorb_file}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"解析过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
