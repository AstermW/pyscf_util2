from math import sqrt
import numpy as np


def parse_bdf_chkfil(file_path, int_size=4, maxrecord=1024):
    """
    解析BDF的chkfil文件。

    参数:
    - file_path: 二进制文件路径
    - int_size: 整数大小（4或8字节）
    - maxrecord: 最大记录数（默认1024）

    返回:
    - 读取的整数数组和字符数组
    """
    import struct

    # 打开二进制文件
    with open(file_path, "rb") as f:
        # 读取整数数组
        int_format = "i" if int_size == 4 else "q"
        int_array = struct.unpack(
            f"{maxrecord}{int_format}", f.read(maxrecord * int_size)
        )

        # 读取字符数组
        char_array = f.read(maxrecord * 8).decode("utf-8")

    return int_array, char_array


def search_keyword_in_chkfil(int_array, char_array, keyword):
    """
    在chkfil中搜索特定关键字。

    参数:
    - int_array: 整数数组
    - char_array: 字符数组
    - keyword: 要搜索的关键字，长度必须为8

    返回:
    - 关键字在整数数组中的位置，如果未找到则返回-1
    """
    if len(keyword) != 8:
        raise ValueError("关键字长度必须为8")

    keyword = keyword.upper()
    loc = char_array.find(keyword)

    if loc == -1:
        return None

    loc = loc // 8

    begin = int_array[loc]
    end = int_array[loc + 1]

    return begin, end


def read_ao2somat_from_chkfil(file_path, int_size=8, maxrecord=1024):
    import struct

    int_array, char_array = parse_bdf_chkfil(file_path, int_size, maxrecord)
    loc = search_keyword_in_chkfil(int_array, char_array, "AO2SOMAT")
    if loc is None:
        raise ValueError("未找到AO2SOMAT关键字")
    begin, end = loc
    print(begin, end)
    nbas = int(sqrt((end - begin) // 8))
    data = np.fromfile(
        file_path, dtype=np.float64, count=(end - begin) // 8, offset=begin
    )
    ao2somat = data.reshape((nbas, nbas))
    return ao2somat.T


def ao2somat_split_based_on_irrep(ao2somat, Mol):

    from pyscf_util.misc.mole import get_orbsym

    irreps = Mol.irrep_name

    res = {}

    loc_now = 0

    for irrep in range(len(irreps)):
        nbas_tmp = Mol.symm_orb[irrep].shape[1]
        res[irrep] = ao2somat[:, loc_now : loc_now + nbas_tmp]
        loc_now += nbas_tmp

    print(res)

    return res


if __name__ == "__main__":

    import pyscf

    ao2somat = read_ao2somat_from_chkfil("02S.chkfil")

    ### check whether it is compatible with pyscf

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

    mf = pyscf.scf.RHF(Mol)
    symm_ao2somat = ao2somat_split_based_on_irrep(ao2somat, Mol)

    # read in casorb file of bdf #

    from _parse_bdf_orbfile import BDFOrbParser

    # parser = BDFOrbParser("02S.casorb.old")  # 读的是一个旧的 casorb 文件
    parser = BDFOrbParser("bdf_test/02S.scforb")  # 读的是一个新的 scforb 文件
    parser.parse_file()
    # parser.BDFold_2_pyscf()
    # parser.BDFnew_2_pyscf()

    ovlp_ao = Mol.intor("int1e_ovlp")

    mo_coeffs = []
    energies = []
    occupancies = []

    for irrep in range(len(Mol.irrep_name)):
        # print(irrep)
        assert symm_ao2somat[irrep].shape[1] == parser.get_sym_data(irrep).shape[1]
        mo_coeff_tmp = symm_ao2somat[irrep] @ parser.get_sym_data(irrep)
        mo_coeffs.append(mo_coeff_tmp)
        energies.append(parser.get_sym_energies(irrep))
        occupancies.append(parser.get_sym_occupations(irrep))

    mo_coeffs = np.hstack(mo_coeffs)
    energies = np.hstack(energies)
    occupancies = np.hstack(occupancies)

    print(mo_coeffs.shape)
    print(energies.shape)
    print(occupancies.shape)

    print(energies)
    print(occupancies)

    from pyscf_util.misc.dump_to_bdforb import dump_to_scforb

    dump_to_scforb(Mol, mo_coeffs, energies, occupancies, "02S.scforb.nosymm")
