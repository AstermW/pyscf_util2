# coding=UTF-8
from pyscf import tools
from pyscf import mcscf, gto, scf

from pyscf_util.MeanField.iciscf import iCI

BASIS = "ccpvtz"

cas_space_symmetry = {
    "A1": 5,  # 5
    "E1x": 2,  # 7
    "E1y": 2,  # 3
}

mol = gto.M(
    verbose=4,
    atom="""
            C   0.000000000000       0.000000000000      -0.621265
            O   0.000000000000       0.000000000000       0.621265
            """,
    basis=BASIS,
    spin=0,
    charge=0,
    symmetry="coov",
    unit="angstrom",
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf.analyze()

# exit(1)

init_guess = mf.mo_coeff.copy()

norb = 9
nelec = 12
mymc2step = mcscf.CASSCF(mf, norb, nelec)

mo_init = mcscf.sort_mo_by_irrep(mymc2step, init_guess, cas_space_symmetry)  # right!

mymc2step.fcisolver = iCI(
    mol=mol,
    cmin=0.0,
    state=[[0, 10, 1], [0, 11, 1]],
    tol=1e-12,
    mo_coeff=mf.mo_coeff,
    taskname="iCI0",
    CVS=True,
)
mymc2step.fcisolver.config["segment"] = "0 0 1 4 4 0 0 0"
mymc2step.fcisolver.config["selection"] = 1
mymc2step.fcisolver.config["nvalelec"] = 12
mymc2step.mc1step()

exit(1)

# dump fcidump #

mf.mo_coeff = mymc2step.mo_coeff
tools.fcidump.from_scf(mf, "FCIDUMP_CO_C_Kedge_%s" % BASIS, tol=1e-10)

# O edge #

init_guess_O = init_guess.copy()
init_guess_O[:, 0] = init_guess[:, 1]
init_guess_O[:, 1] = init_guess[:, 0]

mf.mo_coeff = init_guess_O
mymc2step = mcscf.CASSCF(mf, norb, nelec)
mymc2step.fcisolver = iCI(
    mol=mol,
    cmin=0.0,
    state=[[0, 0, 1]],
    tol=1e-12,
    mo_coeff=mf.mo_coeff,
    taskname="iCI0",
    CVS=True,
)
mymc2step.fcisolver.config["segment"] = "0 0 1 4 4 0 0 0"
mymc2step.fcisolver.config["selection"] = 1
mymc2step.fcisolver.config["nvalelec"] = 12
mymc2step.mc1step()

# dump fcidump #

mf.mo_coeff = mymc2step.mo_coeff
tools.fcidump.from_scf(mf, "FCIDUMP_CO_O_Kedge_%s" % BASIS, tol=1e-10)
