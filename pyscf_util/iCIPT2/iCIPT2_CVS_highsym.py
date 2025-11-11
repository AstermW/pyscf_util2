from pyscf_util.iCIPT2.iCIPT2 import temporary_update, _FILE_NOT_REMOVE
from pyscf_util.misc.icipt2_inputfile_generator import _Generate_InputFile_iCI_CVS
import os

from pyscf_util.iCIPT2.iCIPT2_CVS import _iCIPT2_CVS_Driver, _load_app


# DRIVER High Sym #


ICIPT2_CSF_CVS_DOOH_DRIVER = _iCIPT2_CVS_Driver(
    _load_app("ICI_CSF_DOOH_CVS"), _FILE_NOT_REMOVE
)

ICIPT2_CSF_CVS_COOV_DRIVER = _iCIPT2_CVS_Driver(
    _load_app("ICI_CSF_COOV_CVS"), _FILE_NOT_REMOVE
)


def kernel_coov(
    IsCSF: bool,
    task_name,
    fcidump,
    segment,
    nelec_val,
    rotatemo=0,
    cmin: str | float = 1e-4,
    perturbation=0,
    dumprdm=0,
    relative=0,
    Task: str = None,
    inputocfg=0,
    etol=1e-7,
    selection=1,
    doublegroup=None,
    direct=None,
    start_with=None,
    end_with=None,
    relaxcore=None,
):
    if IsCSF:
        ICIPT2_CSF_CVS_COOV_DRIVER.run(
            task_name,
            fcidump,
            segment,
            nelec_val,
            rotatemo,
            cmin,
            perturbation,
            dumprdm,
            relative,
            Task,
            inputocfg,
            etol,
            selection,
            doublegroup,
            direct,
            start_with,
            end_with,
            relaxcore,
        )
    else:
        print("WARNING: DET is not supported in CVS yet!")
        ICIPT2_CSF_CVS_COOV_DRIVER.run(
            task_name,
            fcidump,
            segment,
            nelec_val,
            rotatemo,
            cmin,
            perturbation,
            dumprdm,
            relative,
            Task,
            inputocfg,
            etol,
            selection,
            doublegroup,
            direct,
            start_with,
            end_with,
            relaxcore,
        )


def kernel_dooh(
    IsCSF: bool,
    task_name,
    fcidump,
    segment,
    nelec_val,
    rotatemo=0,
    cmin: str | float = 1e-4,
    perturbation=0,
    dumprdm=0,
    relative=0,
    Task: str = None,
    inputocfg=0,
    etol=1e-7,
    selection=1,
    doublegroup=None,
    direct=None,
    start_with=None,
    end_with=None,
    relaxcore=None,
):
    if IsCSF:
        ICIPT2_CSF_CVS_DOOH_DRIVER.run(
            task_name,
            fcidump,
            segment,
            nelec_val,
            rotatemo,
            cmin,
            perturbation,
            dumprdm,
            relative,
            Task,
            inputocfg,
            etol,
            selection,
            doublegroup,
            direct,
            start_with,
            end_with,
            relaxcore,
        )
    else:
        print("WARNING: DET is not supported in CVS yet!")
        ICIPT2_CSF_CVS_DOOH_DRIVER.run(
            task_name,
            fcidump,
            segment,
            nelec_val,
            rotatemo,
            cmin,
            perturbation,
            dumprdm,
            relative,
            Task,
            inputocfg,
            etol,
            selection,
            doublegroup,
            direct,
            start_with,
            end_with,
            relaxcore,
        )
