from pyscf_util.iCIPT2.iCIPT2 import _iCIPT2_Driver, _load_app, _FILE_NOT_REMOVE

ICIPT2_4C_COULOMB_D2H_DRIVER = _iCIPT2_Driver(
    _load_app("ICI_4C_D2H_COULOMB"), _FILE_NOT_REMOVE
)
ICIPT2_4C_BREIT_D2H_DRIVER = _iCIPT2_Driver(
    _load_app("ICI_4C_D2H_BREIT"), _FILE_NOT_REMOVE
)


def kernel(
    HasBreit: bool,
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
    direct=None,
    start_with=None,
    end_with=None,
):
    if HasBreit:
        ICIPT2_4C_BREIT_D2H_DRIVER.run(
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
            None,
            direct,
            start_with,
            end_with,
        )
    else:
        ICIPT2_4C_COULOMB_D2H_DRIVER.run(
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
            None,
            direct,
            start_with,
            end_with,
        )
