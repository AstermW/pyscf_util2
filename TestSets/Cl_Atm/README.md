Procedure:

1. download the experimental branch of iCIPT2
2. compile iCIPT2
3. set the env 
   e.g. 
export ICI_CPP        =yourpath/iCIPT2_CXX/bin/iCI_CPP_NEW.exe
export ICI_CSF_CPP    =yourpath/iCIPT2_CXX/bin/iCI_CPP_NEW.exe
export ICI_DET_CPP    =yourpath/iCIPT2_CXX/bin/iCIPT2_D2h_Det.exe
export ICI_DET_COOV   =yourpath/iCIPT2_CXX/bin/iCIPT2_Coov_Det.exe
export ICI_DET_DOOH   =yourpath/iCIPT2_CXX/bin/iCIPT2_Dooh_Det.exe
export ICI_CSF_COOV   =yourpath/iCIPT2_CXX/bin/iCIPT2_Coov_CSF.exe
export ICI_CSF_DOOH   =yourpath/iCIPT2_CXX/bin/iCIPT2_Dooh_CSF.exe
export ICI_CSF_CVS_CPP=yourpath/iCIPT2_CXX/bin/iCIPT2_D2h_CSF_CoreExt.exe
export ICI_4C_D2H_COULOMB =yourpath/iCIPT2_CXX/bin/iCIPT2_Spinor_D2h_Coulomb.exe
export ICI_4C_D2H_BREIT   =yourpath/iCIPT2_CXX/bin/iCIPT2_Spinor_D2h.exe
export ICI_4C_COULOMB     =yourpath/iCIPT2_CXX/bin/iCIPT2_Spinor_Coulomb.exe
export ICI_4C_BREIT       =yourpath/iCIPT2_CXX/bin/iCIPT2_Spinor.exe
4. run 00-integrals.py to generate  the integrals.
5. run the driver