import pyFuzzyME as fme


def SysEq(x):
    BraidMEM = fme.BraidME()
    BraidMEM.CreateAndSaveBraidFIS()
    [ME, dMEdx, Reasoning] = BraidMEM.ComputeRespAndSens(x)
    return(ME)


x = [25.0, 2.7, 10, 5.0, 2.0, 5.0, 0.0]
Effort = SysEq(x)
print(Effort)
