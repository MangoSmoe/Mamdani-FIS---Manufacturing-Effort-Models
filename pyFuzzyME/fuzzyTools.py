#------------------------------------------------------------------#
#******************************************************************#
#            Fuzzy Base CLASSES: FuzzyTools.py                     #
#******************************************************************#
#------------------------------------------------------------------#
# Ersteller:     Markus Schatz                                     #
# Mail:          schatz@llb.mw.tum.de                              #
#------------------------------------------------------------------#
# Change-Log:                                                      #
# 2015-02-02     Definition of class:                              #
#                o First definition of class                       #
#------------------------------------------------------------------#
#------------------------------------------------------------------#
# Included classes are:                                            #
#    o FIS                                                         #
#      - Class for creating a FIS object                           #
#    o MembershipFunction                                          #
#      - Class for handling membership functions                   #
#    o FuzzyTools                                                  #
#      - Class for fuzzification, Rule eval and defuzzification    #
#    o Pimf                                                        #
#    o Zmf                                                         #
#    o Trimf                                                       #
#    o Smf                                                         #
#    o Const                                                       #
#    o Gaussmf                                                     #
#    o Gauss2mf                                                    #
#------------------------------------------------------------------#
#------------------------------------------------------------------#

# Imports
#------------------------------------------------------------------#
import numpy as np
import copy as cp
import math
import sys

# Definition of orthogonal projection
#------------------------------------------------------------------#
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, a, b],
                        [0, 0, -0.0001, zback]])

# Global function definitions
#------------------------------------------------------------------#
def SurfacePlotter(fig, ax, X, Y, Z, SubTitel, xLabel, yLabel, AlphaVal=1.):
    try:
        from matplotlib import cm
        from mpl_toolkits.mplot3d import proj3d
    except:
        logging.error('Matplotlib could not be imported!')
    surf = ax.plot_surface(X, Y, Z, alpha = AlphaVal, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    proj3d.persp_transformation = orthogonal_proj
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.tick_params(labelsize=20)
    if SubTitel:
        ax.set_title(SubTitel, fontsize=22)
    if xLabel:
        ax.set_xlabel(xLabel, fontsize=22)
    if yLabel:
        ax.set_ylabel(yLabel, fontsize=22)


def smfFCT(x, Para, Parb):
    if Para >= Parb:
        smf = int(x >= (Para+Parb)/2.0)
    else:
        smf = 0.0
        if (x <= Para):
            smf = 0.0
        if ((Para < x) & (x <= (Para+Parb)/2.0)):
            smf = 2.0*((x-Para)/(Parb-Para))**2
        if (((Para+Parb)/2.0 < x) & (x <= Parb)):
            smf = 1.0-2.0*((x-Parb)/(Parb-Para))**2
        if (Parb <= x):
            smf = 1.0
    return smf


def zmfFCT(x, Parc, Pard):
    if Parc >= Pard:
        zmf = int(x <= (Pard+Parc)/2.0)
    else:
        zmf = 0.0
        if x <= Parc:
            zmf = 1.0
        if ((Parc < x) & (x <= (Parc+Pard)/2)):
            zmf = 1.0-2.0*((x-Parc)/(Parc-Pard))**2
        if (((Parc+Pard)/2.0 < x) & (x <= Pard)):
            zmf = 2.0*((Pard-x)/(Parc-Pard))**2
        if (Pard <= x):
            zmf = 0.0
    return zmf

# Base class definition
#------------------------------------------------------------------#
class MembershipFunction(object): # Base class for all membership functions
    NumInfiSlice = 1e3

    def __init__(self):
        pass

    def CompArea(self, xL, xU, Ful):
        Area = 0.0
        xSample = np.linspace(xL, xU, num=self.NumInfiSlice)
        for ixSample in range(1, np.size(xSample)):
            xP = xSample[ixSample-1]
            xN = xSample[ixSample]
            yP = self(xP)
            if yP > Ful:
                yP = Ful
            yN = self(xN)
            if yN > Ful:
                yN = Ful
            Area += (xN-xP)*(yN+yP)*.5
        return Area

    def CompCoA(self, xL, xU, Ful):
        Area = 0.0
        CoG = 0.0
        xSample = np.linspace(xL, xU, num=self.NumInfiSlice)
        for ixSample in range(1, np.size(xSample)):
            xP = xSample[ixSample-1]
            xN = xSample[ixSample]
            yP = self(xP)
            if yP > Ful:
                yP = Ful
            yN = self(xN)
            if yN > Ful:
                yN = Ful
            Area += (xN-xP)*(yN+yP)*.5
            CoG += (xN-xP)*(yN+yP)*.5*((xN-xP)*.5+xP)
        return CoG/Area


class Gauss2mf(MembershipFunction):
    def __init__(self, Sig1, c1, Sig2, c2):
        self.Sig1 = Sig1+0.0
        self.c1 = c1+0.0
        self.Sig2 = Sig2+0.0
        self.c2 = c2+0.0

    def __call__(self, x):
        c1i = (x <= self.c1)
        c2i = (x >= self.c2)
        y1 = math.exp(-(x-self.c1)**2/(2.0*self.Sig1**2))*c1i+(1-c1i)
        y2 = math.exp(-(x-self.c2)**2/(2.0*self.Sig2**2))*c2i+(1-c2i)
        return y1*y2

    def __str__(self):
        return 'Combined gaussian mf'


class Gaussmf(MembershipFunction):
    def __init__(self, Sig, c):
        self.Sig = Sig+0.0
        self.c = c+0.0

    def __call__(self, x):
        return math.exp(-(x-self.c)**2/(2.0*self.Sig**2))

    def __str__(self):
        return 'Gaussian mf'


class Const(MembershipFunction):
    def __init__(self, c):
        self.c = c+0.0

    def __call__(self, x):
        return self.c

    def __str__(self):
        return 'Constant mf'


class Smf(MembershipFunction):
    def __init__(self, a, b):
        self.a = a+0.0
        self.b = b+0.0

    def __call__(self, x):
        return smfFCT(x, self.a, self.b)

    def __str__(self):
        return 'S-shaped mf'


class Trimf(MembershipFunction):
    def __init__(self, a, b, c):
        self.a = a+0.0
        self.b = b+0.0
        self.c = c+0.0

    def __call__(self, x):
        if (x <= self.a or self.c <= x):
            return 0.0
        if self.a != self.b:
            if self.a < x and x < self.b:
                return (x-self.a)/(self.b-self.a)
        if self.b != self.c:
            if self.b < x and x < self.c:
                return (self.c-x)/(self.c-self.b)
        if x == self.b:
            return 1.0

    def __str__(self):
        return 'S-shaped mf'


class Zmf(MembershipFunction):
    def __init__(self, c, d):
        self.c = c+0.0
        self.d = d+0.0

    def __call__(self, x):
        return zmfFCT(x, self.c, self.d)

    def __str__(self):
        return 'Z-shaped mf'


class Pimf(MembershipFunction):
    def __init__(self, a, b, c, d):
        self.a = a+0.0
        self.b = b+0.0
        self.c = c+0.0
        self.d = d+0.0

    def __call__(self, x):
        return smfFCT(x, self.a, self.b)*zmfFCT(x, self.c, self.d)

    def __str__(self):
        return 'Pi-shaped mf'


class FuzzyTools(object):        # Base class for all fuzzy tools
    NumInfiSlice = 1e3

    def __init__(self, Name, Type, InfoUpperStr):
        self.Name = Name
        self.Type = Type
        self.InfoStr = ''
        self.InfoUpperStr = InfoUpperStr

    def WriteInfo(self):
        self.InfoStr = 'Name \t: '+self.Name+'\nType \t: '+self.Type+'\n'
        if not self.Input:
            self.InfoStr = self.InfoStr+'No inputs yet!\n'
        else:
            self.ResetInput()
            self.InfoStr = self.InfoStr+'NumInputs \t: '+str(self.nInpRules)+'\n'
        if not self.Output:
            self.InfoStr = self.InfoStr+'No outputs yet!\n'
        else:
            self.ResetOutput()
            self.InfoStr = self.InfoStr+'NumOutputs \t: '+str(self.nOutRules)+'\n'
        if not self.Rules:
            self.InfoStr = self.InfoStr+'No outputs yet!\n'
        else:
            for iRule in self.Rules.keys():
                InpStr = 'If '
                for iStr in range(self.nInpRules):
                    if iStr > 0:
                        InpStr = InpStr+' '+self.Rules[iRule][0]+' '
                    InpStr = InpStr + self.Rules[iRule][iStr*2+1]+' '+self.Rules[iRule][iStr*2+2]
                self.InfoStr = self.InfoStr+'Rule '+iRule+': ('+self.Rules[iRule][0]+') '+InpStr+' '+self.Rules[iRule][self.nInpRules*2+1]+' '+self.Rules[iRule][self.nInpRules*2+2]+' '+self.Rules[iRule][self.nInpRules*2+3]+'\n'
        self.InfoStr = self.InfoStr+self.InfoUpperStr

    def ResetInput(self):
        self.nInpRules = len(self.Input.keys())

    def ResetOutput(self):
        self.nOutRules = len(self.Output.keys())

    def ResetRules(self):
        self.nRules = len(self.Rules.keys())

    def Prod(self, List):
        Prod = 1
        for i in List:
            Prod *= i
        return Prod

    def Sum(self, List):
        Sum = 0
        for i in List:
            Sum += i
        return Sum

    def AND(self, InputVals):
        if self.AndMethod == 'min':
            return min(InputVals)
        elif self.AndMethod == 'prod':
            return self.Prod(InputVals)
        else:
            if self.PrintInfo:
                print('Desired AND-Rule not implemented yet!')
            return []

    def OR(self, InputVals):
        if self.OrMethod == 'max':
            return max(InputVals)
        elif self.OrMethod == 'sum':
            return self.Sum(InputVals)
        else:
            if self.PrintInfo:
                print('Desired OR-Rule not implemented yet!')
            return []

    def Fuzzify(self, InputDict):
        self.InputFuzzyVals = dict()
        for iInput in self.Input.keys():
            self.InputFuzzyVals[iInput] = dict()
            for iMF in self.Input[iInput]['MF'].keys():
                self.InputFuzzyVals[iInput][iMF] = self.Input[iInput]['MF'][iMF](InputDict[iInput])

    def EvalImpl(self, CurRule):
        InpVals = []
        for iInput in range(self.nInpRules):
            InpVals.append(self.InputFuzzyVals[CurRule[iInput*2+1]][CurRule[iInput*2+2]])
        self.InpValsTMP = cp.deepcopy(InpVals)
        if CurRule[0] == 'AND':
            ImpVal = self.AND(InpVals)
        elif CurRule[0] == 'OR':
            ImpVal = self.OR(InpVals)
        else:
            if self.PrintInfo:
                print('Desired rule not implemented yet!')
            return []
        return ImpVal

    def Defuzzify(self, ImpVal):
        Area = 0.0
        CoG = 0.0
        #iKey = self.Output.keys()[0]
        iKey = list(self.Output)[0]
        xSample = np.linspace(self.Output[iKey]['Range'][0],
                              self.Output[iKey]['Range'][1],
                              num=self.NumInfiSlice)
        for ixSample in range(1, np.size(xSample)):
            xP = xSample[ixSample-1]
            xN = xSample[ixSample]
            if self.AggRule == 'sum':
                yP = 0
                yN = 0
                iR = 0
                for iRule in self.Rules.keys():
                    yTMP=self.Output[self.Rules[iRule][self.nInpRules*2+2]]['MF'][self.Rules[iRule][self.nInpRules*2+3]](xP)
                    if self.ImpRule == 'min':
                        if yTMP > ImpVal[iR]:
                            yP+=ImpVal[iR]
                        else:
                            yP+=yTMP
                    elif self.ImpRule == 'prod':
                        yP+=yTMP*ImpVal[iR]
                    yTMP=self.Output[self.Rules[iRule][self.nInpRules*2+2]]['MF'][self.Rules[iRule][self.nInpRules*2+3]](xN)
                    if self.ImpRule == 'min':
                        if yTMP > ImpVal[iR]:
                            yN+=ImpVal[iR]
                        else:
                            yN+=yTMP
                    elif self.ImpRule == 'prod':
                        yN+=yTMP*ImpVal[iR]
                    iR+=1
            elif self.AggRule == 'max':
                yPL = []
                yNL = []
                iR = 0
                for iRule in self.Rules.keys():
                    yTMP = self.Output[self.Rules[iRule][self.nInpRules*2+2]]['MF'][self.Rules[iRule][self.nInpRules*2+3]](xP)
                    if self.ImpRule == 'min':
                        if yTMP > ImpVal[iR]:
                            yPL.append(ImpVal[iR])
                        else:
                            yPL.append(yTMP)
                    elif self.ImpRule == 'prod':
                        yPL.append(yTMP*ImpVal[iR])
                    yTMP = self.Output[self.Rules[iRule][self.nInpRules*2+2]]['MF'][self.Rules[iRule][self.nInpRules*2+3]](xN)
                    if self.ImpRule == 'min':
                        if yTMP > ImpVal[iR]:
                            yNL.append(ImpVal[iR])
                        else:
                            yNL.append(yTMP)
                    elif self.ImpRule == 'prod':
                        yNL.append(yTMP*ImpVal[iR])
                    iR += 1
                yP = max(yPL)
                yN = max(yNL)
            else:
                if self.PrintInfo:
                    print('Desired Agg-Rule not implemented yet!')
                return []
            Area += (xN-xP)*(yN+yP)*.5
            CoG += (xN-xP)*(yN+yP)*.5*((xN-xP)*.5+xP)
        return CoG/Area


class FIS(FuzzyTools):
    Input = dict()
    Output = dict()
    Rules = dict()

    def __init__(self, FISname, PrintInfo=False, AndMethod='prod',
                 OrMethod='max', ImpRule='min', AggRule='sum',
                 DefuzzMethod='CoA'):
        # Alternatives are                    AndMethod='prod/min', OrMethod='max/sum', ImpRule='prod', AggRule='max/sum', DefuzzMethod='CoA'
        InfoStr = 'AndMethod \t: '+AndMethod+'\nOrMethod \t: '+OrMethod+'\nImpRule \t:'+ImpRule+'\nAggRule \t:'+AggRule+'\nDefuzzMethod \t:'+DefuzzMethod+'\n'
        FuzzyTools.__init__(self, FISname, 'Mandani', InfoStr)
        self.FISname = FISname
        self.AndMethod = AndMethod
        self.OrMethod = OrMethod
        self.ImpRule = ImpRule
        self.AggRule = AggRule
        self.PrintInfo = PrintInfo

    def __str__(self):
        self.WriteInfo()
        return '-###############-\n-# Mandani-FIS #-\n-###############-\n'+self.InfoStr

    def GetOutputImplicationVals(self, InputDict):
        self.EvalFIS(InputDict)
        return self.OutImpl

    def EvalFIS(self, InputDict):
        #x=np.array(x)
        self.ResetInput()
        self.ResetOutput()
        self.ResetRules()
        #if len(InputDict.keys())!=self.nInpRules:
        #    if self.PrintInfo:
        #        print 'Length of X differs from number of existing input rules!'
        #    return []
        self.Fuzzify(InputDict)
        self.OutImpl = []
        self.InpPerRuleVals = dict()
        for iRule in self.Rules.keys():
            self.OutImpl.append(self.EvalImpl(self.Rules[iRule]))
            self.InpPerRuleVals[iRule] = self.InpValsTMP
        self.FISValue = cp.deepcopy(self.Defuzzify(self.OutImpl))
        return self.FISValue

    def GetMaxAntecentKey(self):
        return max(self.InputFuzzyVals.iterkeys(),
                   key=lambda k: self.InputFuzzyVals[k])

    def GetMaxAntecentOfMaxImplicationKey(self):
        AntecentList = self.InpPerRuleVals[self.GetMaxImplicationKey()]
        InpMax = AntecentList.index(max(AntecentList))
        return [self.Rules[self.GetMaxImplicationKey()][InpMax*2+1],
                self.Rules[self.GetMaxImplicationKey()][InpMax*2+2]]

    def GetMaxImplicationKey(self):
        return list(self.Rules)[self.OutImpl.index(max(self.OutImpl))]

    def GetInputVariables(self):
        return self.Input.keys()

    def PlotRules(self, NumSupportPoints=50):
        '''
        Plots all rules of the FIS
        '''
        for iRule in self.Rules.keys():
            RuleDef = self.Rules[iRule]
            nRules = (len(RuleDef)-4)/2
            MFdict = dict()
            for iInput in range(nRules):
                iInputKey = RuleDef[1+iInput*2]
                iMFKey = RuleDef[2+iInput*2]
                MFdict['In "'+iInputKey+'-'+iMFKey+'"'] = list()
                iRange = self.Input[iInputKey]['Range']
                xVals = iRange[0] + (iRange[1]-iRange[0])*np.linspace(0., 1., num=NumSupportPoints)
                for iX in xVals:
                    MFdict['In "'+iInputKey+'-'+iMFKey+'"'].append(self.Input[iInputKey]['MF'][iMFKey](iX))
            iOutoutKey = RuleDef[-2]
            iMFKey = RuleDef[-1]
            MFdict['Out "'+iOutoutKey+'-'+iMFKey+'"'] = list()
            iRange = self.Output[iOutoutKey]['Range']
            xVals = iRange[0] + (iRange[1]-iRange[0])*np.linspace(0., 1., num=NumSupportPoints)
            for iX in xVals:
                MFdict['Out "'+iOutoutKey+'-'+iMFKey+'"'].append(self.Output[iOutoutKey]['MF'][iMFKey](iX))
            Iter = np.linspace(0., 1., num=NumSupportPoints)
            self._InputOutputPlot(Iter, MFdict, Label='Rule "'+iRule+'" with "'+RuleDef[0]+'"')

    def PlotInputOutputMFs(self, NumSupportPoints=50):
        '''
        Plots input and output membership functions of the FIS
        '''
        for iInput in self.Input.keys():
            iRange = self.Input[iInput]['Range']
            xVals = iRange[0] + (iRange[1]-iRange[0])*np.linspace(0., 1., num=NumSupportPoints)
            MFdict = dict()
            for iMF in self.Input[iInput]['MF'].keys():
                MFdict['MF "'+iMF+'"'] = list()
                for iX in xVals:
                    MFdict['MF "'+iMF+'"'].append(self.Input[iInput]['MF'][iMF](iX))
            self._InputOutputPlot(xVals, MFdict, Label='Fuzzy input membership functions of '+iInput)
        for iOutput in self.Output.keys():
            iRange = self.Output[iOutput]['Range']
            xVals = iRange[0] + (iRange[1]-iRange[0])*np.linspace(0., 1., num=NumSupportPoints)
            MFdict = dict()
            for iMF in self.Output[iOutput]['MF'].keys():
                MFdict['MF "'+iMF+'"'] = list()
                for iX in xVals:
                    MFdict['MF "'+iMF+'"'].append(self.Output[iOutput]['MF'][iMF](iX))
            self._InputOutputPlot(xVals, MFdict, Label='Fuzzy output membership functions of '+iOutput)

    def PlotFuzzyInferenceSystem(self, UseTex=False, NumSupportPoints=20,
                                 ShowFig=True, FigFile=None, nRulesPerPlot=7.):
        '''
        Plot the whole fuzzy inference system over rules
        '''
        try:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except:
            logging.error('Matplotlib could not be imported!')
        # Initializations
        plt.rc('text', usetex=UseTex)
        Rules = self.Rules.keys()
        nRules = len(Rules)
        iPlot = 1
        nPlots = math.floor(nRules/nRulesPerPlot)+math.ceil(nRules%nRulesPerPlot/10.)
        Inputs = self.Input.keys()
        nInputs = len(Inputs)
        FigureTitle = 'FIS of '+self.FISname
        if hasattr(self, 'FISValue'):
            PlotMEinfo = True
            FigureTitle += ' - %.3f' % (self.FISValue)
        else:
            PlotMEinfo = False
        # FIS plotting
        for RuleID, iRule in enumerate(Rules):
            if RuleID == 0:
                if nRules > nRulesPerPlot:
                    fig = plt.figure(figsize=plt.figaspect(nRulesPerPlot/(nInputs+1.)))
                    fig.suptitle(FigureTitle+'(Part %i/%i)'%(iPlot, nPlots),
                                 fontsize=14, fontweight='bold')
                    nSubPlotY = nRulesPerPlot
                    iPlot += 1
                else:
                    fig = plt.figure(figsize=plt.figaspect(nRules/(nInputs+1.)))
                    fig.suptitle(FigureTitle, fontsize=14, fontweight='bold')
                    nSubPlotY = nRules
                    iPlot += 1
            elif (RuleID) % (nRulesPerPlot) == 0:
                if math.floor((nRules - RuleID)/nRulesPerPlot) > 0:
                    nRest = nRulesPerPlot
                else:
                    nRest = int((nRules - RuleID) % nRulesPerPlot)
                fig = plt.figure(figsize=plt.figaspect(nRest/(nInputs+1.)))
                fig.suptitle(FigureTitle+'(Part %i/%i)' % (iPlot, nPlots),
                             fontsize=14, fontweight='bold')
                nSubPlotY = nRest
                iPlot += 1
            for InputID, jInput in enumerate(Inputs):
                iInput = self.Rules[iRule][InputID*2+1]
                InputMFID = self.Rules[iRule][InputID*2+2]
                ax = fig.add_subplot(nSubPlotY, (nInputs+1.), RuleID*(nInputs+1.)+InputID+1-(iPlot-2)*nRulesPerPlot*(nInputs+1))
                X = np.linspace(self.Input[iInput]['Range'][0],
                                self.Input[iInput]['Range'][1],
                                num=NumSupportPoints)
                Y = list()
                for iX in X:
                    Y.append(self.Input[iInput]['MF'][InputMFID](iX))
                if InputID == 0:
                    yLabel = 'Rule %i:' % (RuleID)
                else:
                    yLabel = []
                self._SubPlot(fig, ax, X, Y, iInput+' is "'+InputMFID+'"', [],
                              yLabel)
                if PlotMEinfo:
                    YTilde = [self.InpPerRuleVals[iRule][InputID]]*np.size(X)
                    ax.plot(X, YTilde, 'r--')
                if RuleID < nRules-1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if InputID > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
            # plot output
            Output = self.Rules[iRule][-2]
            OutputMFID = self.Rules[iRule][-1]
            ax = fig.add_subplot(nSubPlotY, (nInputs+1.), RuleID*(nInputs+1.)+nInputs+1 - (iPlot-2)*nRulesPerPlot*(nInputs+1))
            X = np.linspace(self.Output[Output]['Range'][0], self.Output[Output]['Range'][1], num=NumSupportPoints)
            Y = list()
            for iX in X:
                Y.append(self.Output[Output]['MF'][OutputMFID](iX))
            SubTitel = 'Output'
            self._SubPlot(fig, ax, X, Y, Output+': "'+OutputMFID+'"', [], [],
                          ColorCode='r')
            if PlotMEinfo:
                YTilde = [self.OutImpl[RuleID]]*np.size(X)
                ax.plot(X, YTilde, 'r--')
            if RuleID < nRules-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
        if FigFile != None:
            plt.savefig(FigFile)
        if ShowFig:
            plt.show()
        else:
            plt.close()

    def _SubPlot(self, fig, ax, xVals, yVals, SubTitel, xLabel, yLabel,
                 ColorCode='b'):
        ax.plot(xVals, yVals, ColorCode)
        ax.set_autoscaley_on(False)
        ax.set_ylim([-.1, 1.1])
        if SubTitel:
            ax.set_title(SubTitel)
        if xLabel:
            ax.set_xlabel(xLabel, fontsize=12)
        if yLabel:
            ax.set_ylabel(yLabel, fontsize=12)

    def _InputOutputPlot(self, xVals, MFdict, Label='Fuzzy variable'):
        try:
            import matplotlib.pyplot as plt
        except:
            logging.fatal('Could not import matplotlib modules!')
            sys.exit('Failed to import matlotlib package!')

        num_plots = len(MFdict.keys())
        plt.figure()
        plt.title(Label)
        colormap = plt.cm.gist_rainbow
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

        # Plot several different functions...
        labels = []
        for iMF in MFdict.keys():
            plt.plot(xVals, MFdict[iMF])
            labels.append(iMF)
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
