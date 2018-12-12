#------------------------------------------------------------------#
#******************************************************************#
#                  BraidME CLASS: BraidME.py                       #
#******************************************************************#
#------------------------------------------------------------------#
# Author:        Dr Markus Edwin Schatz                            #
# Mail:          schatz@llb.mw.tum.de                              #
#------------------------------------------------------------------#
# Change-Log:                                                      #
# 2016-08-28     Definition of class:                              # 
#                o First definition of class                       #
# 2017-09-24     Unifying class:                                   # 
#                o Incorporated FuzzyME classes as base classes    #
#                o BraidME inherited all instances of FuzzyME      #
#                o Added description and usage outline             #
# 2018-12-12     Publicate work:                                   # 
#                o MDPI: Journal of Composite Science              #
#------------------------------------------------------------------#
# Title: Enabling Composite Optimization Through Soft Computing Of #
#        Manufacturing Restrictions And Costs Via A Narrow         #
#        Artificial Intelligence                                   #
#        - Journal of Composite Science -                          #
#------------------------------------------------------------------#
# Usage of inherited class BraidME:                                #
#    Import Braid-FIS class:                                       # 
#         from BraidME import BraidME                              #
#    Define a Braid-FIS object:                                    # 
#         BraidMEM = BraidME()                                     #
#    Initialize and save the Braid-FIS:                            # 
#         BraidMEM.CreateAndSaveBraidFIS()                         #
#         -> BraidMEM.LoadBraidFIS() would then load FIS           #
#    Compute efforts based on last input x (see below):            # 
#         ME=BraidMEM.ComputeResponse(x)                           #
#         -> You can pass x as list as defined below or as an      #
#            dictionary which contains lists as follows:           #
#            o BraidingAngle         ... Braiding angle along prof.#
#            o ProfileCircumferences ... Circumference of profile  #
#            o ProfileMinRadius      ... Min. radius of profile    #
#            o PathLength            ... List of length inbetween  #
#            o PathRadii             ... Radius of path inbetween  #
#            o SupPathRadii (optional).. Radius of support path    #
#            o ProfileAspect (optional). Aspect ratio of profile   #
#            o PlyNum (optional)     ... Number of plies           #
#            o PatchNum (optional)   ... Number of patches         #
#    Provide reasoning for input x (see below):                    # 
#         Reasoning=BraidMEM.Reasoning()                           #
#    Compute ME response and its derivative including reasoning    #
#         [ME, dMEdx, Reasoning] = BraidMEM.ComputeRespAndSens(x)  #
#    PrintMEInfoList()                                             #
#         prints all MEList entries computed once a dict has been  #
#         passed to compute ME! -> First .ComputeResponse(MEInp)   #
#    Plot all ME response surfaces                                 #
#         BraidMEM.PlotAllResponseSurfaces(x,PlotSamplePerAx=100)  #
#------------------------------------------------------------------#
# Input x for BraidME class:                                       #
#    BraidAngle [Deg] ... [15 75]) -> 25                           # 
#    YarnWidth  [mm]  ... [1.5 4]  -> 2.7                          #    
#    Curvature  [-]   ... [0 10]   -> 10  R/d                      #      
#    EdgeRadius [mm]  ... [3 5]    -> 5                            #    
#    AspectRatio[-]   ... [2 4]    -> 2                            #   
#    PlyNum     [-]   ... [5 20]   -> 5                            #    
#    PatchNum   [-]   ... [0 5]    -> 0                            #   
#------------------------------------------------------------------#
#------------------------------------------------------------------#

# Imports
#------------------------------------------------------------------#
import numpy as np
import os
import copy as cp
import pickle
import math
import sys

# Import fuzzy stuff from FuzzyTools
#------------------------------------------------------------------#
from FuzzyTools import FIS
from FuzzyTools import SurfacePlotter
from FuzzyTools import Gauss2mf
from FuzzyTools import Gaussmf
from FuzzyTools import Trimf
from FuzzyTools import Pimf
from FuzzyTools import Const
''' Alternative membership functions
from FuzzyTools import Zmf
from FuzzyTools import Smf 
'''
	
class BraidME(FIS,Gauss2mf,Gaussmf,Trimf,Pimf,Const):        # Base class for all fuzzy tools
    nBobins         = 16.0*12
    MaxPatches      = 15.

    def __init__(self,FDTol=1.0e-4):
        self.FDTol = FDTol
        self.CurPath = os.getcwd()
        self.SupportDictKeys = ['ProfileCircumferences', 'PathLength', \
            'ProfileMinRadius', 'PathRadii', 'ProfileAspect', 'PlyNum', \
            'PatchNum', 'BraidingAngle']
        self.UpdateHornGearParams()
            
    def UpdateHornGearParams(self):
        self.nHornGears      = cp.deepcopy(self.nBobins/2.0)
        self.HornGearSpeed   = 120.0*1.0/60.0 # 120 rmp is common!

    def CreateAndSaveBraidFIS(self):
        # Sub-FIS 1
        # ------------------------------------------------------------------- #
        self.SubFIS1        = FIS(FISname='SubFIS1:OverCompacting,BraidOpening,UnrealisticMachine,ProductionTimes')
        self.SubFIS1.Input  = dict()
        self.SubFIS1.Output = dict()
        self.SubFIS1.Rules  = dict()
        # Input - Braiding angle
        self.SubFIS1.Input['BraidAngle'] = dict()
        self.SubFIS1.Input['BraidAngle']['Range'] = [15.0, 75.0]
        self.SubFIS1.Input['BraidAngle']['MF'] = dict()
        self.SubFIS1.Input['BraidAngle']['MF']['VerySmall'] = Gauss2mf(2.0, 10.0, 4.0, 15.0)
        self.SubFIS1.Input['BraidAngle']['MF']['Small'] = Gaussmf(4.0, 25.0)
        self.SubFIS1.Input['BraidAngle']['MF']['Moderatesmall'] = Gaussmf(4.0, 35.0)
        self.SubFIS1.Input['BraidAngle']['MF']['Moderate'] = Gaussmf(4.0, 45.0)
        self.SubFIS1.Input['BraidAngle']['MF']['ModerateBig'] = Gaussmf(4.0, 55.0)
        self.SubFIS1.Input['BraidAngle']['MF']['Big'] = Gaussmf(4.0, 65.0)
        self.SubFIS1.Input['BraidAngle']['MF']['VeryBig'] = Gauss2mf(4.0, 75.0, 2.0, 80.0)
        self.SubFIS1.Input['BraidAngle']['MF']['Any'] = Const(1.0)
        # Input - Yarn width
        self.SubFIS1.Input['YarnWidth'] = dict()
        self.SubFIS1.Input['YarnWidth']['Range'] = [1.5, 4.0]
        self.SubFIS1.Input['YarnWidth']['MF'] = dict()
        self.SubFIS1.Input['YarnWidth']['MF']['TooSmall'] = Gauss2mf(.22, 1.0, .22, 1.5)
        self.SubFIS1.Input['YarnWidth']['MF']['Moderate'] = Gauss2mf(.22, 2.25, .22, 3.25)
        self.SubFIS1.Input['YarnWidth']['MF']['TooBig']   = Gauss2mf(.22, 4.0, .22, 4.5)
        self.SubFIS1.Input['YarnWidth']['MF']['Any'] = Const(1.0)
        # Output - Sub1
        self.SubFIS1.Output['Sub1'] = dict()
        self.SubFIS1.Output['Sub1']['Range'] = [0.0, 1.0]
        self.SubFIS1.Output['Sub1']['MF'] = dict()
        self.SubFIS1.Output['Sub1']['MF']['VeryLow'] = Gaussmf(0.082, 0.0)
        self.SubFIS1.Output['Sub1']['MF']['Low'] = Gaussmf(0.082, 0.1)
        self.SubFIS1.Output['Sub1']['MF']['Moderate'] = Gaussmf(0.082, 0.2)
        self.SubFIS1.Output['Sub1']['MF']['High'] = Gaussmf(0.082, 0.4)
        self.SubFIS1.Output['Sub1']['MF']['VeryHigh'] = Gaussmf(0.082, 0.8)
        self.SubFIS1.Output['Sub1']['MF']['NotManufacturable'] = Gaussmf(0.082, 1.0)
        # Rules Sub-FIS1
        self.SubFIS1.Rules['OverCompTakeUpSpeedInfinity'] = ['OR','BraidAngle','VerySmall','YarnWidth','TooSmall','THEN','Sub1','NotManufacturable']
        self.SubFIS1.Rules['BraidOpenHornGearSpeedInfinity'] = ['OR','BraidAngle','VeryBig','YarnWidth','TooBig','THEN','Sub1','NotManufacturable']
        self.SubFIS1.Rules['VLowestPTimes'] = ['AND','BraidAngle','Small','YarnWidth','Moderate','THEN','Sub1','VeryLow']
        self.SubFIS1.Rules['LowPTimes'] = ['AND','BraidAngle','Moderatesmall','YarnWidth','Moderate','THEN','Sub1','Low']
        self.SubFIS1.Rules['ModeratePTimes'] = ['AND','BraidAngle','Moderate','YarnWidth','Moderate','THEN','Sub1','Moderate']
        self.SubFIS1.Rules['HighPTimes'] = ['AND','BraidAngle','ModerateBig','YarnWidth','Moderate','THEN','Sub1','High']
        self.SubFIS1.Rules['VeryHighPTimes'] = ['AND','BraidAngle','Big','YarnWidth','Moderate','THEN','Sub1','VeryHigh']
        
        # Sub-FIS 2
        # ------------------------------------------------------------------- #
        self.SubFIS2        = FIS(FISname='SubFIS2:CombinationBraidAngleAndRatioOfRadiusDiameter')
        self.SubFIS2.Input  = dict()
        self.SubFIS2.Output = dict()
        self.SubFIS2.Rules  = dict()
        # Input - Braiding angle
        self.SubFIS2.Input['BraidAngle'] = dict()
        self.SubFIS2.Input['BraidAngle']['Range'] = [15.0, 75.0]
        self.SubFIS2.Input['BraidAngle']['MF'] = dict()
        self.SubFIS2.Input['BraidAngle']['MF']['VerySmall'] = Gauss2mf(2.0, 10.0, 4.0, 15.0)
        self.SubFIS2.Input['BraidAngle']['MF']['Small'] = Gaussmf(4.0, 25.0)
        self.SubFIS2.Input['BraidAngle']['MF']['Moderatesmall'] = Gaussmf(4.0, 35.0)
        self.SubFIS2.Input['BraidAngle']['MF']['Moderate'] = Gaussmf(4.0, 45.0)
        self.SubFIS2.Input['BraidAngle']['MF']['ModerateBig'] = Gaussmf(4.0, 55.0)
        self.SubFIS2.Input['BraidAngle']['MF']['Big'] = Gaussmf(4.0, 65.0)
        self.SubFIS2.Input['BraidAngle']['MF']['VeryBig'] = Gauss2mf(4.0, 75.0, 2.0, 80.0)
        self.SubFIS2.Input['BraidAngle']['MF']['Any'] = Const(1.0)
        # Input - Ratio of radius to diameter
        self.SubFIS2.Input['RadiusDiameterRatio'] = dict()
        self.SubFIS2.Input['RadiusDiameterRatio']['Range'] = [0.5, 10.0]
        self.SubFIS2.Input['RadiusDiameterRatio']['MF'] = dict()
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['VerySmall'] = Gaussmf(.5, 0.0)
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['Small'] = Gaussmf(.5, 1.2)
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['Moderate'] = Gaussmf(.6, 2.4)
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['Big'] = Gaussmf(.7, 4.0)
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['Bigger'] = Gaussmf(.8, 6.0)
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['VeryBig'] = Gaussmf(.8, 7.8)
        self.SubFIS2.Input['RadiusDiameterRatio']['MF']['NoCurvature'] = Gaussmf(.9, 10)
        # Output - Sub2
        self.SubFIS2.Output['Sub2'] = dict()
        self.SubFIS2.Output['Sub2']['Range'] = [0, 1.0]
        self.SubFIS2.Output['Sub2']['MF'] = dict()
        self.SubFIS2.Output['Sub2']['MF']['VeryLow'] = Gaussmf(0.082, 0.0)
        self.SubFIS2.Output['Sub2']['MF']['Low'] = Gaussmf(0.082, 0.1)
        self.SubFIS2.Output['Sub2']['MF']['Moderate'] = Gaussmf(0.082, 0.2)
        self.SubFIS2.Output['Sub2']['MF']['High'] = Gaussmf(0.082, 0.4)
        self.SubFIS2.Output['Sub2']['MF']['VeryHigh'] = Gaussmf(0.082, 0.8)
        self.SubFIS2.Output['Sub2']['MF']['NotManufacturable'] = Gaussmf(0.082, 1.0)
        # Rules Sub-FIS2
        self.SubFIS2.Rules['CurvatureTooGreatTakeUp'] = ['OR','BraidAngle','VerySmall','RadiusDiameterRatio','VerySmall','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['CurvatureTooGreatHornGear'] = ['OR','BraidAngle','VeryBig','RadiusDiameterRatio','VerySmall','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['OverCriticalCombAngleCurvature1'] = ['AND','BraidAngle','Small','RadiusDiameterRatio','Small','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['OverCriticalCombAngleCurvature2'] = ['AND','BraidAngle','Small','RadiusDiameterRatio','Moderate','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['CriticalCombAngleCurvature1'] = ['AND','BraidAngle','Small','RadiusDiameterRatio','Big','THEN','Sub2','VeryHigh']
        self.SubFIS2.Rules['ModerateCombAngleCurvature1'] = ['AND','BraidAngle','Small','RadiusDiameterRatio','Bigger','THEN','Sub2','Moderate']
        self.SubFIS2.Rules['AlmostNoCurvature1'] = ['AND','BraidAngle','Small','RadiusDiameterRatio','VeryBig','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['NoCurvature1'] = ['AND','BraidAngle','Small','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['OverCriticalCombAngleCurvature3'] = ['AND','BraidAngle','Moderatesmall','RadiusDiameterRatio','Small','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['CriticalCombAngleCurvature2'] = ['AND','BraidAngle','Moderatesmall','RadiusDiameterRatio','Moderate','THEN','Sub2','VeryHigh']
        self.SubFIS2.Rules['CombAngleCurvature1'] = ['AND','BraidAngle','Moderatesmall','RadiusDiameterRatio','Big','THEN','Sub2','Moderate']
        self.SubFIS2.Rules['AlmostNoCurvature2'] = ['AND','BraidAngle','Moderatesmall','RadiusDiameterRatio','Bigger','THEN','Sub2','Low']
        self.SubFIS2.Rules['AlmostNoCurvature3'] = ['AND','BraidAngle','Moderatesmall','RadiusDiameterRatio','VeryBig','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['NoCurvature2'] = ['AND','BraidAngle','Moderatesmall','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['CriticalCombAngleCurvature3'] = ['AND','BraidAngle','Moderate','RadiusDiameterRatio','Small','THEN','Sub2','VeryHigh']
        self.SubFIS2.Rules['CriticalCombAngleCurvature4'] = ['AND','BraidAngle','Moderate','RadiusDiameterRatio','Moderate','THEN','Sub2','High']
        self.SubFIS2.Rules['CombAngleCurvature2'] = ['AND','BraidAngle','Moderate','RadiusDiameterRatio','Big','THEN','Sub2','Moderate']
        self.SubFIS2.Rules['AlmostNoCurvature4'] = ['AND','BraidAngle','Moderate','RadiusDiameterRatio','Bigger','THEN','Sub2','Low']
        self.SubFIS2.Rules['AlmostNoCurvature5'] = ['AND','BraidAngle','Moderate','RadiusDiameterRatio','VeryBig','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['NoCurvature3'] = ['AND','BraidAngle','Moderate','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['OverCriticalCombAngleCurvature4'] = ['AND','BraidAngle','ModerateBig','RadiusDiameterRatio','Small','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['CriticalCombAngleCurvature5'] = ['AND','BraidAngle','ModerateBig','RadiusDiameterRatio','Moderate','THEN','Sub2','VeryHigh']
        self.SubFIS2.Rules['CombAngleCurvature3'] = ['AND','BraidAngle','ModerateBig','RadiusDiameterRatio','Big','THEN','Sub2','Moderate']
        self.SubFIS2.Rules['AlmostNoCurvature6'] = ['AND','BraidAngle','ModerateBig','RadiusDiameterRatio','Bigger','THEN','Sub2','Low']
        self.SubFIS2.Rules['AlmostNoCurvature7'] = ['AND','BraidAngle','ModerateBig','RadiusDiameterRatio','VeryBig','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['NoCurvature4'] = ['AND','BraidAngle','ModerateBig','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['OverCriticalCombAngleCurvature5'] = ['AND','BraidAngle','Big','RadiusDiameterRatio','Small','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['OverCriticalCombAngleCurvature6'] = ['AND','BraidAngle','Big','RadiusDiameterRatio','Moderate','THEN','Sub2','NotManufacturable']
        self.SubFIS2.Rules['CriticalCombAngleCurvature6'] = ['AND','BraidAngle','Big','RadiusDiameterRatio','Big','THEN','Sub2','VeryHigh']
        self.SubFIS2.Rules['ModerateCombAngleCurvature2'] = ['AND','BraidAngle','Big','RadiusDiameterRatio','Bigger','THEN','Sub2','Moderate']
        self.SubFIS2.Rules['AlmostNoCurvature8'] = ['AND','BraidAngle','Big','RadiusDiameterRatio','VeryBig','THEN','Sub2','VeryLow']
        self.SubFIS2.Rules['NoCurvature5'] = ['AND','BraidAngle','Big','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        
        # Sub-FIS 3
        # ------------------------------------------------------------------- #
        self.SubFIS3        = FIS(FISname='SubFIS3:CombinationYarnWidthAndRatioOfRadiusDiameter')
        self.SubFIS3.Input  = dict()
        self.SubFIS3.Output = dict()
        self.SubFIS3.Rules  = dict()
        # Input - Yarn width
        self.SubFIS3.Input['YarnWidth'] = dict()
        self.SubFIS3.Input['YarnWidth']['Range'] = [1.5, 4.0]
        self.SubFIS3.Input['YarnWidth']['MF'] = dict()
        self.SubFIS3.Input['YarnWidth']['MF']['TooSmall'] = Gauss2mf(.28, 1.0, .28, 1.5)
        self.SubFIS3.Input['YarnWidth']['MF']['Small']    = Gaussmf(.28, 2.14)
        self.SubFIS3.Input['YarnWidth']['MF']['Moderate'] = Gauss2mf(.28, 2.7, .28, 2.8)
        self.SubFIS3.Input['YarnWidth']['MF']['Big']      = Gaussmf(.28, 3.36)
        self.SubFIS3.Input['YarnWidth']['MF']['TooBig']   = Gauss2mf(.28, 4.0, .28, 4.5)
        # Input - Ratio of radius to diameter
        self.SubFIS3.Input['RadiusDiameterRatio'] = dict()
        self.SubFIS3.Input['RadiusDiameterRatio']['Range'] = [0.5, 10.0]
        self.SubFIS3.Input['RadiusDiameterRatio']['MF'] = dict()
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['VerySmall'] = Gaussmf(.5, 0.0)
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['Small'] = Gaussmf(.5, 1.2)
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['Moderate'] = Gaussmf(.6, 2.4)
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['Big'] = Gaussmf(.7, 4.0)
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['Bigger'] = Gaussmf(.8, 6.0)
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['VeryBig'] = Gaussmf(.8, 7.8)
        self.SubFIS3.Input['RadiusDiameterRatio']['MF']['NoCurvature'] = Gaussmf(.9, 10)
        # Output - Sub3
        self.SubFIS3.Output['Sub2'] = dict()
        self.SubFIS3.Output['Sub2']['Range'] = [0, 1.0]
        self.SubFIS3.Output['Sub2']['MF'] = dict()
        self.SubFIS3.Output['Sub2']['MF']['VeryLow'] = Gaussmf(0.082, 0.0)
        self.SubFIS3.Output['Sub2']['MF']['Low'] = Gaussmf(0.082, 0.1)
        self.SubFIS3.Output['Sub2']['MF']['Moderate'] = Gaussmf(0.082, 0.2)
        self.SubFIS3.Output['Sub2']['MF']['High'] = Gaussmf(0.082, 0.4)
        self.SubFIS3.Output['Sub2']['MF']['VeryHigh'] = Gaussmf(0.082, 0.8)
        self.SubFIS3.Output['Sub2']['MF']['NotManufacturable'] = Gaussmf(0.082, 1.0)
        # Rules Sub-FIS3
        self.SubFIS3.Rules['CurvatureTooGreatOverCompact'] = ['OR','YarnWidth','TooSmall','RadiusDiameterRatio','VerySmall','THEN','Sub2','NotManufacturable']
        self.SubFIS3.Rules['CurvatureTooGreatBraidOpen'] = ['OR','YarnWidth','TooBig','RadiusDiameterRatio','VerySmall','THEN','Sub2','NotManufacturable']
        self.SubFIS3.Rules['OverCriticalCombAngleCurvature1'] = ['AND','YarnWidth','Small','RadiusDiameterRatio','Small','THEN','Sub2','NotManufacturable']
        self.SubFIS3.Rules['OverCriticalCombAngleCurvature2'] = ['AND','YarnWidth','Small','RadiusDiameterRatio','Moderate','THEN','Sub2','NotManufacturable']
        self.SubFIS3.Rules['CriticalCombAngleCurvature1'] = ['AND','YarnWidth','Small','RadiusDiameterRatio','Big','THEN','Sub2','High']
        self.SubFIS3.Rules['LowCurvature1'] = ['AND','YarnWidth','Small','RadiusDiameterRatio','Bigger','THEN','Sub2','Low']
        self.SubFIS3.Rules['LowCurvature2'] = ['AND','YarnWidth','Small','RadiusDiameterRatio','VeryBig','THEN','Sub2','Low']
        self.SubFIS3.Rules['NoCurvature1'] = ['AND','YarnWidth','Small','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        self.SubFIS3.Rules['OverCriticalCombAngleCurvature3'] = ['AND','YarnWidth','Moderate','RadiusDiameterRatio','Small','THEN','Sub2','VeryHigh']
        self.SubFIS3.Rules['CriticalCombAngleCurvature2'] = ['AND','YarnWidth','Moderate','RadiusDiameterRatio','Moderate','THEN','Sub2','High']
        self.SubFIS3.Rules['ModerateCurvature1'] = ['AND','YarnWidth','Moderate','RadiusDiameterRatio','Big','THEN','Sub2','Moderate']
        self.SubFIS3.Rules['LowCurvature3'] = ['AND','YarnWidth','Moderate','RadiusDiameterRatio','Bigger','THEN','Sub2','Low']
        self.SubFIS3.Rules['NoCurvature2'] = ['AND','YarnWidth','Moderate','RadiusDiameterRatio','VeryBig','THEN','Sub2','VeryLow']
        self.SubFIS3.Rules['NoCurvature3'] = ['AND','YarnWidth','Moderate','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        self.SubFIS3.Rules['OverCriticalCombAngleCurvature4'] = ['AND','YarnWidth','Big','RadiusDiameterRatio','Small','THEN','Sub2','NotManufacturable']
        self.SubFIS3.Rules['OverCriticalCombAngleCurvature5'] = ['AND','YarnWidth','Big','RadiusDiameterRatio','Moderate','THEN','Sub2','NotManufacturable']
        self.SubFIS3.Rules['CriticalCombAngleCurvature3'] = ['AND','YarnWidth','Big','RadiusDiameterRatio','Big','THEN','Sub2','High']
        self.SubFIS3.Rules['LowCurvature4'] = ['AND','YarnWidth','Big','RadiusDiameterRatio','Bigger','THEN','Sub2','Low']
        self.SubFIS3.Rules['LowCurvature5'] = ['AND','YarnWidth','Big','RadiusDiameterRatio','VeryBig','THEN','Sub2','Low']
        self.SubFIS3.Rules['NoCurvature4'] = ['AND','YarnWidth','Big','RadiusDiameterRatio','NoCurvature','THEN','Sub2','VeryLow']
        
        # MAIN-FIS
        # ------------------------------------------------------------------- #
        self.MainFIS        = FIS(FISname='Manufacturing Effort Model For Braiding',AndMethod='min')
        self.MainFIS.Input  = dict()
        self.MainFIS.Output = dict()
        self.MainFIS.Rules  = dict()
        # Input - Sub1
        self.MainFIS.Input['Sub1'] = dict()
        self.MainFIS.Input['Sub1']['Range'] = [0.0, 1.0]
        self.MainFIS.Input['Sub1']['MF'] = dict()
        self.MainFIS.Input['Sub1']['MF']['Good'] = Trimf(-1.0, 0.0, 1.0)
        self.MainFIS.Input['Sub1']['MF']['Bad']  = Trimf(0.0, 1.0, 2.0)
        self.MainFIS.Input['Sub1']['MF']['Any']  = Const(1.0)
        # Input - Sub2
        self.MainFIS.Input['Sub2'] = dict()
        self.MainFIS.Input['Sub2']['Range'] = [0.0, 1.0]
        self.MainFIS.Input['Sub2']['MF'] = dict()
        self.MainFIS.Input['Sub2']['MF']['Good'] = Trimf(-1.0, 0.0, 1.0)
        self.MainFIS.Input['Sub2']['MF']['Bad']  = Trimf(0.0, 1.0, 2.0)
        self.MainFIS.Input['Sub2']['MF']['Any']  = Const(1.0)
        # Input - Sub3
        self.MainFIS.Input['Sub3'] = dict()
        self.MainFIS.Input['Sub3']['Range'] = [0.0, 1.0]
        self.MainFIS.Input['Sub3']['MF'] = dict()
        self.MainFIS.Input['Sub3']['MF']['Good'] = Trimf(-1.0, 0.0, 1.0)
        self.MainFIS.Input['Sub3']['MF']['Bad']  = Trimf(0.0, 1.0, 2.0)
        self.MainFIS.Input['Sub3']['MF']['Any']  = Const(1.0)
        # Input - Edge radius
        self.MainFIS.Input['EdgeRadius'] = dict()
        self.MainFIS.Input['EdgeRadius']['Range'] = [3.0, 5.0]
        self.MainFIS.Input['EdgeRadius']['MF'] = dict()
        self.MainFIS.Input['EdgeRadius']['MF']['TooSmall'] = Pimf(1.0, 2.8, 2.9, 5.1)
        self.MainFIS.Input['EdgeRadius']['MF']['Moderate'] = Pimf(2.9, 5.1, 5.317, 6.8)
        self.MainFIS.Input['EdgeRadius']['MF']['Any']      = Const(1.0)
        # Input - Aspect ratio
        self.MainFIS.Input['AspectRatio'] = dict()
        self.MainFIS.Input['AspectRatio']['Range'] = [2.0, 4.0]
        self.MainFIS.Input['AspectRatio']['MF'] = dict()
        self.MainFIS.Input['AspectRatio']['MF']['Moderate'] = Pimf(0.0, 1.8, 1.9, 4.1)
        self.MainFIS.Input['AspectRatio']['MF']['TooBig']   = Pimf(1.9, 4.1, 4.317, 5.8)
        self.MainFIS.Input['AspectRatio']['MF']['Any']      = Const(1.0)
        # Input - Number of plies
        self.MainFIS.Input['PlyNum'] = dict()
        self.MainFIS.Input['PlyNum']['Range'] = [5.0, 20.]
        self.MainFIS.Input['PlyNum']['MF'] = dict()
        self.MainFIS.Input['PlyNum']['MF']['Few']     = Pimf(-16.9, -.1, 4.1, 20.9 )
        self.MainFIS.Input['PlyNum']['MF']['TooMany'] = Pimf(4.1, 20.9, 25.1, 41.9)
        self.MainFIS.Input['PlyNum']['MF']['Any']     = Const(1.0)
        # Input - Number of UD-patches
        self.MainFIS.Input['PatchNum'] = dict()
        self.MainFIS.Input['PatchNum']['Range'] = [0.0, 5.]
        self.MainFIS.Input['PatchNum']['MF'] = dict()
        self.MainFIS.Input['PatchNum']['MF']['Few']     = Pimf(-7.3, -1.7, -0.3, 5.3)
        self.MainFIS.Input['PatchNum']['MF']['TooMany'] = Pimf(-.3, 5.3, 6.7, 12.3)
        self.MainFIS.Input['PatchNum']['MF']['Any']     = Const(1.0)
        # Output - Manufacturing effort
        self.MainFIS.Output['ME'] = dict()
        self.MainFIS.Output['ME']['Range'] = [0, 1.0]
        self.MainFIS.Output['ME']['MF'] = dict()
        self.MainFIS.Output['ME']['MF']['VeryLow'] = Gaussmf(0.082, 0.0)
        self.MainFIS.Output['ME']['MF']['Low'] = Gaussmf(0.082, 0.1)
        self.MainFIS.Output['ME']['MF']['Moderate'] = Gaussmf(0.082, 0.2)
        self.MainFIS.Output['ME']['MF']['High'] = Gaussmf(0.082, 0.4)
        self.MainFIS.Output['ME']['MF']['VeryHigh'] = Gaussmf(0.082, 0.8)
        self.MainFIS.Output['ME']['MF']['NotManufacturable'] = Gaussmf(0.082, 1.0)
        # Rules Sub-FIS3
        self.MainFIS.Rules['AllGood'] = ['AND','Sub1','Good','Sub2','Good','Sub3','Good','EdgeRadius','Moderate','AspectRatio','Moderate','PlyNum','Few','PatchNum','Few','THEN','ME','VeryLow']
        self.MainFIS.Rules['AllBad'] = ['OR','Sub1','Bad','Sub2','Bad','Sub3','Bad','EdgeRadius','TooSmall','AspectRatio','TooBig','PlyNum','TooMany','PatchNum','TooMany','THEN','ME','NotManufacturable']

        # Store BraidFIS
        # ------------------------------------------------------------------- #
        self.BraidFIS = [self.SubFIS1,self.SubFIS2,self.SubFIS3,self.MainFIS]
        pickle.dump(self.BraidFIS,open(self.CurPath+'/BraidFIS.p','wb'))
        
        # Store optimal values
        # ------------------------------------------------------------------- #        
        self.FISExtremalMinInput = dict()        # Best configuration
        self.FISExtremalMinInput['BraidAngle']            = 25.
        self.FISExtremalMinInput['YarnWidth']             = 2.7
        self.FISExtremalMinInput['RadiusDiameterRatio']   = 10.
        self.FISExtremalMinInput['EdgeRadius']            = 3.
        self.FISExtremalMinInput['AspectRatio']           = 2.  
        self.FISExtremalMinInput['PlyNum']                = 5.   
        self.FISExtremalMinInput['PatchNum']              = 0.
        self.FISExtremalMaxInput = dict()        # Worst configuration
        self.FISExtremalMaxInput['BraidAngle']            = 75.
        self.FISExtremalMaxInput['YarnWidth']             = 4.
        self.FISExtremalMaxInput['RadiusDiameterRatio']   = 0.
        self.FISExtremalMaxInput['EdgeRadius']            = 5.
        self.FISExtremalMaxInput['AspectRatio']           = 4.
        self.FISExtremalMaxInput['PlyNum']                = 20.
        self.FISExtremalMaxInput['PatchNum']              = self.MaxPatches
        self.FISxMinInputList = [25., 2.7, 10., 3., 2.,  5., 0.]        
        self.FISxMaxInputList = [75., 4.,   0., 5., 4., 20., self.MaxPatches]
        
        # Store elaboration hints
        # ------------------------------------------------------------------- #  
        ElaborationHints = dict()
        # Hint 1
        Hint = 'Increase take-up speed {1}; Reduce horn gear speed {2}'
        ElaborationHints[Hint] = {'Rule': ['BraidOpenHornGearSpeedInfinity', 'CurvatureTooGreatHornGear', \
            'OverCriticalCombAngleCurvature4', 'CriticalCombAngleCurvature5', 'CombAngleCurvature3', \
            'AlmostNoCurvature6', 'OverCriticalCombAngleCurvature5', 'OverCriticalCombAngleCurvature6', \
            'CriticalCombAngleCurvature6', 'ModerateCombAngleCurvature2'], \
            'VerbalVariable': ['BraidAngle']}
        # Hint 2 -> YarnWidth is Big
        Hint = 'Increase take-up speed {1}; Roving with more filaments {1}; Reduce horn gear speed {2}; Increase carrier number {3}'
        ElaborationHints[Hint] = {'Rule': ['BraidOpenHornGearSpeedInfinity', 'CurvatureTooGreatBraidOpen', \
            'OverCriticalCombAngleCurvature4', 'OverCriticalCombAngleCurvature5', 'CriticalCombAngleCurvature3', \
            'LowCurvature4', 'LowCurvature5'], \
            'VerbalVariable': ['YarnWidth']}
        # Hint 3
        Hint = 'Reduce take-up speed {1} Increase horn gear speed {2}'
        ElaborationHints[Hint] = {'Rule': ['OverCompTakeUpSpeedInfinity', 'CurvatureTooGreatTakeUp', \
            'OverCriticalCombAngleCurvature1', 'OverCriticalCombAngleCurvature2', 'CriticalCombAngleCurvature1', \
            'ModerateCombAngleCurvature1', 'AlmostNoCurvature1', 'OverCriticalCombAngleCurvature3', 'CriticalCombAngleCurvature2', \
            'CombAngleCurvature1', 'AlmostNoCurvature2'], \
            'VerbalVariable': ['BraidAngle']}
        # Hint 4 -> YarnWidth is Small
        Hint = 'Reduce take-up speed {1}; Roving with less filaments {1}; Increase horn gear speed {2}; Reduce carrier number {3}'
        ElaborationHints[Hint] = {'Rule': ['OverCompTakeUpSpeedInfinity', 'CurvatureTooGreatOverCompact', \
            'OverCriticalCombAngleCurvature1', 'OverCriticalCombAngleCurvature2', 'CriticalCombAngleCurvature1', \
            'LowCurvature1', 'LowCurvature2', 'NoCurvature1'], \
            'VerbalVariable': ['YarnWidth']}
        # Hint 5
        Hint = 'Reduce braid layers (if possible)'
        ElaborationHints[Hint] = {'Rule': ['VLowestPTimes', 'NoCurvature2', 'NoCurvature3', 'NoCurvature4', 'AllGood'], \
            'VerbalVariable': ['BraidAngle', 'YarnWidth', 'EdgeRadius', 'AspectRatio', 'PlyNum', 'PatchNum']}
        # Hint 6
        Hint = 'Increase take-up speed {1}'
        ElaborationHints[Hint] = {'Rule': ['LowPTimes', 'ModeratePTimes', 'HighPTimes', 'VeryHighPTimes', \
            'NoCurvature1', 'AlmostNoCurvature3', 'NoCurvature2', 'AlmostNoCurvature5', 'NoCurvature3', \
            'AlmostNoCurvature7', 'NoCurvature4', 'AlmostNoCurvature8', 'NoCurvature5'], \
            'VerbalVariable': ['BraidAngle', 'YarnWidth']}
        # Hint 7
        Hint = 'Increase radius of lengthwise curvature {1}; Reduce diameter of mandrel {1}'
        ElaborationHints[Hint] = {'Rule': ['CurvatureTooGreatTakeUp', 'OverCriticalCombAngleCurvature1', \
            'OverCriticalCombAngleCurvature2', 'CriticalCombAngleCurvature1', 'ModerateCombAngleCurvature1', \
            'CurvatureTooGreatHornGear', 'AlmostNoCurvature1', 'OverCriticalCombAngleCurvature3', 'CriticalCombAngleCurvature2', \
            'CombAngleCurvature1', 'AlmostNoCurvature2', 'CriticalCombAngleCurvature3', 'CriticalCombAngleCurvature4', \
            'CombAngleCurvature2', 'AlmostNoCurvature4', 'OverCriticalCombAngleCurvature4', 'CriticalCombAngleCurvature5', \
            'CombAngleCurvature3', 'AlmostNoCurvature6', 'OverCriticalCombAngleCurvature5', 'OverCriticalCombAngleCurvature6', \
            'CriticalCombAngleCurvature6', 'ModerateCombAngleCurvature2', 'CurvatureTooGreatOverCompact', 'CurvatureTooGreatBraidOpen', \
            'OverCriticalCombAngleCurvature1', 'OverCriticalCombAngleCurvature2', 'CriticalCombAngleCurvature1', 'LowCurvature1', \
            'LowCurvature2', 'NoCurvature1', 'OverCriticalCombAngleCurvature3', 'CriticalCombAngleCurvature2', 'ModerateCurvature1', \
            'LowCurvature3', 'OverCriticalCombAngleCurvature4', 'OverCriticalCombAngleCurvature5', 'CriticalCombAngleCurvature3', \
            'LowCurvature4', 'LowCurvature5'], \
            'VerbalVariable': ['RadiusDiameterRatio']}
        # Hint 8
        Hint = 'May reduce number of layers (if possible)'
        ElaborationHints[Hint] = {'Rule': ['NoCurvature1', 'AlmostNoCurvature3', 'NoCurvature2', 'AlmostNoCurvature5', \
            'NoCurvature3', 'AlmostNoCurvature7', 'NoCurvature4', 'AlmostNoCurvature8', 'NoCurvature5', 'NoCurvature2', \
            'NoCurvature3', 'NoCurvature4'], \
            'VerbalVariable': ['RadiusDiameterRatio']}
        # Hint 9
        Hint = 'Increase radius of lengthwise curvature {1}; Reduce diameter of mandrel {1}'
        ElaborationHints[Hint] = {'Rule': ['CriticalCombAngleCurvature3', 'CriticalCombAngleCurvature4', \
            'CombAngleCurvature2', 'AlmostNoCurvature4'], \
            'VerbalVariable': ['BraidAngle']}    
        # Hint 10
        Hint = 'Increase radius of lengthwise curvature {1}; Reduce diameter of mandrel {1}'
        ElaborationHints[Hint] = {'Rule': ['OverCriticalCombAngleCurvature3', 'CriticalCombAngleCurvature2', \
            'ModerateCurvature1', 'LowCurvature3'], \
            'VerbalVariable': ['YarnWidth']} 
        # Hint 11
        Hint = 'Increase edge radii {1}'
        ElaborationHints[Hint] = {'Rule': ['AllBad'], \
            'VerbalVariable': ['EdgeRadius']} 
        # Hint 12
        Hint = 'Reduce aspect ratio {1}'
        ElaborationHints[Hint] = {'Rule': ['AllBad'], \
            'VerbalVariable': ['AspectRatio']} 
        # Hint 13
        Hint = 'Reduce number of braid layers {1}'
        ElaborationHints[Hint] = {'Rule': ['AllBad'], \
            'VerbalVariable': ['PlyNum']} 
        # Hint 14
        Hint = 'Reduce number of UD patches {1}'
        ElaborationHints[Hint] = {'Rule': ['AllBad'], \
            'VerbalVariable': ['PatchNum']} 
        self._ElaborationHints = ElaborationHints
        
    def LoadBraidFIS(self,FileName='BraidFIS',Path=[]):
        if not Path:
            Path = self.CurPath
        self.BraidFIS = pickle.load(open(Path+'/'+FileName+'.p','rb') )
    def _WriteXinDict(self,x):
        # Map input
        # ------------------------------------------------------------------- #
        InVals                         = dict()
        InVals['BraidAngle']           = x[0]   # [Deg] ... [15 75]) -> 25
        InVals['YarnWidth']            = x[1]   # [mm]  ... [1.5 4]  -> 2.7
        InVals['RadiusDiameterRatio']  = x[2]   # [-]   ... [0 10]   -> 10  R/d
        InVals['EdgeRadius']           = x[3]   # [mm]  ... [3 5]    -> 5
        InVals['AspectRatio']          = x[4]   # [-]   ... [2 4]    -> 2
        InVals['PlyNum']               = x[5]   # [-]   ... [5 20]   -> 5
        InVals['PatchNum']             = x[6]*5./self.MaxPatches # [-]   ... [0 MaxPatches]    -> 0
        self.InVals                    = InVals
    def __call__(self,x):
        if isinstance(x, dict):
            UList = x['ProfileCircumferences']       
            PhiList = x['BraidingAngle']
            lList = x['PathLength']
            bList = list()
            for (iU, iPhi) in zip(UList, PhiList):
                bList.append(self.ComputeYarnWidth(iU,iPhi))
            if 'ProfileMinRadius' in x.keys():
                rList = x['ProfileMinRadius']
            else:
                rList = [5.]*len(UList)
            if 'PathRadii' in x.keys():
                if len(x['PathRadii'])==len(x['BraidingAngle']):
                    RList = x['PathRadii']
                else:
                    RList = list()
                    RList.append(x['PathRadii'][0])
                    for (ix0, ix1) in zip(x['PathRadii'][:-1], x['PathRadii'][1:]):
                        RList.append(min(ix0,ix1))
                    RList.append(x['PathRadii'][-1])
                RDList = list()
                for (R, u) in zip(RList, UList):
                    RDList.append(R/(u/np.pi))
            else:
                RDList = [10.]*len(UList)
            if 'ProfileAspect' in x.keys():
                abList = x['ProfileAspect']
            else:
                abList = [2.]*len(UList)
            if 'PlyNum' in x.keys():
                plyNumList = x['PlyNum']
            else:
                plyNumList = [5.]*len(UList)
            if 'PatchNum' in x.keys():
                patchNumList = x['PatchNum']
            else:
                patchNumList = [0.]*len(UList)
            if 'SupPathRadii' in x.keys():
                RSupList = x['SupPathRadii']
                
            (MEList, MEReasonList, MEHintList) = (list(), list(), list())
            xList = list()
            for (iPhi, ib, iRD, ir, iab, iPlyNum, iPatch) in zip(PhiList, bList, RDList, rList, abList, plyNumList, patchNumList): 
                x = cp.deepcopy([iPhi, ib, iRD, ir, iab, iPlyNum, iPatch])
                MEList.append(self._ComputeME(x))
                MEReasonList.append(self.Reasoning())
                MEHintList.append(self.ElaborationHints())
                xList.append(x)
            self.MEList = MEList
            self.xList = xList
            self.MEReasonList = MEReasonList
            self.MEHintList = MEHintList
            
            METrap = 0.
            for (iL, iMEb, iMEa) in zip(lList, MEList[1:], MEList[:-1]):
                METrap += iL/2*(iMEa+iMEb)
            METrap /= sum(lList)
            return METrap
        else:
            return self._ComputeME(x)
        
    def _ComputeME(self,x):
        # Initializations
        # ------------------------------------------------------------------- #
        yL = [15, 1.5,  0., 3., 2.,  5., 0.]
        yU = [75,  4., 10., 5., 4., 20., 5.]
        (MEmin, MEmax) = (0.1, 1.1)
        #      Phi,   b,     R/D,   r,     a/b,   Ply#,  Patch#
        y0  = [   0.,   0.1,    0.,   0.1, -1000, -40.0,    0.]
        ME0 = [MEmax, MEmax, MEmax, MEmax, MEmin, MEmin, MEmin]
        y1  = [  90.,  10.0, 1.0e6, 1.0e6, 1.0e3, 100.0,   50.]
        ME1 = [MEmax, MEmax, MEmin, MEmin, MEmax, MEmax, MEmax]
        # Conduct bound check
        # ------------------------------------------------------------------- #
        xOrg = list()
        BoundViolation = False
        for (iX, iyL, iyU) in zip(x, yL, yU):
            if (iX < iyL) or (iX > iyU):
                BoundViolation = True
                if (iX < iyL):
                    xOrg.append(iyL)
                else:
                    xOrg.append(iyU)
            else:
                xOrg.append(iX)
                
        # Compute FIS response at bound
        # ------------------------------------------------------------------- #
        self._WriteXinDict(xOrg)
        self.InVals['Sub1']            = self.BraidFIS[0].EvalFIS(self.InVals)
        self.InVals['Sub2']            = self.BraidFIS[1].EvalFIS(self.InVals)
        self.InVals['Sub3']            = self.BraidFIS[2].EvalFIS(self.InVals)
        MEOrg = self.BraidFIS[3].EvalFIS(self.InVals)
        
        # Bound has been violated
        # ------------------------------------------------------------------- #        
        if BoundViolation:
            MEReturn = MEOrg
            for (iX, iyL, iyU, iy0, iy1, iME0, iME1) in zip(x, yL, yU, y0, y1, ME0, ME1):
                if (iX < iyL):
                    MEReturn += (iME0-MEOrg)/(iy0-iyL)*(iX-iyL)
                elif (iX > iyU):
                    MEReturn += (iME1-MEOrg)/(iy1-iyU)*(iX-iyU)
            return MEReturn
        else:
            return MEOrg
        
    def VarInfo(self):
        VarInfoStr = '# BraidAngle [Deg] ... [15 75]) -> 25\n'
        VarInfoStr = VarInfoStr+'# YarnWidth  [mm]  ... [1.5 4]  -> 2.7\n'
        VarInfoStr = VarInfoStr+'# Curvature  [-]   ... [0 10]   -> 10  R/d\n'
        VarInfoStr = VarInfoStr+'# EdgeRadius [mm]  ... [3 5]    -> 5\n'
        VarInfoStr = VarInfoStr+'# AspectRatio[-]   ... [2 4]    -> 2\n'
        VarInfoStr = VarInfoStr+'# PlyNum     [-]   ... [5 20]   -> 5\n'
        VarInfoStr = VarInfoStr+'# PatchNum   [-]   ... [0 MaxPatches] -> 0\n' 
        self.VarInfoStr
        return self.VarInfoStr
        
    def ComputeResponse(self,x):
        return self(x)
        
    def ComputeRespAndSens(self,x):
        '''
        Computes sensitivities! 
        - if x is a dict, than all sensitivities need be passed with SENS attached to key
        - returns: ME, MEsens, Reason (if dict, than last one is a list!)
        '''
        if isinstance(x, dict):
            # Initializations
            # --------------------------------------------------------------- #
            SupportedDictEntries = self.SupportDictKeys
            try:
                nDV = np.size(x['BraidingAngleSENS'][0])
            except:
                ErrorMSG = 'No sensitivities in input dictionary or wrong defined!'
                print ErrorMSG 
                sys.exit(ErrorMSG)
                return ErrorMSG 
            MEsens = np.zeros(np.shape(x['BraidingAngleSENS'][0]))
            (RespKeys, SensKeys) = (list(), list())
            for iKey in x.keys():
                if (iKey in SupportedDictEntries) or (iKey[:-4] in SupportedDictEntries):
                    if 'SENS' not in iKey:
                        RespKeys.append(iKey)
                    else:
                        SensKeys.append(iKey)
            '''
            if len(RespKeys)!=len(SensKeys):
                ErrorMSG = 'Not all sensitivities via SENS are given!'
                print ErrorMSG 
                sys.exit(ErrorMSG)
                return ErrorMSG 
            '''
            
            # Compute and store base results
            # --------------------------------------------------------------- #
            MEorg = self(x)
            MEList = self.MEList
            xList = self.xList
            MEReasonList = self.MEReasonList
            MEHintList = self.MEHintList
            
            # Compute FIS sensitivities
            # --------------------------------------------------------------- #
            for DictKey in SensKeys:
                for iME in range(len(x[DictKey])):
                    MEInp = cp.deepcopy(x)
                    MEInp[DictKey][iME] += self.FDTol
                    MEsens += ((self(MEInp)-MEorg)/self.FDTol)*x[DictKey+'SENS'][iME]
            
            # Store results in object
            # --------------------------------------------------------------- #
            self.MEList = MEList
            self.xList = xList
            self.MEReasonList = MEReasonList
            self.MEHintList = MEHintList
            return [MEorg, MEsens, MEReasonList]
        else:
            # Compute FIS sensitivities
            # --------------------------------------------------------------- #
            r0 = self(x)
            Reason = self.Reasoning()
            rSens = []
            for i in range(np.size(x)):
                X = cp.deepcopy(x)
                X[i]+= self.FDTol
                rSens.append((self(X)-r0)/self.FDTol)
            return [r0,np.array(rSens),Reason]
        
    def Reasoning(self):
        # Compute FIS reasoning
        # ------------------------------------------------------------------- #
        CritRule                        = self.BraidFIS[3].GetMaxImplicationKey()
        Reason                          = self.BraidFIS[3].GetMaxAntecentOfMaxImplicationKey()
        if 'Sub' in Reason[0]:
            if Reason[0]=='Sub1':
                CritRule                = self.BraidFIS[0].GetMaxImplicationKey()
                Reason                  = self.BraidFIS[0].GetMaxAntecentOfMaxImplicationKey()
            elif Reason[0]=='Sub2':
                CritRule                = self.BraidFIS[1].GetMaxImplicationKey()
                Reason                  = self.BraidFIS[1].GetMaxAntecentOfMaxImplicationKey()
            elif Reason[0]=='Sub3':
                CritRule                = self.BraidFIS[2].GetMaxImplicationKey()
                Reason                  = self.BraidFIS[2].GetMaxAntecentOfMaxImplicationKey()
        self._ElaborationList = [CritRule, Reason[0], Reason[1]]
        return 'In rule "'+CritRule+'": "'+Reason[0]+'" is "'+Reason[1]+'"'
        
    def ElaborationHints(self): 
        '''
        Gives hints on how to finally elaborate design such that lowest manufacturing effort level
        German: Ausgestaltungshinweise, sodass Herstellungsaufwaende minimal -> Konstruktionslehre: 'Ausarbeiten' 
        '''
        # Load elaboration hints
        ElaborationHints = self._ElaborationHints
        # Evaluate elaboration hint based on critical rule
        [CritRule, VerbalVariable] = self._ElaborationList[:2]
        for iHint in ElaborationHints.keys():
            if CritRule in ElaborationHints[iHint]['Rule']:
                if VerbalVariable in ElaborationHints[iHint]['VerbalVariable']:
                    return iHint
        return 'Eorror in hint computation for RULE: "%s" and VERBAL VARIABLE: "%s"'%(CritRule, VerbalVariable)
        
    def ComputeYarnWidth(self,U,Phi):
        return math.cos(Phi*math.pi/180.0)*2.0*(U)/self.nBobins
        
    def PlotAllResponseSurfaces(self,x,PlotSamplePerAx=100, UseTex=True, Extremal=True, \
            Language='Eng'): # 'Eng' 'Ger'
        try:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except:
            logging.error('Matplotlib could not be imported!')
        plt.rc('text', usetex=UseTex)
        #plt.rc('font', family='serif')
        fig = plt.figure(figsize=plt.figaspect(0.5))
        if Language=='Eng':
            fig.suptitle(r'Manufacturing Effort Model', fontsize=28, fontweight='bold')
        else:
            fig.suptitle(r'Fertigungsaufwandsmodell', fontsize=28, fontweight='bold')
        
        # BraidAngle vs YarnWidth
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        X = np.linspace(self.SubFIS1.Input['BraidAngle']['Range'][0], \
            self.SubFIS1.Input['BraidAngle']['Range'][1],num=PlotSamplePerAx)
        Y = np.linspace(self.SubFIS1.Input['YarnWidth']['Range'][0], \
            self.SubFIS1.Input['YarnWidth']['Range'][1],num=PlotSamplePerAx)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        if Extremal:
            ZMin = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
            ZMax = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        for ix in range(PlotSamplePerAx):
            for jy in range(PlotSamplePerAx):
                xTmp = cp.deepcopy(x)
                xTmp[0] = X[ix,jy]
                xTmp[1] = Y[ix,jy]
                Z[ix,jy] = self(xTmp)
                if Extremal:
                    TMPInput = cp.deepcopy(self.FISxMinInputList )
                    TMPInput[0] = X[ix,jy]
                    TMPInput[1] = Y[ix,jy]
                    ZMin[ix,jy] = self(TMPInput)
                    TMPInput = cp.deepcopy(self.FISxMaxInputList)
                    TMPInput[0] = X[ix,jy]
                    TMPInput[1] = Y[ix,jy]
                    ZMax[ix,jy] = self(TMPInput) 
        if Extremal:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Braiding angles [DEG]',r'Yarn width [mm]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Flechtwinkel [Grad]',r'Ablegebreite [mm]', AlphaVal=1.)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, Z, alpha = .667, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, ZMax, alpha = .333, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Braiding angles [DEG]',r'Yarn width [mm]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Flechtwinkel [Grad]',r'Ablegebreite [mm]', AlphaVal=1.)
        # BraidAngle vs RadiusDiameterRatio 
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        X = np.linspace(self.SubFIS2.Input['RadiusDiameterRatio']['Range'][0], \
            self.SubFIS2.Input['RadiusDiameterRatio']['Range'][1],num=PlotSamplePerAx)        
        Y = np.linspace(self.SubFIS1.Input['BraidAngle']['Range'][0], \
            self.SubFIS1.Input['BraidAngle']['Range'][1],num=PlotSamplePerAx)   
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        if Extremal:
            ZMin = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
            ZMax = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        for ix in range(PlotSamplePerAx):
            for jy in range(PlotSamplePerAx):
                xTmp = cp.deepcopy(x)
                xTmp[2] = X[ix,jy]
                xTmp[0] = Y[ix,jy]
                Z[ix,jy] = self(xTmp)
                if Extremal:
                    TMPInput = cp.deepcopy(self.FISxMinInputList )
                    TMPInput[2] = X[ix,jy]
                    TMPInput[0] = Y[ix,jy]
                    ZMin[ix,jy] = self(TMPInput)
                    TMPInput = cp.deepcopy(self.FISxMaxInputList)
                    TMPInput[2] = X[ix,jy]
                    TMPInput[0] = Y[ix,jy]
                    ZMax[ix,jy] = self(TMPInput) 
        if Extremal:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Ratio of radius to diameter [-]',r'Braiding angles [DEG]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Radiusdurchmesserverhaeltnis [-]',r'Flechtwinkel [Grad]', AlphaVal=1.)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, Z, alpha = .667, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, ZMax, alpha = .333, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Ratio of radius to diameter [-]',r'Braiding angles [DEG]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Radiusdurchmesserverhaeltnis [-]',r'Flechtwinkel [Grad]', AlphaVal=1.)
        # EdgeRadius vs AspectRatio
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        #X = np.linspace(self.MainFIS.Input['EdgeRadius']['Range'][0], \
        #    self.MainFIS.Input['EdgeRadius']['Range'][1],num=PlotSamplePerAx)
        #Y = np.linspace(self.MainFIS.Input['AspectRatio']['Range'][0], \
        #    self.MainFIS.Input['AspectRatio']['Range'][1],num=PlotSamplePerAx)
        X = np.linspace(3.0,5.0,num=PlotSamplePerAx)
        Y = np.linspace(2.5,4.0,num=PlotSamplePerAx)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        if Extremal:
            ZMin = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
            ZMax = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        for ix in range(PlotSamplePerAx):
            for jy in range(PlotSamplePerAx):
                xTmp = cp.deepcopy(x)
                xTmp[3] = X[ix,jy]
                xTmp[4] = Y[ix,jy]
                Z[ix,jy] = self(xTmp)
                if Extremal:
                    TMPInput = cp.deepcopy(self.FISxMinInputList )
                    TMPInput[3] = X[ix,jy]
                    TMPInput[4] = Y[ix,jy]
                    ZMin[ix,jy] = self(TMPInput)
                    TMPInput = cp.deepcopy(self.FISxMaxInputList)
                    TMPInput[3] = X[ix,jy]
                    TMPInput[4] = Y[ix,jy]
                    ZMax[ix,jy] = self(TMPInput) 
        if Extremal:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Edge radius [mm]',r'Aspect ratio [-]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Kantenradius [mm]',r'Seitenverhaeltnis [-]', AlphaVal=1.)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, Z, alpha = .667, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, ZMax, alpha = .333, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Edge radius [mm]',r'Aspect ratio [-]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Kantenradius [mm]',r'Seitenverhaeltnis [-]', AlphaVal=1.)
        # PlyNum vs PatchNum
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        #X = np.linspace(self.MainFIS.Input['PlyNum']['Range'][0], \
        #    self.MainFIS.Input['PlyNum']['Range'][1],num=PlotSamplePerAx)
        #Y = np.linspace(self.MainFIS.Input['PatchNum']['Range'][0], \
        #    self.MainFIS.Input['PatchNum']['Range'][1]/5.*self.MaxPatches,num=PlotSamplePerAx)
        X = np.linspace(8.,20.,num=PlotSamplePerAx)
        Y = np.linspace(2.,12,num=PlotSamplePerAx)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        if Extremal:
            ZMin = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
            ZMax = np.zeros([int(PlotSamplePerAx),int(PlotSamplePerAx)])
        for ix in range(PlotSamplePerAx):
            for jy in range(PlotSamplePerAx):
                xTmp = cp.deepcopy(x)
                xTmp[5] = X[ix,jy]
                xTmp[6] = Y[ix,jy]
                Z[ix,jy] = self(xTmp)
                if Extremal:
                    TMPInput = cp.deepcopy(self.FISxMinInputList )
                    TMPInput[5] = X[ix,jy]
                    TMPInput[6] = Y[ix,jy]
                    ZMin[ix,jy] = self(TMPInput)
                    TMPInput = cp.deepcopy(self.FISxMaxInputList)
                    TMPInput[5] = X[ix,jy]
                    TMPInput[6] = Y[ix,jy]
                    ZMax[ix,jy] = self(TMPInput) 
        if Extremal:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Number of plies [-]',r'Number of patches [-]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,ZMin,[],r'Anzahl Umflechtungen [mm]',r'Anzahl Verstaerkungslagen [-]', AlphaVal=1.)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, Z, alpha = .667, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.hold(True)
            surf = ax.plot_surface(X, Y, ZMax, alpha = .333, rstride=1, \
                cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            if Language=='Eng':
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Number of plies [-]',r'Number of patches [-]', AlphaVal=1.)
            else:
                SurfacePlotter(fig,ax,X,Y,Z,[],r'Anzahl Umflechtungen [mm]',r'Anzahl Verstaerkungslagen [-]', AlphaVal=1.)
        # Define last plot settings
        plt.tight_layout()
        plt.show()
        
    def PlotBraidFISOverRules(self,NumSupportPoints=20, ShowFig=True, FigFile=None, nRulesPerPlot=7.):
        if FigFile == None:
            FigList = [None]*4
        else:
            FigList = [FigFile+'-SUB1-FIS', FigFile+'-SUB2-FIS', FigFile+'-SUB3-FIS', FigFile+'-MAIN-FIS']
        self.BraidFIS[0].PlotFuzzyInferenceSystem(NumSupportPoints=NumSupportPoints, \
            ShowFig=ShowFig, FigFile=FigList[0], nRulesPerPlot=nRulesPerPlot)
        self.BraidFIS[1].PlotFuzzyInferenceSystem(NumSupportPoints=NumSupportPoints, \
            ShowFig=ShowFig, FigFile=FigList[1], nRulesPerPlot=nRulesPerPlot)
        self.BraidFIS[2].PlotFuzzyInferenceSystem(NumSupportPoints=NumSupportPoints, \
            ShowFig=ShowFig, FigFile=FigList[2], nRulesPerPlot=nRulesPerPlot)
        self.BraidFIS[3].PlotFuzzyInferenceSystem(NumSupportPoints=NumSupportPoints, \
            ShowFig=ShowFig, FigFile=FigList[3], nRulesPerPlot=nRulesPerPlot)
        
    def PrintMEInfoList(self):
        for (iME, iReason, iX, iHint) in zip(self.MEList, self.MEReasonList, self.xList, self.MEHintList):
            print 'Effort euals ', iME, ' because of :', iReason
            print ' -> ME input is:', iX
            print ' -> Possible Elaboration of design: ', iHint
            
    def ComputeCost(self,MEInp,tTurn=2,aPhi=0.8,aManPower=0.5):
        # tTurn     ... Turning time 10 sec according to LCC
        # aPhi      ... EUR/min for a braiding machine including ... (see paper!)
        # aManPower ... EUR/min for worker -> Europe 30 EUR/hour
        if isinstance(MEInp, dict):
            InfoStr = ''
            # Compute Cost from here on
            if 'PathLength' not in MEInp.keys():
                ErrorMSG = '"PathLength" not defined in MEInp, but needed!'
                print ErrorMSG 
                sys.exit(ErrorMSG)
                return [], [], ErrorMSG
        
            if 'PlyNum' not in MEInp.keys():
                MEInp['PlyNum'] = [1.]*len(MEInp['PathLength'])
                InfoStr += 'WARNING BraidME: No ply number given with MEInp["PlyNum"]\n'
                InfoStr += '  -> Set to MEInp["PlyNum"] = [1.]*Sections\n'
                
            LoopSet = zip(MEInp['BraidingAngle'][:-1], MEInp['BraidingAngle'][1:], \
                MEInp['ProfileCircumferences'][:-1], MEInp['ProfileCircumferences'][1:], \
                MEInp['PlyNum'], MEInp['PathLength'])
            self.UpdateHornGearParams()
            nHornGears = self.nHornGears
            HornGearSpeed = self.HornGearSpeed
            # Compute times
            tPhi = 0.
            for (Phi1, Phi2, U1, U2, nLayer, iLength) in LoopSet:
                PhiMid = 0.5*(Phi1+Phi2)
                UMid = 0.5*(U1+U2)
                tPhi += nLayer*math.tan(PhiMid*math.pi/180.)*nHornGears*iLength/(UMid*HornGearSpeed)    
            tDT = max(MEInp['PlyNum'])*tTurn
            tManPower = 0
            # Assembly costs
            Costs = (tPhi+tDT)*aPhi+tManPower*aManPower
            InfoStr += 'Cost computed for %i Bobins, %.2f EUR per machine time and %.2f EUR for man power'%(int(self.nBobins),aPhi,aManPower)
            self.Costs = Costs
            self.tPhi = tPhi
            self.InfoStr = InfoStr
            return Costs, tPhi, InfoStr
        else:
            ErrorMSG = 'Method needs MEInp because of length information!\n'
            ErrorMSG += '    -> See class description for MEInp definition'
            print ErrorMSG 
            sys.exit(ErrorMSG)
            return [], [], ErrorMSG
    
if __name__ == '__main__':
    x = [25.0, 2.7, 10, 5.0, 2.0, 5.0, 0.0]
    #x = [40.0, 3.0, 6., 4.4, 2.5, 7.0, 4.0]
    # BraidAngle [Deg] ... [15 75]) -> 25
    # YarnWidth  [mm]  ... [1.5 4]  -> 2.7
    # Curvature  [-]   ... [0 10]   -> 10  R/d
    # EdgeRadius [mm]  ... [3 5]    -> 5
    # AspectRatio[-]   ... [2 4]    -> 2
    # PlyNum     [-]   ... [5 20]   -> 5
    # PatchNum   [-]   ... [0 MaxPatches] -> 0
    
    from BraidME import BraidME
    
    BraidMEM = BraidME()
    if os.path.isfile('BraidFIS.p'):
        # Loads existing FIS
        BraidMEM.LoadBraidFIS()
    else:
        BraidMEM.CreateAndSaveBraidFIS()
    
    TestCase = 'TestXSens' 
    # 'TestX', 'TestXSens', 'PlotSurfaces', 'PlotRules', 'MEInpDict', 'MEInpDictSens', 'Costs'
    
    if TestCase == 'TestX':
        ME=BraidMEM.ComputeResponse(x)
        print 'ME has been computed to be:'
        print ME
        Reasoning=BraidMEM.Reasoning()
        print 'Reasoning is:'
        print Reasoning
        BraidMEM.PlotBraidFISOverRules(NumSupportPoints=40, ShowFig=True, FigFile=None, nRulesPerPlot=7.)
        
    elif TestCase == 'TestXSens':
        [ME, dMEdx, Reasoning] = BraidMEM.ComputeRespAndSens(x)
        print 'ME has been computed to be:'
        print ME
        print 'Reasoning is:'
        print Reasoning
        print 'Sensitivity is:'
        print dMEdx
    
    elif TestCase == 'PlotSurfaces':
        BraidMEM.PlotAllResponseSurfaces(x,PlotSamplePerAx=80, UseTex=False, Extremal=False, \
            Language='Eng')
            
    elif TestCase == 'PlotRules':
        BraidMEM.PlotBraidFISOverRules(NumSupportPoints=200, ShowFig=True, FigFile=None, nRulesPerPlot=7.)
        
    elif TestCase == 'MEInpDict':
        MEInp = pickle.load(open(os.path.join('01-BraidME-Data','MEInp.p'),'r') )
        #print BraidMEM.ComputeResponse(MEInp)
        ME = BraidMEM.ComputeResponse(MEInp)
        print 'ME has been computed to be:'
        print ME
        BraidMEM.PrintMEInfoList()   
        
    elif TestCase == 'MEInpDictSens':
        MEInp = pickle.load(open(os.path.join('01-BraidME-Data','MEInp.p'),'r') )
        #print BraidMEM.ComputeResponse(MEInp)
        [ME, MEsens, MEReasonList] = BraidMEM.ComputeRespAndSens(MEInp)
        print 'ME has been computed to be:'
        print ME
        BraidMEM.PrintMEInfoList()  
        
    elif TestCase == 'Costs':
        MEInp = pickle.load(open(os.path.join('01-BraidME-Data','MEInp.p'),'rb') )
        #print BraidMEM.ComputeResponse(MEInp)
        [Costs, tPhi, InfoStr] = BraidMEM.ComputeCost(MEInp)
        print 'Braiding costs are [EUR]:'
        print Costs
        print 'Braiding time is [sec]:'
        print tPhi
        print InfoStr
        
    print '----- Debug Mode -----'
    print 'DONE'