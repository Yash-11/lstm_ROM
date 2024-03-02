"""
Set of Plots for Heat results 
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np

from os.path import dirname, realpath, join
import sys
filePath = realpath(__file__)
srcDir = dirname(dirname(filePath))
sys.path.append(srcDir)

from Utils import Dict2Class


class Plots:

    def save_show(self, plot, save_Path, fig, format='pdf', bbox_inches=None, pad_inches=0.1):
        if save_Path:
            Path = save_Path+ f'.{format}'
            fig.savefig(Path, format=f'{format}', bbox_inches=bbox_inches, pad_inches=pad_inches) 
            print(f'saved plot: {Path}')
        # fig.show() if plot else 0
        plt.close(fig)
        plt.close('all')


    def imshow(self, imData, axes, imParams):
        ip = imParams
        alpha = imParams.alpha if hasattr(imParams, 'alpha') else 1
        axes.imshow(imData, interpolation='nearest', cmap=ip.cmap, vmin=ip.v_min, vmax=ip.v_max, alpha=alpha)


    def Simulation(self, plotData, plotParams, savePath):
        """
        Vars:
            imData (ndarray): [maxtimeStep, numNodes]
            idxLs (list[int]): idx of InitConds to plot; len=4
        """
        Data = plotData

        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)
        cmap = mpl.cm.get_cmap('jet') 

        fig, ax = plt.subplots()
        plt.imshow(plotData)
        self.save_show(1, savePath, fig, bbox_inches='tight', format='png')


    def plotPercentError(self, plotData, plotParams):
        pp = plotParams
        fig, ax = plt.subplots()
        ax.plot(plotData)
        # ax.set(xticks=pp.xticks, xlabel=pp.xlabel,
        #         yticks=pp.yticks, ylabel=pp.ylabel, 
        #         xticklabels=pp.xticklabels, yticklabels=pp.yticklabels,
        #         title=pp.title)
        plt.ylabel(pp.ylabel)
        plt.xlabel("timestep")

        self.save_show(1, pp.savePath, fig, bbox_inches='tight', format='png')


    
    def plotPred(self, plotData: Dict, plotParams, savePath: str):
        """
        Args:
            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, timeStepModel, 1, numNodes)
            target (ndarray): (numSampTest, timeStepModel, 1, numNodes)
        """

        pp = plotParams

        pred = plotData['pred'][:]
        target = plotData['target'][:]
        error = np.abs(pred[:] - target[:])

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)        
        cmap = mpl.cm.get_cmap('jet')  
        cmap_error = mpl.cm.get_cmap('inferno') 
        
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(pred.T)
        ax[1].plot(target.T)

        ax[0].set_ylabel('prediction', fontsize=8)
        ax[1].set_ylabel('true', fontsize=8)

        self.save_show(1, savePath, fig, bbox_inches='tight', format='png')

    def plotPredSingle(self, plotData: Dict, plotParams, savePath: str, timestep):
        """
        Args:
            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, timeStepModel, 1, numNodes)
            target (ndarray): (numSampTest, timeStepModel, 1, numNodes)
        """

        pp = plotParams

        pred = plotData['pred'][:]
        target = plotData['target'][:]
        
        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)    

        sid = pred.shape[1]//2      
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.diagonal(target[pp.tend]), label = 'training end')
        ax.plot(np.diagonal(target[timestep]), label = 'ground truth')
        ax.plot(np.diagonal(pred[timestep]), label = 'prediction')
        ax.legend()
        ax.set_ylabel('u', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.set_title(f'Prediction at timestep = {pp.tlable}')
        ax.set_ylim([27, 30])

        self.save_show(1, savePath, fig, bbox_inches='tight', format='png')

    def plotPredSingleVar(self, plotData: Dict, plotParams, savePath: str, timestep):
        """
        Args:
            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, timeStepModel, 1, numNodes)
            target (ndarray): (numSampTest, timeStepModel, 1, numNodes)
        """

        pp = plotParams

        pred = plotData['pred'][:]
        target = plotData['target'][:]
        var = plotData['var'][:]
        sig = var**0.5
        x = np.arange(pred.shape[1])
        
        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)        

        sid = pred.shape[1]//2     
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(target[0][sid], label = 'training end')
        ax.plot(target[timestep][sid], label = 'ground truth')
        ax.plot(pred[timestep][sid], label = 'prediction')
        # pdb.set_trace()
        ax.fill_between(x, pred[timestep]+sig[timestep], pred[timestep]-sig[timestep], facecolor='blue', alpha=0.3)
        ax.legend()
        ax.set_ylabel('u', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.set_title(f'Prediction at timestep = {timestep}')

        self.save_show(1, savePath, fig, bbox_inches='tight', format='png')

    def implotPred(self, plotData: Dict, plotParams, savePath: str):
        """
        Args:

            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, timeStepModel, 1, numNodes)
            target (ndarray): (numSampTest, timeStepModel, 1, numNodes)
        """

        pp = plotParams

        pred = plotData['pred'].T
        target = plotData['target'].T
        error = pred[:] - target[:]

        plt.close("all")
        # mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        # mpl.rc('text', usetex=False)
        # mpl.rc('font', size=8)        
        cmap = mpl.cm.get_cmap('plasma')  
        cmapError = mpl.cm.get_cmap('viridis') 
        
        fig, ax = plt.subplots(3, 1, figsize=(7, 6))
        
        for i in range(1):

            row1 = ax[0]; row2 = ax[1]; row3 = ax[2]

            c_max = np.max(np.array([ pred, target]))
            c_min = np.min(np.array([ pred, target]))
            imParams = {'cmap':cmap, 'v_min':c_min, 'v_max':c_max}

            c_minError = np.min(error)
            c_maxError = np.max(error)
            imParamsError = {'cmap':cmapError, 'v_min':c_minError, 'v_max':c_maxError}

            self.imshow(pred, row1, Dict2Class(imParams))
            self.imshow(target, row2, Dict2Class(imParams))
            self.imshow(error, row3, Dict2Class(imParamsError))

            # --------------------- colorbar image
            p0 = row1.get_position().get_points().flatten()
            p1 = row2.get_position().get_points().flatten()
            w = (p0[2]-p0[0])*0.05
            ax_cbar = fig.add_axes([p1[2]+w, p1[1], w, p0[3]-p1[1]])
            ticks = np.linspace(0, 1, 5)
            tickLabels = [f'{t0:02.2f}' for t0 in np.linspace(c_min, c_max, 5)]

            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), 
                    orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)
            

            # ----------------------- colorbar error
            p0 = row3.get_position().get_points().flatten()
            w = (p0[2]-p0[0])*0.05
            ax_cbar = fig.add_axes([p0[2]+w, p0[1], w, p0[3]-p0[1]])
            ticks = np.linspace(0, 1, 5)
            tickLabels = [f'{t0:02.2f}' for t0 in np.linspace(c_minError, c_maxError, 5)]

            cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmapError), 
                    orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)

            row1.set_ylabel('prediction')
            row2.set_ylabel('true')
            row3.set_ylabel('L1 Error')
            row3.set_xlabel('Time steps')

        self.save_show(1, savePath, fig, bbox_inches='tight', format='png') 

    def imgPlot(self, plotData: Dict, plotParams, savePath: str):
        """
        Args:

            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, timeStepModel, 1, numNodes)
            target (ndarray): (numSampTest, timeStepModel, 1, numNodes)
        """

        pp = plotParams

        im = plotData['data'].T

        plt.close("all") 
        cmap = mpl.cm.get_cmap('plasma')  
        
        fig, ax = plt.subplots(1, 1)#, figsize=(7, 6))
        imParams = {'cmap':cmap, 'v_min':pp.v_min, 'v_max':pp.v_max}
        self.imshow(im, ax, Dict2Class(imParams))
        self.save_show(1, savePath, fig, bbox_inches='tight', format='png') 

    def cbar(self, plotData: Dict, plotParams, savePath: str):

        pp = plotParams
        plt.close("all") 
        cmap = mpl.cm.get_cmap('plasma')  
        
        fig, ax = plt.subplots(1, 1)#, figsize=(7, 6))
        # --------------------- colorbar image
        p0 = ax.get_position().get_points().flatten()
        w = (p0[2]-p0[0])*0.05
        ax_cbar = fig.add_axes([p0[2]+w, p0[1], w, p0[3]-p0[1]])
        ticks = np.linspace(0, 1, 5)
        tickLabels = [f'{t0:02.2f}' for t0 in np.linspace(pp.v_min, pp.v_max, 5)]

        cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), 
                orientation='vertical', ticks=ticks)
        cbar.set_ticklabels(tickLabels)
        self.save_show(1, savePath, fig, bbox_inches='tight', format='png') 


    def plotPredSingleAE(self, plotData: Dict, plotParams, savePath: str, timestep):
        """
        Args:
            plotData (Dict):
        Vars:
            pred (ndarray): (numSampTest, H, W)
            target (ndarray): (numSampTest, H, W)
        """

        pp = plotParams

        pred = plotData['pred'][:]
        target = plotData['target'][:]
        
        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)   

        sid = pred.shape[0]//2     
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.diagonal(target[timestep]), label = 'ground truth')
        ax.plot(np.diagonal(pred[timestep]), label = 'prediction')
        ax.legend()
        ax.set_ylabel('u', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.set_title(f'Prediction at timestep = {timestep}')

        self.save_show(1, savePath, fig, bbox_inches='tight', format='png')