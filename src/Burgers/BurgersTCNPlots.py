"""
Set of Plots for Heat results 
"""
import pdb
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

        pred = plotData['pred']
        target = plotData['target']
        error = np.abs(pred[:] - target[:])

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)        
        cmap = mpl.cm.get_cmap('jet')  
        cmap_error = mpl.cm.get_cmap('inferno') 
        
        fig, ax = plt.subplots(2, 1)
        # pdb.set_trace()
        ax[0].plot(pred.T)
        ax[1].plot(target.T)

        ax[0].set_ylabel('prediction', fontsize=8)
        ax[1].set_ylabel('true', fontsize=8)

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
        
        fig, ax = plt.subplots(3, 1, figsize=(4, 7))
        
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


    def violinplot(self, l2Error, plotParams, savePath):
        """
        Args:
            plotParams (Dict):
        Vars:
            l2Error (ndarray): (M, numNoiseLevels, numSampTest*numNodes)
        """
        import matplotlib.patches as mpatches
        pp = plotParams

        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        fig, ax = plt.subplots()

        for m in range(l2Error.shape[0]):
            vp = ax.violinplot(l2Error[m], pp.xticksPlot[m], widths=3, showmeans=True)#,showmeans=False, showmedians=False, showextrema=False)
            
            add_label(vp, pp.label[m])
            # styling:
            for body in vp['bodies']:
                body.set_facecolor(pp.facecolor[m])
                body.set_edgecolor('black')
                body.set_alpha(0.4)
        ax.set(xlim=(0, 50), xticks=pp.xticks, xlabel=pp.xlabel, xticklabels=pp.xticklabels,
                ylim=(0, 0.5), yticks=np.arange(0, 0.5, 0.05), ylabel=pp.ylabel, title=pp.title)
        
        plt.legend(*zip(*labels), loc=2)
        self.save_show(1, savePath, fig)


    def noisePlots(self, plotData, plotParams, savePath):
        pp = plotParams
        self.imDim = pp.imDim

        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=4)
        cmap = mpl.cm.get_cmap('jet')  

        fig, ax = plt.subplots(1, len(plotData), figsize=(len(plotData)*2, 1*3))

        for i in range(len(plotData)):
            c_max = np.max(plotData[i])
            c_min = np.min(plotData[i])
            imParams = {'cmap':cmap, 'v_min':c_min, 'v_max':c_max}
            self.Heat_imshow(plotData[i], ax[i], Dict2Class(imParams))
            ax[i].set(title=f'{pp.titleLs[i]}')
            ax[i].axis('off')

        self.save_show(1, savePath, fig, bbox_inches='tight', pad_inches=0.1)
    