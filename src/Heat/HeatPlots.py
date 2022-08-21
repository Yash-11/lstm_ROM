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


    def Heat_imshow(self, imData, axes, imParams):
        H, w = self.imDim
        ip = imParams
        alpha = imParams.alpha if hasattr(imParams, 'alpha') else 1
        axes.imshow(imData.reshape((H, w)), interpolation='nearest', cmap=ip.cmap, vmin=ip.v_min, vmax=ip.v_max, alpha=alpha)


    def femSimulation(self, plotData, plotParams, savePath):
        """
        Vars:
            imData (ndarray): [maxtimeStep, 1, numNodes]
            idxLs (list[int]): idx of InitConds to plot; len=4
        """
        imData = plotData
        idxLs, self.imDim = plotParams

        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)
        cmap = mpl.cm.get_cmap('jet') 

        fig, ax = plt.subplots(1, len(idxLs))

        for i, idx in enumerate(idxLs):
            imParams = {'cmap':cmap, 'v_min':np.min(imData), 'v_max':np.max(imData)}

            self.Heat_imshow(imData[idx, 0], ax[i], Dict2Class(imParams))
            
            ax[i].axis('off')

        self.save_show(1, savePath, fig, bbox_inches='tight')


    def plotPercentError(self, plotData, plotParams):
        pp = plotParams
        fig, ax = plt.subplots()
        ax.plot(plotData)
        ax.set(xticks=pp.xticks, xlabel=pp.xlabel,
                yticks=pp.yticks, ylabel=pp.ylabel, 
                xticklabels=pp.xticklabels, yticklabels=pp.yticklabels,
                title=pp.title)

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
        self.imDim = pp.imDim
        idxLs = pp.tStepModelPlot

        pred = plotData['pred'][pp.tStepPlot]
        target = plotData['target'][pp.tStepPlot]
        error = np.abs(pred[:] - target[:])

        numPlots = pred.shape[0]
        ch = pred.shape[2]

        plt.close("all")
        mpl.rcParams['font.family'] = ['serif']  # default is sans-serif
        mpl.rc('text', usetex=False)
        mpl.rc('font', size=8)        
        cmap = mpl.cm.get_cmap('jet')  
        cmap_error = mpl.cm.get_cmap('inferno') 
        
        fig, ax = plt.subplots(3*ch, numPlots, figsize=(numPlots*2, ch*3))

        for i in range(numPlots):
            for j in range(ch):
                t_ij = target[i, idxLs[i], j]
                p_ij = pred[i, idxLs[i], j]
                c_max = np.max(np.array([ t_ij, p_ij]))
                c_min = np.min(np.array([ t_ij, p_ij]))

                imParams = {'cmap':cmap, 'v_min':c_min, 'v_max':c_max}
                self.Heat_imshow(t_ij, ax[3*j, i], Dict2Class(imParams))
                self.Heat_imshow(p_ij, ax[3*j+1, i], Dict2Class(imParams))

                e_ij = p_ij - t_ij
                c_max_error = np.max(e_ij)
                c_min_error = np.min(e_ij)
                imParams = {'cmap':cmap_error, 'v_min':c_min_error, 'v_max':c_max_error}
                self.Heat_imshow(e_ij, ax[3*j+2, i], Dict2Class(imParams))

                # colorbar image
                p0 =ax[3*j, i].get_position().get_points().flatten()
                p1 = ax[3*j+1, i].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p1[2]+0.0075, p1[1], 0.005, p0[3]-p1[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min, c_max, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                
                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                # colorbar error
                p0 =ax[3*j+2, i].get_position().get_points().flatten()
                ax_cbar = fig.add_axes([p0[2]+0.0075, p0[1], 0.005, p0[3]-p0[1]])
                ticks = np.linspace(0, 1, 5)
                tickLabels = np.linspace(c_min_error, c_max_error, 5)
                tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]

                cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap_error), orientation='vertical', ticks=ticks)
                cbar.set_ticklabels(tickLabels)

                for ax0 in ax[:,i]:
                    ax0.xaxis.set_ticks([])
                    ax0.yaxis.set_ticks([])

                if i==0:
                    ax[3*j, i].set_ylabel('true', fontsize=8)
                    ax[3*j+1, i].set_ylabel('prediction', fontsize=8)
                    ax[3*j+2, i].set_ylabel('L1 error', fontsize=8)
                        
            # ax[0, i].set_title(f'idx={idxLs[i]}', fontsize=14)
            # ax[-1, i].set_xlabel('x', fontsize=14)
        
        self.save_show(1, savePath, fig, bbox_inches='tight')


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
    