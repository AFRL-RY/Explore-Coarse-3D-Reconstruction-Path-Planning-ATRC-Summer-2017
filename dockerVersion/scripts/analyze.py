import numpy as np

def analyzeResults(visibility,c_min,dd,cfg):

    # Find points not covered by initial set for plotting
    initialNotVis = np.where(np.sum(visibility, axis=0) == 0)[0]

    # Get numbers of points covered from different angles for plots
    stackAll = c_min[:, 0:dd.size]
    totalBins = cfg['angle_sorting']['vertical']+cfg['angle_sorting']['horizontal']
    for i in range(1, totalBins):
        stackAll = np.r_[stackAll, c_min[:, dd.size * i:dd.size * (i + 1)]]
    visStatsFinal = np.sum(stackAll, axis=0)
    visStatsInitial = np.sum(visibility[:, 0:dd.size], axis=0)
    visStatsFinal[initialNotVis] = -1  # Mark points which are not visible from original images
    visStatsInitial[initialNotVis] = -1

    return visStatsInitial,visStatsFinal,initialNotVis
