# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:56:15 2017

@author: student
"""

import dubins
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipy
from munkres import Munkres
from datetime import datetime

def findPath(camIdx,cameras,cfg):

    # Unpack cameras
    cn = cameras.cn.values
    ce = cameras.ce.values
    cor_n = cameras.cor_n.values
    cor_e = cameras.cor_e.values
    
    qlist = np.zeros([len(cn[camIdx]),3])
    

    for i, row in enumerate(qlist):
        theta = np.arctan2(cor_e[camIdx][i],cor_n[camIdx][i])
        theta = theta + 3*np.pi/2.0
        qlist[i,:] = np.array([cn[camIdx][i],ce[camIdx][i],theta])

    TSP = Christofides(qlist,cfg)
    
    qlist = qlist[TSP.finalSolution]
    
    return qlist, TSP.finalSolution
        
class Christofides:
    # Initial Code Credit: Landen Blackburn - BYU 2015
    
    finalSolution=None #indexes of the given initial point set in the correct order. Does not have first point on the end
    finalPointSet=None #the initial point set ordered by the final solution
    finalDistance=None #the total distance through the entire final point set
    # Final point set broken down into x y and z arrays. These have the starting point on the end
    xValues=None
    yValues=None
    thetaValues=None

    #accepts a numpy array with one row for each point. The array should have three columns for x, y, and z values respectively
    def __init__(self, initialPoints, cfg):
#        print("Go-go gadget Christofides!")

        startTime = datetime.now()
        numpoints=initialPoints.shape[0]
        #print("Initial Point Set: \n"+ str(initialPoints))
        distanceMatrix = np.zeros((numpoints,numpoints))#, dtype=np.complex128)
        #create distance matrix
        rowCounter=0
        for currentRow in initialPoints:
            colCounter=rowCounter
            for otherRow in initialPoints[rowCounter:]:
                #print(str(currentRow) +" "+ str(otherRow))
                distanceMatrix[rowCounter,colCounter]= self.dubinsDist(currentRow, otherRow, cfg)
                colCounter+=1
            rowCounter+=1

        #create a distance matrix where mat[A][B]==mat[B][A]; needed for munkres
        distanceMatrixRedundant = distanceMatrix + distanceMatrix.transpose()
        for counter in range(0,numpoints):
            distanceMatrixRedundant[counter,counter]=np.inf

#        print("Finding Minimum Spanning Tree")
        minimumSpanningPointConnections=self.getMinSpanTree(distanceMatrix,numpoints)

#        print("Time elapsed finding MST: "+str( (datetime.now()-startTime).seconds) +" seconds, "+ str( (datetime.now()-startTime).microseconds ) +" microseconds")
#        print("Finding Minimum Perfect Matching Set")
        minimumPerfectMatching=self.getMinPerfMatching(minimumSpanningPointConnections,distanceMatrixRedundant,numpoints)

        minimumPerfectMatching=self.revmoveDuplicates(minimumPerfectMatching)

        # combine the two sets
        combinedPathwaySet=np.concatenate((minimumSpanningPointConnections, minimumPerfectMatching), axis=0)
        combinedPathwaySet=combinedPathwaySet[combinedPathwaySet[:,0].argsort()]

#        print("Time elapsed finding MST and MPMS: "+str( (datetime.now()-startTime).seconds) +" seconds, "+ str( (datetime.now()-startTime).microseconds ) +" microseconds")
#        print("Finding Euler Path")
        #find EulerPath
        #solutionWithDuplicates=findEulerPath(combinedPathwaySet,pointOrder,currentPoint,startPoint,firstIteration=True)
        solutionWithDuplicates=self.findEulerPath(combinedPathwaySet)

        #remove the end point (the checkTotalDistance method adds it back in when calculating the distance)
        solutionWithDuplicates=np.delete(solutionWithDuplicates,-1)

        self.finalSolution, self.finalPointSet, self.finalDistance = self.removeDuplicatePoints(solutionWithDuplicates,distanceMatrixRedundant,initialPoints)

#        print("Total time elapsed :"+str( (datetime.now()-startTime).seconds) +" seconds, "+ str( (datetime.now()-startTime).microseconds ) +" microseconds")
        #print("Final Solution Order: " +str(self.finalSolution))
        #print("points in Order: \n"+str(finalPointSet))
#        print("Total Distance: " + str(self.finalDistance))

        #print("Python solution graphs: ")
        self.xValues=np.append(self.finalPointSet[:,0],self.finalPointSet[0,0])
        self.yValues=np.append(self.finalPointSet[:,1],self.finalPointSet[0,1])
        self.thetaValues=np.append(self.finalPointSet[:,2],self.finalPointSet[0,2])

#        fig=plt.figure()
#        ax=fig.gca(projection='3d')
#        ax.plot(xValues, yValues, thetaValues)
#        print("Generating Pathway Plot")
#        #plt.plot(xValues, yValues, thetaValues)
#        plt.show()

    #finds minimum spanning tree for point set
    def getMinSpanTree(self,distanceMatrix,numpoints):
        #these are the group assignments. Each points begins in its own group and converges to just one group
        groupMatrix=np.array(range(numpoints)) #dtype = [('pointIndex', dtype=int64), ('groupNumber', dtype=int64)]
        minimumSpanningPointConnections = np.zeros((numpoints-1,2),dtype=int)
        #find minimum spanning
        for counter in range(0,numpoints-1):
            successful = False
            while not successful:
                minval = np.min(distanceMatrix[np.nonzero(distanceMatrix)])
                #arbitrarily takes the first value that is found
                pointA=np.where(distanceMatrix == minval)[0][0]
                pointB=np.where(distanceMatrix == minval)[1][0]
                #update groups
                newGroupVal=groupMatrix[pointA]
                oldGroupVal=groupMatrix[pointB]
                distanceMatrix[pointA,pointB]=0 #changes distance value to zero so np.nonzero will ignore it
                #only happens when two different groups are combined
                if(newGroupVal!=oldGroupVal):
                    successful = True
                    groupMatrix[groupMatrix==oldGroupVal]=newGroupVal
                    minimumSpanningPointConnections[counter,:]=[pointA,pointB]
        return minimumSpanningPointConnections

    #finds minimum perfect matching set given minimum spanning tree and the redundant distance matrix
    def getMinPerfMatching(self,minimumSpanningPointConnections,distanceMatrixRedundant,numpoints):
        m=Munkres()

        #find odd degree set of points
        degree=np.bincount(minimumSpanningPointConnections.reshape(minimumSpanningPointConnections.size))
        isOddDegree = np.mod(degree,2)!=0
        oddDegreeIndexes=np.extract(isOddDegree,range(numpoints))
        temp=distanceMatrixRedundant[oddDegreeIndexes,:]
        munkresMat=temp[:,oddDegreeIndexes]

        #find the minimum perfect matchign set of odd-degree points
        #need to loop in case of odd numbered subtours that are formed
        subtourGlitchCounter=0
        solutionFound=False
        while not solutionFound:
            indexes = np.array(m.compute(munkresMat.copy()))
            solutionFound=True
            indexesClone=indexes.copy()

            #each number shows up exactly once in the first column and once in the second column
            #this for loop sorts the array into it's propper order of points
            SortedOrder=np.zeros(indexes.shape[0])
            currentPoint=0
            for c in range(indexes.max()+1):

                firstCol=np.where(indexes[:,0] == currentPoint)[0]
                secondCol=np.where(indexes[:,1] == currentPoint)[0]
                InFirstCol=firstCol.size!=0
                InSecondCol=secondCol.size!=0

                if not InFirstCol and not InSecondCol: #previousRow==-2:
                    firstCol=np.nonzero(indexes+1)[0][0]
                    SortedOrder[c]=firstCol
                    currentPoint=indexes[firstCol,1]
                    indexes[firstCol,:]=np.array([-1,-1])
                elif InFirstCol:
                    SortedOrder[c]=firstCol[0]
                    currentPoint=indexes[firstCol[0],1]
                    indexes[firstCol[0],:]=np.array([-1,-1])
                elif InSecondCol:
                    SortedOrder[c]=secondCol[0]
                    currentPoint=indexes[secondCol[0],0]
                    indexes[secondCol[0],:]=np.array([-1,-1])
            SortedOrder=SortedOrder.astype(int)
            indexes=indexesClone[SortedOrder]
            #check for odd subtours
            subtourSize=1
            maxDistance=0
            maxDistanceLeftCol=0
            maxDistanceRightCol=0
            beginningOfSubtour=0

            for c in range(1,indexes.shape[0]):
                if subtourSize==0:
                    subtourSize+=1
                    beginningOfSubtour=indexes[c][0]
                    maxDistanceLeftCol=indexes[c][0]
                    maxDistanceRightCol=indexes[c][1]
                    maxDistance=munkresMat[maxDistanceLeftCol,maxDistanceRightCol]
                else:
                    subtourSize+=1
                    newDisLeftCol=indexes[c][0]
                    newDisRightCol=indexes[c][1]
                    newDis=munkresMat[newDisLeftCol,newDisRightCol]
                    if newDis>maxDistance:
                        newDis=maxDistance
                        maxDistanceLeftCol=newDisLeftCol
                        maxDistanceRightCol=newDisRightCol

                if indexes[c][1]==beginningOfSubtour:
                    if subtourSize%2==1:
                        solutionFound=False
                        subtourGlitchCounter+=1
                        #print("Munnkres failed, modify and retry")
                        munkresMat[maxDistanceLeftCol][maxDistanceRightCol]=np.inf
                        munkresMat[maxDistanceRightCol][maxDistanceLeftCol]=np.inf
                    else:
                        subtourSize=0
#        print("Required "+str(subtourGlitchCounter) +" attempts to find MPMS")

        #shift the indexes back to the values we need. This turns [a,b,c] into [[a,b],[b,c],[c,a]]
        minimumPerfectMatching=np.array([oddDegreeIndexes[indexes[:,0]],oddDegreeIndexes[indexes[:,1]]])
        minimumPerfectMatching=minimumPerfectMatching.transpose()
        return minimumPerfectMatching

    #remove duplicates by sorting each row, then sorting by column, then taking the even verticies
    def revmoveDuplicates(self,minimumPerfectMatching):
        rowCounter=0
        for row in minimumPerfectMatching:
            if row[0]>row[1]:
                minimumPerfectMatching[rowCounter]=row[::-1]
            rowCounter+=1
        minimumPerfectMatching=minimumPerfectMatching[minimumPerfectMatching[:,0].argsort()]
        numRows=(minimumPerfectMatching.shape[0]-1)
        minimumPerfectMatching=minimumPerfectMatching[range(0,numRows+1,2)]
        return minimumPerfectMatching

    #this is the driving method that uses findCircuitRecursive to find Euler tour
    def findEulerPath(self,combinedPathwaySet):
        startPoint=combinedPathwaySet[0,0]
        combinedPathwaySetCopy=combinedPathwaySet.copy() #done to preserve original combinedPathwaySet
        #find first ciruit
        pointOrder, combinedPathwaySetCopy = self.findCircuitRecursive(combinedPathwaySetCopy,np.array([startPoint]),startPoint,startPoint,firstIteration=True)

        while combinedPathwaySetCopy.size>0:
            newStartPoint=-1
            newStartPointIndex=np.zeros(2)
            pointOrderCounter=0
            for usedPoint in pointOrder:
                LHSindexes=np.where(combinedPathwaySetCopy[:,0] == usedPoint)[0]
                RHSindexes=np.where(combinedPathwaySetCopy[:,1] == usedPoint)[0]
                if LHSindexes.size>0:
                    newStartPoint=usedPoint
                    newStartPointIndex=pointOrderCounter#LHSindexes[0]
                    break
                elif RHSindexes.size>0:
                    newStartPoint=usedPoint
                    newStartPointIndex=pointOrderCounter#RHSindexes[0]
                    break
                pointOrderCounter+=1
            #find next circuit
            nextCircuit, combinedPathwaySetCopy = self.findCircuitRecursive(combinedPathwaySetCopy,np.array([newStartPoint]),newStartPoint,newStartPoint,firstIteration=True)
            pointOrder=np.delete(pointOrder,newStartPointIndex)
            pointOrder=np.insert(pointOrder, newStartPointIndex, nextCircuit)
        return pointOrder

    #note:this method heavily modifies combinedPathwaySetCopy. Send a copy if you want the original preserved
    def findCircuitRecursive(self,combinedPathwaySetCopy,pointOrder,currentPoint,startPoint,firstIteration=None):
        if currentPoint==startPoint and not firstIteration:
            return (pointOrder, combinedPathwaySetCopy)
        else:
            LHSindexes=np.where(combinedPathwaySetCopy[:,0] == currentPoint)[0]
            RHSindexes=np.where(combinedPathwaySetCopy[:,1] == currentPoint)[0]
            if LHSindexes.size>0:
                nextPoint=combinedPathwaySetCopy[LHSindexes[0],1]
                combinedPathwaySetCopy=np.delete(combinedPathwaySetCopy,LHSindexes[0],0)
            else:
                nextPoint=combinedPathwaySetCopy[RHSindexes[0],0]
                combinedPathwaySetCopy=np.delete(combinedPathwaySetCopy,RHSindexes[0],0)
            newPointOrder =np.append(pointOrder,nextPoint)
            return self.findCircuitRecursive(combinedPathwaySetCopy,newPointOrder,nextPoint,startPoint)

    def removeDuplicatePoints(self,solutionWithDuplicates,distanceMatrixRedundant,initialPoints):
        #remove points that are visited multiple times, favoring the removals that decrease total distance most
        for point in range(solutionWithDuplicates.max()+1):
            indexesOfCurrentElement=np.argwhere(solutionWithDuplicates==point) #convert to int() or float() if this is giving us trouble
            if(indexesOfCurrentElement.size==1):
                continue
            bestNewSolution=None
            for c in range(indexesOfCurrentElement.size):
                elementIndexesToDelete=np.delete(indexesOfCurrentElement,c) #this will make us keep one of them
                currentAttempt=np.delete(solutionWithDuplicates,elementIndexesToDelete)
                if bestNewSolution is None:
                    bestNewSolution=currentAttempt
                elif self.checkTotalDistance(distanceMatrixRedundant,currentAttempt) < self.checkTotalDistance(distanceMatrixRedundant,bestNewSolution):
                    bestNewSolution=currentAttempt
            solutionWithDuplicates=bestNewSolution
        finalSolution=solutionWithDuplicates
        finalDistance=self.checkTotalDistance(distanceMatrixRedundant,finalSolution)
        finalPointSet=initialPoints[finalSolution,:]
        return (finalSolution, finalPointSet, finalDistance)

    def checkTotalDistance(self,distanceMatrixRedundant,pointOrder):
        #assumes that final point is left off, so it adds it back in to calculate total distance
        pointOrder=np.append(pointOrder,pointOrder[0])
        totalDistance=0
        for c in range(pointOrder.size-1):
            totalDistance+=distanceMatrixRedundant[pointOrder[c],pointOrder[c+1]]
        return totalDistance
    
    def dubinsDist(self,a,b,cfg):
        turning_radius = cfg['flightPath']['min_turn']
        step_size = 5
        qs, _ = dubins.path_sample(a, b, turning_radius, step_size)
        qs = np.asarray(qs)
        dist = 0
        for i in range(len(qs)-1):
            dist += np.sqrt((qs[i,0]-qs[i+1,0])**2+(qs[i,1]-qs[i+1,1])**2)
        return dist