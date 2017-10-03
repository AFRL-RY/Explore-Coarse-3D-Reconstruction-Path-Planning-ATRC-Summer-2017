import os                               # Enables directory search items

## Utility function to get list of the immediate dub dirs in folder
# @param aDir - the directory to search for sub directories from
# @return an array of sub directories
def getImmediateSubDirs(aDir):
    return [name for name in os.listdir(aDir)
            if os.path.isdir(os.path.join(aDir,name))]

## Utility function to get list of the immediate files in folder
# @param aDir - the directory to search for the immediate files
# @return an array of immediate files contained in the directory
def getImmediateFiles(aDir):
    """
    Utility function to get list of the immediate files in folder.
    """
    return [name for name in os.listdir(aDir)
            if os.path.isfile(os.path.join(aDir,name))]
