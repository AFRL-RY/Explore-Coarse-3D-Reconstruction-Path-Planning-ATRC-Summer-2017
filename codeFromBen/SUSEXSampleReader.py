
import argparse                         # Enables the python arg parser
import json                             # Enables the use of JSON objects
import xmltodict                        # Enables the reading of XML files
import csv                              # Enables the CSV reading
import cv2 as cv                        # Enables the OpenCV Library use 
import time                             # Enables the use of timing
import os                               # Enables File IO Items

# Utilities
from FileUtilities import getImmediateSubDirs
from FileUtilities import getImmediateFiles
#from Utilities.GirderUtilities import getCollectionID
#from Utilities.GirderUtilities import getItemID

class SUSEXDataset:
   def getImageList(self,aDir):
      tSubFolderList = getImmediateSubDirs(aDir=aDir)
      tTargetFiles = [];
      for tSubFolder in tSubFolderList:
         tFolderListItems = getImmediateFiles(aDir=aDir+'/'+tSubFolder)
         if not tFolderListItems is None and tFolderListItems[0].find("bmp") >= 0:
            tTargetFiles = tTargetFiles + tFolderListItems
            for tTargetFile in tTargetFiles:
               tTargetFile = aDir + '/' + tSubFolder + '/' +  tTargetFile
      return tTargetFiles

   def findImagePath(self, aName, aPath):
      for root, dirs, files in os.walk(aPath):
         if aName in files:
            return os.path.join(root,aName)
      return []

   def getGeoJson(self, aLat, aLon):
      # Construct the geo point from the XML/dictionary
      tGeoJSONPoint = '{ "geometry": { "type": "Point", "coordinates": [' + aLat  + ', ' + aLon +  '] } }'
      tGeoJsonObjectPoint = json.loads(tGeoJSONPoint)
      return tGeoJsonObjectPoint

   def getBagPrefix(self, aDirMetadata):
      tBagPrefix = ''

      tSplitMetadataPath = aDirMetadata.split("\\")

      if len(tSplitMetadataPath[len(tSplitMetadataPath)-1]) == 0:
        tBagPrefix = tSplitMetadataPath[len(tSplitMetadataPath)-2]
      else:
        tBagPrefix = tSplitMetadataPath[len(tSplitMetadataPath)-1]

      print(tBagPrefix)
      return tBagPrefix

   def getCSVDict(self, aRaw):
       # Read the raw image data from the files...
      print("Reading the RAW image CSV")
      tFileHandleCSV = open(aRaw,'r')                                                           # Open the file
      tHeaderLine = tFileHandleCSV.readline().replace("\n","").replace("%","").split(",");      # get the header
      #print "      Raw CSV Header (Raw Image) = ", tHeaderLine
      return csv.DictReader(tFileHandleCSV, tHeaderLine)                                        # Convert the CSV to a header file.

   def getMatchSeq(self, aReader, aRow):
     tFound = False;
     while tFound == False:
         #print "getting next mark entry"
         tIter = next(aReader)
         if tIter['field.header.seq'] == aRow["field.FrameNumber"]:
             tFound = True
     return tIter, tFound
     
   
   # The following function is used to read SUSEX CSV/PNG/BAG FILES
   # @param self required field for processing
   # @param aDirImages   the target directory (Images)
   # @param aDirMetadata the target directory (Metadata)
   # @return N/A
   def processFolder(self, aDirImages, aDirMetadata, aIgnoreHeight = 0, aSaveDir=None):
      print("Setting up the RUN of the SUSEX INGEST")
      
      # Get lists of files to process
      #tCurrentImages   = getImmediateFiles(aDir=aDirImages)
      #tCurrentMetadata = getImmediateFiles(aDir=aDirMetadata)

      # get the bag prefix from the path
      tBagPrefix = self.getBagPrefix(aDirMetadata)

      #print "Metadata Files ---> ", tCurrentMetadata
      #print "Image Files ---> ", tCurrentImages

      # Define the files of interest.... Note not all are used but will be created non the less
      tSpanIMU  = aDirMetadata + "/" + tBagPrefix + "nov_span_1_imu.csv"
      tSpanPVA  = aDirMetadata + "/" + tBagPrefix + "nov_span_2_pva.csv"
      tSpanFIX  = aDirMetadata + "/" + tBagPrefix + "nov_span_2_fix.csv"
      tSpanOBS  = aDirMetadata + "/" + tBagPrefix + "nov_span_2_obs.csv"
      tSpanVEL  = aDirMetadata + "/" + tBagPrefix + "nov_span_2_vel.csv"
      tSpanSTAT = aDirMetadata + "/" + tBagPrefix + "nov_span_1_ins_status.csv"
      tSpanEPH  = aDirMetadata + "/" + tBagPrefix + "nov_span_2_eph.csv"
      tSpanPO   = aDirMetadata + "/" + tBagPrefix + "ParameterOutput.csv"
      tSpanAI   = aDirMetadata + "/" + tBagPrefix + "AtmosphericInfo.csv"
      tSpanMT   = aDirMetadata + "/" + tBagPrefix + "nov_span_2_mark_time_1.csv"
      tSpanPOS  = aDirMetadata + "/" + tBagPrefix + "nov_span_2_mark_pos_1.csv"
      tSpanMPVA = aDirMetadata + "/" + tBagPrefix + "nov_span_2_mark_1_pva.csv"
      tSpanRaw  = aDirMetadata + "/" + tBagPrefix + "RawImageTopic.csv"

      # Create the readers
      tReaderRawImage = self.getCSVDict(aRaw=tSpanRaw)
      tReaderMark = self.getCSVDict(aRaw=tSpanPOS)
      tReaderPVA = self.getCSVDict(aRaw=tSpanMPVA)
      
      print("Processing Images")
      for tRow in tReaderRawImage:
        tImageName = "frame"+format(int(tRow["field.FrameNumber"]), '05')+"_"+str(int(float(tRow["field.GPSTime"])))+".bmp"
        if not os.path.isfile(aDirImages+'/'+tImageName):
            tImageName = "frame"+format(int(tRow["field.FrameNumber"]), '05')+"_"+str(int(float(tRow["field.GPSTime"])))+".jpg"    
        
        print("      Processing Image ", tImageName)
        aFullImagePath = self.findImagePath(tImageName, aDirImages);

        #TODO break this out so that it is a function
        tFound = False;
        try:
           tIterMark, tFoundMark = self.getMatchSeq(tReaderMark, tRow)
           tIterPVA, tFoundPVA = self.getMatchSeq(tReaderPVA, tRow)
           tFound = tFoundMark and tFoundPVA
        except:
           pass

        tAlt = float(tIterMark["field.altitude"])

        if len(aFullImagePath) > 0 and tFound and tAlt > aIgnoreHeight:
           aColorImagePath=aFullImagePath[0:len(aFullImagePath)-4]+".jpg"
           if aSaveDir != None:
              tSplit = aColorImagePath.split("/")
              aColorImagePath = aSaveDir + "/" + tSplit[len(tSplit)-1]
           print("      ", aColorImagePath)
           if not os.path.isfile(aColorImagePath) and os.path.isfile(aFullImagePath):
              tImageCV_GRAY = cv.imread(aFullImagePath,0)
              tImageCV_RGB  = cv.cvtColor(tImageCV_GRAY,cv.COLOR_BAYER_GR2RGB);
              cv.imwrite(aColorImagePath, tImageCV_RGB)
              #os.remove(aFullImagePath)
           else:
              print("      Image already converted to Save Format(", aColorImagePath, ")")

           print("      ", {"ALT":float(tIterMark["field.altitude"]),
                            "LAT":float(tIterMark['field.latitude']),
                            "LON":float(tIterMark['field.longitude']),
                            "ROLL":float(tIterPVA['field.roll']),
                            "PITCH":float(tIterPVA['field.pitch']),
                            "AZIMUTH":float(tIterPVA['field.azimuth'])})
           
            #BKH - TODO Add in four corner calculation... for the data.
           
        else:
                #if not len(aFullImagePath) > 0:
                        #print "Failed the image path check..."
                if not tFound:
                        print("Failed the found path check...")
                if not tAlt > aIgnoreHeight:
                        print("failed the height check...")
                

# Main entry point for the application
def main():
    # Parse the inputs. Note the --collection parameter is not really needed
    # assuming the directroy name is correct.
    parser = argparse.ArgumentParser(description='This is an example application for reading the metadata from SUSEX in Python')
    parser.add_argument('--directory-images'  ,  required=True,   default=None)
    parser.add_argument('--directory-metadata',  required=True,   default=None)
    parser.add_argument('--ignore-height',       required=False,  default=0)   
    parser.add_argument('--save-dir',            required=False,  default=None) 
    args = parser.parse_args()
    
    print("Processing Image Directory of ", args.directory_images)
    print("Processing Metadata Directory of ", args.directory_metadata)

    # Create the object for processing the data...
    tSUSEXDataset = SUSEXDataset()

    # Process the data 
    tSUSEXDataset.processFolder(aDirImages = args.directory_images, aDirMetadata = args.directory_metadata, aIgnoreHeight=args.ignore_height, aSaveDir = args.save_dir);

if __name__ == '__main__':
    main()  # pragma: no cover
