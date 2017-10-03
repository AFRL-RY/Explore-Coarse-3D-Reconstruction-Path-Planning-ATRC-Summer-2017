# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:37:33 2017

@author: student
"""

import xml.etree.ElementTree as ET
import utm

kml_file = './POI/muscat.kml'

tree = ET.parse(kml_file) 
#lineStrings = tree.findall('.//{http://www.opengis.net/kml/2.2}LinearRing')
namespace = tree.getroot().tag
namespace = namespace[namespace.find("{")+1:namespace.find("}")]
lineStrings = tree.findall('.//{' + str(namespace) + '}LinearRing')

eastList = []
northList =[]
for attributes in lineStrings:
    for subAttribute in attributes:
        if subAttribute.tag == '{http://www.opengis.net/kml/2.2}coordinates':
            for line in subAttribute.text.split():
                print(line)
                lon,lat,alt = line.split(',')
                east, north, zone, zone_letter = utm.from_latlon(float(lat),float(lon))
                eastList.append(east)
                northList.append(north)
