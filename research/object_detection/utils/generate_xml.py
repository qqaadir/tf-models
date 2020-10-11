import xml.etree.cElementTree as ET
from random import randrange
import os
os.chdir(os.getcwd())
class GenerateXml(object):
    def __init__(self, box_array, im_width, im_height, inferred_class, file_name):
        self.inferred_class = inferred_class
        self.box_array = box_array
        self.im_width = im_width
        self.im_height = im_height
        self.file_name = file_name

    def gerenate_basic_structure(self):
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = self.file_name + ".jpg"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(self.im_width)
        ET.SubElement(size, "height").text = str(self.im_height)
        ET.SubElement(size, "depth").text = "3"
        
        for i in self.box_array:
            # this a necessary structure for the XML file
            objectBox = ET.SubElement(annotation, "object")
            ET.SubElement(objectBox, "name").text = self.inferred_class
            ET.SubElement(objectBox, "pose").text = "Unspecified"
            ET.SubElement(objectBox, "truncated").text = "0"
            ET.SubElement(objectBox, "difficult").text = "0"
            bndBox = ET.SubElement(objectBox, "bndbox")
            ET.SubElement(bndBox, "xmin").text = str(round(i['xmin']))
            ET.SubElement(bndBox, "ymin").text = str(round(i['ymin']))
            ET.SubElement(bndBox, "xmax").text = str(round(i['xmax']))
            ET.SubElement(bndBox, "ymax").text = str(round(i['ymax']))

        arquivo = ET.ElementTree(annotation)
        print('CREATING XML FILE')
        arquivo.write("/content/tf-models/research/object_detection/labeled_data/" + self.file_name + ".xml")

# just for debuggind
def main():
    xml = GenerateXml([{'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}], '4000', '2000', 'miner', 'miner') 
    xml.gerenate_basic_structure()    