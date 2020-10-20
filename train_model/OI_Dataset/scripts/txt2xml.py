import os
from tqdm import tqdm
from sys import exit
import argparse
import cv2
from textwrap import dedent
from lxml import etree
from argparse import ArgumentParser

XML_DIR = 'To_PASCAL_XML'

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Dataset path")
    return parser

args = build_argparser().parse_args()
root = os.listdir(args.input)

for dir in root:
    sub_dir = args.input + dir
    if os.path.isdir(sub_dir):
        os.chdir(sub_dir)
        class_dirs = os.listdir(os.getcwd())
        for class_dir in class_dirs:
            if os.path.isdir(class_dir):
                os.chdir(class_dir)
                
                if not os.path.exists(XML_DIR):
                    os.makedirs(XML_DIR)
                
                for filename in tqdm(os.listdir(os.getcwd())):
                    if filename.endswith(".txt"):
                        filename_str = str.split(filename, ".")[0]
                        
                        annotation = etree.Element("annotation")
                        
                        #os.chdir("..")
                        folder = etree.Element("folder")
                        folder.text = os.path.basename(os.getcwd())
                        annotation.append(folder)

                        filename_xml = etree.Element("filename")
                        filename_xml.text = filename_str + ".jpg"
                        annotation.append(filename_xml)

                        path = etree.Element("path")
                        path.text = os.path.join(os.path.dirname(os.path.abspath(filename)), filename_str + ".jpg")
                        annotation.append(path)

                        source = etree.Element("source")
                        annotation.append(source)

                        database = etree.Element("database")
                        database.text = "Unknown"
                        source.append(database)

                        size = etree.Element("size")
                        annotation.append(size)
                
                        width = etree.Element("width")
                        height = etree.Element("height")
                        depth = etree.Element("depth")
                        
                        img = cv2.imread(filename_xml.text)
      
                        width.text = str(img.shape[1])
                        height.text = str(img.shape[0])
                        depth.text = str(img.shape[2])

                        size.append(width)
                        size.append(height)
                        size.append(depth)

                        segmented = etree.Element("segmented")
                        segmented.text = "0"
                        annotation.append(segmented)
    
                        label_original = open(filename, 'r')

                        # Labels from OIDv4 Toolkit: name_of_class X_min Y_min X_max Y_max
                        for line in label_original:
                            line = line.strip()
                            l = line.split(' ')
                            len_str = len(l)
                            if len_str == 6:
                                class_name = l[0]+ "_" + l[1]
                                xmin_l = str(int(float(l[2])))
                                ymin_l = str(int(float(l[3])))
                                xmax_l = str(int(float(l[4])))
                                ymax_l = str(int(float(l[5])))
                            else:
                                class_name = l[0]
                                xmin_l = str(int(float(l[1])))
                                ymin_l = str(int(float(l[2])))
                                xmax_l = str(int(float(l[3])))
                                ymax_l = str(int(float(l[4])))
                            
                            obj = etree.Element("object")
                            annotation.append(obj)

                            name = etree.Element("name")
                            name.text = class_name
                            obj.append(name)

                            pose = etree.Element("pose")
                            pose.text = "Unspecified"
                            obj.append(pose)

                            truncated = etree.Element("truncated")
                            truncated.text = "0"
                            obj.append(truncated)

                            difficult = etree.Element("difficult")
                            difficult.text = "0"
                            obj.append(difficult)

                            bndbox = etree.Element("bndbox")
                            obj.append(bndbox)

                            xmin = etree.Element("xmin")
                            xmin.text = xmin_l
                            bndbox.append(xmin)

                            ymin = etree.Element("ymin")
                            ymin.text = ymin_l
                            bndbox.append(ymin)

                            xmax = etree.Element("xmax")
                            xmax.text = xmax_l
                            bndbox.append(xmax)

                            ymax = etree.Element("ymax")
                            ymax.text = ymax_l
                            bndbox.append(ymax)

                        os.chdir(XML_DIR)

                        # write xml to file
                        s = etree.tostring(annotation, pretty_print=True)
                        with open(filename_str + ".xml", 'wb') as f:
                            f.write(s)
                            f.close()

                        os.chdir("..")
                
                os.chdir("..") 
        os.chdir("..")
