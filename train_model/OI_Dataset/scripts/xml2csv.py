import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

dataset = " "

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Dataset path")
    return parser


def xml_to_csv(path):
    xml_list = []
    classes_names = []
    os.chdir(path)
    CLASS_DIRS = os.listdir(os.getcwd())
    
    print(CLASS_DIRS)
    for CLASS_DIR in CLASS_DIRS:
      if os.path.isdir(CLASS_DIR):
        os.chdir(CLASS_DIR)
        print("Currently in Subdirectory:", os.path.join(os.getcwd()))
        path = os.path.join(os.getcwd()) + "/To_PASCAL_XML"
        print(path)
        for xml_file in glob.glob(path + '/*.xml'):
          #print(xml_file)
          tree = ET.parse(xml_file)
          #print(tree)
          root = tree.getroot()
          for member in root.findall('object'):
              classes_names.append(member[0].text)
              print(root.find('path').text)
              value = (root.find('path').text,
                      int(root.find('size')[0].text),
                      int(root.find('size')[1].text),
                      member[0].text,
                      int(member[4][0].text),
                      int(member[4][1].text),
                      int(member[4][2].text),
                      int(member[4][3].text)
                      )
              xml_list.append(value)
        os.chdir("..")
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return xml_df, classes_names

def main():
    args = build_argparser().parse_args()
    dataset = args.input
    os.chdir(dataset)
    for folder in ['train','test', 'validation']:
        image_path = os.path.join(os.getcwd(), (folder))
        print(image_path)
        DIR = image_path
        if os.path.isdir(DIR):
           os.chdir(DIR)
           print("Currently in Subdirectory:", DIR)
           xml_df, classes_names  = xml_to_csv(DIR)
           path = DIR + "/" + folder + '_labels.csv'
           print(path)
           xml_df.to_csv((path), index=None)
           print('Successfully converted xml to csv.')

           # Labelmap generation
           l_path = DIR + "/" + "label_map.pbtxt"
           pbtxt_content = ""
           for i, class_name in enumerate(classes_names):
              pbtxt_content = (
                pbtxt_content
                + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
                    i + 1, class_name
                )
             )
           pbtxt_content = pbtxt_content.strip()
           with open(l_path, "w") as f:
             f.write(pbtxt_content)
        os.chdir("..")
    os.chdir("..")
main()