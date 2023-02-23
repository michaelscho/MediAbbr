import lxml.etree as LET # for handling xslt
import os
import config
import re
import random

# creates ground truth data from xml files

# load xml file in input folder
# create txt files from that
# line by line, break = no als eine line
# one file abbreviated
# one file expanded 
# append txt files of xml files
# append abbreviated with expanded separated by ';'
# store in training



def xml_to_ground_gruth(file):
    tree = LET.parse(file)
    root = tree.getroot()
    xslt_abbr = LET.XML(config.xslt_abbr)
    transform = LET.XSLT(xslt_abbr)
    plain_text_abbr = str(transform(root))
    xslt_expan = LET.XML(config.xslt_expan)
    transform = LET.XSLT(xslt_expan)
    plain_text_expan = str(transform(root))

    plain_text_expan = clean_xslt_output(plain_text_expan)
    plain_text_abbr = clean_xslt_output(plain_text_abbr)

    data_to_be_paired = [plain_text_abbr, plain_text_expan]
    pairs = zip(*(s.splitlines() for s in data_to_be_paired))
    ground_truth = '\n'.join(';'.join(pair) for pair in pairs).replace(' ;',';')

    return ground_truth

def clean_xslt_output(input):
    output = re.sub('\+\s+\n\s+§', '',input)
    output = output.replace('\n+',' ')
    output = re.sub('\s+', ' ',output)
    output = output.replace('§','\n')
    output = re.sub('^\s', '',output)
    output = re.sub('\+\n', '\n',output)
    output = re.sub('\+\s+\n', '\n',output)
    output = re.sub('\n\s+', '\n',output)
    output = re.sub('\s+\+', '',output)
    output = re.sub('\+\s+', '',output)
    output = re.sub('\+', '',output)

    return output

def main():

    files = [os.path.join(os.getcwd(),'..','data','input', f) for f in os.listdir(os.path.join(os.getcwd(),'..','data','input')) if 
    os.path.isfile(os.path.join(os.getcwd(),'..','data','input', f)) and '.xml' in f]

    ground_truth_data = ''
    for file in files:
        ground_truth = xml_to_ground_gruth(file)
        ground_truth_data = ground_truth_data + ground_truth
        
    with open(os.path.join(os.getcwd(),'..','data','training','trainingset.txt'), 'w', encoding = 'utf8') as trainigset:
        trainigset.write(ground_truth_data)
    
if __name__ == '__main__':
    main()

