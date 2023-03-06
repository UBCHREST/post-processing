import numpy as np
from lxml import etree

def ReadFile(file):
    tree = etree.parse(file)
    root = tree.getroot()
    flames = dict()
    flamekeys = []
    for child in root:
        key=child.attrib['key']
        flamekeys.append(str(key))
        varslabel = child.attrib['vars'].split(',')
        varsdata = child.getchildren()
        flame = dict()
        ind=0
        for v in varsdata:
          flame[varslabel[ind]] = np.array(v.text.split(','),float)
          ind=ind+1
        flames[str(key)] = flame
    return [flames,flamekeys]