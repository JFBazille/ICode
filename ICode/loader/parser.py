import xlrd
import os, glob

from sklearn.datasets.base import Bunch

def xlsparser(document='/volatile/hubert/Decode_servier2_16022015_v17.xls'):
    workbook = xlrd.open_workbook(document)
    sheet = workbook.sheet_by_index(0)
    header = sheet.row_values(1)
    NIP = header.index('NIP')
    Period = header.index('Period')
    Treatment = header.index('Treatment')
    current_NIP = ''
    nrows = sheet.nrows
    OUTPUT_dictionnary = dict()

    for idx in xrange(2, nrows):
        row = sheet.row_values(idx)
        if row[NIP]!='':
            current_NIP = row[NIP]

        if row[Treatment]=='Placebo':
            if row[Period]=='P1':
                OUTPUT_dictionnary[current_NIP] = '11'
            if row[Period]=='P1BIS':
                OUTPUT_dictionnary[current_NIP] = '12'
            if row[Period]=='P2':
                OUTPUT_dictionnary[current_NIP] = '21'
            if row[Period]=='P2BIS':
                OUTPUT_dictionnary[current_NIP] = '22'
    return OUTPUT_dictionnary


def base_dir():
    return os.path.join('/volatile', 'hubert', 'scaling', 'study')

def load_servier_data(document=None, prefix='rs1'):
    if document is None:
       parsexls = xlsparser()
    else:
        parsexls = xlsparser(document)
    session1_files = []
    subjects = []

    BASE_DIR = base_dir()
    for (NIP, Period) in parsexls.items():
        session1_files += glob.glob(os.path.join(BASE_DIR, NIP, 'fMRI',
                                                     'acquisition' + Period,
                                                     prefix+'*.nii'))
        subjects.append(NIP)
    return Bunch(func1=session1_files, subjects=subjects)