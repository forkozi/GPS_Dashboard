#from bokeh.charts import BoxPlot, Scatter, Bar, output_file, show
from bokeh.sampledata.autompg import autompg as df
from bokeh.io import vform
from bokeh.plotting import figure, show, output_file

import os
import numpy as np
import pandas as pd

DATA_dir = r"C:\Users\Nick.Forfinski-Sarko\Downloads\PPP_Project"
#DATA_dir = r"C:\Users\Nick\SkyDrive\ONEDRIVE_DOCS\Classes\OSU\CE640_GpsTheory\PPP_Project"
data_files = [i for i in os.listdir(DATA_dir) \
                if i.endswith("csv")]  

sta_qual = {'BICK':'good',
                   'U727':'good',
                   'B726':'bad',
                   'GLAS':'bad',
                   'LBCC':'med',
                   'Y683':'med',}

# used in makePlotGui.py
def getUniqueFields():

    field_lists = []
    data = []
    services = []

    for data_file in data_files:

        path = os.path.join(DATA_dir, data_file)
        f = open(path, "r")
        
        print data_file + " has the following fields:  "
        field_lists.append(f.readline().replace("\n","").split(","))

    # get single list of all fields in all csv files
    all_fields = []
    for fl in field_lists:
        all_fields.extend([s.lower() for s in fl])

    all_fields = list(set(all_fields))
    all_fields.append("service")
    all_fields.append("sta_qual")
    all_fields = [
            'de (cm)',
            'dn (cm)',
            'dh (cm)',
            'nominalt',
            'station',
            'service',
            'sta_qual'
        ]

    return all_fields
    

def getDataFrame():
        
    field_lists = []
    data = []

    for data_file in data_files:

        print "getting data from " + data_file + "..."
        path = os.path.join(DATA_dir, data_file)
        f = open(path, "r")
        
        field_lists.append(f.readline().replace("\n","").split(","))

        # add list for each .csv data file
        data.append([])

        # add list for each field within each csv data file
        for i in range(len(field_lists[-1])):
            data[-1].append([])

        for line in f:
            row_data = line.replace("\n","").split(",")
            row_data = ["NaN" if rd == "" else rd for rd in row_data]

            for j, field in enumerate(field_lists[-1]):            
                data[-1][j].append(row_data[j])
                
    # get single list of all fields in all csv files
    all_fields = []
    for fl in field_lists:
        all_fields.extend([s.lower() for s in fl])

    all_fields = list(set(all_fields))
    all_fields.append("service")
    all_fields.append("sta_qual")

    # build list with column for each unique field
    DF_data = []
    all_fields = [
            'de (cm)',
            'dn (cm)',
            'dh (cm)',
            'nominalt',
            'station',
            'service',
            'sta_qual'
        ]
    for unq_field in all_fields:
        DF_data.append([])
        print unq_field

        # concatenate like fields from each data file
        for i, fl in enumerate(field_lists):

            fl_lower = [l.lower() for l in fl]
            if unq_field in fl_lower:
                DF_data[-1].extend(data[i][fl_lower.index(unq_field)])

            # append list of services
            elif unq_field == "service":
                DF_data[-1].extend([data_files[i].replace(".csv","")] * len(data[i][1]))

            # append list of station quality
            elif unq_field == "sta_qual":
                sta_ind = fl_lower.index("station")
                sta_qual_list = []

                for r in data[i][sta_ind]:
                    sta_qual_list.append(sta_qual[r])
    
                    
                DF_data[-1].extend(sta_qual_list)
                    
            else:
                DF_data[-1].extend([np.nan] * len(data[i][1]))


    # convert fields to appropriate data types
    for i, r in enumerate(DF_data):
        for j, f in enumerate(r):
            try:
                DF_data[i][j] = float(DF_data[i][j])
            except:
                break

    # build data frame for Bokeh plotting
    DFstr = ""
       
    for i, field in enumerate(all_fields):

        # build string to create DataFrame
        DFstr += "'" + field + "':" + "DF_data[" + str(i) + "],"
        
    # create DataFrame for Bokeh plotting
    DATA = eval("pd.DataFrame({" + DFstr[:-1] + "})")


    return DATA


