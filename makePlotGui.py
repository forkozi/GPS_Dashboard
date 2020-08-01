from bokeh.plotting import Figure
from bokeh.models import HBox, VBoxForm, FactorRange, SingleIntervalTicker, FixedTicker, CategoricalTicker, CategoricalAxis
from bokeh.models.widgets import Select, Panel, Tabs, CheckboxGroup, Toggle
from bokeh.io import curdoc
from bokeh.client import push_session
from bokeh.models.glyphs import Circle
from bokeh.models import CustomJS, ColumnDataSource, GMapPlot, GMapOptions, Range1d, PanTool, WheelZoomTool, BoxSelectTool
from bokeh.models.widgets import DataTable, TableColumn, RadioGroup
from numpy import mean, sqrt, square
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

import matplotlib.path as mpath

#import scipy
import getDataFrame as g
import numpy as np
import warnings

time = 2

# need to run 'python -m bokeh serve' at C:\Python27 command line

# open a session to keep our local document in sync with server
session = push_session(curdoc())

avail_fields = g.getUniqueFields()



axis_map_num = {}
axis_map_cat = {}
stats = ['Mean','Min','Max','Range','Std. Dev.','RMS']
var_list = ['de (cm)',
          'dn (cm)',
          'dh (cm)',
          'Hor. Dist. 1',
          'Hor. Dist. 2',
          '3D Dist. 1',
          '3D Dist. 2']

sta_qual = {'BICK':'good',
                   'U727':'good',
                   'B726':'bad',
                   'GLAS':'bad',
                   'LBCC':'med',
                   'Y683':'med',}

qual_colors = {'good':'green',
               'med':'yellow',
               'bad':'red'}


defaults = []
df_source = g.getDataFrame()
df = df_source
services = list(set(df['service']))
stations = list(set(df['station']))

table_columns = []
unq_cats = {}
avail_colors = ['orange', 'blue', 'red', 'cyan', 'green', 'purple']

nominalts = list(set(df['nominalt']))

def defineDataDict():
    data=dict(x=[], y=[], radius=[], color=[])
    for k in df:
        data.update({k:[]})

    return data


source = ColumnDataSource(defineDataDict())
source_unq_cats = ColumnDataSource(unq_cats)
source_hist = ColumnDataSource({'top':[],
                           'bottom':[],
                           'left':[],
                           'right':[],
                           'color':[],
                           'line_color':[]})

# Confidence Radius
data_dict_CR = {'x_CI':[],'y_CI':[],'color_CI':[],'x_mean':[],'y_mean':[],'size_':[]}
source_CR = ColumnDataSource(data_dict_CR)

plotBy_dict = {'By Station': 'station','By Service': 'service', 'Station Quality': 'sta_qual'}

for f in avail_fields:
    if f == "station" or f == "service" or f == "sta_qual":
        axis_map_cat.update({f:f})

        # get list of unique field values
        values = list(set(df[f]))
        unq_cats.update({f:values})
    else:
        axis_map_num.update({f: f})

    defaults.append(f)
    table_columns.append(TableColumn(field=f, title=f))


x_axis = Select(title="X Axis", options=sorted(axis_map_num.keys()), value=defaults[0])
y_axis = Select(title="Y Axis", options=sorted(axis_map_num.keys()), value=defaults[1])
radius = Select(title="Radius", options=sorted(axis_map_num.keys()), value=defaults[3])
color = Select(title="Color", options=sorted(axis_map_cat.keys()), value=defaults[5])
print defaults
print axis_map_cat.keys()
print 'color.value: ' + color.value
print 'unq_cats:' + str(unq_cats)
checkboxes = CheckboxGroup(labels=sorted(unq_cats[color.value]), 
                               active=[i for i, x in enumerate(sorted(unq_cats[color.value]))])
checkboxes_bulls_eye = CheckboxGroup(labels=[s + " 95% Error Ellipse" for s in sorted(unq_cats[color.value])],
                               active=[])


def setCR_source():
    num = len(checkboxes_bulls_eye.labels)
    x_cr = [[0] for c in range(num)]
    y_cr = [[0] for c in range(num)]
    color_cr = [0 for c in range(num)]
    x_mean = [0 for c in range(num)]
    y_mean = [0 for c in range(num)]
    cross_size = [0 for c in range(num)]
    dict_cr = {'x_CI':x_cr,'y_CI':y_cr,'color_CI':color_cr,'x_mean':x_mean,'y_mean':y_mean,'size_':cross_size}
    source_CR.data.update(dict_cr)


setCR_source()

var_control = Select(title="Variable", options=sorted(var_list), value="dn (cm)")
stat_control = Select(title="Statistic", options=sorted(stats), value="RMS")
byPlot_control = Select(title="Plot histograms by...",
                        options=plotBy_dict.keys(),
                        value="By Station")

byPlot_series_control= RadioGroup(labels=[str(n) for n in nominalts], active=0)
hist_series_select = CheckboxGroup(labels=sorted(unq_cats['station']), 
                               active=[i for i, x in enumerate(unq_cats['station'])])

TOOLS = "resize,crosshair,pan,wheel_zoom,box_zoom,reset,tap,previewsave,box_select,poly_select,lasso_select"
p = Figure(plot_height=600, plot_width=600, title="Position Residuals (" + str(time) + '-hr duration)', tools=TOOLS)
p.title_text_font_size = '17pt'
p.title_text_font_style = "bold"

p.x_range = Range1d(start=-20, end=20)
p.y_range = Range1d(start=-20, end=20)
p.circle(x="x", y="y", source=source, size="radius", color="color", line_color=None, fill_alpha=0.2)



# confidence radius
p.circle_cross(x="x_mean", y="y_mean", source=source_CR, size='size_', color='white', line_color='color_CI', line_width=2, fill_alpha=0.7)
p.multi_line(xs="x_CI", ys="y_CI", source=source_CR, color='color_CI', line_width=4)

p.xaxis.axis_label_text_font_size = "17pt"
p.yaxis.axis_label_text_font_size = "17pt"

p_hist = Figure(plot_height=300, plot_width=600, title="Difference Histograms", tools=TOOLS)
p_hist.quad(top='top', bottom=0, left='left', right='right', fill_alpha=0.5, source=source_hist,
       fill_color='color', line_color='line_color')

p_temp = Figure(plot_height=500, plot_width=600, title="GPS Data Exploration", tools=TOOLS)

summary_station_sortBy = RadioGroup(labels=["Sort & Color By Station", "Sort By Duration"], active=1)
summary_service_sortBy = RadioGroup(labels=["Sort & Color By Service", "Sort By Duration"], active=1)

p_summary = Figure(plot_height=500,
                   plot_width=500,
                   # y_range=[],
                   title="Position Residuals",
                   tools=TOOLS
                   )

p_summary.min_border_right = 0
p_summary.min_border_right = 0
y_labels_station = ['B726 (2)', 'B726 (3)', 'B726 (4)', 'B726 (5)', 'B726 (7)', 'B726 (10)', 'BICK (2)', 'BICK (3)', 'BICK (4)', 'BICK (5)', 'BICK (7)', 'BICK (10)', 'GLAS (2)', 'GLAS (3)', 'GLAS (4)', 'GLAS (5)', 'GLAS (7)', 'GLAS (10)', 'LBCC (2)', 'LBCC (3)', 'LBCC (4)', 'LBCC (5)', 'LBCC (7)', 'LBCC (10)', 'U727 (2)', 'U727 (3)', 'U727 (4)', 'U727 (5)', 'U727 (7)', 'U727 (10)', 'Y683 (2)', 'Y683 (3)', 'Y683 (4)', 'Y683 (5)', 'Y683 (7)', 'Y683 (10)']
y_labels_station_byTime = ['B726 (2)', 'BICK (2)', 'GLAS (2)', 'LBCC (2)', 'U727 (2)', 'Y683 (2)', 'B726 (3)', 'BICK (3)', 'GLAS (3)', 'LBCC (3)', 'U727 (3)', 'Y683 (3)', 'B726 (4)', 'BICK (4)', 'GLAS (4)', 'LBCC (4)', 'U727 (4)', 'Y683 (4)', 'B726 (5)', 'BICK (5)', 'GLAS (5)', 'LBCC (5)', 'U727 (5)', 'Y683 (5)', 'B726 (7)', 'BICK (7)', 'GLAS (7)', 'LBCC (7)', 'U727 (7)', 'Y683 (7)', 'B726 (10)', 'BICK (10)', 'GLAS (10)', 'LBCC (10)', 'U727 (10)', 'Y683 (10)']
p_summary.title_text_font_size = '15pt'
p_summary.title_text_font_style = "bold"
p_summary.xaxis.axis_label_text_font_size  = '12pt'
p_summary.y_range = Range1d(start=0.5, end=36.5, bounds='auto')
p_summary.ygrid.grid_line_alpha = 0.5
p_summary.ygrid.grid_line_dash = [6, 4]
p_summary.yaxis.visible = None
p_summary.ygrid.ticker = SingleIntervalTicker(interval=1)
p_summary.extra_y_ranges = {'byStation':FactorRange(factors=y_labels_station, bounds='auto'),
                            'byDuration': FactorRange(factors=y_labels_station_byTime, bounds='auto')}
cat_axis_station = CategoricalAxis(y_range_name='byStation')
cat_axis_station_byTime = CategoricalAxis(y_range_name='byDuration')
p_summary.add_layout(cat_axis_station,'left')
p_summary.add_layout(cat_axis_station_byTime,'left')
p_summary_dict = {'x':[], 'y':[], 'range':[], 'color':[], 'avgs':[]}
p_summary_source = ColumnDataSource(p_summary_dict)
p_summary.ray(x='x',
              y='y',
              source=p_summary_source,
              length='range',
              angle=0,
              angle_units="deg",
              color='color',
              line_width=5)
p_summary.circle(x='avgs',
                 y='y',
                 source=p_summary_source,
                 fill_color="black",
                 line_color='white')

p_summary_service = Figure(plot_height=500,
                   plot_width=600,
                   # y_range=[],
                   title="Position Residuals",
                   tools=TOOLS
                   )
p_summary_service.min_border_left = 0
y_labels_service = ['AUSPOS (2)', 'AUSPOS (3)', 'AUSPOS (4)', 'AUSPOS (5)', 'AUSPOS (7)', 'AUSPOS (10)', 'CSRS_PPP (2)', 'CSRS_PPP (3)', 'CSRS_PPP (4)', 'CSRS_PPP (5)', 'CSRS_PPP (7)', 'CSRS_PPP (10)', 'GAPS (2)', 'GAPS (3)', 'GAPS (4)', 'GAPS (5)', 'GAPS (7)', 'GAPS (10)', 'OPUS_S (2)', 'OPUS_S (3)', 'OPUS_S (4)', 'OPUS_S (5)', 'OPUS_S (7)', 'OPUS_S (10)', 'Trimble_RTX (2)', 'Trimble_RTX (3)', 'Trimble_RTX (4)', 'Trimble_RTX (5)', 'Trimble_RTX (7)', 'Trimble_RTX (10)']
y_labels_service_byTime = ['AUSPOS (2)', 'CSRS_PPP (2)', 'GAPS (2)', 'OPUS_S (2)', 'Trimble_RTX (2)', 'AUSPOS (3)', 'CSRS_PPP (3)', 'GAPS (3)', 'OPUS_S (3)', 'Trimble_RTX (3)', 'AUSPOS (4)', 'CSRS_PPP (4)', 'GAPS (4)', 'OPUS_S (4)', 'Trimble_RTX (4)', 'AUSPOS (5)', 'CSRS_PPP (5)', 'GAPS (5)', 'OPUS_S (5)', 'Trimble_RTX (5)', 'AUSPOS (7)', 'CSRS_PPP (7)', 'GAPS (7)', 'OPUS_S (7)', 'Trimble_RTX (7)', 'AUSPOS (10)', 'CSRS_PPP (10)', 'GAPS (10)', 'OPUS_S (10)', 'Trimble_RTX (10)']
p_summary_service.title_text_font_size = '15pt'
p_summary_service.title_text_font_style = "bold"
p_summary_service.xaxis.axis_label_text_font_size  = '12pt'
p_summary_service.y_range = Range1d(start=0.5, end=30.5, bounds='auto')
p_summary_service.ygrid.grid_line_alpha = 0.5
p_summary_service.ygrid.grid_line_dash = [6, 4]
p_summary_service.yaxis.visible = None
p_summary_service.ygrid.ticker = SingleIntervalTicker(interval=1)
p_summary_service.extra_y_ranges = {'byService':FactorRange(factors=y_labels_service, bounds='auto'),
                            'byDuration': FactorRange(factors=y_labels_service_byTime, bounds='auto')}
cat_axis_service = CategoricalAxis(y_range_name='byService')
cat_axis_service_byTime = CategoricalAxis(y_range_name='byDuration')
p_summary_service.add_layout(cat_axis_service,'left')
p_summary_service.add_layout(cat_axis_service_byTime,'left')
p_summary_service_source = ColumnDataSource(p_summary_dict)
p_summary_service.ray(x='x',
              y='y',
              source=p_summary_service_source,
              length='range',
              angle=0,
              angle_units="deg",
              color='color',
              line_width=5)
p_summary_service.circle(x='avgs',
                 y='y',
                 source=p_summary_service_source,
                 fill_color="black",
                 line_color='white')


# kludgy work-around for legend limitation.  dummy sources for "invisible" series
dummy_sources = []
dummy_sources_hist = []

tabs_plot_dict = {'x':[0], 'y':[0]}
figs_byStation = []
figs_byStation_sources = []
tabs_byStation = []
red_line_sources = [[],[]]
for station in sorted(unq_cats['station']):
    figs_byStation_sources.append([])
    red_line_sources[0].append(ColumnDataSource({'x':[],'y':[],'text':[]}))
    figs_byStation.append(Figure(plot_width=550,
                               plot_height=400,
                               title="",
                               tools=TOOLS))

    figs_byStation[-1].line(x='x', y='y',
                            source=red_line_sources[0][-1],
                            line_color="red",
                            line_width=10,
                            line_alpha=0.2)

    figs_byStation[-1].text(x='x', y='y',
                            source=red_line_sources[0][-1],
                            text='text',
                            text_color="red",
                            text_align="left",
                            text_font_size="10pt",
                            text_alpha=0.2)

    figs_byStation[-1].xaxis.axis_label = 'nominalt (hrs)'
    figs_byStation[-1].xaxis.axis_label_text_font_size  = '12pt'


    for c, service in enumerate(sorted(unq_cats['service'])):
        col = avail_colors[c]
        figs_byStation_sources[-1].append(ColumnDataSource(tabs_plot_dict))
        figs_byStation[-1].line(x=nominalts,
                 y="y",
                 source=figs_byStation_sources[-1][-1],
                 line_dash=(4, 4),
                 line_color=col,
                 line_width=2)
                   
        figs_byStation[-1].circle(x=nominalts,
                   y="y",
                   source=figs_byStation_sources[-1][-1],
                   fill_color=col,
                   line_color=col,
                   legend=service)

    tabs_byStation.append(Panel(child=figs_byStation[-1], title=station))
                      
station_tabs = Tabs(tabs=tabs_byStation)

                          
figs_byService = []
figs_byService_sources = []
tabs_byService = []
for service in sorted(unq_cats['service']):
    figs_byService_sources.append([])
    red_line_sources[1].append(ColumnDataSource({'x':[],'y':[],'text':[]}))
    figs_byService.append(Figure(plot_width=550,
                               plot_height=400,
                               title="",
                               tools=TOOLS))

    figs_byService[-1].line(x='x', y='y',
                            source=red_line_sources[1][-1],
                            line_color="red",
                            line_width=10,
                            line_alpha=0.2)

    figs_byService[-1].text(x='x', y='y',
                            source=red_line_sources[1][-1],
                            text='text',
                            text_color="red",
                            text_align="left",
                            text_font_size="10pt",
                            text_alpha=0.2)

    figs_byService[-1].xaxis.axis_label = 'nominalt (hrs)'
    figs_byService[-1].xaxis.axis_label_text_font_size  = '12pt'

    for c, station in enumerate(sorted(unq_cats['station'])):
        col = avail_colors[c]
        figs_byService_sources[-1].append(ColumnDataSource(tabs_plot_dict))
        figs_byService[-1].line(x=nominalts,
                 y="y",
                 source=figs_byService_sources[-1][-1],
                 line_dash=(4, 4),
                 line_color=col,
                 line_width=2)
                   
        figs_byService[-1].circle(x=nominalts,
                   y="y",
                   source=figs_byService_sources[-1][-1],
                   fill_color=col,
                   line_color=col,
                   legend=station)

    tabs_byService.append(Panel(child=figs_byService[-1], title=service))
                      
service_tabs = Tabs(tabs=tabs_byService)


def makeLegend1(active_indices):
    for i, c in enumerate(avail_colors[:-1]):
        p.circle(x=[],
                 y=[],
                 color=c,
                 legend=sorted(unq_cats['service'])[i])
    p.legend.label_text_font_size = "13pt"
    p.xaxis.major_label_text_font_size = "17pt"
    p.yaxis.major_label_text_font_size = "17pt"

def makeLegend2(active_indices):
    for i, c in enumerate(avail_colors):
        p_hist.circle(x=[],
                 y=[],
                 color=c,
                 legend="series" + str(i + 1))

makeLegend1(checkboxes.active)
makeLegend2(hist_series_select.active)


def updateAxis(attribute, old, new):         
    x_name = axis_map_num[x_axis.value]
    y_name = axis_map_num[y_axis.value]
    p.xaxis.axis_label = x_axis.value
    p.xaxis.axis_label_text_font_size  = '12pt'
    p.yaxis.axis_label = y_axis.value
    p.yaxis.axis_label_text_font_size  = '12pt'
    source.data.update({'x':df[x_name]})
    source.data.update({'y':df[y_name]})


def updateRadii(attribute, old, new):
    radius_field = axis_map_num[radius.value]
    radius_values = df[radius_field]
    old_min = min(radius_values)
    old_max = max(radius_values)
    new_min = 5.0
    new_max = 15.0
    old_range = old_max - old_min
    new_range = new_max - new_min
    radii = []
    
    for r in radius_values:
        if old_range == 0:
            radii.append(new_min)
        else:
            # from stackexchange
            radii.append((((r - old_min) * new_range) / old_range) + new_min)
    source.data.update({'radius':radii})


def updateColor(attribute, old, new):
    color_field = axis_map_cat[color.value]

    if old != new:
        checkboxes.labels = sorted(unq_cats[color.value])
        checkboxes.active = [x for x in range(len(unq_cats[color.value]))]

        checkboxes_bulls_eye.labels = ["95% C.R." + s for s in sorted(unq_cats[color.value])]
        checkboxes_bulls_eye.active = []
        setCR_source()

    # make color list based on selected field
    colors = [avail_colors[sorted(unq_cats[color_field]).index(x)]
                  for x in df[color_field]]
    source.data.update({'color':colors})




def updateSeries(attribute, old, new):
    active_cats=[]
    active_inds = checkboxes.active
    try:
        for ind in active_inds:
            for i, c in enumerate(sorted(unq_cats[color.value])):
                if ind == i:
                    active_cats.append(c)

        global df
        df_temp = df_source.loc[df_source[color.value].isin(active_cats)]
        df = df_temp.loc[df_temp['nominalt'] == time]
        updateSource()
        
    except:
        print "there are no series selected to display"


def updateSource():

    source.data = defineDataDict()
    updateAxis(None,None,None)
    updateRadii(None,None,None)
    updateColor(None,None,None)

    for key in df:
        source.data.update({key:df[key]})


def updateLegend():
    num_legend_items = len(dummy_sources)
    num_active_series = len(checkboxes.active)

    for i, ds in enumerate(dummy_sources):

        if i <= num_active_series:
            
            ds.data.update({'legend_str':["dfdf"]})
            print dummy_sources[i].data
        else:
            print dummy_sources[i].data
            ds.data.update({'legend_str':[""]})


def updateConRad(attribute, old, new):

    dd = source_CR.to_df()
    x_cr = list(dd['x_CI'])
    y_cr = list(dd['y_CI'])
    color_cr = list(dd['color_CI'])
    x_mean = list(dd['x_mean'])
    y_mean = list(dd['y_mean'])
    cross_size = list(dd['size_'])

    if len(str(new)[1:-1]) == 0 or str(new) == 'None':
        print "nothing selected"

        if len(str(old)[1:-1]) == 1:
            old_ind = list(set(old) - set(new))[0]
            print str(old_ind) + " was unchecked"

            cross_size[old_ind] = 0
            x_cr[old_ind] = [np.nan]
            y_cr[old_ind] = [np.nan]
            x_mean[old_ind] = np.nan
            y_mean[old_ind] = np.nan


    else:

        if len(set(new) - set(old)) != 0: # something checked

            new_ind = list(set(new) - set(old))[0]

            # get data frame for checked-series source
            cat_to_get = sorted(checkboxes.labels)[new_ind]

            df_cr_temp = df_source.loc[df_source[color.value] == cat_to_get]
            df_cr = df_cr_temp.loc[df_cr_temp['nominalt'] == time]


            # calc mean residual
            des = df_cr['de (cm)']
            dns = df_cr['dn (cm)']
            # print ','.join(['cat_to_get','duration','mean de','mean dn','var_de','var_dn','cov_de_dn','sqrt(lambda1)','sqrt(lamba2)','theta'])

            mean_pos, a, b, c = calculateErrorEllipsePoints(cat_to_get,list(des),list(dns),2.45)


            ellip_points = get_ellipse_coords(a=a, b=b, x=mean_pos[0], y=mean_pos[1], angle=-1*c, k=1)

            # update CR source
            cross_size[new_ind] = 10
            x_mean[new_ind] = mean_pos[0]
            y_mean[new_ind] = mean_pos[1]
            x_cr[new_ind] = list(zip(*ellip_points.tolist())[0])
            y_cr[new_ind] = list(zip(*ellip_points.tolist())[1])
            color_cr[new_ind] = avail_colors[new_ind]


        else: # something unchecked
            old_ind = list(set(old) - set(new))[0]

            cross_size[old_ind] = 0
            x_cr[old_ind] = [np.nan]
            y_cr[old_ind] = [np.nan]
            x_mean[old_ind] = np.nan
            y_mean[old_ind] = np.nan



    lists = [x_cr,y_cr]
    for l1 in lists:
        for l2_i, l2 in enumerate(l1):
            isNaN = np.isnan(l2)
            for r, v in enumerate(isNaN):
                if v:
                    l2[r] = 0


    for l in [x_mean,y_mean,cross_size]:

        isNaN = np.isnan(l)
        for r, v in enumerate(isNaN):
            if v:
                l[r] = 0

    data_dict_CR = {'x_CI':x_cr,'y_CI':y_cr,'color_CI':color_cr,'x_mean':x_mean,'y_mean':y_mean,'size_':cross_size}
    source_CR.data.update(data_dict_CR)

# key concepts taken from
# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def calculateErrorEllipsePoints(cat, x,y,nstd):
    points = np.asarray(zip(x,y))
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]

    vals = vals[order]
    vecs = vecs[:,order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = nstd * np.sqrt(vals)

    data_str = str([cat,time,pos[0],pos[1],cov[0][0],cov[1][1],cov[0][1],width,height,theta])
    print data_str

    return pos, width, height, theta


# http://central.scipy.org/item/23/2/plot-an-ellipse
######  with one modification:  multiply angle by -1 ######
def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360*k+1, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)

    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts


def makeHTML_DataTable():
    # make table source data
    data = {}
    for k in df:
        data.update({k:[]})

    s2 = ColumnDataSource(data)
    dt = DataTable(source=s2,
                   columns=table_columns,
                   width=600,
                   height=250)

    source.callback = CustomJS(args=dict(s2=s2, dt=dt), code="""
            var inds = cb_obj.get('selected')['1d'].indices;
            var d1 = cb_obj.get('data');
            var d2 = s2.get('data');

            for (var key in d1) {
                d2[key] = [];

                for (i = 0; i < inds.length; i++) {
                    try {
                        d2[key].push(d1[key][inds[i]]);
                    }
                    catch(err) {
                        console.log("this key isn't in table source");
                    }
                }
            }
            console.log(d2);
            s2.trigger('change');
            dt.trigger('change');
        """)
    return dt


def calcStat(data, stat):
    result = np.nan
    if stat == "Min":
        result = np.amin(data)

    elif stat == "Max":
        result = np.amax(data)

    elif stat == "Range":
        result = np.amax(data) - np.amin(data)

    elif stat == "Std. Dev.":
        result = np.std(data)

    elif stat == "Mean":
        result = np.mean(data)

    elif stat == "RMS":
        result = sqrt(mean(square(data)))

    return result


def getVarData(df, var):

    result = np.nan
    if var == "dn (cm)" or var == "de (cm)" or var == "dh (cm)":
        data = df[var]

    else:
        n = list(df["dn (cm)"])
        e = list(df["de (cm)"])
        h = list(df["dh (cm)"])

        n.extend(e)
        
        if var == "Hor. Dist. 1": 
            data = n

        elif var == "3D Dist. 1":
            n.extend(h)
            data = n

        elif var == "Hor. Dist. 2":
            data = sqrt(square(df["dn (cm)"]) + square(df["de (cm)"]))

        elif var == "3D Dist. 2":
            data = sqrt(square(df["dn (cm)"]) + square(df["de (cm)"]) + square(df["dh (cm)"]))

    return data


def updateTabPlot(attribute, old, new):
    var = var_control.value
    stat = stat_control.value
    x_axis= 'nominalt'
    x_axis_values = sorted(list(set(df[x_axis])))
    df_tabs = df_source

    red_line_nominalt = x_axis_values[byPlot_series_control.active]

    for n in range(2):

        if n == 0:
            for s, sta in enumerate(sorted(unq_cats['station'])):

                for i, ser in enumerate(sorted(unq_cats['service'])):
                    stat_values = []

                    for x_val in x_axis_values:
                        # get data from df_source
                        df_tab_plot = getTabPlotDf(df_tabs, sta, ser, x_val)
                        data = getVarData(df_tab_plot, var)

                        if len(data) == 0:
                            stat_value = np.nan
                        else:
                            stat_value = calcStat(data, stat)
                        stat_values.append(stat_value)

                    isNaN = list(np.isnan(stat_values))
                    for r, sv in enumerate(isNaN):
                        if sv:
                            stat_values[r] = "NaN"

                    figs_byStation_sources[s][i].data.update({'y':stat_values})
                    figs_byStation[s].title = "Variable:  " + var
                    figs_byStation[s].title_text_font_size = '15pt'
                    figs_byStation[s].title_text_font_style = "bold"
                    figs_byStation[s].yaxis.axis_label = stat
                    figs_byStation[s].yaxis.axis_label_text_font_size  = '12pt'


        elif n == 1:
            for s, ser in enumerate(sorted(unq_cats['service'])):
                for i, sta in enumerate(sorted(unq_cats['station'])):
                    stat_values = []

                    for x_val in x_axis_values:
                        
                        # get data from df_source
                        df_tab_plot = getTabPlotDf(df_tabs, sta, ser, x_val)
                        data = getVarData(df_tab_plot, var)
                        if len(data) == 0:
                            stat_value = np.nan
                        else:
                            stat_value = calcStat(data, stat)
                        stat_values.append(stat_value)

                    isNaN = list(np.isnan(stat_values))
                    for r, sv in enumerate(isNaN):
                        if sv:
                            stat_values[r] = 0

                    figs_byService_sources[s][i].data.update({'y':stat_values})
                    figs_byService[s].title = "Variable:  " + var
                    figs_byService[s].title_text_font_size = '15pt'
                    figs_byService[s].title_text_font_style = "bold"
                    figs_byService[s].yaxis.axis_label = stat
                    figs_byService[s].yaxis.axis_label_text_font_size  = '12pt'


def updateHistPlot(attribute, old, new):
    plotBy = byPlot_control.value
    hist_series_select.labels = sorted(unq_cats[plotBy_dict[byPlot_control.value]])
    nominal_time = nominalts[byPlot_series_control.active]
    var = var_control.value
    activeSeries_indices = hist_series_select.active
    avail_series = hist_series_select.labels
    activeSeries = [s for i, s in enumerate(sorted(avail_series)) if i in activeSeries_indices ]
    plotBy_series = activeSeries
    df_nominalt = df_source.loc[df_source['nominalt'] == nominal_time]

    # get min and max values of all var for selected nominal time
    if var in ['Hor. Dist. 1', 'Hor. Dist. 2', '3D Dist. 1', '3D Dist. 2']:
        data = getVarData(df_nominalt, var)
    else:
        data = df_nominalt[var]
        
    min_ = int(np.amin(data) - 2)
    max_ = int(np.amax(data) + 2)

    # make 0.5-cm bins
    top = []
    left = []
    right = []
    colors = []
    line_color = []
    
    bin_size = 0.5
    bins = [float(x)/10 for x in range(int(min_*10),int(max_*10),int(bin_size*10))]

    for i, s in enumerate(sorted(plotBy_series)):

        # get df rows that satisfy series
        df_s = df_nominalt.loc[df_nominalt[plotBy_dict[plotBy]] == s]
        var_data = getVarData(df_s, var)

        # calculate counts for histogram
        counts, bins = np.histogram(var_data, bins=bins)

        # build lists to comprise data dict
        top.extend(counts.tolist())
        colors.extend([avail_colors[activeSeries_indices[i]] for x in range(len(counts))])
        left.extend(bins[:-1])
        right.extend(bins[1:])
        line_color.extend(['gray' for x in range(len(counts))])

    data = {'top':top,
           'left':left, 
           'right':right,
           'color':colors,
           'line_color':line_color}

    source_hist.data.update(data)

    p_hist.title = "'" + var + "'" + " Distribution (0.5-cm bins)"
    p_hist.title_text_font_size = '15pt'
    p_hist.title_text_font_style = "bold"
    p_hist.yaxis.axis_label = "count"
    p_hist.yaxis.axis_label_text_font_size  = '12pt'


def updateRedLine():
    # draw red line of tab plots to correspond with histogram series
    #plotBy_dict = {'By Station': 'station','By Service': 'service', 'Station Quality': 'sta_qual'}
    nominal_time_ind = byPlot_series_control.active
    nominal_time = nominalts[nominal_time_ind]

    # csb = color series by
    csb = byPlot_control.value
    if csb == 'By Station':
        for i, uc in enumerate(list(unq_cats[plotBy_dict[csb]])):
            print figs_byStation[i].y_range.end
            red_line_sources[0][i].data.update({'x':[nominal_time,nominal_time],
                                                'y':[0, figs_byStation[i].y_range.end],
                                                'text':['','(Histograms)']})
            for rls in red_line_sources[1]:
                rls.data.update({'x':[],'y':[]})

    elif csb == 'By Service':
        for i, uc in enumerate(list(unq_cats[plotBy_dict[csb]])):
            red_line_sources[1][i].data.update({'x':[nominal_time,nominal_time],
                                                'y':[0, figs_byService[i].y_range.end],
                                                'text':['','(Histograms)']})
            for rls in red_line_sources[0]:
                rls.data.update({'x':[],'y':[],'text':[]})

    elif csb == 'Station Quality':
        for i, uc in enumerate(list(unq_cats[plotBy_dict[csb]])):
            for rls in red_line_sources[0]:
                rls.data.update({'x':[],'y':[],'text':[]})
            for rls in red_line_sources[1]:
                rls.data.update({'x':[],'y':[],'text':[]})


def getTabPlotDf(df, station, service, nominalt):
    data = df[(df.station == station) \
              & (df.service == service) \
              & (df.nominalt == nominalt)]

    return data


def getTabPlotDf_QUAL(df, qual, service, nominalt):
    data = df[(df.sta_qual == qual) \
              & (df.service == service) \
              & (df.nominalt == nominalt)]
    return data


def makeMap():
    map_options = GMapOptions(lat=30.29,
                              lng=-97.73,
                              map_type="satellite",
                              zoom=11)

    mapPlot = GMapPlot(
        x_range=Range1d(),
        y_range=Range1d(),
        map_options=map_options,
        title="Austin"
    )

    source = ColumnDataSource(
        data=dict(
            lat=[30.29, 30.20, 30.29],
            lon=[-97.70, -97.74, -97.78],
        )
    )

    circle = Circle(x="lon",
                    y="lat",
                    size=15,
                    fill_color="blue",
                    fill_alpha=0.8,
                    line_color=None)
    
    mapPlot.add_glyph(source, circle)
    mapPlot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

    return mapPlot


# def makeSummaryPlot(attribute, old, new):
def makeSummaryStationPlot(attribute, old, new):
    y1 = 'station'
    x = []
    y = []
    rang = []
    color = []
    y_labels = []
    to_be_avg = []
    avgs = []

    y1_cats = sorted(unq_cats['station'])
    y2_source = figs_byStation_sources

    for z, y1_cat in enumerate(y1_cats):
        mins = []
        maxs = []

        # build to be averaged list of lists
        to_be_avg.append([])
        for t in nominalts:
            to_be_avg[-1].append([])

        for j, y2 in enumerate(y2_source[z]):
            for ind, value in enumerate(y2.data['y']):
                if value == 'NaN':
                    to_be_avg[-1][ind].extend([np.nan])
                else:
                    to_be_avg[-1][ind].extend([value])

            if j == 0:
                mins = y2.data['y']
                maxs = y2.data['y']

            else:
                mins = [s if (s < mins[i] and s != 'NaN')
                                 or (s != 'NaN' and mins[i] == 'NaN') else mins[i]
                            for i, s in enumerate(y2.data['y'])]

                maxs = [s if (s > maxs[i] and s != 'NaN')
                                 or (s != 'NaN' and maxs[i] == 'NaN') else maxs[i]
                            for i, s in enumerate(y2.data['y'])]

        for time in to_be_avg[-1]:
            if np.isnan(time).all():
                avgs.extend([np.nan])
            else:
                avgs.extend([np.nanmean(time)])

        mins=np.array([np.float64(n) for n in mins])
        maxs=np.array([np.float64(n) for n in maxs])
        ran = maxs - mins

        x.extend(list(mins))
        y.extend([n + 1 for n in range(len(rang),len(rang)+len(y2.data['y']),1)])
        y_labels.extend([y1_cat + " (" + str(int(n)) + ")" for n in nominalts])
        rang.extend(list(ran))
        color.extend([avail_colors[z] for e in mins])

    for e in x:

        isNaN_rang = np.isnan(rang)
        rang_ = [rang[i] if not n else "NaN" for i, n in enumerate(isNaN_rang)]

        isNaN_x = np.isnan(x)
        x_ = [x[i] if not n else "NaN" for i, n in enumerate(isNaN_x)]

        isNaN_avgs = np.isnan(avgs)
        avgs_ = [avgs[i] if not n else "NaN" for i, n in enumerate(isNaN_avgs)]

        isNaN_avgs = np.isnan(avgs)
        color_ = [color[i] if not n else "NaN" for i, n in enumerate(isNaN_avgs)]

    # update source based on sort-by selection
    if summary_station_sortBy.active == 0:
        p_summary_source.data.update({'x':x_,'y':y,'range':rang_,'color':color_,'avgs':avgs_})
        p_summary.title = y1 + " summary"
        p_summary.xaxis.axis_label = "variable: " + var_control.value + ", statistic: " + stat_control.value
        p_summary.yaxis[2].visible = False
        p_summary.yaxis[1].visible = True

    else:
        x = []
        y_new = []
        rang = []
        avgs = []
        color = []
        y_labels_ = []

        for n in range(len(nominalts)):
            x.append([])
            y_new.append([])
            rang.append([])
            avgs.append([])
            color.append([])
            y_labels_.append([])

        for i in y:
            if (i-1) % 6 == 0:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i-1) % 6].extend([y_labels[i-1]])
            elif (i-1) % 6 == 1:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 2:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 3:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 4:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 5:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])

        x_ = []
        rang_ = []
        avgs_ = []
        color_ = []
        y_labels = []
        for n in range(len(nominalts)):
            x_.extend(x[n])
            rang_.extend(rang[n])
            avgs_.extend(avgs[n])
            color_.extend(color[n])
            y_labels.extend(y_labels_[n])

        p_summary.yaxis[1].visible = None
        p_summary.yaxis[2].visible = True
        p_summary_source.data.update({'x':x_,'y':[x+1 for x in range(len(x_))],'range':rang_,'color':color_,'avgs':avgs_})
        p_summary.title = y1 + " summary"
        p_summary.xaxis.axis_label = "variable: " + var_control.value + ", statistic: " + stat_control.value


def makeSummaryServicePlot(attribute, old, new):
    y1 = 'service'
    x = []
    y = []
    rang = []
    color = []
    y_labels = []
    to_be_avg = []
    avgs = []

    y1_cats = sorted(unq_cats['service'])
    y2_source = figs_byService_sources

    for z, y1_cat in enumerate(y1_cats):
        mins = []
        maxs = []

        # build to be averaged list of lists
        to_be_avg.append([])
        for t in nominalts:
            to_be_avg[-1].append([])

        for j, y2 in enumerate(y2_source[z]):

            for ind, value in enumerate(y2.data['y']):
                if value == 'NaN':
                    to_be_avg[-1][ind].extend([np.nan])
                else:
                    to_be_avg[-1][ind].extend([value])

            if j == 0:
                mins = y2.data['y']
                maxs = y2.data['y']

            else:
                mins = [s if (s < mins[i] and s != 'NaN')
                                 or (s != 'NaN' and mins[i] == 'NaN') else mins[i]
                            for i, s in enumerate(y2.data['y'])]

                maxs = [s if (s > maxs[i] and s != 'NaN')
                                 or (s != 'NaN' and maxs[i] == 'NaN') else maxs[i]
                            for i, s in enumerate(y2.data['y'])]

        for time in to_be_avg[-1]:
            if np.isnan(time).all():
                avgs.extend([np.nan])
            else:
                avgs.extend([np.nanmean(time)])

        mins=np.array([np.float64(n) for n in mins])
        maxs=np.array([np.float64(n) for n in maxs])
        ran = maxs - mins

        x.extend(list(mins))
        y.extend([n + 1 for n in range(len(rang),len(rang)+len(y2.data['y']),1)])
        y_labels.extend([y1_cat + " (" + str(int(n)) + ")" for n in nominalts])
        rang.extend(list(ran))
        color.extend([avail_colors[z] for e in mins])
    for e in x:

        isNaN_rang = np.isnan(rang)
        rang_ = [rang[i] if not n else "NaN" for i, n in enumerate(isNaN_rang)]

        isNaN_x = np.isnan(x)
        x_ = [x[i] if not n else "NaN" for i, n in enumerate(isNaN_x)]

        isNaN_avgs = np.isnan(avgs)
        avgs_ = [avgs[i] if not n else "NaN" for i, n in enumerate(isNaN_avgs)]

        isNaN_avgs = np.isnan(avgs)
        color_ = [color[i] if not n else "NaN" for i, n in enumerate(isNaN_avgs)]

    # update source based on sort-by selection
    if summary_service_sortBy.active == 0:
        p_summary_service_source.data.update({'x':x_,'y':y,'range':rang_,'color':color_,'avgs':avgs_})
        p_summary_service.title = y1 + " summary"
        p_summary_service.xaxis.axis_label = "variable: " + var_control.value + ", statistic: " + stat_control.value
        p_summary_service.yaxis[2].visible = False
        p_summary_service.yaxis[1].visible = True

    else:
        x = []
        y_new = []
        rang = []
        avgs = []
        color = []
        y_labels_ = []

        for n in range(len(nominalts)):
            x.append([])
            y_new.append([])
            rang.append([])
            avgs.append([])
            color.append([])
            y_labels_.append([])

        for i in y:
            if (i-1) % 6 == 0:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i-1) % 6].extend([y_labels[i-1]])
            elif (i-1) % 6 == 1:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 2:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 3:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 4:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])
            elif (i-1) % 6 == 5:
                x[(i-1) % 6].extend([x_[i-1]])
                rang[(i-1) % 6].extend([rang_[i-1]])
                avgs[(i-1) % 6].extend([avgs_[i-1]])
                color[(i-1) % 6].extend([avail_colors[(i-1) % 6]])
                y_labels_[(i - 1) % 6].extend([y_labels[i - 1]])

        x_ = []
        rang_ = []
        avgs_ = []
        color_ = []
        y_labels = []
        for n in range(len(nominalts)):
            x_.extend(x[n])
            rang_.extend(rang[n])
            avgs_.extend(avgs[n])
            color_.extend(color[n])
            y_labels.extend(y_labels_[n])

        p_summary_service.yaxis[1].visible = None
        p_summary_service.yaxis[2].visible = True
        p_summary_service_source.data.update({'x':x_,'y':[x+1 for x in range(len(x_))],'range':rang_,'color':color_,'avgs':avgs_})
        p_summary_service.title = y1 + " summary"
        p_summary_service.xaxis.axis_label = "variable: " + var_control.value + ", statistic: " + stat_control.value


# assigen event handlers
controls1 = [x_axis,
            y_axis,
            radius,
            color]

controls2 = [var_control,
            stat_control,
            byPlot_control]

x_axis.on_change('value', updateAxis)
y_axis.on_change('value', updateAxis)
radius.on_change('value', updateRadii)
color.on_change('value', updateColor)
#color.on_change('labels', updateCR_source)
checkboxes.on_change('active', updateSeries)
var_control.on_change('value', updateTabPlot)
var_control.on_change('value', updateHistPlot)
stat_control.on_change('value', updateTabPlot)
byPlot_control.on_change('value', updateHistPlot)
byPlot_series_control.on_change('active', updateHistPlot)
hist_series_select.on_change('active', updateHistPlot)
summary_station_sortBy.on_change('active', makeSummaryStationPlot)
summary_service_sortBy.on_change('active', makeSummaryServicePlot)
checkboxes_bulls_eye.on_change('active', updateConRad)

# make html content
dt = makeHTML_DataTable()

inputs_left1 = HBox(VBoxForm(*controls1))
inputs_left2 = HBox(VBoxForm(*controls2))
scatter_controls = HBox(VBoxForm(checkboxes),VBoxForm(checkboxes_bulls_eye))
hist_controls = HBox(VBoxForm(byPlot_series_control),VBoxForm(hist_series_select))

updateSource()
updateTabPlot(None,None,None)
updateHistPlot(None,None,None)
makeSummaryStationPlot(None,None,None)
makeSummaryServicePlot(None,None,None)


LP_Graphs_upper_tab1 = Panel(child=p, title='Scatter Plot')
LP_Graphs_upper_tab2 = Panel(child=p_temp, title='Stations Map (in the works))')
LP_Graphs_upper_tab3 = Panel(child=HBox(VBoxForm(p_summary,summary_station_sortBy),
                                        VBoxForm(p_summary_service, summary_service_sortBy)),
                             title='Station/Service Summary')

LP_Graphs_upper = Tabs(tabs=[LP_Graphs_upper_tab1,
                             LP_Graphs_upper_tab3,
                             LP_Graphs_upper_tab2])

LP_Graphs_lower_tab1 = Panel(child=p_hist, title='Histograms')
LP_Graphs_lower_tab2 = Panel(child=dt, title='Query Data Frame')

LP_Graphs_lower = Tabs(tabs=[LP_Graphs_lower_tab1,
                             LP_Graphs_lower_tab2])

LP_Graphs = HBox(VBoxForm(LP_Graphs_upper, LP_Graphs_lower))
RP_Graphs = HBox(VBoxForm(station_tabs, service_tabs))

curdoc().add_root(HBox(HBox(VBoxForm(inputs_left1,scatter_controls,inputs_left2,hist_controls)), LP_Graphs, RP_Graphs))
# curdoc().add_root(HBox(makeMap()))

session.show() # open the document in a browser
session.loop_until_closed() # run forever
