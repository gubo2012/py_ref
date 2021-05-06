
import matplotlib 
from cycler import cycler
from functools import wraps

#################################################################
#Setting up hms colors

#Taken from https://buzz.hms.com/sites/marcom/Guides/HMS-Style-Guide.pdf
#Also see Branding quick links, https://buzz.hms.com/sites/marcom/SitePages/Home.aspx
#Prefer Orange and DarkGrey, should sparingly use the others

hms_colors = {'Orange'       : '#ff6633',
              'LightBlue'    : '#00a9e0',
              'DarkGrey'     : '#5b6770',
              'Teal'         : '#00c7b1',
              'LightPurple'  : '#8687c2',
              'Green'        : '#63be60',
              'LightGrey'    : '#c1c6c8',
              'Black'        : '#000000',
              'DarkBlue'     : '#265d77',
              'DarkPurple'   : '#3b2a71'}

font_color = hms_colors['DarkBlue']
hms_cycle = cycler(color=list(hms_colors.values()))

#Do you need more colors?
#I've seen them use 
# dark purple, [59, 42, 113] '#3b2a71'
# light purple [134, 135, 194] '#8687c2'
# green        [99, 190, 96] '#63be60'
# or black     [0, 0, 0] '#000000'
##################################################################

##################################################################
#These are my suggested changes to the default plot

hms_style = {'font.sans-serif' : "Arial",
             'font.family' : "sans-serif",
             'axes.grid' : True,
             'grid.linestyle' : '--',
             'grid.color' : hms_colors['LightGrey'],
             'legend.framealpha' : 1,
             'legend.facecolor' : 'white',
             'legend.shadow' : True,
             'legend.fontsize' : 14,
             'legend.title_fontsize' : 16,
             'xtick.labelsize' : 14,
             'ytick.labelsize' : 14,
             'axes.labelsize' : 16,
             'axes.titlesize' : 20,
             'axes.axisbelow' : True,
             'figure.figsize' : [6, 4],
             'figure.dpi' : 100,
             'text.color' : font_color,
             'axes.labelcolor' : font_color,
             'axes.edgecolor' : font_color,
             'xtick.color' : font_color,
             'ytick.color' : font_color,
             'legend.edgecolor' : font_color,
             'axes.prop_cycle' : hms_cycle}

#matplotlib.rcParams.update(hms_style)

#To set back to default
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
#Or
#matplotlib.pyplot.style.use('default') #better for Jupyter notebook
##################################################################


##################################################################
#Functions to change the style

#can pass a dictionary to up_dict to change the defaults
#in the style sheet, style=False sets it back to defaults
def hms_plots(style=True, up_dict={}):
    if style:
        #if you pass in other options
        if up_dict:
            up_style = hms_style.copy()
            up_style.update(up_dict)
            matplotlib.rcParams.update(up_style)
        else:
            matplotlib.rcParams.update(hms_style)
    else:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.pyplot.style.use('default')
#had a hard time passing in as kwargs, since the style sheet 
#items are all ???.??? format, so would get an error


#adding in a decorator
def style_wrap(orig_func):
    #get the current global style
    cur_style = matplotlib.rcParams.copy()
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        hms_plots(style=True) 
        result = orig_func(*args, **kwargs)
        hms_plots(style=True, up_dict=cur_style) #changes the state back
        return result
    return wrapper
##################################################################