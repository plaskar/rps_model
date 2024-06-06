
#############################################
############     HELPER FUNCTIONS ############
#############################################

# Function to determine numerical stability
def stability_colour(val_list, c_stable='blue', c_unstable='red', c_other='black'):
    n = len(val_list)
    out = []
    for i in range(n):
        temp = [c_stable if val<0 else c_unstable if val>0 else c_other for val in val_list[i]]
        out.append(temp)
    return out 



####################### 0. Making Progress Bar ##############################
import time
from IPython.display import display, HTML, clear_output

# Function to display the progress bar
def display_progress(progress, total):
    percent = int((progress / total) * 100)
    bar_length = 30
    bar = '█' * int(bar_length * percent / 100) + '-' * (bar_length - int(bar_length * percent / 100))
    display(HTML(f"<div style='width: 50%;'><div style='width: {percent}%; background-color: blue; color: white; text-align: center;'>{percent}%</div></div>"))
    clear_output(wait=True)
    
