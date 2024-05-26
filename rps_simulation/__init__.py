
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


# convert input string to learning_curve class
#def string_to_learning(string):
    