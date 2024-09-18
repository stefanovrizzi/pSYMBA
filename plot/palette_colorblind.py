# Step 1: Define the verbose descriptions and corresponding LaTeX labels in a list
conditions_order = [r'$B_{+}$', r'$B_{0}$', r'$B_{-}$', r'$B_{00}$', r'$Z$', r'$f$', 'LSE']
cognitive_traits = ['optimistic bias', 'no bias', 'pessimistic bias', 'no bias individual', 'no learning', 'fundamental value', 'LSE']

# Step 2: Create a direct mapping from verbose descriptions to LaTeX labels
# Map the verbose descriptions to LaTeX labels directly using the order
biasDict = dict(zip(cognitive_traits, conditions_order))

# Step 3: Generate the numeric mappings based on the order
# Create the reverse mapping to generate dict_num2label
dict_string2num = {description: idx for idx, description in enumerate(cognitive_traits)}
dict_num2label = {idx: label for idx, label in enumerate(conditions_order)}
dict_label2num = {v: k for k, v in dict_num2label.items()}

# Output dictionaries for verification (optional)
#print("dict_string2num:", dict_string2num)
#print("dict_num2label:", dict_num2label)
#print("biasDict:", biasDict)

#palette = [
#    "#AA3377",  # Dark Magenta (Aubergine)
#    "#66CCEE",  # Sky Blue (Cyan)    
#    "#4477AA",  # Muted Blue
#    "#EE6677",  # Salmon Pink (Coral)    
#    "#EECC66",  # Sandy Yellow
#    "#BBBBBB"   # Light Grey
#    "#997700",  # Olive Brown (Golden Brown)
#    ]
   
palette = [
    "#AA3377",  # Dark Magenta (Aubergine)
    "#4477AA",  # Muted Blue
    "#66CCEE",  # Sky Blue (Cyan)    
    "#228833",  # Grass Green (Forest Green)
    "#EECC66",  # Sandy Yellow
    "#BBBBBB",   # Light Grey
    "#000000"  # Black
    #"#997700",  # Olive Brown (Golden Brown)
    ]
    

