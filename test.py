'''

with open("out0.chroma") as f:
    for line in f:
        if "146/146056.mp3" in line:
            print(line)

# Open file
fileHandler = open("out0.chroma", "r")
while True:
    # Get next line from file
    line = fileHandler.readline()
    try:
        feat = str(line.split(";",1)[1]).replace(' ','')
        # If line is empty then end of file reached
        if((not str(feat).startswith("[[0")) and (not str(feat).startswith("[[1"))):
            print("OUT: \n")
            print(line)
        if not line:
            break;
    except: 
        print("ERROR: \n")
        print(line) 
# Close Close
fileHandler.close()

'''

#146/146056.mp3; [] 
#114/114497.mp3; []
#145/145056.mp3; []
#080/080237.mp3; []


# Open file
fileHandler = open("out0.chroma", "r")
while True:
    # Get next line from file
    line = fileHandler.readline()
    if not line:
        break;

    feat = str(line).replace(' ','')
    # If line is empty then end of file reached
    if(not "[[0" in line) and (not "[[1" in line):
        print("OUT: \n")
        print(line)

    feat = str(line.split(";",1)[1]).replace(' ','')

    try: 
        feat = str(line.split(";",1)[1]).replace(' ','')
    except: 
        print("ERROR: \n")
        print(feat)    

    if(not ";" in line):
        print("NO ; : \n")
        print(line)
       
# Close Close
fileHandler.close()

