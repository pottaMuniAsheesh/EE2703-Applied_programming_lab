import sys # importing sys module to take arguments from commandline

CIRCUIT = '.circuit'
END = '.end'

# a function which returns a list of lists which contains each token of each line that describes the circuit. The comments are also removed.
def process_lines(lines):
    flag = False
    data = []
    
    for i in range(len(lines)):
        if lines[i] == CIRCUIT:
            flag = True
            continue
        elif lines[i] == END:
            flag = False
            break
        elif flag:
            data.append(lines[i].split('#')[0].split())
    return data

# a function which returns a string which when printed shows each line in reversed order with tokens of each line also in reversed order.
def string_to_print(data):
    reversed_lines = []
    for i in range(-1, -len(data)-1, -1):
        reversed_lines.append(' '.join(data[i][::-1]))
    return '\n'.join(reversed_lines)

# the main function.
def main():
    # try to open the file given as argument and process it.
    try:
        # check if only one file is given, if not print the below message.
        if len(sys.argv) != 2:
            print("No arguments or more than required number of arguments given")
            return
        # open the file, read from it and create a list containing each line.
        with open(sys.argv[1], 'r') as fh:
            lines = fh.read()
            lines = lines.split('\n')

        data = process_lines(lines) # call the above function which returns a relevant data.
        print(string_to_print(data)) # print the data in reversed order.

    # if an invalid file is given as argument, print the error message
    except IOError as e:
        print('Invalid input')

main() # calling the main function.
