import sys # importing sys module to take arguments from commandline
import numpy as np

CIRCUIT = '.circuit'
END = '.end'

class Component:
    name = ''
    n1 = None
    n2 = None
    n3 = None
    n4 = None
    control_current = ''
    value = 0
    # p_p_value = 0
    # phase = 0

    def __init__(self, data):
        self.name = data[0]
        self.n1 = data[1]# int(data[1]) if data[1]!="GND" else 0
        self.n2 = data[2]# int(data[2]) if data[2]!="GND" else 0
        if self.name[0] in set(['V','I']):
            if data[3] == 'ac':
                self.value = complex((float(data[4])/2)*np.cos(float(data[5])*(np.pi/180)),(float(data[4])/2)*np.sin(float(data[5])*(np.pi/180)))
            elif data[3] == 'dc':
                self.value = float(data[4])
        elif self.name[0] in set(['E','G']):
            self.n3 = data[3]# int(data[3]) if data[3]!="GND" else 0
            self.n4 = data[4]# int(data[4]) if data[4]!="GND" else 0
            self.value = float(data[5])
        elif self.name[0] in set(['H','F']):
            self.control_current = data[3]
            self.value = float(data[4])
        else:
            self.value = float(data[3])
    
class DC_Circuit:
    nodes = 0
    voltage_sources = 0
    components = None
    M = None
    x = None
    b = None
    Vs_current = dict()
    current_label = []
    node_table = dict()

    def __init__(self, comps):
        self.components = comps
        for comp in comps:
            if comp.name[0] in set(['V','E','H','L']):
                self.voltage_sources += 1
                self.Vs_current[comp.name] = self.voltage_sources
                self.current_label.append('I_'+comp.name)
            # self.nodes = max(self.nodes, comp.n1, comp.n2, comp.n3, comp.n4)
            if comp.n1 != "GND" and self.node_table.get(comp.n1,0) == 0:
                self.nodes += 1
                self.node_table[comp.n1] = self.nodes
            elif comp.n2 != "GND" and self.node_table.get(comp.n2,0) == 0:
                self.nodes += 1
                self.node_table[comp.n2] = self.nodes
            elif comp.n3 != None and comp.n3 != "GND" and self.node_table.get(comp.n3,0) == 0:
                self.nodes += 1
                self.node_table[comp.n3] = self.nodes
            elif comp.n4 != None and comp.n4 != "GND" and self.node_table.get(comp.n4,0) == 0:
                self.nodes += 1
                self.node_table[comp.n4] = self.nodes
        self.nodes += 1
        self.node_table["GND"] = 0
        self.M = np.zeros((self.nodes+self.voltage_sources, self.nodes+self.voltage_sources))
        self.b = np.zeros(self.nodes+self.voltage_sources)
        for comp in comps:
            if comp.name[0] == 'R':
                self.M[self.node_table[comp.n1],self.node_table[comp.n1]] += 1/comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n2]] += 1/comp.value
                self.M[self.node_table[comp.n1],self.node_table[comp.n2]] -= 1/comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n1]] -= 1/comp.value
            elif comp.name[0] == 'L':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.b[self.nodes-1+self.Vs_current[comp.name]] += 0
            elif comp.name[0] == 'C':
                self.b[self.node_table[comp.n1]] -= 0
                self.b[self.node_table[comp.n2]] += 0
            elif comp.name[0] == 'V':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.b[self.nodes-1+self.Vs_current[comp.name]] += comp.value
            elif comp.name[0] == 'I':
                self.b[self.node_table[comp.n1]] -= comp.value
                self.b[self.node_table[comp.n2]] += comp.value
            elif comp.name[0] == 'E':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n3]] -= comp.value
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n4]] += comp.value
            elif comp.name[0] == 'H':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.nodes-1+self.Vs_current[comp.control_current]] -= comp.value
            elif comp.name[0] == 'G':
                self.M[self.node_table[comp.n1],self.node_table[comp.n3]] += comp.value
                self.M[self.node_table[comp.n1],self.node_table[comp.n4]] -= comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n3]] -= comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n4]] += comp.value
            elif comp.name[0] == 'F':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.control_current]] += comp.value
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.control_current]] -= comp.value
        self.x = np.linalg.solve(self.M[1:,1:],self.b[1:])
    
    def solution(self):
        result = ''
        for node, node_value in self.node_table.items():
            if node != 'GND':
                result += 'Vn'+node+' '+str(self.x[node_value-1])+'\n'
        for i in range(self.nodes-1,self.x.size):
            result += self.current_label[i-self.nodes+1]+' '+str(self.x[i])+'\n'
        return result

class AC_Circuit:
    nodes = 0
    voltage_sources = 0
    frequency = 0
    components = None
    M = None
    x = None
    b = None
    Vs_current = dict()
    current_label = []
    node_table = dict()

    def __init__(self, comps, freq):
        self.components = comps
        self.frequency = freq
        for comp in comps:
            if comp.name[0] in set(['V','E','H']):
                self.voltage_sources += 1
                self.Vs_current[comp.name] = self.voltage_sources
                self.current_label.append('I_'+comp.name)
            # self.nodes = max(self.nodes, comp.n1, comp.n2, comp.n3, comp.n4)
            if comp.n1 != "GND" and self.node_table.get(comp.n1,0) == 0:
                self.nodes += 1
                self.node_table[comp.n1] = self.nodes
            elif comp.n2 != "GND" and self.node_table.get(comp.n2,0) == 0:
                self.nodes += 1
                self.node_table[comp.n2] = self.nodes
            elif comp.n3 != None and comp.n3 != "GND" and self.node_table.get(comp.n3,0) == 0:
                self.nodes += 1
                self.node_table[comp.n3] = self.nodes
            elif comp.n4 != None and comp.n4 != "GND" and self.node_table.get(comp.n4,0) == 0:
                self.nodes += 1
                self.node_table[comp.n4] = self.nodes
        self.nodes += 1
        self.node_table["GND"] = 0
        self.M = np.zeros((self.nodes+self.voltage_sources, self.nodes+self.voltage_sources), dtype=complex)
        self.b = np.zeros(self.nodes+self.voltage_sources, dtype=complex)
        for comp in comps:
            if comp.name[0] == 'R':
                self.M[self.node_table[comp.n1],self.node_table[comp.n1]] += 1/comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n2]] += 1/comp.value
                self.M[self.node_table[comp.n1],self.node_table[comp.n2]] -= 1/comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n1]] -= 1/comp.value
            elif comp.name[0] == 'L':
                self.M[self.node_table[comp.n1],self.node_table[comp.n1]] += 1/complex(0,2*np.pi*self.frequency*comp.value)
                self.M[self.node_table[comp.n2],self.node_table[comp.n2]] += 1/complex(0,2*np.pi*self.frequency*comp.value)
                self.M[self.node_table[comp.n1],self.node_table[comp.n2]] -= 1/complex(0,2*np.pi*self.frequency*comp.value)
                self.M[self.node_table[comp.n2],self.node_table[comp.n1]] -= 1/complex(0,2*np.pi*self.frequency*comp.value)
            elif comp.name[0] == 'C':
                self.M[self.node_table[comp.n1],self.node_table[comp.n1]] += complex(0,2*np.pi*self.frequency*comp.value)
                self.M[self.node_table[comp.n2],self.node_table[comp.n2]] += complex(0,2*np.pi*self.frequency*comp.value)
                self.M[self.node_table[comp.n1],self.node_table[comp.n2]] -= complex(0,2*np.pi*self.frequency*comp.value)
                self.M[self.node_table[comp.n2],self.node_table[comp.n1]] -= complex(0,2*np.pi*self.frequency*comp.value)
            elif comp.name[0] == 'V':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.b[self.nodes-1+self.Vs_current[comp.name]] += comp.value
            elif comp.name[0] == 'I':
                self.b[self.node_table[comp.n1]] -= comp.value
                self.b[self.node_table[comp.n2]] += comp.value
            elif comp.name[0] == 'E':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n3]] -= comp.value
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n4]] += comp.value
            elif comp.name[0] == 'H':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.name]] += 1
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.name]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n1]] += 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.node_table[comp.n2]] -= 1
                self.M[self.nodes-1+self.Vs_current[comp.name],self.nodes-1+self.Vs_current[comp.control_current]] -= comp.value
            elif comp.name[0] == 'G':
                self.M[self.node_table[comp.n1],self.node_table[comp.n3]] += comp.value
                self.M[self.node_table[comp.n1],self.node_table[comp.n4]] -= comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n3]] -= comp.value
                self.M[self.node_table[comp.n2],self.node_table[comp.n4]] += comp.value
            elif comp.name[0] == 'F':
                self.M[self.node_table[comp.n1],self.nodes-1+self.Vs_current[comp.control_current]] += comp.value
                self.M[self.node_table[comp.n2],self.nodes-1+self.Vs_current[comp.control_current]] -= comp.value
        self.x = np.linalg.solve(self.M[1:,1:],self.b[1:])
    
    def solution(self):
        result = ''
        for node, node_value in self.node_table.items():
            if node != 'GND':
                result += 'Vn'+node+' '+str(self.x[node_value-1])+'\n'
        for i in range(self.nodes-1,self.x.size):
            result += self.current_label[i-self.nodes+1]+' '+str(self.x[i])+'\n'
        return result

# a function which returns a list of lists which contains each token of each line that describes the circuit. The comments are also removed.
def process_lines(lines):
    flag = False
    data = []
    freq = 0
    
    for i in range(len(lines)):
        if lines[i] == CIRCUIT:
            flag = True
            continue
        elif lines[i] == END:
            flag = False
        elif flag:
            data.append(lines[i].split('#')[0].split())
        elif lines[i].startswith('.ac'):
            freq = float(lines[i].split('#')[0].split()[-1])
    return data, freq

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

        data, freq = process_lines(lines) # call the above function which returns a relevant data.
        # print(string_to_print(data)) # print the data in reversed order.
        comps = []
        for line in data:
            comps.append(Component(line))
        if freq == 0:
            ckt = DC_Circuit(comps)
        else:
            ckt = AC_Circuit(comps, freq)
        print(ckt.solution())

    # if an invalid file is given as argument, print the error message
    except IOError as e:
        print('Invalid input')
    except np.linalg.LinAlgError:
        print(e)
        print('Invalid Input - circuit cannot be solved')

main() # calling the main function.