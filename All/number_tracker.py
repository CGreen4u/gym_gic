import pandas as pd
import os
import pandas
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")


class number_tracking:

    #be sure to put the location of the file out in front
    os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")
    def __init__(self):
        self.graph_data = pd.read_csv(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\example_data_in_order.csv')
        self.does_the_csv_exist()
        self.lst = []
        self.list_of_gic_actual = []
        self.value = 0
        self.number_of_times_script_ran()
        self.new_df()
        #self.list_for_active_chart()


    def number_of_times_script_ran(self):
        '''keeping track of counts in df'''
        df = pandas.read_csv('counter.csv')
        df = df.add(1)
        self.value = df.iat[0,0]
        #print(value)
        df.to_csv('counter.csv',index=False)
        
    def new_df(self):
        '''we are getting the data in list adding 1 to count to go forward'''
        self.lst = self.graph_data['gic'].tolist()
        #print(self.lst[self.value])
        return self.lst[self.value]

    def does_the_csv_exist(self):
        # if file does not exist write header 
        if not os.path.isfile('counter.csv'):
            d = {'count': [0]}
            df = pd.DataFrame(data=d)
            df.to_csv('counter.csv',index=False)
        else: # else it exists so append without writing the header
           pass

    def other_variables_in_list(self):
        '''We are bringing in a big data frame so i wanted to separate out the values
        we are grabing them and puting them into a separte list'''
        #print(self.graph_data)
        np_graph_data = self.graph_data.as_matrix(columns=self.graph_data.columns[1:])
        return np_graph_data[self.value]

    def list_for_active_chart(self):
        return self.list_of_gic_actual.append(self.new_df())

'''
import numpy as np
obj = number_tracking()
y = 3
x = obj.other_variables_in_list()
z = np.array([1, x])
w = np.array([1,0,0,0])
print(z[0])
print(z[1][0])
print(z[1][1])
print(z[1][2])
print(z)
print(x)
print(np.array([z[0],z[1][0],z[1][1],z[1][2]]))
print(w[0])


observation_if_outside_target = 13
observation_view_of_alternative_variables = np.array([observation_if_outside_target, x])
print(observation_view_of_alternative_variables[0],observation_view_of_alternative_variables[1][0])
observation_space = tuple((observation_view_of_alternative_variables[0],observation_view_of_alternative_variables[1][0],observation_view_of_alternative_variables[1][1],observation_view_of_alternative_variables[1][2]))

print(observation_space)
'''

##obj = number_tracking()
##    #def numbers_to_track(self):
##
##print(obj.new_df())

        

