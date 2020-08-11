
import pandas as pd
import os
import pandas
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")


class number_tracking:

    #be sure to put the location of the file out in front
    os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")
    def __init__(self):
        self.graph_data = pd.read_csv(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\example2.txt')
        self.does_the_csv_exist()
        self.lst = []
        self.value = 0
        self.number_of_times_script_ran()
        self.new_df()
        

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

##obj = number_tracking()
##    #def numbers_to_track(self):
##
##print(obj.new_df())

        

