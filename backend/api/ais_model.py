from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

class aisModel:
    def __init__(self,aisFeatures):
        self.ais_features=aisFeatures
        self.df=pd.read_csv('https://raw.githubusercontent.com/abhinavanagarajan/election/main/oilspilldataset.csv')
        self.y=self.df['SpillCond']
        self.x=self.df.drop('SpillCond',axis=1)
    def train_Model(self):
        x_train,x_test,y_train,y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=155)
        self.model=RandomForestClassifier(n_estimators=175)
        self.model.fit(x_train,y_train)
        self.prediction_Tests(x_train,x_test,y_train,y_test)
    def prediction_Tests(self,x_train,x_test,y_train,y_test):
        y_lr_train_pred=self.model.predict(x_train)
        y_lr_test_pred=self.model.predict(x_test)

        lr_train_mse=mean_squared_error(y_train,y_lr_train_pred)
        lr_train_r2=r2_score(y_train,y_lr_train_pred)
        lr_test_mse=mean_squared_error(y_test,y_lr_test_pred)
        lr_test_r2=r2_score(y_test,y_lr_test_pred)

        adj_lr_test_r2=1-(1-lr_test_r2)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
        print('RFC MSE (Train):',lr_train_mse)
        #print('RFC R2 (Train):',lr_train_r2)
        print('RFC MSE (Test):',lr_test_mse)
        #print('RFC R2 (Test):',lr_test_r2)
        #print('RFC Adjusted R2 (Test):',adj_lr_test_r2)

        print('RFC Accuracy:',accuracy_score(y_test,y_lr_test_pred))

        lr_results=pd.DataFrame(['Random Forest Classification',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
        lr_results.columns=['Method','Training MSE','Training R2','Test MSE','Test R2']

        plt.figure(figsize=(5,5))
        plt.scatter(y_train,y_lr_train_pred,c='black',alpha=0.3)
        plt.plot(y_train,y_train,c='blue',linewidth=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        #plt.show()
    def simulate_Input(self):
        l=[2,22325,79.11,841.03,180,55812500,51.11,1.21,61900,0.02,901.7,0.02,0.03,0.11,0.01,0.11,6058.23,4061.15,2.3,0.02,0.02,87.65,0,0.58,132.78,-0.01,3.78,0.84,7.09,-2.21,0,0,0,0,704,40140,0,68.65,89,69,5750,11500,9593.48,1648.8,0.6,0,51572.04,65.73,6.26,0]
        #print(len(l))
        outp=[]
        for ea in range(5):
            k=[]
            for i in range(len(l)-1):
                if (type(l[i]) == int):
                    k.append(l[i]+random.randint(0,9))
                else:
                    k.append(round(l[i]+random.random(),3))
            outp.append(k)
        outp.append([4,1201,1562.53,295.65,66,3002500,42.4,7.97,18030,0.19,166.5,0.21,0.26,0.48,0.1,0.38,120.22,33.47,1.91,0.16,0.21,87.65,0,0.48,132.78,-0.01,3.78,0.84,6.78,-3.54,-0.33,2.2,0,2.2,183,10080,0,108.27,89,69,6041.52,761.58,453.21,144.97,13.33,1,37696.21,65.67,8.07])

    def getPredictions(self):
        self.train_Model()
        #self.simulate_Input()
        input_data = pd.DataFrame(self.ais_features,columns=['Course', 'Speed', 'Heading', 'RateofTurn', 'PositionAccuracy', 'SOG', 'Latitude', 'Longitude', 'COG', 'AISType', 'NavigationalStatus', 'IMOMessage', 'Timestamp', 'Date', 'WeatherConditions', 'DistanceTraveled', 'TimeToDestination', 'DeviationfromCourse', 'VesselTrafficDensity', 'VesselCollisions', 'VesselEncounters', 'PortTraffic', 'VesselSpeedViolations', 'VesselTracking', 'VesselIdentification', 'VesselClassification', 'VesselBehaviorAnalysis', 'VesselSafety', 'VesselSecurity', 'VesselEfficiency', 'VesselEmissions', 'VesselFuelConsumption', 'VesselMaintenance', 'VesselInsurance', 'VesselFinancing', 'VesselValuation', 'VesselOwnership', 'VesselDisplacement', 'VesselDeadweight', 'VesselGrossTonnage', 'VesselNetTonnage', 'VesselDraft', 'VesselBeam', 'VesselLength', 'VesselHeight', 'VesselEnginePower', 'VesselFuelCapacity', 'VesselCargoCapacity', 'VesselCrewSize'])
        predictions = self.model.predict(input_data)
        # Print the prediction
        print(predictions)
        return predictions

aism=aisModel([[4,1201,1562.53,295.65,66,3002500,42.4,7.97,18030,0.19,166.5,0.21,0.26,0.48,0.1,0.38,120.22,33.47,1.91,0.16,0.21,87.65,0,0.48,132.78,-0.01,3.78,0.84,6.78,-3.54,-0.33,2.2,0,2.2,183,10080,0,108.27,89,69,6041.52,761.58,453.21,144.97,13.33,1,37696.21,65.67,8.07]])
res=aism.getPredictions()
print(res)