from abc import ABC, abstractmethod


# interface for Model

class MLmodel(ABC):
    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def test(self):
        pass
# concrete productS

class DecisionTreeML(MLmodel):
    def train(self):
        return "Decision tree model training..."
    def test(self):
        return "Decision tree model testing..."
class KNNML(MLmodel):
    def train(self):
        return "KNN model training..."
    def test(self):
        return "KNN model testing..." 
# factory Model traing and testing:

class TrainAndTestModel:
    def startTrainingAndTesting(self,type_model):
        if type_model=="DecisionTreeML":
            a=DecisionTreeML().train()
            b=DecisionTreeML().test()
            return (a,b)
        elif type_model=="KNNML":
            return (KNNML().train(),KNNML().test())
        else:
            return ("nothing trained..","nothing tested..")
# A user of the factory

if __name__=="__main__":
    model=TrainAndTestModel()
    (a,b)=model.startTrainingAndTesting("DecisionTreeML")
    print(a)
    print(b)
    
    
    