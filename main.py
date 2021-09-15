from Expediency import *
import json
import cv2
if __name__ == "__main__":

    my_Data = Data_Loader()
    image_Array, label_Array = my_Data.load_Images(debug=True,load=False)
    print(image_Array.shape)
    print(label_Array.shape)
    
    RF = RF_Classifier(X=image_Array,y=label_Array)
    RF.create_Classifier()
    RF.data_Splitting()
    RF.train()
    a = RF.evaluate()
    DT = DT_Classifier(X=image_Array,y=label_Array)
    DT.create_Classifier()
    DT.data_Splitting()
    DT.train()
    b = DT.evaluate()
    SVM = SVM_Classifier(X=image_Array,y=label_Array)
    SVM.create_Classifier()
    SVM.data_Splitting()
    SVM.train()
    c = SVM.evaluate()
    dictionary ={
    "SVM" : c,
    "DT" : b,
    "RF" : a,
    "DEN" : d,
    "YOL" :e,
    }
    
    with open("result.json", "w") as outfile:
        json.dump(dictionary, outfile)

    plot_Metrics(jsonFile="result.json")

    

