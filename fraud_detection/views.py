from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
import os
import json
from .mlmodel import credit_card_fraud_detection as cc
# Create your views here.

def home(request):
    if request.method == 'POST':
        data = [[]]
        for i in range(0,30):
            data[0].append(float(request.POST.get("transaction["+str(i)+"]")))
        print(data)
        result = cc.find(data)
        return JsonResponse({'result':result})
    data = pd.read_csv('fraud_detection/mlmodel/creditcard.csv')
    data=data[data['Class']==1]
    data=data.head()
    json_records = data.reset_index().to_json(orient ='records') 
    data = json.loads(json_records) 
    # print(data)
    return render(request,'fraud_detection/index.html',{"data" : data})
    