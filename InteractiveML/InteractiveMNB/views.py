from django.shortcuts import render
from django.http import HttpResponse
from MLmodels import MNB

# Create your views here.
model = MNB.MNBmodel()

def predictor(request):
    if request.method == 'GET':
        model.getTrainTestData()
        model.train()
        model.test()
        
    
    elif request.method == 'POST':
        txtDoc = request.POST['txtdoc']
        prediction = model.predict(txtDoc)
        prediction['accuracy'] = model.acc
        return render(request, 'main.html', prediction)
    
    return render(request, 'main.html', {'accuracy': model.acc})