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

def add_word(request):
    word = request.POST['word']
    model.add_word_vocabulary(word)
    model.getTrainTestData()
    model.train()
    model.test()
    prediction = model.predict()
    prediction['accuracy'] = model.acc
    return render(request, 'main.html', prediction)

def rem_word(request):
    word = request.POST['word']
    model.rem_word_vocabulary(word)
    model.getTrainTestData()
    model.train()
    model.test()
    prediction = model.predict()
    prediction['accuracy'] = model.acc
    return render(request, 'main.html', prediction)

def adj_weight(request):
    word = request.POST['word']
    weight = request.POST['weight']
    model.adj_weight(word, weight)
    model.getTrainTestData()
    model.train()
    model.test()
    prediction = model.predict()
    prediction['accuracy'] = model.acc
    return render(request, 'main.html', prediction)