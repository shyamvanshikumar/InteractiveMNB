from django.shortcuts import render
from django.http import HttpResponse
from MLmodels import MNB

# Create your views here.
model = MNB.MNBmodel()

def predictor(request):
    if request.method == 'GET':
        model.getTrainTestData()
        k_fold_accuracy = model.trainAndValidate()
        model.train()
        model.test()
        
    
    elif request.method == 'POST':
        txtDoc = request.POST['txtdoc']
        prediction = model.predict(txtDoc)
        prediction['accuracy'] = model.acc
        return render(request, 'main.html', prediction)
    
    return render(request, 'main.html', {'accuracy': k_fold_accuracy})

def add_word(request):
    print("here")
    word = request.POST.get('word', False)
    if word == False:
        return predictor(request)
    model.add_word_vocabulary(word)
    model.getTrainTestData()
    model.train()
    model.test()
    prediction = model.predict()
    prediction['accuracy'] = model.acc
    return render(request, 'main.html', prediction)

def rem_word(request):
    word = request.POST.get('word', False)
    if word == False:
        return predictor(request)
    model.rem_word_vocabulary(word)
    model.getTrainTestData()
    model.train()
    model.test()
    prediction = model.predict()
    prediction['accuracy'] = model.acc
    return render(request, 'main.html', prediction)

def adj_weight(request):
    word = request.POST.get('word', False)
    weight = request.POST.get('weight', False)
    if word == False:
        return predictor(request)
    if weight == False:
        weight = 1
    model.adj_weight(word, weight)
    model.getTrainTestData()
    model.train()
    model.test()
    prediction = model.predict()
    prediction['accuracy'] = model.acc
    return render(request, 'main.html', prediction)