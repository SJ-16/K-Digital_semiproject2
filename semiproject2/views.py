from django.shortcuts import render, redirect

from frame.custdb import CustDB
from frame.error import ErrorCode
from frame.linkdb import LinkDB
from myanalysis.best import Analysis
# Create your views here.

def index(request):
    return render(request, 'index.html');

def home2(request):
    return render(request, 'home2.html')

def login(request):
    return render(request, 'login.html')

def loginimpl(request):
    id = request.POST['id']
    pwd = request.POST['pwd']

    try:
        cust = CustDB().selectOne(id)
        if pwd == cust.getPwd():
            request.session['logininfo'] = {'id': cust.getId(), 'name': cust.getName()}
            print(request.session['logininfo'])
            return redirect('index')
        else:
            raise Exception
    except:
        next = 'loginfail.html'
        context = {'msg': ErrorCode.e02}
        return render(request, next, context)

def logout(request):
    if request.session['logininfo'] != None:
        del request.session['logininfo']
    return redirect('index')


def signup(request):
    return render(request, 'signup.html')

def signupimpl(request):
    try:
        id = request.POST['id']
        pwd = request.POST['pwd']
        name = request.POST['name']
        age = int(request.POST['age'])
        height = float(request.POST['ht'])
        weight = int(request.POST['wt'])
        CustDB().insert(id, pwd, name, age, height, weight)
        return render(request, 'signupsuccess.html')
    except Exception as err:
        print(err)
        context = {'msg': ErrorCode.e01}
        return render(request, 'signupfail.html', context)

def myinfo(request):
    id = request.session['logininfo']['id']
    cust = CustDB().selectOne(id)
    context = {'cust': cust}
    return render(request, 'myinfo.html', context)

def infoupdate(request):
    try:
        id = request.POST['id']
        pwd = request.POST['pwd']
        name = request.POST['name']
        age = int(request.POST['age'])
        height = float(request.POST['ht'])
        weight = int(request.POST['wt'])
        CustDB().update(id, pwd, name, age, height, weight)
        return redirect('index')
    except Exception as err:
        print('에러:', err)
        context= {'msg': ErrorCode.e03}
        return render(request, 'updatefail.html', context)

def infodelete(request):
    id = request.GET['id']
    CustDB().delete(id)
    return redirect('index')

def recommend(request):
    id = request.session['logininfo']['id']
    cust = CustDB().selectOne(id)
    age = cust.getAge()
    height = cust.getHt()
    weight = cust.getWt()
    # 분석내용이랑 연결해야 함
    size = Analysis().sizeRecomm(age, height, weight)
    # 해당 사이즈에 맞는 웹사이트 링크 가져오기
    links = LinkDB().selectOne(size)
    context = {
        'size': size,
        'mf': links[0],
        'yoox': links[1]
    }
    return render(request, 'recommend.html', context)
