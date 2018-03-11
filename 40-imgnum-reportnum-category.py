import bs4

import lxml

import MeCab

import dbm

import os

import pickle

import glob

import concurrent.futures

import hashlib

import json

import os

import re

import datetime

import sys

import math

import numpy as np

m = MeCab.Tagger("-Owakati")

def _map(arg):
  key,names = arg

  results = []
  for name in names:
    save = name.split('/').pop()
    if os.path.exists(f'contents/{save}'):
      continue
    soup = bs4.BeautifulSoup(open(name).read(), "lxml")

    number = soup.find('strong', {'class':'number'})
    if number is None:
      continue

    bar = soup.find('div', {'class':'bar'})
    target = soup.find('div', {'class':'target'})
    sub = soup.find('div', {'class':'subtitle'})
    patron = soup.find('div', {'class':'patron'})
    
    try:
      profile, category = sub.find_all('a')
    except:
      continue

    # img属性の数をカウント
    imgs = soup.find_all('img')
    imgs_size = len(imgs)

    # statusコード{おめでとうございます:green, 終了しました:gray} 
    status = soup.find('section', {'class':'status'})
    try:
      number = re.search(r'\d{1,}', number.text.replace(',', '')).group(0)
      target = re.search(r'\d{1,}', target.text.replace(',', '')).group(0)
      category = category.text.replace(',', '')
      patron = re.search(r'\d{1,}', patron.text.replace(',', '')).group(0)
      status = status.text.strip().replace('\n', '')

    except:
      continue

    success = 'success' if 'おめでとうございます' in status else 'fail'
    days = re.findall(r'\d{4}/\d{2}/\d{2}', status)
    days = [ datetime.datetime.strptime(day, '%Y/%m/%d') for day in days]
    delta = re.search(r'\d{1,}', f'{days[1] - days[0]}').group(0)
    result = f'test {target} {category} {imgs_size} {patron} {success} {delta}'
    print(result)
    result = {'imgs_size':imgs_size, 'patron':patron, 'success':success, 'category':category, 'target':target, 'delta':delta}
    results.append(result)

  return results
  #o = {"time":time, "titles":titles, "bodies":bodies }
  #save = name.split('/').pop()
  #open(f'contents/{save}', 'w').write( json.dumps(o, indent=2, ensure_ascii=False) )

# step 1 parse html -> jsonp
if '--step1' in sys.argv:
  args = {}
  for index, name in enumerate(glob.glob('htmls/*')):
    key = index%32
    if args.get(key) is None:
      args[key] = []
    args[key].append( name )
  args = [(key,names) for key,names in args.items()]
  #_map(args[0])

  results = []
  fp = open('param.jsonp', 'w')
  with concurrent.futures.ProcessPoolExecutor(max_workers=18) as exe:
    for _results in exe.map(_map, args):
      for result in _results:
        fp.write(f'{json.dumps(result, ensure_ascii=False)}\n')

# step 2 parse jsonp -> vector  
if '--step2' in sys.argv:
  feat_index = {}
  for line in open('./param.jsonp'):
    line = line.strip()
    obj = json.loads(line)
    feat = obj['category']
    if feat_index.get(f'cat_{feat}') is None:
      feat_index[f'cat_{feat}'] = len(feat_index) 
   
  feat_index[f'imgs_size'] = len(feat_index) 
  feat_index[f'patron'] = len(feat_index) 
  feat_index[f'delta'] = len(feat_index) 
  feat_index[f'target'] = len(feat_index) 

  json.dump(feat_index, fp=open('feat_index.json', 'w'), indent=2)
  Xs, Ys = [], []
  for line in open('./param.jsonp'):
    obj     = json.loads(line)
    imgs_size = math.log(int(obj['imgs_size']))
    patron  = math.log(int(obj['patron'])+1)
    success = obj['success']
    cat     = 'cat_' + obj["category"]
    delta   = math.log(int(obj['delta'])+1)
    target  = math.log(int(obj['target']))
    xs = [0.0]*len(feat_index)

    xs[ feat_index['imgs_size'] ] = imgs_size
    xs[ feat_index['patron'] ] = 0. #patron
    # xs[ feat_index['success'] ] = 
    xs[ feat_index[cat] ]      =  1.0
    xs[ feat_index['delta'] ]  = delta
    xs[ feat_index['target'] ] = target
    
    print(1.0 if success == 'success' else 0.0, xs)
    Ys.append( 1.0 if success == 'success' else 0.0 )
    Xs.append( xs )  

  open('dataset.pkl', 'wb').write( pickle.dumps( [np.array(x) for x in [Ys, Xs]] ) ) 

if '--step3' in sys.argv:
  Ys, Xs = pickle.load( open('dataset.pkl', 'rb') )
  
  split = int( len(Ys) * 0.8 )
  Ys, Xs, Yst, Xst = Ys[:split], Xs[:split], Ys[split:], Xs[split:]

  feat_index = json.load(fp=open('feat_index.json'))
  index_feat = {index:feat for feat, index in feat_index.items() }
  #for y in Ys.tolist():
  #  print(y)

  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score

  patterns = []
  for pat1 in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    model = LogisticRegression(max_iter=1000, penalty='l2', solver=pat1)
    model.fit(Xs, Ys)
    Ysp = model.predict(Xst)
    acc = accuracy_score(Ysp, Yst)
    outname = f'{acc:0.09f}_{pat1}_model.pkl'
    pickle.dump(model, open(f'models/{outname}', 'wb'))
    print(acc, pat1)
    coef = model.coef_
    coef = coef.tolist().pop()
    for index, co in enumerate(coef):
      print(co, index_feat[index])

if '--step4' in sys.argv:
  feat_index = json.load(fp=open('feat_index.json'))
  index_feat = {index:feat for feat, index in feat_index.items() }

  model = pickle.load(open(sorted(glob.glob('./models/*')).pop(), 'rb'))
  coef = model.coef_
  coef = coef.tolist().pop()
  for index, co in enumerate(coef):
    print(co, index_feat[index])
