import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100] #indices dos exemplos que serao retirados do conjunto de treinamento

#dados para treinamento
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#dados para teste
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target) #treinando o classificador com os dados de treinamento

print test_target      #os labels esperados(tipos de flores)
print clf.predict(test_data)  #passando novo exemplo para que o classificador indique o tipo

print test_data[2], test_target[2]  #mudar indices para fazer os testes

print iris.feature_names, iris.target_names

