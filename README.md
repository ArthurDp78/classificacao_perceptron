# Trabalho Prático 03: MLPClassifier

## GCC 128 - Inteligência Artificial

Implementação de um classificador baseado em rede neural utilizando o `MLPClassifier` do Scikit-learn nas bases **Iris** e **Wine**, com análise exploratória, pré-processamento, avaliação por métricas e matrizes de confusão.

> Notebook preparado para execução no Google Colab, Jupyter Notebook ou VS Code.

## Objetivo

O trabalho tem como objetivo aplicar um classificador multiclasses para compreender o funcionamento prático de um perceptron multicamadas e comparar seu desempenho com o KNN desenvolvido no Trabalho Prático 01.

O notebook contempla:

- carregamento das bases por meio do KaggleHub;
- análise da estrutura e da distribuição dos dados;
- identificação da variável alvo e das classes;
- verificação de valores ausentes;
- separação entre treinamento e teste;
- padronização dos atributos sem data leakage;
- treinamento do `MLPClassifier`;
- cálculo de precisão, revocação e acurácia;
- visualização das matrizes de confusão;
- comparação documentada com os resultados do KNN.

## Bases de dados

### Iris Flower Dataset

- Dataset Kaggle: `arshid/iris-flower-dataset`
- Arquivo encontrado: `IRIS.csv`
- Dimensões: **150 linhas e 5 colunas**
- Variável alvo: `species`
- Classes: `Iris-setosa`, `Iris-versicolor` e `Iris-virginica`
- Distribuição: 50 amostras por classe
- Valores ausentes: nenhum

### Wine Dataset

- Dataset Kaggle: `tawfikelmetwally/wine-dataset`
- Arquivo encontrado: `Wine dataset.csv`
- Dimensões: **178 linhas e 14 colunas**
- Variável alvo: `class`
- Classes: `1`, `2` e `3`
- Distribuição: 59, 71 e 48 amostras, respectivamente
- Valores ausentes: nenhum

> O notebook identifica o arquivo CSV dentro do diretório retornado pelo KaggleHub e adapta a coluna alvo aos nomes presentes no dataset baixado.

## Metodologia

### Pré-processamento

1. A coluna alvo é separada dos atributos de entrada.
2. Os rótulos das classes são convertidos para valores numéricos com `LabelEncoder`.
3. Os dados são divididos em treinamento e teste com:
   - `test_size=0.20`;
   - `random_state=42`;
   - divisão estratificada por classe.
4. Os atributos são padronizados com `StandardScaler`.

O scaler está dentro de um `Pipeline`, portanto é ajustado somente com os dados de treinamento. Essa organização evita que informações do conjunto de teste influenciem o treinamento.

### Configuração do MLPClassifier

```python
MLPClassifier(
	hidden_layer_sizes=(50, 25),
	activation="relu",
	solver="adam",
	max_iter=2000,
	early_stopping=True,
	validation_fraction=0.15,
	n_iter_no_change=30,
	random_state=42,
)
```

- `(50, 25)`: duas camadas ocultas com capacidade suficiente para as bases utilizadas;
- `relu`: função de ativação adequada para aprender relações não lineares;
- `adam`: otimizador eficiente para treinamento de redes neurais;
- `early_stopping`: interrompe o treinamento quando a validação deixa de melhorar;
- `random_state=42`: favorece a reprodutibilidade dos resultados.

## Resultados do MLPClassifier

As métricas abaixo foram obtidas após a execução do notebook, usando a média macro para precisão e revocação.

| Base | Precisão | Revocação | Acurácia |
|---|---:|---:|---:|
| Iris | 0.8727 | 0.8667 | 0.8667 |
| Wine | 1.0000 | 1.0000 | 1.0000 |

Na Iris, a maior dificuldade ocorreu na distinção entre `Iris-versicolor` e `Iris-virginica`. A classe `Iris-setosa` apresentou desempenho mais consistente. Na Wine, não foram observados erros no conjunto de teste desta execução.

## Comparação com o KNN

Os resultados fornecidos do Trabalho Prático 01 para a base Iris foram:

| `k` | Acurácia | Precisão | Revocação |
|---:|---:|---:|---:|
| 1 | 0.9333 | 0.9444 | 0.9408 |
| 3 | 0.9778 | 0.9833 | 0.9792 |
| 5 | 0.9778 | 0.9833 | 0.9792 |
| 7 | 0.9778 | 0.9833 | 0.9792 |

Para `k=3`, a comparação registrada no notebook é:

| Modelo | Precisão | Revocação | Acurácia |
|---|---:|---:|---:|
| MLPClassifier | 0.8727 | 0.8667 | 0.8667 |
| KNN do TP01 | 0.9833 | 0.9792 | 0.9778 |

Com base nos resultados informados, o KNN apresentou métricas superiores ao MLP na Iris. Entretanto, a comparação definitiva deve confirmar se a divisão dos dados, o pré-processamento e os demais parâmetros foram iguais nos dois trabalhos.

Também foi registrada a comparação entre as implementações hardcore e Scikit-learn do KNN:

- os resultados foram idênticos para `k=1`, `k=3`, `k=5` e `k=7`;
- em `k=3`, o tempo de `fit + predict` foi de **19.05 ms** na implementação hardcore;
- em `k=3`, o tempo foi de **2.22 ms** no Scikit-learn.

O resultado do KNN para a base Wine ainda não foi fornecido. Por isso, o notebook não declara um vencedor para essa base nem inventa métricas ausentes.

## Como executar

### Google Colab

1. Abra o arquivo [TP03_MLPClassifier_Iris_Wine.ipynb](TP03_MLPClassifier_Iris_Wine.ipynb) no Google Colab.
2. Execute as células em ordem, do início ao fim.
3. Autorize o acesso à internet quando necessário para o download via KaggleHub.
4. Confira as tabelas, os relatórios e as matrizes de confusão geradas.

### Jupyter ou VS Code

Instale as dependências:

```bash
pip install kagglehub matplotlib pandas seaborn scikit-learn jinja2
```

Depois, abra o notebook e execute as células sequencialmente.

## Organização do notebook

| Seção | Conteúdo |
|---:|---|
| 1 | Importação das bibliotecas |
| 2 | Funções auxiliares |
| 3-7 | Carregamento, análise, treinamento e avaliação da Iris |
| 8-12 | Carregamento, análise, treinamento e avaliação da Wine |
| 13 | Resumo das métricas do MLP |
| 14 | Comparação entre MLPClassifier e KNN |
| 15 | Texto-base para o relatório |
| Final | Roteiro do vídeo e checklist de entrega |

## Arquivo de entrega

O notebook deve ser acompanhado dos demais itens solicitados na atividade:

- código em `.ipynb` ou `.py`;
- relatório de até uma página em PDF;
- vídeo de apresentação;
- link do vídeo inserido no relatório ou no slide final.

## Referência principal

- [Notebook do trabalho](TP03_MLPClassifier_Iris_Wine.ipynb)
