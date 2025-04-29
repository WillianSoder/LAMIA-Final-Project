# Loan AI V2

#### O objetivo deste projeto é desenvolver um modelo de machine learning capaz de determinar a elegibilidade de indivíduos ou organizações para a obtenção de empréstimos em uma instituição de crédito, utilizando dados como renda, grau de escolaridade, pontuação de crédito, entre outros.
#### O modelo escolhido foi o Sequencial, e para o treinamento, utilizou-se um dataset do Kaggle. Após o treinamento, a rede neural foi aplicada para classificar novos dados fornecidos pelos usuários.

# Informações do projeto:

## DataSet:
  Foi utilizado um dataset disponivel no Kaggle que contém 4269 registros de pedidos de emprestimos, o DataSet é dividido em 13 colunas sendo elas:
  - Id do empréstimo; 
  - Número de dependentes;
  - Escolaridade;
  - Autônomo;
  - Renda Anual;
  - Valor do empréstimo;
  - Prazo do empréstimo;
  - Pontuação de crédito;
  - Ativos residenciais;
  - Ativos comerciais; 
  - Ativos de luxo;
  - Ativos bancários;
  - Status do empréstimo.

  O objetivo deste dataset é determinar a elegibilidade de indivíduos ou organizações para a obtenção de empréstimos em uma instituição de crédito, por meio de uma coleção de registros financeiros e informações associadas.


## Pré-processamento dos dados: 
- *LabelEncoder():* foi utilizado no projeto para converter as colunas que possuiam dados categóricos em valores inteiros

- *StandardScaler():* é utilizado para padronizar os dados, ajustando-os para que tenham média zero e desvio padrão igual a um. Isso é feito subtraindo a média e dividindo pelo desvio padrão de cada característica.

## Construção do Modelo: 
```bash
    modelo = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation='relu', input_shape=(X_treino_scaler.shape[1],)),
      tf.keras.layers.Dense(16, activation='relu'),
      tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- *Modelo Sequential:* é uma maneira simples e linear de empilhar camadas em uma rede neural. Como o projeto é de classificação binaria é vantajoso usar o modelo sequential porque facilita a construção de um modelo direto e eficiente, onde se pode adicionar camadas de forma sequencial, incluindo uma camada final com uma função de ativação sigmoide. 

- *Ativação ReLU:* função de ativação utilizada na camada de entrada e intermediarias, que retorna o valor de entrada se for positivo, e zero se for negativo. Ela é amplamente utilizada em redes neurais devido à sua simplicidade computacional e eficiência, ajudando a resolver o problema do gradiente desvanecente, comum em funções como a sigmoide.

- *Ativação Sigmoid:* função de ativação utilizada na camada de saída, ela transforma a entrada em um valor entre 0 e 1, sendo útil para tarefas de classificação binária, pois o resultado pode ser interpretado como uma probabilidade.

## Compilação do Modelo: 
```bash
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
```
- *Otimizador Adam:* é um dos otimizadores mais utilizados, ele ajusta a taxa de aprendizado para cada parâmetro com base nas primeiras e segundas momentas do gradiente (média e variância), o que permite uma adaptação eficiente do passo de atualização ao longo do tempo.

- *binary_crossentropy:* é uma função de perda utilizada em tarefas de classificação binária. Ela mede a diferença entre as previsões feitas pelo modelo e os rótulos reais, avaliando a precisão da classificação. A função calcula a perda comparando as probabilidades previstas (geralmente saídas de uma função sigmoide) com os valores reais (0 ou 1).

- *Métricas:* foi utilizada a acurácia que é uma métrica de avaliação que mede a proporção de previsões corretas de um modelo em relação ao total de previsões realizadas. Ela é calculada dividindo o número de previsões corretas pelo total de previsões e é particularmente útil em tarefas de classificação, pois fornece uma visão geral do desempenho do modelo. 

## Avaliando o Modelo: 

#### Treino:
- Acurácia: 98% 
- Perda: 0.041

#### Teste: 
- Acurácia: 94% 
- Perda: 0.1158

#### Matriz de confusão: 

- *Classe 'Approved':* total de 536, onde destes foram 521 acertos e 15 erros de previsão, menos de 3% de erro

- *Classse 'Rejected':* total de 318,onde deste foram 290 acertos e 28 erros  de previsão, menos de 8% de erro

**OBS: os valores de acurácia, perda e a matriz de confusão podem variar a cada nova execução.**
## Utilização do modelo para usuários:

  Para utilizar o modelo, basta realizar o download da pasta 'MODELO', que contém os arquivos 'LoanAI.ipynb' e 'loan_approval_dataset.csv'. Após isso, abra o arquivo no Jupyter e execute as células na sequência apresentada. Ao chegar na seção 'Perguntas para o usuário', será necessário responder às perguntas necessárias para a classificação dos novos dados.
