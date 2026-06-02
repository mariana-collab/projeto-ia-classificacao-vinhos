# 🍷 Inteligência Artificial para Classificação de Qualidade de Vinhos

👉 **LINK DO APLICATIVO EM PRODUÇÃO:**
(https://projeto-ia-classificacao-vinhos-hwrccxatkazvscq8sy54ga.streamlit.app/)

---

## Sobre o Projeto

Este projeto foi desenvolvido como requisito avaliativo para a Nota P2 da disciplina de Algoritmos de Inteligência Artificial (Classificadores), no 3° Termo do curso de Inteligência Artificial da Universidade de Marília (UNIMAR).

O objetivo principal é aplicar algoritmos de Machine Learning para prever se um vinho tinto é de alta qualidade (Vinho Excelente) ou comum (Vinho Regular), baseando-se estritamente em suas características físico-químicas.

---

## Integrantes do Grupo

* Maria Cláudia Geloneze Cangussú - RA: 208730-6
* Mariana da Silva - RA: 203757-5
* Murilo Tozoni - RA: 203885-0

---

## O Problema de Negócio e o Dataset

Utilizando o dataset público *Wine Quality*, o projeto realiza uma Classificação Binária onde a variável alvo original (`quality`) foi transformada em `quality_bin`:

* Classe 1 (Excelente): Vinhos com nota original de qualidade ≥ 7
* Classe 0 (Regular/Ruim): Vinhos com nota original de qualidade < 7

O modelo recebe 11 atributos preditores químicos de entrada para realizar a classificação automática:

1. Acidez Fixa (*fixed acidity*)
2. Acidez Volátil (*volatile acidity*)
3. Ácido Cítrico (*citric acid*)
4. Açúcar Residual (*residual sugar*)
5. Cloretos (*chlorides*)
6. Dióxido de Enxofre Livre (*free sulfur dioxide*)
7. Dióxido de Enxofre Total (*total sulfur dioxide*)
8. Densidade (*density*)
9. pH
10. Sulfatos (*sulphates*)
11. Teor Alcoólico (*alcohol*)

---

## Tipo de Problema de Machine Learning

O projeto aborda um problema de Classificação Binária Supervisionada, onde o objetivo é prever se um vinho é excelente ou regular com base em características físico-químicas.

---

## Evoluções Técnicas Aplicadas da P1 para a P2 - Metodologia 

Este trabalho é a continuação direta do projeto iniciado no primeiro bimestre. Durante a P2, foram aplicadas melhorias técnicas com base nas orientações da professora e nos feedbacks disponibilizados no Moodle.

As principais evoluções implementadas foram:

### Eliminação de Data Leakage (Vazamento de Dados)

Foi implementada a arquitetura de Pipelines (`sklearn.pipeline`) para garantir que o `StandardScaler` aprendesse os parâmetros estatísticos apenas nos dados de treino, aplicando a transformação de forma correta aos dados de teste.

### Tratamento de Dados Desbalanceados

Foi utilizado o parâmetro `class_weight='balanced'` nos modelos de Regressão Logística e Random Forest para melhorar o tratamento da classe minoritária (vinhos excelentes).

### Remoção de Registros Duplicados

Durante a análise exploratória foram identificados 240 registros duplicados no dataset. Essas duplicatas foram removidas utilizando `drop_duplicates()`, reduzindo possíveis vieses e melhorando a confiabilidade das métricas.

### Validação Cruzada Estratificada

Foi implementado o `StratifiedKFold` com 5 dobras para garantir uma validação mais robusta dos modelos e reduzir problemas de overfitting.

### Novas Métricas Visuais

Foram adicionadas:

* Matrizes de Confusão
* Curvas ROC (AUC)
* Comparações de métricas entre modelos

Esses recursos permitiram análises mais completas sobre o desempenho dos classificadores.

### Salvamento do Modelo para Produção

O pipeline do modelo final foi serializado utilizando Joblib, gerando o arquivo:

```bash
modelo_classificacao_vinhos.pkl
```

Isso permitiu a integração do modelo com a aplicação desenvolvida em Streamlit para execução de predições em tempo real.

---

## Modelos Treinados

Os seguintes modelos de Machine Learning foram treinados e avaliados:

* Regressão Logística
* K-Nearest Neighbors (KNN)
* Random Forest

---

## Modelo Final Escolhido

Após os testes comparativos entre Regressão Logística, KNN e Random Forest, o modelo final escolhido foi a Regressão Logística.

Mesmo após a remoção das duplicatas do dataset, a Regressão Logística apresentou o melhor equilíbrio entre:

* Precisão
* Recall
* F1-Score

Além disso, demonstrou maior capacidade de identificar corretamente vinhos excelentes e apresentou melhor estabilidade geral em relação aos demais classificadores.

---

## Métricas de Avaliação

Os modelos foram avaliados utilizando as seguintes métricas:

* Accuracy (Acurácia)
* Precision (Precisão)
* Recall
* F1-Score
* ROC-AUC
* Matriz de Confusão

---

## Principais Resultados

Os modelos apresentaram bons resultados na tarefa de classificação binária dos vinhos.

Após a remoção de 240 registros duplicados, houve uma leve redução em algumas métricas, tornando a avaliação mais realista e reduzindo possíveis vieses causados por padrões repetidos.

A Regressão Logística apresentou o melhor equilíbrio geral entre as métricas de classificação, especialmente para identificação da classe positiva (vinhos excelentes).

---

## Tecnologias e Bibliotecas Utilizadas

* Linguagem: Python 3
* Manipulação de Dados: Pandas, NumPy
* Visualização de Dados: Matplotlib, Seaborn
* Machine Learning: Scikit-Learn
* Salvamento do Modelo: Joblib
* Aplicação Web: Streamlit

---

## Limitações

Apesar dos bons resultados, o projeto ainda apresenta algumas limitações:

* Dataset relativamente pequeno após a remoção das duplicatas
* Desbalanceamento entre as classes
* Predições baseadas apenas em atributos físico-químicos
* Possibilidade de melhoria através de ajuste de hiperparâmetros e técnicas mais avançadas de balanceamento

---

## Conclusão

O projeto demonstrou que algoritmos de Machine Learning podem ser aplicados com eficiência na classificação da qualidade de vinhos utilizando características físico-químicas.

Além da melhoria das métricas e da organização do notebook em relação à P1, a P2 permitiu aplicar técnicas mais robustas de validação, tratamento de inconsistências e interpretação dos resultados, tornando o modelo mais confiável para uso em aplicações reais.

O desenvolvimento da aplicação em Streamlit também possibilitou transformar o modelo treinado em uma solução interativa e acessível para realização de predições em tempo real.
