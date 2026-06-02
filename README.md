# # 🍷 Inteligência Artificial para Classificação de Qualidade de Vinhos

👉 **LINK DO APLICATIVO EM PRODUÇÃO:** [Cole Aqui o Link do seu Streamlit quando ficar pronto]

---

Este projeto foi desenvolvido como requisito avaliativo para a **Nota P2 da disciplina de Inteligência Artificial**, no 3° Termo do curso de Engenharia de Computação da **Universidade de Marília (UNIMAR)**.

O objetivo principal é aplicar algoritmos de Machine Learning para prever se um vinho tinto é de alta qualidade (Vinho Excelente) ou comum (Vinho Regular), baseando-se estritamente em suas características físico-químicas.

---

## 👥 Integrantes do Grupo
* **Maria Cláudia Geloneze Cangussú** - RA: 208730-6
* **Mariana da Silva** - RA: 203757-5
* **Murilo Tozoni** - RA: 203885-0

---

## 🎯 O Problema de Negócio e o Dataset
Utilizando o dataset público *Wine Quality*, o projeto realiza uma **Classificação Binária** onde a variável alvo original (`quality`) foi transformada em `quality_bin`:
* **Classe 1 (Excelente):** Vinhos com nota original de qualidade $\ge 7$.
* **Classe 0 (Regular/Ruim):** Vinhos com nota original de qualidade $< 7$.

O modelo recebe 11 atributos preditores químicos de entrada para realizar o diagnóstico automático:
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

## 🚀 Evoluções Técnicas Aplicadas da P1 para a P2
Para atender aos feedbacks e critérios rigorosos da banca examinadora, reformulamos completamente a estrutura do código da P1, aplicando as seguintes melhorias na P2:

1. **Eliminação de Data Leakage (Vazamento de Dados):** Implementamos a arquitetura de **Pipelines** (`sklearn.pipeline`), garantindo que o escalonamento dos dados (`StandardScaler`) aprenda os parâmetros estatísticos exclusivamente com os dados de treino e apenas aplique a transformação nos dados de teste.
2. **Tratamento de Dados Desbalanceados:** Corrigimos o problema do F1-Score baixo ativando o rebalanceamento algorítmico através do parâmetro `class_weight='balanced'` nos modelos. Isso forçou os algoritmos a penalizarem os erros cometidos na classe minoritária (Vinhos Excelentes).
3. **Validação Cruzada Estratificada:** Adotamos o `StratifiedKFold` com 5 dobras para garantir que as métricas reportadas em treino fossem estatisticamente robustas, provando que a disparidade acentuada (overfitting) da P1 foi mitigada.
4. **Novas Métricas Visuais:** Adicionamos a plotagem obrigatória de **Matrizes de Confusão** e **Curvas ROC (AUC)** para todos os modelos avaliados (Regressão Logística, KNN e Random Forest).
5. **Salvamento do Modelo:** Serializamos o pipeline do modelo campeão (**Random Forest**) gerando o arquivo binário `modelo_classificacao_vinhos.pkl` para consumo externo.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python 3
* **Manipulação de Dados:** Pandas, NumPy
* **Gráficos e Visualização:** Matplotlib, Seaborn
* **Machine Learning & Validação:** Scikit-Learn (Sklearn)
* **Salvamento do Modelo:** Joblib
* **Interface Web:** Streamlit
