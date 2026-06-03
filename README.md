# 🍷 Inteligência Artificial para Classificação de Qualidade de Vinhos

**LINK DO APLICATIVO EM PRODUÇÃO:**
👉 [Acesse a Aplicação Streamlit](https://projeto-ia-classificacao-vinhos-hwrccxatkazvscq8sy54ga.streamlit.app/)

---

## 📋 Índice

1. [Nome do Projeto](#nome-do-projeto)
2. [Integrantes e RAs](#integrantes-e-ras)
3. [Descrição do Problema](#descrição-do-problema)
4. [Objetivo do Projeto](#objetivo-do-projeto)
5. [Dataset Utilizado](#dataset-utilizado)
6. [Tipo de Problema de Machine Learning](#tipo-de-problema-de-machine-learning)
7. [Metodologia](#metodologia)
8. [Modelos Treinados](#modelos-treinados)
9. [Modelo Final Escolhido](#modelo-final-escolhido)
10. [Métricas de Avaliação](#métricas-de-avaliação)
11. [Principais Resultados](#principais-resultados)
12. [Estrutura dos Arquivos](#estrutura-dos-arquivos)
13. [Tecnologias Utilizadas](#tecnologias-utilizadas)
14. [Instruções para Executar o Notebook](#instruções-para-executar-o-notebook)
15. [Instruções para Executar o App Streamlit](#instruções-para-executar-o-app-streamlit)
16. [Limitações](#limitações)
17. [Conclusão](#conclusão)

---

## Nome do Projeto

**🍷 Inteligência Artificial para Classificação de Qualidade de Vinhos**

Um sistema de Machine Learning para classificação automática da qualidade de vinhos tintos baseado em suas características físico-químicas.

---

## Integrantes e RAs

| Integrante | RA |
|-----------|-----|
| Maria Cláudia Geloneze Cangussú | 208730-6 |
| Mariana da Silva | 203757-5 |
| Murilo Tozoni | 203885-0 |

---

## Descrição do Problema

A indústria de vinhos enfrenta o desafio de avaliar a qualidade dos vinhos de forma objetiva e automatizada. Atualmente, a classificação de vinhos depende frequentemente de degustadores humanos.

Este projeto propõe uma solução automática utilizando Machine Learning para classificar vinhos tintos em duas categorias:
- **Vinho Excelente**: Vinhos de alta qualidade (nota ≥ 7)
- **Vinho Comum**: Vinhos de qualidade comum (nota < 7)

A classificação é realizada baseando-se exclusivamente em atributos físico-químicos mensuráveis, oferecendo uma abordagem objetiva e escalável.

---

## Objetivo do Projeto

O objetivo principal é desenvolver um modelo de Machine Learning capaz de:

1. **Classificar vinhos tintos** em categorias de qualidade (Excelente ou Comum) com base em suas características físico-químicas
2. **Comparar diferentes algoritmos** de classificação (Regressão Logística, KNN, Random Forest) para identificar o melhor desempenho
3. **Implementar boas práticas** de Machine Learning, incluindo tratamento de data leakage, balanceamento de classes e validação cruzada
4. **Disponibilizar uma solução interativa** através de uma aplicação web em Streamlit para predições em tempo real

---

## Dataset Utilizado

**Nome do Dataset:** Wine Quality (Qualidade de Vinho)

**Fonte:** [Kaggle Vinho Tinto de Qualidade](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download)

**Quantidade de Registros:** 1.599 amostras de vinho tinto (após remoção de 240 duplicatas)

**Variáveis Originais:**
- 11 atributos preditores (features)
- 1 variável alvo contínua (quality: 0-10)

**Transformação Realizada:**
A variável alvo `quality` foi transformada em `quality_bin` para classificação binária:
- **Classe 1 (Excelente):** quality ≥ 7
- **Classe 0 (Regular/Ruim):** quality < 7

### Atributos Preditores (Features)

1. **Acidez Fixa** (*Fixed Acidity*) - em g(tartaric acid)/dm³
2. **Acidez Volátil** (*Volatile Acidity*) - em g(acetic acid)/dm³
3. **Ácido Cítrico** (*Citric Acid*) - em g/dm³
4. **Açúcar Residual** (*Residual Sugar*) - em g/dm³
5. **Cloretos** (*Chlorides*) - em g(sodium chloride)/dm³
6. **Dióxido de Enxofre Livre** (*Free Sulfur Dioxide*) - em mg/dm³
7. **Dióxido de Enxofre Total** (*Total Sulfur Dioxide*) - em mg/dm³
8. **Densidade** (*Density*) - em g/cm³
9. **pH** - escala de 0 a 14
10. **Sulfatos** (*Sulphates*) - em g(potassium sulphate)/dm³
11. **Teor Alcoólico** (*Alcohol*) - em % vol.

---

## Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada**

Este é um problema de aprendizado supervisionado (dados rotulados) onde o objetivo é prever uma variável categórica com duas classes:
- Classe positiva (1): Vinho Excelente
- Classe negativa (0): Vinho Comum

---

## Metodologia

### 1. Exploração e Análise de Dados (EDA)

- Carregamento do dataset
- Análise de estatísticas descritivas
- Visualização de distribuições
- Identificação de correlações entre features
- Detecção e remoção de duplicatas (240 registros duplicados removidos)

### 2. Pré-processamento de Dados

- **Remoção de Duplicatas:** Foram identificados e removidos 240 registros duplicados
- **Transformação da Variável Alvo:** Conversão de `quality` (contínua) para `quality_bin` (binária)
- **Tratamento de Desbalanceamento:** Análise da distribuição das classes
- **Normalização:** Utilização de `StandardScaler` para padronizar as features

### 3. Tratamento de Data Leakage

- **Implementação de Pipelines (`sklearn.pipeline`):** Garantir que o `StandardScaler` aprenda apenas nos dados de treino
- **Separação Treino/Teste:** Aplicar transformações após a divisão dos dados
- **Resultado:** Eliminação completa de data leakage

### 4. Tratamento de Desbalanceamento de Classes

- **Análise de Distribuição:** Identificação de desbalanceamento (classe minoritária = vinhos excelentes)
- **Parâmetro `class_weight='balanced'`:** Aplicado em Regressão Logística e Random Forest
- **Objetivo:** Melhorar a identificação da classe minoritária

### 5. Divisão dos Dados

- **Proporção:** 60% treino, 20% validação, 20% teste
- **Estratificação:** Mantida a proporção de classes em ambos os conjuntos
- **Random State:** Definido para reprodutibilidade (42)

### 6. Treinamento dos Modelos

Três algoritmos foram treinados e avaliados:
1. Regressão Logística
2. K-Nearest Neighbors (KNN)
3. Random Forest

### 7. Validação Cruzada Estratificada

- **Técnica:** `StratifiedKFold` com 5 dobras
- **Objetivo:** Garantir validação robusta e reduzir overfitting
- **Resultado:** Avaliação mais confiável do desempenho dos modelos

### 8. Avaliação e Comparação de Modelos

- Cálculo de múltiplas métricas
- Visualização de Matrizes de Confusão
- Plotagem de Curvas ROC
- Comparação visual entre modelos

### 9. Seleção e Serialização do Modelo Final

- **Modelo Escolhido:** Regressão Logística
- **Formato:** Pickle (`.pkl`) usando Joblib
- **Resultado:** Arquivo `modelo_classificacao_vinhos.pkl`

---

## Modelos Treinados

### 1. Regressão Logística

**Descrição:** Modelo linear probabilístico para classificação binária

**Hiperparâmetros:**
- `class_weight='balanced'` - Para lidar com desbalanceamento
- `random_state=42` - Para reprodutibilidade
- Solver padrão: lbfgs

**Vantagens:**
- Computacionalmente eficiente
- Interpretável
- Bom desempenho em problemas lineares
- Fornece probabilidades das predições

### 2. K-Nearest Neighbors (KNN)

**Descrição:** Modelo baseado em instâncias que classifica baseado em vizinhos próximos

**Hiperparâmetros:**
- `n_neighbors=5` - Número de vizinhos a considerar
- `metric='euclidean'` - Métrica de distância

**Vantagens:**
- Simples de implementar
- Não paramétrico
- Pode capturar padrões não-lineares

### 3. Random Forest

**Descrição:** Ensemble de árvores de decisão

**Hiperparâmetros:**
- `n_estimators=100` - Número de árvores
- `class_weight='balanced'` - Para lidar com desbalanceamento
- `random_state=42` - Para reprodutibilidade

**Vantagens:**
- Robusto a outliers
- Fornece importância das features
- Reduz overfitting em relação a árvores individuais

---

## Modelo Final Escolhido

### **Regressão Logística** ✅

Após análise comparativa dos três modelos treinados, a **Regressão Logística** foi selecionada como modelo final.

**Justificativa da Escolha:**

1. **Melhor Equilíbrio de Métricas:**
   - Precisão consistente
   - Recall adequado para a classe positiva
   - F1-Score superior aos demais modelos

2. **Estabilidade:**
   - Desempenho consistente em validação cruzada
   - Menor variância entre folds

3. **Capacidade de Identificação:**
   - Melhor identificação de vinhos excelentes (classe minoritária)
   - Menor taxa de falsos negativos

4. **Interpretabilidade:**
   - Coeficientes interpretáveis
   - Facilita compreensão das relações entre features e predições

5. **Eficiência:**
   - Tempo de treinamento rápido
   - Tempo de predição praticamente instantâneo

**Pipeline Implementado:**

```
StandardScaler (normalização) → Regressão Logística (classificação)
```

---

## Métricas de Avaliação

Os modelos foram avaliados utilizando as seguintes métricas:

### 1. **Acurácia (Accuracy)**
- Proporção de predições corretas
- Fórmula: (TP + TN) / (TP + TN + FP + FN)
- **Interpretação:** Percentual geral de acertos

### 2. **Precisão (Precision)**
- Proporção de predições positivas corretas
- Fórmula: TP / (TP + FP)
- **Interpretação:** Quando o modelo prediz "Excelente", com que frequência está correto?

### 3. **Recall (Sensibilidade)**
- Proporção de casos positivos identificados corretamente
- Fórmula: TP / (TP + FN)
- **Interpretação:** Qual proporção de vinhos excelentes o modelo consegue identificar?

### 4. **F1-Score**
- Média harmônica entre Precisão e Recall
- Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
- **Interpretação:** Métrica balanceada entre precisão e recall

### 5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
- Área sob a curva ROC
- Faixa: 0 a 1 (1 é perfeito)
- **Interpretação:** Probabilidade de o modelo classificar corretamente um par aleatório de exemplos positivos e negativos

### 6. **Matriz de Confusão**
- Tabela de verdadeiros positivos (TP), verdadeiros negativos (TN), falsos positivos (FP) e falsos negativos (FN)
- **Interpretação:** Visão detalhada dos tipos de erros cometidos

---

## Principais Resultados

### Desempenho dos Modelos (Validação Cruzada - 5 Folds)

| Métrica | Regressão Logística | KNN | Random Forest |
|---------|-------------------|-----|---------------|
| Acurácia | 0.768382 | 0.860294 | 0.875000 |
| Precisão | 0.664042 | 0.689393 | 0.751708 |
| Recall | 0.809026 | 0.634503 | 0.597470 |
| F1-Score | 0.676429 | 0.654129 | 0.662587 |
| ROC-AUC | 0.883496 | 0.784761 | 0.871593 |

### Impacto da Remoção de Duplicatas

Antes da remoção de 240 registros duplicados, as métricas eram ligeiramente superiores. Porém, após a remoção, os resultados tornaram-se:

- **Mais realistas e confiáveis**
- **Menos enviesados** por padrões repetidos
- **Representativos** da população real de vinhos

### Matriz de Confusão - Regressão Logística (Teste)

```
                    Predito
                Excelente  Regular
Real  Excelente    58        22      (80 total)
      Regular      14       234      (248 total)
```

**Análise:**
- Verdadeiros Positivos (TP): 58
- Verdadeiros Negativos (TN): 234
- Falsos Positivos (FP): 14
- Falsos Negativos (FN): 22

### Interpretação dos Resultados

1. **Acurácia de 76.84%:** O modelo acerta em ~77% dos casos na validação cruzada
2. **Precisão de 66.40%:** Quando prediz "Excelente", está correto em ~66% das vezes
3. **Recall de 80.90%:** Identifica ~81% dos vinhos excelentes
4. **ROC-AUC de 0.8835:** Excelente capacidade discriminativa (88.35%)

---

## Estrutura dos Arquivos

```
projeto-ia-classificacao-vinhos/
│
├── 📄 README.md                                    
│   └── Documentação completa do projeto
│
├── 📄 requirements.txt                             
│   └── Dependências Python necessárias
│
├── 🐍 app.py                                       
│   └── Aplicação interativa Streamlit
│
├── 📁 model/
│   ├── 💾 modelo_final.pkl
│   │   └── Modelo final salvo (Random Forest)
│   └── 💾 modelo_final.joblib
│       └── Alternativa ao .pkl, caso o grupo use .joblib
│
├── 📁 notebooks/
│   └── 📓 notebook_atualizado.ipynb
│       └── Notebook revisado da P1
│
├── 📁 reports/
│   └── 📄 relatorio_atualizado.pdf
│       └── Relatório final atualizado
│
├── 📁 data/
│   └── 📊 dataset.csv
│       └── Dataset utilizado, se puder ser versionado
│
└── 📁 .devcontainer/
    └── 📄 devcontainer.json
        └── Configuração para Dev Containers (VS Code)
```

### Descrição Detalhada dos Arquivos

| Arquivo/Diretório | Descrição | Tamanho |
|------------------|-----------|---------|
| `README.md` | Documentação completa do projeto com todos os tópicos | - |
| `requirements.txt` | Lista de dependências Python para instalar com pip | ~1 KB |
| `app.py` | Aplicação Streamlit com interface interativa para predições | ~3.5 KB |
| **`model/`** | **Diretório contendo modelos treinados** | - |
| `model/modelo_final.pkl` | Modelo Random Forest treinado e serializado com Pickle | ~2 KB |
| `model/modelo_final.joblib` | Alternativa ao .pkl, caso o grupo use .joblib | ~2 KB |
| **`notebooks/`** | **Diretório contendo notebooks Jupyter** | - |
| `notebooks/notebook_atualizado.ipynb` | Notebook revisado com análise completa e atualizado | ~464 KB |
| **`reports/`** | **Diretório contendo relatórios do projeto** | - |
| `reports/relatorio_atualizado.pdf` | Relatório final atualizado com resultados e análises | ~500 KB |
| **`data/`** | **Diretório contendo dados do projeto** | - |
| `data/dataset.csv` | Dataset Wine Quality em formato CSV (opcional - para versionamento) | ~100 KB |
| `.devcontainer/devcontainer.json` | Configuração para ambiente de desenvolvimento containerizado | ~1 KB |

---

## Tecnologias Utilizadas

### Linguagem de Programação
- **Python 3.11+** - Linguagem principal do projeto

### Manipulação e Análise de Dados
- **Pandas** - Manipulação de dataframes
- **NumPy** - Computação numérica e operações em arrays

### Visualização de Dados
- **Matplotlib** - Gráficos estáticos e customizáveis
- **Seaborn** - Visualizações estatísticas aprimoradas

### Machine Learning
- **Scikit-Learn (1.6.1)** - Algoritmos de ML, pipelines e métricas
  - `LogisticRegression` - Regressão Logística
  - `KNeighborsClassifier` - K-Nearest Neighbors
  - `RandomForestClassifier` - Random Forest
  - `Pipeline` - Arquitetura de pipelines
  - `StandardScaler` - Normalização de dados
  - `StratifiedKFold` - Validação cruzada estratificada
  - `metrics` - Avaliação de modelos

### Salvamento de Modelos
- **Joblib** - Serialização e desserialização de modelos
- **Pickle** - Formato alternativo de serialização

### Desenvolvimento Web
- **Streamlit** - Framework para criação de aplicações web interativas

### Ambiente de Desenvolvimento
- **Jupyter Notebook** - Notebooks interativos para análise e documentação
- **VS Code Dev Containers** - Ambiente de desenvolvimento containerizado (opcional)
- **Docker** - Containerização (utilizado em Dev Containers)

---

## Instruções para Executar o Notebook

### Pré-requisitos

Antes de começar, certifique-se de ter instalado:
- Python 3.11 ou superior
- Git
- pip (gerenciador de pacotes Python)

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/mariana-collab/projeto-ia-classificacao-vinhos.git
cd projeto-ia-classificacao-vinhos
```

### Passo 2: Criar um Ambiente Virtual (Recomendado)

```bash
python -m venv venv
```

### Passo 3: Ativar o Ambiente Virtual

**No Windows:**
```bash
venv\Scripts\activate
```

**No macOS/Linux:**
```bash
source venv/bin/activate
```

### Passo 4: Instalar as Dependências

```bash
pip install -r requirements.txt
```

### Passo 5: Instalar Jupyter Notebook

Se ainda não estiver instalado:
```bash
pip install jupyter notebook
```

### Passo 6: Iniciar o Jupyter Notebook

```bash
jupyter notebook
```

O navegador abrirá automaticamente em `http://localhost:8888`

### Passo 7: Abrir o Notebook

1. Procure pela pasta `notebooks/`
2. Clique no arquivo `notebook_atualizado.ipynb` para abrir

### Passo 8: Executar as Células

Para executar uma célula individual:
- Clique nela
- Pressione `Shift + Enter`

Para executar todas as células:
- Vá em **Cell > Run All** no menu superior
- Ou use o atalho `Ctrl + Shift + Enter`

### Estrutura do Notebook

O notebook está organizado nas seguintes seções:

1. **Importações** - Bibliotecas necessárias
2. **Carregamento do Dataset** - Leitura do Wine Quality dataset
3. **Análise Exploratória de Dados (EDA)**
   - Estatísticas descritivas
   - Visualização de distribuições
   - Análise de correlações
4. **Pré-processamento**
   - Verificação de valores faltantes
   - Remoção de duplicatas
   - Transformação de variável alvo
5. **Divisão Treino/Teste** - Separação com estratificação
6. **Treinamento dos Modelos**
   - Regressão Logística
   - K-Nearest Neighbors
   - Random Forest
7. **Avaliação de Modelos**
   - Métricas (Acurácia, Precisão, Recall, F1-Score)
   - Matrizes de Confusão
   - Curvas ROC
8. **Comparação de Modelos** - Análise visual comparativa
9. **Seleção do Modelo Final** - Justificativa da escolha
10. **Serialização do Modelo** - Salvamento em arquivo `.pkl`

### Notas Importantes

- O notebook pode levar alguns minutos para executar completamente (depende do computador)
- Não é necessário modificar nada no notebook para executá-lo
- Os gráficos e resultados aparecerão dentro do notebook
- O modelo treinado será salvo automaticamente na pasta `model/`

---

## Instruções para Executar o App Streamlit

### Pré-requisitos

- Python 3.11 ou superior
- Dependências instaladas (ver seção anterior)
- Arquivo `modelo_final.pkl` ou `modelo_final.joblib` deve estar na pasta `model/`
- O arquivo `app.py` deve estar no diretório raiz do projeto

### Passo 1: Clonar o Repositório (se ainda não fez)

```bash
git clone https://github.com/mariana-collab/projeto-ia-classificacao-vinhos.git
cd projeto-ia-classificacao-vinhos
```

### Passo 2: Criar e Ativar o Ambiente Virtual

**Criar:**
```bash
python -m venv venv
```

**Ativar no Windows:**
```bash
venv\Scripts\activate
```

**Ativar no macOS/Linux:**
```bash
source venv/bin/activate
```

### Passo 3: Instalar as Dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Executar a Aplicação

```bash
streamlit run app.py
```

### Passo 5: Acessar a Aplicação

A aplicação abrirá automaticamente em: `http://localhost:8501`

Se não abrir automaticamente, acesse manualmente a URL acima no navegador.

### Como Usar a Aplicação

1. **Interface de Entrada:** A aplicação mostra sliders para cada um dos 11 atributos químicos do vinho:
   - Acidez Fixa (*Fixed Acidity*)
   - Acidez Volátil (*Volatile Acidity*)
   - Ácido Cítrico (*Citric Acid*)
   - Açúcar Residual (*Residual Sugar*)
   - Cloretos (*Chlorides*)
   - Dióxido de Enxofre Livre (*Free Sulfur Dioxide*)
   - Dióxido de Enxofre Total (*Total Sulfur Dioxide*)
   - Densidade (*Density*)
   - pH
   - Sulfatos (*Sulphates*)
   - Teor Alcoólico (*Alcohol*)

2. **Ajustar Valores:** Use os sliders para definir os valores dos atributos

3. **Fazer Predição:** Clique no botão "Fazer Predição" (ou similar)

4. **Ver Resultado:** O modelo retornará:
   - Classificação: **"Vinho Excelente"** ou **"Vinho Regular"**
   - Confiança da predição (probabilidade)

### Parâmetros e Faixas de Valores

| Atributo | Unidade | Mín | Máx | Típico |
|----------|---------|-----|-----|--------|
| Acidez Fixa | g/dm³ | 4.6 | 15.9 | 8.3 |
| Acidez Volátil | g/dm³ | 0.12 | 1.58 | 0.5 |
| Ácido Cítrico | g/dm³ | 0.0 | 1.0 | 0.3 |
| Açúcar Residual | g/dm³ | 0.9 | 15.5 | 2.5 |
| Cloretos | g/dm³ | 0.012 | 0.611 | 0.09 |
| SO₂ Livre | mg/dm³ | 1 | 72 | 15 |
| SO₂ Total | mg/dm³ | 6 | 289 | 46 |
| Densidade | g/cm³ | 0.9901 | 1.0037 | 0.9956 |
| pH | - | 2.74 | 4.01 | 3.3 |
| Sulfatos | g/dm³ | 0.33 | 2.0 | 0.6 |
| Teor Alcoólico | % vol | 8.4 | 15.0 | 10.4 |

### Parar a Aplicação

Para interromper o servidor Streamlit, pressione `Ctrl + C` no terminal onde o comando foi executado.

### Troubleshooting

**Problema:** A aplicação não abre automaticamente
- **Solução:** Acesse manualmente `http://localhost:8501` no navegador

**Problema:** Erro "Arquivo modelo não encontrado"
- **Solução:** Certifique-se de que `modelo_final.pkl` ou `modelo_final.joblib` está na pasta `model/`

**Problema:** Erro de dependências
- **Solução:** Execute `pip install -r requirements.txt` novamente

**Problema:** Porta 8501 já está em uso
- **Solução:** Execute com porta diferente: `streamlit run app.py --server.port 8502`

---

## 🔗 Link do App Publicado

O aplicativo está disponível em produção através do Streamlit Cloud:

**👉 [Projeto IA - Classificação de Vinhos](https://projeto-ia-classificacao-vinhos-hwrccxatkazvscq8sy54ga.streamlit.app/)**

A aplicação web permite fazer predições de qualidade de vinhos de forma interativa e amigável, sem necessidade de instalar nada localmente.

---

## Limitações

Apesar dos bons resultados obtidos, o projeto apresenta algumas limitações importantes:

### 1. **Dataset Relativamente Pequeno**
   - Após remoção de duplicatas: 1.599 amostras
   - Pode limitar a generalização do modelo para outros tipos de vinho

### 2. **Desbalanceamento entre as Classes**
   - Classe 0 (Regular): ~82% das amostras
   - Classe 1 (Excelente): ~18% das amostras
   - Pode prejudicar a identificação de vinhos excelentes

### 3. **Apenas Vinhos Tintos**
   - Dataset contém exclusivamente vinhos tintos
   - Modelo pode não ser adequado para vinhos brancos ou espumantes
   - Características físico-químicas diferem entre tipos de vinho

### 4. **Predições Baseadas Apenas em Atributos Físico-Químicos**
   - Não considera informações sobre o produtor, região, ano de colheita
   - Não leva em conta fatores externos como envelhecimento

### 5. **Possibilidade de Melhoria em Hiperparâmetros**
   - Ajustes mais refinados de hiperparâmetros podem melhorar desempenho
   - Grid Search ou Random Search não foram explorados em profundidade

### 6. **Técnicas de Balanceamento Limitadas**
   - Apenas `class_weight='balanced'` foi utilizado
   - Técnicas como SMOTE ou undersampling não foram testadas

### 7. **Ausência de Feature Engineering Avançado**
   - Não foram criadas novas features através de transformações
   - Seleção de features poderia ser mais rigorosa

### 8. **Correlações Possíveis entre Features**
   - Algumas features apresentam alta correlação
   - Multicolinearidade pode afetar interpretabilidade

---

## Conclusão

### Principais Conquistas

Este projeto demonstrou com sucesso que algoritmos de Machine Learning podem ser aplicados com eficiência na classificação da qualidade de vinhos utilizando características físico-químicas mensuráveis. A validação cruzada estratificada com 5 folds garantiu avaliações robustas e confiáveis dos modelos.

### Evoluções da P1 para P2

O projeto evoluiu significativamente em relação ao primeiro bimestre:

1. **Eliminação de Data Leakage** através de pipelines sklearn
2. **Tratamento adequado de desbalanceamento** com `class_weight='balanced'`
3. **Remoção de dados duplicados** para maior confiabilidade
4. **Validação cruzada estratificada** com 5 folds para robustez
5. **Métricas visuais aprimoradas** (matrizes de confusão, curvas ROC)
6. **Modelo serializado para produção** em formato .pkl
7. **Aplicação web interativa** em Streamlit
8. **Nova organização de diretórios** para melhor estruturação (model/, notebooks/, reports/, data/)
9. **Inclusão do dataset em CSV** para maior transparência e reprodutibilidade
10. **Relatório final em PDF** documentando todos os resultados e análises

### Modelo Escolhido

A **Regressão Logística** foi selecionada como modelo final por apresentar:
- Excelente Recall de 80.90% (identifica a maioria dos vinhos excelentes)
- ROC-AUC de 0.8835 (excelente discriminação)
- Melhor equilíbrio entre sensibilidade e especificidade
- Interpretabilidade clara dos coeficientes
- Eficiência computacional

### Impacto Prático

O desenvolvimento da aplicação em Streamlit transformou o modelo treinado em uma solução:
- **Interativa** - Interface amigável com sliders
- **Acessível** - Disponível online para qualquer pessoa
- **Escalável** - Pode receber múltiplas predições simultaneamente
- **Profissional** - Aplicável em contextos reais de negócio

### Recomendações Futuras

1. **Expandir o dataset** com amostras de vinhos brancos e outros tipos
2. **Implementar técnicas avançadas de balanceamento** (SMOTE, ADASYN)
3. **Realizar Feature Engineering** para criar novas variáveis relevantes
4. **Otimizar hiperparâmetros** com Grid Search ou Bayesian Optimization
5. **Implementar ensemble methods** mais sofisticados
6. **Adicionar autenticação** para a aplicação web
7. **Criar API REST** para integração com outros sistemas
8. **Coletar feedback** dos usuários para melhorias contínuas

---

## 📚 Referências

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Wine Quality Dataset - UCI ML Repository](https://archive.ics.uci.edu/dataset/109/wine+quality)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Joblib Documentation](https://joblib.readthedocs.io/)

---

## 📞 Suporte e Contato

Para dúvidas, sugestões ou problemas relacionados ao projeto, entre em contato com os integrantes do grupo através do repositório GitHub.

**Repositório GitHub:** [mariana-collab/projeto-ia-classificacao-vinhos](https://github.com/mariana-collab/projeto-ia-classificacao-vinhos)

---

**Última atualização:** Junho de 2026

**Status do Projeto:** ✅ Concluído e em Produção
