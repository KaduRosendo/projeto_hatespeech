# 🧠 Projeto Semestral de Inteligência Artificial  
Detecção automática de discurso ofensivo em textos em língua portuguesa utilizando técnicas de *Machine Learning*.

---

## 👥 Integrantes do Grupo
- **Carlos Eduardo Rosendo Basseto** – RA: 10409941
- **Matheus Santiago de Brito** - RA: 10408953

---

# 📘 1. Descrição do Projeto

Este projeto tem como objetivo **classificar automaticamente textos como ofensivos ou não-ofensivos**, utilizando um modelo de IA treinado sobre o dataset **OLID-BR**, uma base brasileira de comentários rotulados.

Foram testados três modelos:
- **KNN**
- **Árvore de Decisão**
- **MLPClassifier** (melhor desempenho)

O modelo final (MLP + TF-IDF) foi integrado a uma aplicação **Streamlit** que permite ao usuário digitar qualquer texto e obter uma **probabilidade de ser ofensivo** junto com a classificação.

---

# 📓 2. Código completo (Notebook)

👉 **Acesse o notebook aqui:**  
[colab](https://github.com/KaduRosendo/projeto_hatespeech.git)

---

# 🌐 3. Aplicação Streamlit

👉 **Acesse o Streamlit aqui:**
[link](http://192.168.15.63:8501/)  

---

# ▶️ 4. Como executar o projeto

### 1. Clone o repositório

```

git clone https://github.com/KaduRosendo/projeto_hatespeech.git

cd projeto_hatespeech

```

### 2. Instale as dependências

```

pip install -r requirements.txt

```

### 3. Instale as dependências

```

pip install -r requirements.txt

```

### 4. Execute a aplicação Streamlit

```

streamlit run app_streamlit.py

```

---

# 📊 6. Dataset Utilizado

O projeto utiliza o OLID-BR (Offensive Language Identification Dataset - Brasil):

Comentários rotulados como:

OFF → ofensivo

NOT → não ofensivo

Dataset originalmente desbalanceado → foi aplicado balanceamento (undersampling)

Fonte oficial:
https://huggingface.co/datasets/dougtrajano/olid-br**

# 🎥 9. Vídeo de Apresentação

[Vídeo]()
