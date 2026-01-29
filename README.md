# COMTRADE Viewer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PySide6](https://img.shields.io/badge/Qt-PySide6-green)
![Matplotlib](https://img.shields.io/badge/plot-matplotlib-orange)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey)

Aplicação desktop para **visualização, análise e exportação de arquivos COMTRADE**
(`.cfg + .dat` / `.cfg + .bdat`), desenvolvida em **Python + PySide6 + Matplotlib**.

O **COMTRADE Viewer** permite:
- Visualizar **tensão e corrente** em gráficos sincronizados
- Selecionar canais individualmente
- Aplicar **PRI/SEC**
- Recortar intervalos de tempo
- Exportar subconjuntos de dados em **COMTRADE ASCII, COMTRADE BINARY ou JSON**

---

## ✨ Principais recursos

### 📂 Leitura COMTRADE
- ASCII: `.cfg + .dat`
- BINARY: `.cfg + .bdat`
  - Timestamp **32-bit** ou **64-bit** (detecção automática)
- Parsing completo do `.cfg` (canais, escalas, PRI/SEC, `fs`, `time_mult`)

### 📊 Visualização
- Gráficos separados e sincronizados:
  - **Tensão** (eixo superior)
  - **Corrente** (eixo inferior)
- Cursor vertical sincronizado
- Autoscale inteligente (somente canais visíveis)
- Botão **Home** corrigido (restaura layout e escalas corretamente)
- Modo de foco:
  - Todos
  - Apenas tensão
  - Apenas corrente

### 🎛️ Controles
- Seleção de canais via checkbox
- Marcar / Desmarcar todos
- Janela de tempo (`tmin`, `tmax`)
- Aplicação opcional de **PRI/SEC**

### 📤 Exportação
Exporta **somente os canais selecionados** e **somente o intervalo visível**:
- COMTRADE ASCII (`.cfg + .dat`)
- COMTRADE BINARY (`.cfg + .bdat`)
  - Timestamp 32-bit
  - Timestamp 64-bit
- JSON (tempo + canais escalonados)

Cada exportação cria automaticamente uma **pasta com timestamp**.

---

## 🧩 Requisitos

- Python **3.9 ou superior**
- Dependências:
  - `PySide6`
  - `numpy`
  - `matplotlib`

---

## 📦 Instalação

Existem **duas formas** de utilizar o **COMTRADE Viewer**:

- ▶️ **Executável pronto** (recomendado para usuários finais)
- 🧑‍💻 **A partir do código-fonte** (para desenvolvimento)

---

## ▶️ Opção 1 — Executável (Recomendado)

1. Acesse a aba **Releases** do projeto:  
   https://github.com/paulotaraujo/comtrade-viewer/releases

2. Baixe o arquivo correspondente ao seu sistema operacional, por exemplo: ComtradeViewer-linux-x86_64-v1.0.0.zip

3. Extraia o arquivo `.zip`

4. No Linux, torne o binário executável (se necessário):
```bash
chmod +x ComtradeViewer
./ComtradeViewer
```

## ▶️ Opção 2 — Executar a partir do código-fonte

Clone o repositório do projeto, acesse o diretório e prepare o ambiente de execução. Recomenda-se o uso de um ambiente virtual Python para evitar conflitos de dependências do sistema.

```bash
git clone https://github.com/paulotaraujo/comtrade-viewer.git
cd comtrade-viewer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py

