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
- Botão **Home** corrigido (restaura layout + escalas corretamente)
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

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/paulotaraujo/comtrade-viewer.git
cd comtrade-viewer

