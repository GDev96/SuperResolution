# Super Resolution

Un progetto di intelligenza artificiale per il miglioramento della risoluzione delle immagini utilizzando tecniche di deep learning.

## 📅 Timeline del Progetto

### Fase 1: Setup e Ricerca (Luglio - Agosto 2025)
- **23 Luglio 2025**: Inizializzazione del progetto
- **24 Luglio - 10 Agosto 2025**: Ricerca su architetture SRCNN, ESRGAN, Real-ESRGAN
- **11-31 Agosto 2025**: Setup ambiente di sviluppo e preparazione dataset

### Fase 2: Implementazione Base (Settembre 2025)
- **1-10 Settembre 2025**: Implementazione modello SRCNN base
- **11-18 Settembre 2025**: Training iniziale e validazione
- **19-30 Settembre 2025**: Ottimizzazione hyperparameters

### Fase 3: Modelli Avanzati (Ottobre 2025)
- **1-10 Ottobre 2025**: Implementazione ESRGAN
- **11-18 Ottobre 2025**: Confronto performance modelli
- **19-25 Ottobre 2025**: Fine-tuning e ottimizzazioni
- **26-31 Ottobre 2025**: Interfaccia utente, testing finale e release

## ✅ Task Completate

- [x] **Inizializzazione repository** - 23 Luglio 2025
  - Creazione struttura base del progetto
  - Setup README iniziale

## 📋 Todo List

### Prerequisiti Mattia
- vedere algoritmi di registrazione astrologici (vedere astroalign)
- capire il significato dei metadati
- procurare le immagini e provare ad aprirle su python

### 🔴 Priorità Alta
- [x] **Setup ambiente di sviluppo**
  - [x] Installazione Python 3.8+
  - [x] Setup virtual environment
  - [x] Installazione dipendenze (PyTorch, OpenCV, PIL)
  
- [ ] **Raccolta e preparazione dataset**
  - [ ] Acquisizione immagini da osservatorio locale
  - [ ] Download immagini dalla galleria Hubble online
  - [ ] Implementazione algoritmi di registrazione immagini (astroalign)
  - [ ] Allineamento e calibrazione immagini astronomiche
  - [ ] Valutazione qualità registrazione e correzione artefatti
  - [ ] Implementazione data loader per immagini astronomiche
  - [ ] Preprocessing pipeline (cropping, normalization, gestione metadati FITS)

### 🟡 Priorità Media
- [ ] **Implementazione modello SRCNN**
  - [ ] Architettura della rete neurale
  - [ ] Funzioni di loss (MSE, PSNR, SSIM)
  - [ ] Training loop
  - [ ] Validation e metriche

- [ ] **Sperimentazione modelli avanzati**
  - [ ] Implementazione ESRGAN
  - [ ] Implementazione Real-ESRGAN
  - [ ] Transfer learning da modelli pre-addestrati

### 🟢 Priorità Bassa
- [ ] **Interfaccia utente**
  - [ ] CLI per processing batch
  - [ ] Web interface con Gradio/Streamlit
  - [ ] API REST per integrazione

- [ ] **Ottimizzazioni e deployment**
  - [ ] Conversione modelli ONNX
  - [ ] Ottimizzazione per inferenza
  - [ ] Docker containerization
  - [ ] CI/CD pipeline

## 📊 Metriche di Progresso

| Fase | Stato | Completamento | Scadenza |
|------|-------|---------------|----------|
| Setup e Ricerca | 🟡 In corso | 10% | 31 Agosto |
| Implementazione Base | ⚪ Non iniziato | 0% | 30 Settembre |
| Modelli Avanzati | ⚪ Non iniziato | 0% | 31 Ottobre |

## 🎯 Obiettivi del Progetto

1. **Obiettivo Primario**: Implementare un sistema di super resolution efficace che migliori la qualità delle immagini di almeno 2x-4x
2. **Obiettivo Secondario**: Confrontare performance di diversi modelli (SRCNN vs ESRGAN)
3. **Obiettivo Terziario**: Creare un'interfaccia user-friendly per l'utilizzo pratico

## 📈 KPI e Metriche di Successo

- **PSNR**: Target > 30 dB su dataset Set5
- **SSIM**: Target > 0.85 su dataset Set14
- **Tempo di inferenza**: < 1 secondo per immagine 512x512
- **Qualità visiva**: Valutazione soggettiva su scala 1-10, target > 7

## 🛠️ Setup Ambiente di Sviluppo

### Creazione e attivazione virtual environment

```bash
# Creazione virtual environment
python -m venv venv

# Attivazione su Windows
venv\Scripts\activate

# Attivazione su Linux/MacOS
source venv/bin/activate

# Disattivazione (quando necessario)
deactivate
```

### Gestione dipendenze

```bash
# Installazione dipendenze da requirements.txt
pip install -r requirements.txt

# Aggiornamento requirements.txt con le dipendenze correnti
pip freeze > requirements.txt

# Installazione nuove dipendenze e aggiornamento requirements
pip install nome_pacchetto
pip freeze > requirements.txt
```

---

*Ultimo aggiornamento: 30 Luglio 2025*

