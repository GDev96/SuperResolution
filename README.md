# Super Resolution per Immagini Astronomiche

Un progetto di intelligenza artificiale per il miglioramento della risoluzione delle immagini astronomiche utilizzando tecniche di deep learning e pipeline di elaborazione avanzate.

## 🔭 Descrizione del Progetto

Questo progetto combina tecniche di **super resolution** con **elaborazione di immagini astronomiche**, creando una pipeline completa per:
- Registrazione e allineamento di immagini HST (Hubble Space Telescope)
- Creazione di mosaici ad alta risoluzione
- Miglioramento della qualità tramite AI
- Interfaccia web user-friendly per l'elaborazione

## 📁 Struttura del Progetto

```
SuperResolution/
├── 📄 README.md                    # Questo file
├── ⚙️ requirements.txt            # Dipendenze principali
├── ⚙️ requirements_ui.txt         # Dipendenze interfaccia utente
├── 🚀 run_interface.py            # Launcher interfaccia web
├── 🎓 StesuraTesi/               # Documentazione tesi
├── 📊 data/                      # Dataset e immagini
│   ├── img_lights_1/             # Immagini originali HST
│   ├── img_plate_2/              # Immagini con WCS risolto
│   ├── img_register_4/           # Immagini registrate/allineate
│   ├── img_preprocessed/         # Mosaici finali
│   ├── dataset_sr_patches/       # Dataset per training SR
│   ├── local_raw/                # Immagini locali grezze
│   └── local_processed/          # Immagini locali elaborate
├── 🧠 models/                    # Modelli AI (vuoto, da popolare)
├── 📝 logs/                      # Log di elaborazione
├── 📈 results/                   # Risultati e visualizzazioni
├── 🔧 scripts/                   # Pipeline di elaborazione
│   ├── analyze_hubble.py         # Analisi immagini Hubble
│   ├── AstroPlateSolver.py       # Risoluzione coordinate (WCS)
│   ├── AstroRegister.py          # Registrazione/allineamento
│   ├── AstroMosaic.py            # Creazione mosaici
│   └── create_sr_dataset.py      # Creazione dataset SR
├── 💻 src/                       # Codice sorgente principale
│   ├── preprocessing/            # Moduli preprocessing
│   ├── ui/                       # Interfaccia utente
│   └── utils/                    # Utilities
└── 🔧 venv/                      # Ambiente virtuale Python
```

## 🚀 Quick Start

### 1. Setup Ambiente

```bash
# Clone del repository
git clone <repository-url>
cd SuperResolution

# Creazione e attivazione ambiente virtuale
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Installazione dipendenze
pip install -r requirements.txt
pip install -r requirements_ui.txt
```

### 2. Avvio Interfaccia Web

```bash
python run_interface.py
```

### 3. Pipeline di Elaborazione Astronomica

```bash
cd scripts

# Passo 1: Risoluzione coordinate WCS
python AstroPlateSolver.py

# Passo 2: Registrazione e allineamento
python AstroRegister.py

# Passo 3: Creazione mosaico finale
python AstroMosaic.py
```

## 🔄 Pipeline di Elaborazione

### 📡 Elaborazione Immagini Astronomiche

#### 1. **AstroPlateSolver.py** - Risoluzione WCS
- **Input**: `img_lights_1/` (immagini HST originali)
- **Output**: `img_plate_2/` (immagini con coordinate risolte)
- **Funzione**: Aggiunge/verifica informazioni coordinate mondiali

#### 2. **AstroRegister.py** - Registrazione
- **Input**: `img_plate_2/` 
- **Output**: `img_register_4/` (immagini allineate)
- **Funzione**: Allinea tutte le immagini su un sistema di riferimento comune

#### 3. **AstroMosaic.py** - Creazione Mosaico
- **Input**: `img_register_4/`
- **Output**: `img_preprocessed/` (mosaico finale)
- **Funzione**: Combina immagini allineate in un unico mosaico

### 🧠 Super Resolution AI

#### 4. **create_sr_dataset.py** - Preparazione Dataset
- Crea patches per training da mosaici ad alta risoluzione
- **Output**: `dataset_sr_patches/`

#### 5. **Training Modelli SR** (da implementare)
- **Architetture**: SRCNN, ESRGAN, Real-ESRGAN
- **Target**: Miglioramento 2x-4x della risoluzione

## 📊 Stato del Progetto

### ✅ Completato

- [x] **Setup ambiente di sviluppo**
  - [x] Struttura progetto
  - [x] Ambiente virtuale e dipendenze
  - [x] Interfaccia web con Gradio

- [x] **Pipeline elaborazione astronomica**
  - [x] Risoluzione WCS con AstroPlateSolver
  - [x] Registrazione immagini con AstroRegister
  - [x] Creazione mosaici con AstroMosaic
  - [x] Analisi immagini Hubble

- [x] **Gestione dati**
  - [x] Struttura directory organizzata
  - [x] Logging completo delle operazioni
  - [x] Fallback intelligenti per path

### 🔄 In Corso

- [ ] **Modelli Super Resolution**
  - [ ] Implementazione architetture SRCNN
  - [ ] Training su dataset astronomico
  - [ ] Validazione e metriche

- [ ] **Ottimizzazioni**
  - [ ] Gestione memoria per immagini grandi
  - [ ] Processing parallelo
  - [ ] Cache intelligente

### 📋 Todo

- [ ] **Modelli Avanzati**
  - [ ] Implementazione ESRGAN
  - [ ] Real-ESRGAN per immagini reali
  - [ ] Transfer learning da modelli pre-addestrati

- [ ] **Interfaccia e Deployment**
  - [ ] Miglioramento UI web
  - [ ] API REST
  - [ ] Docker containerization

## 🎯 Obiettivi

### 🔭 Elaborazione Astronomica
- **Mosaici HST**: Combinazione automatica di survey Hubble
- **Registrazione precisa**: Allineamento sub-pixel di immagini
- **Gestione WCS**: Coordinate mondiali accurate

### 🧠 Super Resolution
- **PSNR Target**: > 30 dB su dataset astronomico
- **SSIM Target**: > 0.85 per qualità visiva
- **Performance**: < 1 secondo per patch 512x512

## 📈 Metriche e KPI

| Componente | Stato | Completamento | Performance |
|------------|-------|---------------|-------------|
| Pipeline Astronomica | ✅ | 90% | Stabile |
| Interfaccia Web | ✅ | 70% | Funzionale |
| Modelli SR | 🔄 | 20% | In sviluppo |
| Dataset Creation | ✅ | 80% | Ottimizzato |

## 🛠️ Tecnologie Utilizzate

### 🔧 Librerie Principali
- **Astropy**: Elaborazione immagini astronomiche e WCS
- **NumPy/SciPy**: Calcoli numerici e processamento array
- **OpenCV**: Processamento immagini
- **PyTorch**: Framework deep learning
- **Gradio**: Interfaccia web interattiva

### 📊 Formati Supportati
- **FITS**: Standard astronomico per immagini scientifiche
- **HST DRZ**: Immagini drizzle Hubble Space Telescope
- **WCS**: World Coordinate System per coordinate celesti

## 🔧 Configurazione Avanzata

### Ottimizzazione Memoria
Per immagini molto grandi, modifica i parametri negli script:

```python
# In AstroRegister.py
MAX_IMAGES = 50          # Riduci per meno memoria
max_size = 8000          # Canvas massimo (pixel)

# In AstroMosaic.py
FEATHER_RADIUS = 100     # Bordi sfumati
SIGMA_CLIP_THRESHOLD = 3.0  # Rimozione outlier
```

### Modalità Debug
Attiva logging dettagliato:

```bash
export PYTHONPATH=./src:$PYTHONPATH
export LOG_LEVEL=DEBUG
python scripts/AstroRegister.py
```

## 📚 Documentazione Aggiuntiva

### 📖 Guide Specifiche
- **StesuraTesi/**: Documentazione accademica completa
- **logs/**: Log dettagliati di ogni operazione
- **README.md** in ogni directory dati per dettagli specifici

### 🔬 Algoritmi Utilizzati
- **Reproject**: Reproiezione accurata con conservazione flusso
- **Sigma Clipping**: Rimozione automatica outlier
- **Edge Feathering**: Bordi sfumati per mosaici seamless
- **WCS Optimization**: Calcolo canvas ottimale automatico

## 🐛 Troubleshooting

### Errori Comuni

#### Memoria Insufficiente
```bash
# Riduci dimensioni canvas
python scripts/AstroRegister.py
# Scegli opzione 1 (Standard) per minor memoria
```

#### Path Non Trovati
```bash
# Verifica struttura directory
ls -la data/
# Assicurati che esistano img_lights_1/ o img_cropped_3/
```

#### Dipendenze Mancanti
```bash
# Reinstalla requirements
pip install --upgrade -r requirements.txt
pip install reproject scipy
```

#### WCS Errors
```bash
# Verifica headers FITS
python -c "from astropy.io import fits; print(fits.getheader('file.fits'))"
```

### 🔍 Logging e Debug
Tutti gli script generano log dettagliati in `logs/` con timestamp:

```bash
# Visualizza log più recente
ls -lt logs/
tail -f logs/registration_*.log
```

### 🆘 Supporto
- **Issues GitHub**: Per bug e feature request
- **Logs Directory**: Per debug dettagliato
- **StesuraTesi/**: Per riferimenti teorici

## 📄 Licenza

Questo progetto è sviluppato per scopi di ricerca accademica.

## 🙏 Riconoscimenti

- **Hubble Space Telescope**: Fonte dei dati astronomici
- **Astropy Project**: Librerie fondamentali per astronomia
- **Community Open Source**: Strumenti e librerie utilizzate

---

## 📅 Cronologia Versioni

- **v1.2** (Novembre 2025): Pipeline completa e documentazione estesa
- **v1.1** (Ottobre 2025): Pipeline astronomica completa
- **v1.0** (Agosto 2025): Setup iniziale e interfaccia base

---

*Ultimo aggiornamento: Novembre 2025*  
*Progetto tesi triennale - Elaborazione immagini astronomiche con AI* per Immagini Astronomiche
Un progetto di intelligenza artificiale per il miglioramento della risoluzione delle immagini astronomiche utilizzando tecniche di deep learning e pipeline di elaborazione avanzate.