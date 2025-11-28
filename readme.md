# 🛠️ Configurazione Ambiente di Sviluppo (Windows)

Questa guida spiega come configurare l'ambiente Python, creare il Virtual Environment (venv) e installare le dipendenze necessarie per il progetto di Super Resolution astronomica.

## 1. Prerequisiti

Assicurati di avere installato:

- **Python** (versione 3.10 o superiore)
- **VS Code** (o il tuo editor preferito)
- **Terminale**: PowerShell o Command Prompt

## 2. Creazione del Virtual Environment (VENV)

È fondamentale usare un ambiente virtuale per non "sporcare" l'installazione globale di Python e gestire le versioni delle librerie.

1. Apri il terminale nella cartella del progetto (es. `F:\SuperRevoltGaia\SuperResolution`)

2. Esegui il seguente comando per creare la cartella `.venv`:

```powershell
# Se hai python nel PATH:
python -m venv venv

# OPPURE, se usi il percorso completo:
C:/Users/dell/AppData/Local/Programs/Python/Python313/python.exe -m venv venv
```

Se il comando ha successo, vedrai apparire una cartella chiamata `venv` nella tua directory.

## 3. Attivazione dell'Ambiente

Questo è il passaggio che spesso crea problemi su Windows a causa dei permessi di sicurezza.

1. Prova ad attivare l'ambiente:

```powershell
.\venv\Scripts\Activate.ps1
```

2. **🛑 Se ricevi un errore** che dice "l'esecuzione di script è disabilitata nel sistema":

   Esegui questo comando per abilitare gli script solo per questa sessione (sicuro):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

   Premi `S` (o `Y`) se richiesto per confermare.

3. Riprova ad attivare:

```powershell
.\venv\Scripts\Activate.ps1
```

**✅ Successo**: Dovresti vedere la scritta `(venv)` verde all'inizio della riga di comando.

## 4. Installazione delle Dipendenze

Ora che sei dentro il `(venv)`, installa tutte le librerie necessarie per la pipeline (Step 1-7).

1. Aggiorna prima pip (opzionale ma consigliato):

```powershell
python -m pip install --upgrade pip
```

2. Installa il pacchetto completo:

```powershell
pip install numpy pandas matplotlib astropy scipy scikit-image tqdm reproject astroalign
```

### Lista delle librerie principali:

- **numpy, pandas**: Gestione dati e calcoli
- **astropy**: Gestione file FITS e coordinate WCS
- **matplotlib**: Grafici e visualizzazione immagini
- **scikit-image**: Elaborazione immagini (resize, metriche)
- **reproject**: Per la registrazione (riproiezione) delle immagini
- **astroalign**: Per l'allineamento geometrico basato su stelle
- **tqdm**: Barre di caricamento

## 5. Verifica della Struttura delle Cartelle

Per far funzionare gli script senza errori, assicurati che la struttura delle cartelle sia così:

```
SuperResolution/
├── venv/                   # Creato al passo 2
├── scripts/                # Dove metti i file .py
│   ├── Dataset_step1_...py
│   └── ...
├── data/                   # Cartella Dati
│   ├── logs/               # Creata automaticamente dagli script
│   ├── M1/                 # Esempio Target
│   │   ├── 1_originarie/
│   │   │   ├── img_lights/ # Metti qui i file Hubble
│   │   │   └── local_raw/  # Metti qui i file Osservatorio
│   │   └── ...             # Altre cartelle create dagli script: 2_wcs, 3_registered...
│   └── NGC7635/            # Altri target...
└── ...
```

## 6. Come Eseguire gli Script

Sempre con il `(venv)` attivo, lancia gli script dalla root del progetto:

**Esempio Step 1** (WCS + Registrazione):

```powershell
python "scripts/Dataset_step1_datasetwcs Gaia.py"
```

**Esempio Step 3** (Estrazione Patch):

```powershell
python "scripts/Dataset_step3_extractpatches_Gaia.py"
```

## 🆘 Risoluzione Problemi Comuni

### Errore `ModuleNotFoundError: No module named 'pandas'`

Significa che non hai attivato il venv prima di lanciare lo script. Rifai il punto 3.

### Errore `SyntaxError: invalid non-printable character`

Hai copiato del codice che contiene spazi "invisibili". Cancella la riga indicata dall'errore e riscrivila a mano.

### Errore `ObjectNotFound` su `Activate.ps1`

Non hai creato il venv. Rifai il punto 2.

---

**Nota**: Ricorda sempre di attivare il virtual environment prima di lavorare sul progetto!




Controllare lo standard delle coordinate ra e dec
conversione wcs in ra, dec

Controllare questo:
        # Pixel scale (negativo per RA per convenzione astronomica)
        wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
        
        # Tipo proiezione (TAN = tangente, standard per campi piccoli)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Sistema di riferimento
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.equinox = 2000.0


Controllare header hubble per plate solving (astronet/astropy)