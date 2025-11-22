M33
CON 95% ESCONO 72 COPPIE 
CON 97% ESCONO 46 
CON 99 ESCONO 31



È un'osservazione eccellente e molto comune quando si lavora con dati astronomici provenienti da sorgenti diverse (Hubble vs Terra).

Il fatto che l'accuratezza sia al 100% (o il filtro di copertura sia >99%) significa che geometricamente le immagini sono sovrapposte correttamente. Tuttavia, il contenuto di ciò che vedi differisce per motivi fisici e tecnici.

Ecco le 3 ragioni principali per cui vedi stelle nell'Osservatorio che sembrano "sparire" in Hubble:

1. La differenza nei Filtri (Larghezza di Banda)
Questa è la causa più probabile.

Hubble (HR): Le immagini di Hubble che stai usando sembrano essere "Narrow Band" (es. filtro F656N per l'Idrogeno Alpha). Questi filtri sono estremamente stretti (lasciano passare pochissima luce, spesso solo 1-2 nanometri). Sono progettati per bloccare quasi tutta la luce delle stelle per far risaltare la struttura del gas (la nebulosa).

Osservatorio (LR): Anche se stai usando un filtro H-Alpha da terra, i filtri amatoriali o semi-professionali sono molto più "larghi" (es. 7nm o 12nm). Questo permette a molta più luce stellare "parassita" (continuum) di passare.

Risultato: L'immagine dell'Osservatorio mostrerà molte più stelle di fondo perché il suo filtro non è selettivo quanto quello di Hubble.

2. Rumore scambiato per Stelle
Guardando i pannelli 2 e 6 ("Obs Input") delle tue immagini, si nota una texture molto "granulosa".

In bassa risoluzione (80x80 pixel), il rumore digitale (pixel caldi o rumore di lettura del sensore CCD) può assomigliare molto a delle stelle deboli.

Hubble ha un rapporto segnale/rumore molto più alto e un fondo cielo molto più pulito (essendo nello spazio). Quello che nell'Osservatorio sembra una "stella", in Hubble potrebbe rivelarsi essere solo un picco di rumore di fondo che sparisce quando si guarda l'immagine pulita.

3. Dinamica e Normalizzazione (Il "Contrasto")
Guardando il Pannello 5 (Hubble Target), si vede che il contrasto è ottimizzato per mostrare i filamenti della nebulosa (le parti gialle/verdi).

Le stelle in Hubble potrebbero esserci, ma essere molto deboli rispetto al gas luminoso.

Se la normalizzazione (il modo in cui il computer decide cosa è bianco e cosa è nero) è impostata per far vedere il gas, le stelle deboli potrebbero finire sotto la soglia del nero ed essere invisibili all'occhio, mentre nell'immagine dell'Osservatorio (che ha meno contrasto dinamico) appaiono come macchie grigie.