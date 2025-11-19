import gc
import sys
import psutil
from typing import Dict

def pulisci_ram(verbose: bool = True, aggressive: bool = False) -> Dict[str, float]:
    """
    Libera memoria RAM forzando il garbage collector e rimuovendo cache.
    
    Args:
        verbose: Se True, stampa informazioni sulla memoria liberata
        aggressive: Se True, esegue pulizia più aggressiva (più lenta)
        
    Returns:
        Dictionary con statistiche memoria prima e dopo la pulizia
    """
    # Misura memoria iniziale
    process = psutil.Process()
    mem_prima = process.memory_info().rss / 1024 / 1024  # MB
    mem_sistema_prima = psutil.virtual_memory().percent
    
    # Forza garbage collection multiplo
    for _ in range(3 if aggressive else 1):
        gc.collect(generation=0)
        gc.collect(generation=1)
        gc.collect(generation=2)
        gc.collect()  # Full collection
    
    # Libera cache numpy se disponibile
    try:
        import numpy as np
        # Forza numpy a liberare memoria interna
        np.core._get_threaded_functions_module()
    except:
        pass
    
    # Misura memoria dopo
    mem_dopo = process.memory_info().rss / 1024 / 1024  # MB
    mem_sistema_dopo = psutil.virtual_memory().percent
    
    risultati = {
        'processo_prima_mb': mem_prima,
        'processo_dopo_mb': mem_dopo,
        'liberata_mb': mem_prima - mem_dopo,
        'sistema_prima_perc': mem_sistema_prima,
        'sistema_dopo_perc': mem_sistema_dopo
    }
    
    if verbose:
        print(f"🧹 Pulizia RAM completata:")
        print(f"  Processo: {mem_prima:.2f} MB → {mem_dopo:.2f} MB")
        print(f"  Liberati: {risultati['liberata_mb']:.2f} MB")
        print(f"  Sistema: {mem_sistema_prima:.1f}% → {mem_sistema_dopo:.1f}%")
    
    return risultati


def cancella_variabili(escludi: list = None):
    """
    Cancella tutte le variabili globali eccetto quelle specificate.
    
    Args:
        escludi: Lista di nomi variabili da non cancellare
    """
    if escludi is None:
        escludi = []
    
    # Aggiungi moduli essenziali alla lista esclusioni
    escludi.extend(['__builtins__', '__name__', '__doc__', '__package__'])
    
    # Ottieni variabili globali
    for var in list(globals().keys()):
        if var not in escludi and not var.startswith('_'):
            try:
                del globals()[var]
            except:
                pass
    
    gc.collect()
    print(f"✨ Variabili cancellate (escluse: {len(escludi)})")


# Esempio d'uso
if __name__ == "__main__":
    # Crea alcune variabili di esempio
    dati_grandi = [i for i in range(1000000)]
    altra_lista = list(range(500000))
    
    print("Prima della pulizia:")
    pulisci_ram()
    
    # Cancella le variabili
    del dati_grandi
    del altra_lista
    
    print("\nDopo aver cancellato variabili:")
    pulisci_ram()