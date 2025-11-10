#!/usr/bin/env python3
"""
Utility per cambiare l'oggetto target in tutti gli script della pipeline
"""

import os
import glob
import re
from pathlib import Path

def update_target_object(new_target):
    """
    Aggiorna la variabile TARGET_OBJECT in tutti gli script della pipeline
    
    Args:
        new_target (str): Nome del nuovo oggetto target (es. 'M42', 'M33', 'NGC2024')
    """
    script_dir = Path(__file__).parent
    
    # Lista degli script da aggiornare
    scripts_to_update = [
        'AstroPlateSolver.py',
        'AstroRegister.py', 
        'AstroMosaic.py',
        'create_sr_dataset.py',
        'analyze_hubble.py'
    ]
    
    updated_files = []
    
    for script_name in scripts_to_update:
        script_path = script_dir / script_name
        
        if not script_path.exists():
            print(f"⚠️  Script {script_name} non trovato, skip...")
            continue
        
        try:
            # Leggi il file
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern per trovare la linea TARGET_OBJECT
            pattern = r'TARGET_OBJECT\s*=\s*["\'][^"\']*["\']'
            new_line = f'TARGET_OBJECT = "{new_target}"'
            
            # Sostituisci
            if re.search(pattern, content):
                new_content = re.sub(pattern, new_line, content)
                
                # Scrivi il file aggiornato
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                updated_files.append(script_name)
                print(f"✅ Aggiornato {script_name}")
            else:
                print(f"⚠️  Pattern TARGET_OBJECT non trovato in {script_name}")
                
        except Exception as e:
            print(f"❌ Errore aggiornando {script_name}: {e}")
    
    return updated_files

def list_available_objects():
    """Lista gli oggetti disponibili nelle directory dati"""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Cerca nelle directory principali
    search_dirs = ['img_lights_1', 'img_plate_2', 'img_register_4', 'img_preprocessed']
    
    available_objects = set()
    
    for search_dir in search_dirs:
        dir_path = data_dir / search_dir
        if dir_path.exists():
            # Cerca sottocartelle (oggetti)
            for item in dir_path.iterdir():
                if item.is_dir():
                    available_objects.add(item.name)
    
    return sorted(list(available_objects))

def create_directory_structure(target_object):
    """Crea la struttura di directory per un nuovo oggetto"""
    data_dir = Path(__file__).parent.parent / 'data'
    
    directories_to_create = [
        f'img_lights_1/{target_object}',
        f'img_plate_2/{target_object}', 
        f'img_register_4/{target_object}',
        f'img_preprocessed/{target_object}',
        f'dataset_sr_patches/{target_object}',
        f'local_raw/{target_object}',
        f'local_processed/{target_object}'
    ]
    
    logs_dir = Path(__file__).parent.parent / 'logs' / target_object
    results_dir = Path(__file__).parent.parent / 'results' / target_object
    
    created_dirs = []
    
    for dir_rel in directories_to_create:
        dir_path = data_dir / dir_rel
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
    
    # Crea anche logs e results
    for dir_path in [logs_dir, results_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True) 
            created_dirs.append(str(dir_path))
    
    return created_dirs

def main():
    """Funzione principale con menu interattivo"""
    print("=" * 60)
    print("🎯 GESTIONE OGGETTI TARGET - PIPELINE ASTRONOMICA")
    print("=" * 60)
    
    # Mostra oggetti disponibili
    available_objects = list_available_objects()
    if available_objects:
        print(f"\n📂 Oggetti trovati nel dataset:")
        for i, obj in enumerate(available_objects, 1):
            print(f"   {i}. {obj}")
    else:
        print(f"\n📂 Nessun oggetto trovato nel dataset")
    
    print(f"\n🔧 OPZIONI:")
    print(f"1. Cambia oggetto target esistente")
    print(f"2. Crea nuovo oggetto (con directory)")
    print(f"3. Solo crea directory per oggetto esistente")
    
    try:
        choice = input(f"\nScegli opzione (1-3): ").strip()
        
        if choice == "1":
            if available_objects:
                print(f"\nOggetti disponibili:")
                for i, obj in enumerate(available_objects, 1):
                    print(f"   {i}. {obj}")
                
                try:
                    obj_choice = int(input(f"\nScegli oggetto (numero): ")) - 1
                    if 0 <= obj_choice < len(available_objects):
                        target_object = available_objects[obj_choice]
                    else:
                        print("❌ Scelta non valida")
                        return
                except ValueError:
                    print("❌ Inserisci un numero valido")
                    return
            else:
                target_object = input(f"\nInserisci nome oggetto: ").strip()
        
        elif choice == "2":
            target_object = input(f"\nInserisci nome nuovo oggetto (es. M42, NGC2024): ").strip()
            if not target_object:
                print("❌ Nome oggetto non può essere vuoto")
                return
            
            # Crea directory
            print(f"\n📁 Creazione struttura directory per {target_object}...")
            created_dirs = create_directory_structure(target_object)
            print(f"✅ Create {len(created_dirs)} directory")
            for dir_path in created_dirs[:5]:  # Mostra solo prime 5
                print(f"   {dir_path}")
            if len(created_dirs) > 5:
                print(f"   ... e altre {len(created_dirs)-5}")
        
        elif choice == "3":
            target_object = input(f"\nInserisci nome oggetto per directory: ").strip()
            if not target_object:
                print("❌ Nome oggetto non può essere vuoto")
                return
            
            created_dirs = create_directory_structure(target_object)
            print(f"\n✅ Create {len(created_dirs)} directory per {target_object}")
            return
        
        else:
            print("❌ Scelta non valida")
            return
        
        # Valida nome oggetto
        if not target_object or not re.match(r'^[A-Za-z0-9_-]+$', target_object):
            print("❌ Nome oggetto non valido. Usa solo lettere, numeri, _ e -")
            return
        
        # Aggiorna gli script
        print(f"\n🔄 Aggiornamento script per oggetto: {target_object}")
        updated_files = update_target_object(target_object)
        
        if updated_files:
            print(f"\n✅ AGGIORNAMENTO COMPLETATO!")
            print(f"   Oggetto target: {target_object}")
            print(f"   Script aggiornati: {len(updated_files)}")
            
            print(f"\n📋 PROSSIMI PASSI:")
            print(f"   1. Inserisci immagini in: data/img_lights_1/{target_object}/")
            print(f"   2. Esegui: python AstroPlateSolver.py")
            print(f"   3. Esegui: python AstroRegister.py") 
            print(f"   4. Esegui: python AstroMosaic.py")
        else:
            print(f"\n⚠️  Nessun script aggiornato")
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Operazione annullata")
    except Exception as e:
        print(f"\n❌ Errore: {e}")

if __name__ == "__main__":
    main()