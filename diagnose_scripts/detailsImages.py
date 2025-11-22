"""
Script per analizzare le dimensioni delle immagini FITS REGISTRATE con coordinate WCS
"""
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import pandas as pd
import numpy as np

ROOT_DATA_DIR = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data")

results = []

def get_wcs_info(header):
    """Estrae informazioni WCS dall'header"""
    try:
        wcs = WCS(header)
        # Centro immagine in coordinate celesti
        ny, nx = header.get('NAXIS2', 0), header.get('NAXIS1', 0)
        if ny > 0 and nx > 0:
            center_sky = wcs.pixel_to_world(nx/2, ny/2)
            ra = center_sky.ra.deg
            dec = center_sky.dec.deg
            
            # Pixel scale in arcsec/pixel
            scales = proj_plane_pixel_scales(wcs)
            pixel_scale = np.mean(scales) * 3600  # deg to arcsec
            
            return {
                'RA_center': f"{ra:.6f}°",
                'DEC_center': f"{dec:.6f}°",
                'Pixel_Scale': f"{pixel_scale:.3f}\"/px"
            }
        else:
            return {'RA_center': 'N/A', 'DEC_center': 'N/A', 'Pixel_Scale': 'N/A'}
    except Exception as e:
        return {'RA_center': 'ERROR', 'DEC_center': 'ERROR', 'Pixel_Scale': 'ERROR'}

# Scansiona tutti i target
for target_dir in ROOT_DATA_DIR.iterdir():
    if not target_dir.is_dir() or target_dir.name in ['logs', '__pycache__']:
        continue
    
    target_name = target_dir.name
    
    # Immagini REGISTRATE Hubble
    hubble_reg_dir = target_dir / '3_registered_native' / 'hubble'
    if hubble_reg_dir.exists():
        for fits_file in hubble_reg_dir.glob('*.fits'):
            try:
                with fits.open(fits_file) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    if data is not None:
                        shape = data.shape
                        if len(shape) == 3:  # Multi-layer
                            shape = shape[1:]
                        
                        wcs_info = get_wcs_info(header)
                        
                        results.append({
                            'Oggetto': target_name,
                            'Origine': 'Hubble (HR)',
                            'File': fits_file.name[:40] + '...' if len(fits_file.name) > 40 else fits_file.name,
                            'Dimensioni': f"{shape[1]}x{shape[0]}",
                            'Pixels': f"{shape[1] * shape[0]:,}",
                            'RA Centro': wcs_info['RA_center'],
                            'DEC Centro': wcs_info['DEC_center'],
                            'Pixel Scale': wcs_info['Pixel_Scale']
                        })
            except Exception as e:
                print(f"⚠️ Errore {fits_file.name}: {e}")
    
    # Immagini REGISTRATE Observatory
    obs_reg_dir = target_dir / '3_registered_native' / 'observatory'
    if obs_reg_dir.exists():
        for fits_file in obs_reg_dir.glob('*.fits'):
            try:
                with fits.open(fits_file) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    if data is not None:
                        shape = data.shape
                        if len(shape) == 3:
                            shape = shape[1:]
                        
                        wcs_info = get_wcs_info(header)
                        
                        results.append({
                            'Oggetto': target_name,
                            'Origine': 'Observatory (LR)',
                            'File': fits_file.name[:40] + '...' if len(fits_file.name) > 40 else fits_file.name,
                            'Dimensioni': f"{shape[1]}x{shape[0]}",
                            'Pixels': f"{shape[1] * shape[0]:,}",
                            'RA Centro': wcs_info['RA_center'],
                            'DEC Centro': wcs_info['DEC_center'],
                            'Pixel Scale': wcs_info['Pixel_Scale']
                        })
            except Exception as e:
                print(f"⚠️ Errore {fits_file.name}: {e}")

# Crea DataFrame e stampa tabella
df = pd.DataFrame(results)
if not df.empty:
    df = df.sort_values(['Oggetto', 'Origine'])
    
    print("\n" + "="*120)
    print("DIMENSIONI IMMAGINI REGISTRATE CON COORDINATE WCS".center(120))
    print("="*120)
    print(df.to_string(index=False))
    print("\n" + "="*120)
    
    # Statistiche per origine
    print("\n📊 STATISTICHE PER ORIGINE:")
    print("-"*120)
    for origine in df['Origine'].unique():
        subset = df[df['Origine'] == origine]
        print(f"\n{origine}:")
        print(f"  • Numero immagini: {len(subset)}")
        dims = subset['Dimensioni'].unique()
        if len(dims) == 1:
            print(f"  • Dimensioni: {dims[0]} (uniforme)")
        else:
            print(f"  • Dimensioni: {', '.join(dims)} (variabili)")
        
        # Pixel scale medio
        scales = subset['Pixel Scale'].str.extract(r'([\d.]+)')[0].astype(float)
        if not scales.empty:
            print(f"  • Pixel Scale medio: {scales.mean():.3f}\"/px")
    
    print("\n" + "="*120)
    print(f"TOTALE IMMAGINI ANALIZZATE: {len(df)}")
    print("="*120)
else:
    print("\n❌ ERRORE: Nessuna immagine trovata in 3_registered_native/")
    print("Verifica che il preprocessing sia stato completato.")