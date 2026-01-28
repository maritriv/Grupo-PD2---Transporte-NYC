# src/features/build_features.py
from pathlib import Path
from src.data.download_tlc_data import download_service_data

# | Tipo servicio           | Descripción breve                                                | Años aproximados disponibles               |
# | ----------------------- | ---------------------------------------------------------------- | ------------------------------------------ |
# | Yellow Taxi             | Taxis amarillos tradicionales en toda la ciudad.                 | Desde 2009–2010 hasta 2025+.               |
# | Green Taxi              | Taxis "boro" (fuera de Manhattan core).                          | Desde ~2013–2014 hasta 2025+.              |
# | FHV                     | For-Hire Vehicles (livery, black car, etc.) no high volume.      | Desde ~2015 hasta 2025+.                   |
# | High Volume FHV (HVFHV) | Uber, Lyft y otras bases de "alto volumen".                      | Desde 2019 hasta 2025+.                    |


def download_all_services(start_year: int = 2023, end_year: int = 2025):
    """
    Descarga datos de todos los servicios de NYC TLC.
    
    Args:
        start_year: Año inicial de descarga
        end_year: Año final de descarga (inclusive)
    """
    services = {
        'yellow': 'Yellow Taxi (taxis amarillos tradicionales)',
        'green': 'Green Taxi (taxis "boro" fuera de Manhattan)',
        'fhv': 'FHV (For-Hire Vehicles, livery, black car)',
        'fhvhv': 'HVFHV (Uber, Lyft, alto volumen)'
    }
    
    total_services = len(services)
    
    print("\n" + "═" * 80)
    print("NYC TAXI & LIMOUSINE COMMISSION - DESCARGA MASIVA DE DATOS")
    print("═" * 80)
    print(f"Período: {start_year} - {end_year}")
    print(f"Servicios a descargar: {total_services}")
    print(f"Destino: data/raw/[servicio]/")
    print("═" * 80 + "\n")
    
    for idx, (service, description) in enumerate(services.items(), 1):
        print(f"\n{'▓' * 80}")
        print(f"  {description}")
        print(f"  Progreso global: [{idx}/{total_services}]")
        print(f"{'▓' * 80}\n")
        
        try:
            download_service_data(
                service=service,
                years=range(start_year, end_year + 1),
                months=range(1, 13),
                data_dir=Path("data/raw")
            )
        except Exception as e:
            print(f"\nERROR al descargar {service}: {e}")
            print("Continuando con el siguiente servicio...\n")
            continue


if __name__ == "__main__":
    download_all_services(start_year=2020, end_year=2025)