# Detección temprana de focos emergentes de desaparición de personas en México (2015-2025)

Proyecto de Análisis y Exploración de Datos, Ingeniería Informática, ITS Teziutlán.

Clasificación a nivel municipal: estima si un municipio que no es foco lo será en los siguientes 6 meses
(foco = 10 % de municipios con más personas desaparecidas y no localizadas), con datos mensuales del RNPDNO.

## Datos (no incluidos)
Escobar, Reyes-Guzman y Sanchez Ruiz (2026). Open Science Framework, CC-BY 4.0.
https://doi.org/10.17605/OSF.IO/4AYMZ
Descomprime el zip en `data/` para que existan los CSV en `data/processed/`.

## Uso
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python rnpdno_pipeline.py all        # panel, EDA, modelos y ranking
python rnpdno_pipeline.py validar    # origen móvil con repeticiones
python rnpdno_pipeline.py mapas      # requiere un mapa municipal en data/geo/
```

## Resultados
Figuras y tablas en `salidas/`. Los rankings son estimaciones del modelo, no hechos que necesariamente ocurrirán.
Los datos son agregados por municipio y mes, sin información de personas.
