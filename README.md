# CoastWatch

CLI Python pour monitorer les conditions des plages de la côte Atlantique en temps réel via des webcams publiques.

Analyse combinée **OpenCV** (pré-traitement local) + **Claude Vision** (analyse IA détaillée). Stockage SQLite, interface CLI avec `rich`.

## Plages couvertes

- Biarritz (Grande Plage, Côte des Basques, Port Vieux)
- Hossegor (Seignosse, Plage)
- Capbreton

## Installation

```bash
git clone https://github.com/nicovlr/coastwatch.git
cd coastwatch
pip install -e .
```

## Configuration

1. **Clé Windy** (gratuit) : https://api.windy.com/webcams
2. **Clé Anthropic** (pour Claude Vision, optionnel)

```bash
export WINDY_API_KEY="votre_clé"
export ANTHROPIC_API_KEY="sk-ant-..."  # optionnel
```

## Utilisation

```bash
# Lister les plages configurées
coastwatch beaches

# Capturer et analyser (une fois, OpenCV seulement)
coastwatch capture --once --no-ai

# Capturer et analyser (une fois, OpenCV + Claude Vision)
coastwatch capture --once

# Capturer une plage spécifique
coastwatch capture --once -b biarritz-grande-plage

# Mode daemon (capture continue toutes les 5 min)
coastwatch capture

# Voir les conditions actuelles
coastwatch status biarritz-grande-plage

# Classement des plages
coastwatch best
coastwatch best --activity surfing

# Historique
coastwatch history biarritz-grande-plage --hours 24
coastwatch history biarritz-grande-plage --format json
```

## Architecture

```
src/coastwatch/
├── cli/            # Interface Click + commandes
├── capture/        # Grabber HTTP (Windy API) + scheduler
├── analysis/       # OpenCV (crowd, waves, weather) + Claude Vision
├── storage/        # SQLite schema + repository
├── config/         # Chargement YAML + modèles Pydantic
└── common/         # Exceptions + rate limiter
```

### Pipeline d'analyse

```
Webcam (Windy API) → Frame JPEG
    → OpenCV : affluence (blobs), vagues (Canny/whitecaps), météo (sky color)
    → Claude Vision : analyse détaillée, scores, recommandations
    → SQLite : observation horodatée
    → CLI : affichage rich
```

## Ajouter une plage

Éditer `config/beaches.yaml` :

```yaml
- id: ma-plage
  name: "Ma Plage"
  region: "Ma Région"
  coordinates:
    latitude: 43.0
    longitude: -1.5
  webcam:
    snapshot_url: "windy://WEBCAM_ID"
    type: snapshot
    refresh_interval_sec: 300
  metadata:
    orientation: west
    surf_spot: true
```

## Dépendances

`anthropic`, `click`, `httpx`, `numpy`, `opencv-python-headless`, `pillow`, `pydantic`, `pyyaml`, `rich`

## Licence

MIT
