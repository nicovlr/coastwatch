# BAYWATCH

CLI Python pour monitorer les conditions des plages de la côte Atlantique en temps réel via des webcams publiques.

**v0.2.0** — Comptage de personnes par IA (YOLOv8), météo réelle (OpenWeatherMap), détection de courants dangereux (Claude Vision), détection de l'état caméra.

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

```bash
export WINDY_API_KEY="votre_clé"                # Webcams (gratuit : https://api.windy.com/webcams)
export OPENWEATHERMAP_API_KEY="votre_clé"       # Météo (gratuit : https://openweathermap.org/api)
export ANTHROPIC_API_KEY="sk-ant-..."           # Claude Vision (optionnel)
```

## Utilisation

```bash
# Lister les plages configurées
baywatch beaches

# Capturer et analyser (YOLO + météo, sans Claude Vision)
baywatch capture --once --no-ai

# Capturer avec analyse complète (YOLO + météo + Claude Vision + courants)
baywatch capture --once

# Capturer une plage spécifique
baywatch capture --once -b hossegor-plage

# Mode daemon (capture continue toutes les 5 min)
baywatch capture

# Voir les conditions actuelles
baywatch status hossegor-plage

# Classement des plages
baywatch best
baywatch best --activity surfing

# Historique
baywatch history hossegor-plage --hours 24
baywatch history hossegor-plage --format json
```

> La commande `coastwatch` reste disponible pour la rétrocompatibilité.

## Architecture

```
src/coastwatch/
├── cli/            # Interface Click + commandes
├── capture/        # Grabber HTTP (Windy API) + scheduler
├── analysis/       # YOLOv8 (personnes) + OpenCV (vagues) + Weather API + Claude Vision
├── storage/        # SQLite schema + repository + migrations
├── config/         # Chargement YAML + modèles Pydantic
└── common/         # Exceptions + rate limiter + calcul solaire
```

### Pipeline d'analyse

```
Webcam (Windy API) → Frame JPEG
    → Détection état caméra (working / night / offline / obstructed)
    → YOLOv8 : comptage de personnes (3MB, ~50ms CPU)
    → OpenCV : analyse de vagues (Canny + whitecaps)
    → OpenWeatherMap : température, vent, humidité, précipitations
    → Claude Vision : analyse détaillée, courants dangereux, scores
    → SQLite : observation horodatée
    → CLI : affichage rich
```

### Détection des courants (baïnes)

Claude Vision analyse les indicateurs visuels de courants de baïne :
- Chenaux d'eau calme et sombre traversant les vagues
- Eau décolorée ou boueuse se dirigeant vers le large
- Mousse, algues ou débris dérivant vers l'océan
- Brèches dans la ligne de déferlement

### État de la caméra

Le système détecte automatiquement l'état de chaque caméra :
- **working** : image exploitable
- **night** : image sombre en dehors des heures de jour (calcul solaire via `astral`)
- **offline** : image sombre en plein jour → caméra HS
- **obstructed** : image uniforme → objectif obstrué

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

`anthropic`, `astral`, `click`, `httpx`, `numpy`, `opencv-python-headless`, `pillow`, `pydantic`, `pyyaml`, `rich`, `ultralytics`

## Licence

MIT
