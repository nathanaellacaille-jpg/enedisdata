# Cave IA — Plan de développement complet

> Document de référence — version 1.0 — 13 juin 2026
> Destinataire : porteur du projet, dev solo JS/TS

---

## Table des matières

1. [Synthèse exécutive](#1-synthèse-exécutive)
2. [Architecture technique détaillée](#2-architecture-technique-détaillée)
3. [Découpage en phases — Roadmap](#3-découpage-en-phases--roadmap)
4. [Backlog initial structuré](#4-backlog-initial-structuré)
5. [Plan d'attaque briques critiques](#5-plan-dattaque-briques-critiques)
6. [Stratégie données et conformité](#6-stratégie-données-et-conformité)
7. [Tests et qualité](#7-tests-et-qualité)
8. [Estimation budgétaire et ressources](#8-estimation-budgétaire-et-ressources)
9. [Risques et plans de mitigation](#9-risques-et-plans-de-mitigation)
10. [Questions ouvertes pour le porteur](#10-questions-ouvertes-pour-le-porteur)

---

## 1. Synthèse exécutive

### Reformulation du projet

Cave IA est un assistant mobile iOS (React Native/Expo) permettant à un amateur de vin francophone de scanner ses bouteilles par photo d'étiquette, de connaître la fenêtre d'apogée de chaque vin, de gérer son inventaire et de recevoir des recommandations met/vin contextuelles. La différenciation repose sur l'intelligence active (conseil et anticipation) versus les concurrents passifs (tracking seul). Cible initiale : France, Italie, Espagne. LLM local (Ollama sur NAS) comme fallback OCR pour maîtriser les coûts au démarrage.

### Risques techniques majeurs identifiés

| # | Risque | Impact | Probabilité |
|---|--------|--------|-------------|
| R1 | Précision OCR insuffisante sur étiquettes artisanales / manuscrites | Élevé | Élevée |
| R2 | Disponibilité NAS Ollama non garantie en production | Élevé | Moyenne |
| R3 | LWIN incomplet sur petits producteurs / vins nature | Moyen | Élevée |
| R4 | Latence pipeline > 3s → friction UX rédhibitoire | Élevé | Moyenne |
| R5 | Qualité du modèle d'apogée sur millésimes peu documentés | Moyen | Moyenne |

### Décisions architecturales à trancher dès le départ

- **Supabase comme backend-as-a-service** : PostgreSQL + Auth + Storage + Edge Functions hébergé EU (Frankfurt). Évite de maintenir une infra à la main en solo. Décision irréversible à la Phase 1 — valider.
- **LWIN bundlé en SQLite dans l'app** : enable le matching offline. Implique de regénérer le bundle à chaque mise à jour LWIN (~trimestrielle). Poids estimé : ~40 Mo compressé.
- **Ollama NAS = dev/staging uniquement** : dès que le volume beta le justifie (>500 utilisateurs actifs), migrer vers Mistral free tier API ou instance GPU dédiée. Prévoir l'abstraction dès le départ.

---

## 2. Architecture technique détaillée

### 2.1 Schéma logique des composants

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAVE IA — Architecture                          │
├────────────────────────┬───────────────────────────┬────────────────────┤
│    MOBILE (iOS)        │    BACKEND (Supabase EU)   │   IA SERVICES      │
│    React Native/Expo   │                            │                    │
│                        │  ┌─────────────────────┐  │  ┌──────────────┐  │
│  ┌──────────────────┐  │  │  Supabase Postgres   │  │  │ Ollama NAS   │  │
│  │ Vision Framework │  │  │  + pgvector          │  │  │ LLaVA 1.6   │  │
│  │ (OCR on-device)  │  │  └──────────┬──────────┘  │  │ (fallback)   │  │
│  └────────┬─────────┘  │             │             │  └──────┬───────┘  │
│           │            │  ┌──────────▼──────────┐  │         │          │
│  ┌────────▼─────────┐  │  │  Supabase Auth      │  │         │          │
│  │ LWIN SQLite      │  │  │  Edge Functions     │  │         │          │
│  │ (bundlé ~40 Mo)  │──┼─▶│  (matching, apogée, │──┼─────────┘          │
│  └──────────────────┘  │  │   recommandations)  │  │                    │
│                        │  └──────────┬──────────┘  │                    │
│  ┌──────────────────┐  │             │             │                    │
│  │ Queue offline    │──┼─────────────┘             │                    │
│  │ (Zustand + MMKV) │  │                            │                    │
│  └──────────────────┘  │  ┌─────────────────────┐  │                    │
│                        │  │  Supabase Storage   │  │                    │
│                        │  │  (photos étiquettes)│  │                    │
│                        │  └─────────────────────┘  │                    │
└────────────────────────┴───────────────────────────┴────────────────────┘
```

### 2.2 Diagramme de flux — Pipeline de scan

```
┌──────────┐
│  CAPTURE │  Utilisateur prend la photo de l'étiquette
│  (Expo   │
│  Camera) │
└────┬─────┘
     │ image HEIC/JPG
     ▼
┌─────────────────────────────┐
│  OCR ON-DEVICE              │  Vision Framework (VNRecognizeTextRequest)
│  Apple Vision Framework     │  Langue : français + italien + espagnol
│  Résultat : blocs de texte  │  Confiance par bloc : 0.0 → 1.0
│  avec coordonnées et scores │
└────┬────────────────────────┘
     │ texte brut + score moyen
     │
     ▼
┌─────────────────────────────┐
│  PRÉ-PROCESSING TEXTE       │  Normalisation unicode, suppression bruit,
│  (on-device, sync)          │  détection langue, extraction tokens clés
│                             │  (producteur, appellation, millésime, pays)
└────┬────────────────────────┘
     │ tokens structurés
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  MATCHING LWIN (SQLite local)                               │
│                                                             │
│  1. Recherche par millésime exact + appellation (exact)     │
│  2. Recherche fuzzy trigrams (FTS + Levenshtein)            │
│  3. Score de confiance composite : 0.0 → 1.0               │
└────────────────┬──────────────────────────────┬────────────┘
                 │                              │
         Score ≥ 0.7                      Score < 0.7
                 │                              │
                 ▼                              ▼
     ┌───────────────────┐         ┌───────────────────────┐
     │  MATCH DIRECT     │         │  FALLBACK LLM         │
     │  Résultat immédiat│         │  (async, avec queue   │
     │  affiché en < 1s  │         │  si offline)          │
     └─────────┬─────────┘         │                       │
               │                   │  Payload: image +     │
               │                   │  texte OCR brut       │
               │                   │  → Ollama LLaVA NAS   │
               │                   └──────────┬────────────┘
               │                              │
               │                    ┌─────────▼─────────┐
               │                    │  Résultat LLM     │
               │                    │  (nom, producteur,│
               │                    │  millésime, région)│
               │                    │  + score confiance│
               │                    └─────────┬─────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  VALIDATION UTILISATEUR       │
               │  Fiche pré-remplie + 1 tap    │
               │  Corriger / Confirmer         │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  ENRICHISSEMENT               │
               │  - Calcul fenêtre d'apogée    │
               │  - Métadonnées (cépage, etc.) │
               │  - Stockage correction →      │
               │    amélioration future        │
               └──────────────────────────────┘
```

### 2.3 Stack technique — Justification

| Couche | Choix retenu | Pourquoi | Alternatives écartées |
|--------|-------------|---------|----------------------|
| **Mobile** | React Native + Expo SDK 51 | Dev JS/TS, cross-platform préservé pour Android Phase 2, Expo EAS Build | Swift natif (learning curve Android), Flutter (Dart) |
| **OCR** | Apple Vision Framework via module natif Expo | Meilleure précision sur iPhone, on-device, gratuit | Tesseract (moins précis), Google ML Kit (Android pour Phase 2) |
| **Backend** | Supabase (EU Frankfurt) | PostgreSQL + Auth + Storage + Edge Functions tout-en-un, managed, solo dev | Neon + Vercel + Clerk (plus modulaire mais +overhead), Railway Node.js (moins intégré) |
| **DB** | PostgreSQL 16 + pgvector | Relationnel structuré + similarité vectorielle pour reco | MongoDB (pas d'avantage sur données structurées), SQLite only (pas assez pour backend) |
| **Auth** | Supabase Auth | Intégré, JWT, RGPD EU, magic link + OAuth | Clerk (excellent mais $25/mo après free), Auth0 (overkill MVP) |
| **Stockage images** | Supabase Storage | Intégré, presigned URLs, EU | Cloudflare R2 (mieux à grande échelle mais +config), S3 Paris (coûteux) |
| **LLM fallback** | Ollama + LLaVA 1.6 (NAS) | Coût nul, multimodal, suffisant pour Phase 0-1 | Claude API (coût variable), GPT-4o (idem), Groq free tier (pas vision) |
| **Cache mobile** | Zustand + MMKV | Performant, TypeScript natif, sync offline | Redux (verbeux), Jotai (moins documenté React Native) |
| **Navigation** | Expo Router v3 | File-based routing, deep links natifs | React Navigation (plus de boilerplate) |
| **Notifications push** | Expo Notifications | Intégré EAS, APNs + FCM | OneSignal (overhead service tiers) |

### 2.4 Modèle de données

```sql
-- =========================================================
-- CORE ENTITIES
-- =========================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Utilisateurs
CREATE TABLE users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT UNIQUE NOT NULL,
  display_name  TEXT,
  subscription  TEXT NOT NULL DEFAULT 'free',  -- free|basic|premium
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at    TIMESTAMPTZ  -- soft delete pour RGPD
);

-- Catalogue vins (partagé entre tous les users, enrichi progressivement)
CREATE TABLE wines (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lwin_id         TEXT UNIQUE,              -- identifiant LWIN si matché
  name            TEXT NOT NULL,
  producer        TEXT,
  region          TEXT,                     -- 'Pauillac', 'Barolo', 'Rioja'
  appellation     TEXT,
  country         TEXT NOT NULL DEFAULT 'France',
  grape_varieties TEXT[],                   -- ['Nebbiolo'], ['Cabernet Sauvignon','Merlot']
  wine_type       TEXT NOT NULL,            -- red|white|rose|sparkling|dessert|fortified
  vintage_year    INT,
  apogee_min      INT,                      -- année calendaire min
  apogee_max      INT,                      -- année calendaire max
  apogee_peak     INT,                      -- année idéale
  apogee_source   TEXT,                     -- 'rules'|'llm'|'user_correction'
  apogee_confidence FLOAT DEFAULT 0.5,
  embedding       vector(1536),             -- recherche sémantique
  source          TEXT DEFAULT 'lwin',      -- 'lwin'|'llm_inferred'|'user_created'
  verified        BOOLEAN DEFAULT false,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX wines_lwin_idx ON wines(lwin_id);
CREATE INDEX wines_trgm_name ON wines USING gin(name gin_trgm_ops);
CREATE INDEX wines_trgm_producer ON wines USING gin(producer gin_trgm_ops);
CREATE INDEX wines_embedding_idx ON wines USING ivfflat(embedding vector_cosine_ops);

-- Cave utilisateur (bouteilles physiques)
CREATE TABLE bottles (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  wine_id           UUID REFERENCES wines(id),
  quantity          INT NOT NULL DEFAULT 1,
  purchase_price    DECIMAL(10,2),
  purchase_date     DATE,
  cellar_position   TEXT,                   -- 'Casier A-3', 'Frigo du bas'
  status            TEXT NOT NULL DEFAULT 'in_cellar',  -- in_cellar|consumed|gifted|sold
  storage_condition TEXT DEFAULT 'good',    -- ideal|good|average|poor
  label_photo_url   TEXT,
  ocr_raw_text      TEXT,
  ocr_confidence    FLOAT,
  match_method      TEXT,                   -- 'lwin_exact'|'lwin_fuzzy'|'llm'|'manual'
  notes             TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX bottles_user_idx ON bottles(user_id);
CREATE INDEX bottles_wine_idx ON bottles(wine_id);
CREATE INDEX bottles_status_idx ON bottles(user_id, status);

-- Notes de dégustation
CREATE TABLE tasting_notes (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  bottle_id   UUID REFERENCES bottles(id) ON DELETE SET NULL,
  wine_id     UUID NOT NULL REFERENCES wines(id),
  tasted_at   DATE NOT NULL DEFAULT CURRENT_DATE,
  score       INT CHECK (score BETWEEN 0 AND 100),
  appearance  TEXT,
  nose        TEXT,
  palate      TEXT,
  finish      TEXT,
  notes       TEXT,
  food_pairing TEXT[],
  is_public   BOOLEAN DEFAULT false,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Recommandations (log pour amélioration continue)
CREATE TABLE recommendations (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  query           TEXT NOT NULL,            -- 'agneau grillé ce soir'
  context         JSONB,                    -- {dish, occasion, guests, budget}
  result_wine_ids UUID[],
  llm_reasoning   TEXT,
  model_version   TEXT,
  feedback        SMALLINT,                 -- -1|0|1
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Corrections utilisateurs (dataset d'amélioration)
CREATE TABLE scan_corrections (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  bottle_id       UUID NOT NULL REFERENCES bottles(id),
  original_wine_id UUID,
  corrected_wine_id UUID REFERENCES wines(id),
  ocr_raw_text    TEXT,
  label_photo_url TEXT,
  correction_type TEXT,  -- 'wrong_wine'|'wrong_vintage'|'wrong_producer'
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Alertes apogée planifiées
CREATE TABLE apogee_alerts (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  bottle_id   UUID NOT NULL REFERENCES bottles(id) ON DELETE CASCADE,
  alert_date  DATE NOT NULL,
  alert_type  TEXT NOT NULL,  -- 'approaching'|'at_peak'|'past_peak'
  sent_at     TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 3. Découpage en phases — Roadmap

### Phase 0 — Validation technique (2-3 semaines)

**Objectif** : Prouver que le pipeline OCR → LWIN → Ollama fonctionne avec un taux de reconnaissance acceptable avant tout investissement produit.

**Livrables**
- Script d'ingestion LWIN → PostgreSQL + SQLite
- Module natif Expo exposant Vision Framework (OCR)
- Pipeline de matching fuzzy (pg_trgm) avec score de confiance
- Intégration Ollama LLaVA sur NAS (timeout, fallback gracieux)
- Jeu de test de 100 étiquettes annotées (France/Italie/Espagne)
- Dashboard de métriques OCR (taux reconnaissance, temps moyen)

**Critères de succès**
- Taux de reconnaissance ≥ 70% sur le jeu de test 100 étiquettes
- Temps pipeline OCR + LWIN < 1,5s sur iPhone (hors LLM)
- LLM fallback ajoute ≤ 30% de reconnaissance correcte supplémentaire
- Ollama disponible et répondant en < 10s sur 80% des requêtes test

**Durée** : 2-3 semaines
**Ressources** : 1 dev solo

---

### Phase 1 — MVP (10-12 semaines)

**Objectif** : Version utilisable par 10-20 beta users (réseau personnel). Couvre le flux complet : scan → cave → recommandation → note.

**Semaines 1-2 : Fondations**
- Setup Expo + Supabase (auth, DB, storage)
- Navigation de base (Expo Router)
- Intégration module OCR Phase 0

**Semaines 3-5 : Scan et inventaire**
- Écran de scan (Expo Camera + Vision Framework)
- Pipeline matching + validation utilisateur
- CRUD bouteilles (ajout/suppression/modification)
- Vue cave (liste triable : apogée, millésime, région)

**Semaines 6-7 : Apogée et détail bouteille**
- Moteur d'apogée V1 (règles heuristiques YAML)
- Fiche bouteille détaillée
- Indicateur visuel apogée (passé / optimal / futur)
- Alertes push pour apogées imminentes (≤ 2 ans)

**Semaines 8-9 : Recommandations**
- Écran "Quel vin ce soir ?"
- Prompt LLM met/vin (Ollama Llama3.2 text)
- Affichage suggestions avec raisonnement
- Feedback thumbs up/down

**Semaines 10-11 : Notes de dégustation**
- Formulaire note structuré (score, nez, bouche, finale)
- Historique des dégustations
- Lien bouteille → note

**Semaine 12 : Stabilisation et beta**
- Correction bugs retours early testers
- Onboarding simplifié (3 écrans)
- TestFlight distribution aux 10-20 betas

**Critères de succès Phase 1**
- 15+ utilisateurs beta actifs
- Taux de scan réussi ≥ 75% (confirmation sans correction)
- NPS interne ≥ 7/10
- Zéro crash bloquant sur iOS 16+
- Temps d'ajout d'une bouteille ≤ 45 secondes end-to-end

**Durée** : 10-12 semaines
**Ressources** : 1 dev solo + beta testers réseau personnel

---

### Phase 2 — Beta publique (6-8 semaines)

**Objectif** : Préparer le lancement App Store, affiner le produit sur retours beta, préparer la monétisation.

**Livrables**
- Amélioration pipeline OCR (fine-tuning sur corrections collectées Phase 1)
- Modèle d'apogée V2 (intégration corrections utilisateurs)
- Import CSV depuis CellarTracker/Vivino (migration utilisateurs)
- Stats cave (valeur estimée, distribution par région/millésime/type)
- Abonnements in-app (RevenueCat)
- Politiques RGPD (export données, suppression compte)
- Submission App Store

**Critères de succès Phase 2**
- App Store approved et live
- 100+ téléchargements semaine 1
- Taux de conversion free → payant ≥ 5%
- Taux de reconnaissance ≥ 85%

**Durée** : 6-8 semaines
**Ressources** : 1 dev solo + 1 bêta testeur UX si possible

---

### Vue d'ensemble Roadmap

```
Semaine  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
         │──────────│──────────────────────────────────│──────────────│
Phase 0  ████████
Phase 1           ██████████████████████████████████████
Phase 2                                                  ████████████
```

---

## 4. Backlog initial structuré

### EPIC-INFRA — Infrastructure et setup

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| INFRA-1 | Setup repo Expo + Supabase projet EU | App se lance, Supabase connecté, auth fonctionnel | S | — |
| INFRA-2 | Module natif OCR Vision Framework | Texte extrait d'une image en < 1s sur iPhone 12+ | M | INFRA-1 |
| INFRA-3 | Ingestion LWIN → PostgreSQL + SQLite | 200k+ vins importés, index trigrams actifs | M | INFRA-1 |
| INFRA-4 | Pipeline matching fuzzy avec score | Score confiance retourné, top 3 candidats | M | INFRA-2, INFRA-3 |
| INFRA-5 | Intégration Ollama NAS (client TS) | Appel LLaVA retourne résultat ou timeout gracieux | M | — |
| INFRA-6 | Queue offline (Zustand + MMKV) | Actions en attente réémises à la reconnexion | L | INFRA-1 |

### EPIC-AUTH — Authentification

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| AUTH-1 | Créer un compte (email + magic link) | Compte créé, token JWT valide, accès app | S | INFRA-1 |
| AUTH-2 | Se connecter sur nouvel appareil | Session persistée, données synchronisées | S | AUTH-1 |
| AUTH-3 | Se déconnecter | Session détruite localement et serveur | S | AUTH-1 |
| AUTH-4 | Supprimer son compte (RGPD) | Toutes données supprimées en < 30 jours | M | AUTH-1 |

### EPIC-SCAN — Scan et reconnaissance

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| SCAN-1 | Photographier une étiquette pour l'identifier | Caméra ouvre, OCR déclenche automatiquement | M | INFRA-2 |
| SCAN-2 | Voir résultat de reconnaissance avec confiance | Fiche pré-remplie affichée < 2s après photo | M | INFRA-4 |
| SCAN-3 | Corriger un résultat erroné (recherche manuelle) | Recherche textuelle retourne résultats LWIN | M | INFRA-3 |
| SCAN-4 | Recevoir résultat LLM si LWIN échoue | Notification/résultat affiché en différé | L | INFRA-5, INFRA-6 |
| SCAN-5 | Ajouter une bouteille non reconnue manuellement | Formulaire minimal (nom, millésime, région) | S | INFRA-1 |
| SCAN-6 | Scanner plusieurs bouteilles en séquence | Bouton "scanner suivant" sans quitter le flux | M | SCAN-2 |

### EPIC-CELLAR — Gestion de cave

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| CELLAR-1 | Voir la liste de mes bouteilles | Liste triable par nom, millésime, apogée, région | M | SCAN-2 |
| CELLAR-2 | Voir le détail d'une bouteille | Toutes métadonnées + indicateur apogée visible | S | CELLAR-1 |
| CELLAR-3 | Modifier la quantité (consommation) | Quantité mise à jour, statut "consumed" si 0 | S | CELLAR-1 |
| CELLAR-4 | Supprimer une bouteille | Bouteille archivée (soft delete) | S | CELLAR-1 |
| CELLAR-5 | Filtrer par région / type / statut apogée | Filtres combinables, résultats instantanés | M | CELLAR-1 |
| CELLAR-6 | Rechercher une bouteille par nom/producteur | Résultats en temps réel (debounce 300ms) | S | CELLAR-1 |
| CELLAR-7 | Recevoir une alerte apogée imminente (push) | Notification push si apogée ≤ 2 ans | M | CELLAR-2 |

### EPIC-APOGEE — Modèle d'apogée

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| APOGEE-1 | Voir la fenêtre d'apogée d'une bouteille | Affichage min/max/peak en années calendaires | S | INFRA-3 |
| APOGEE-2 | Règles heuristiques France/Italie/Espagne | YAML couvrant 20+ régions × cépages principaux | L | — |
| APOGEE-3 | Indicateur visuel (trop tôt / optimal / trop tard) | Code couleur + libellé sur la fiche bouteille | S | APOGEE-1 |
| APOGEE-4 | Génération LLM pour vins non couverts | Ollama génère une estimation pour cas inconnus | M | INFRA-5 |

### EPIC-RECO — Recommandations

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| RECO-1 | Demander "quel vin ce soir ?" avec description plat | LLM propose 1-3 vins de ma cave avec raison | L | EPIC-CELLAR |
| RECO-2 | Voir le raisonnement de la recommandation | Explication textuelle courte (3-5 phrases) | S | RECO-1 |
| RECO-3 | Donner un feedback sur la recommandation | Boutons pouce haut/bas, stocké pour amélioration | S | RECO-1 |
| RECO-4 | Recommandation contextuelle (occasion, nb convives) | Filtres occasion (dîner, apéro, cadeau) dans le prompt | M | RECO-1 |

### EPIC-NOTES — Notes de dégustation

| ID | User Story | Critère d'acceptation | Taille | Dépendances |
|----|-----------|----------------------|--------|-------------|
| NOTES-1 | Créer une note de dégustation pour une bouteille | Formulaire score + notes libres enregistré | S | EPIC-CELLAR |
| NOTES-2 | Consulter l'historique de mes dégustations | Liste chronologique avec score et vin | S | NOTES-1 |
| NOTES-3 | Modifier / supprimer une note | Modifications persistées, suppression confirmée | S | NOTES-1 |

---

## 5. Plan d'attaque briques critiques

### 5.a Pipeline OCR + matching LWIN

#### Ingestion LWIN

LWIN (Liv-ex Wine Index) est téléchargeable en CSV sur `liv-ex.com` (licence CC BY-SA 4.0). Il contient ~200 000 entrées avec les champs : `lwin`, `wine_name`, `producer_name`, `country`, `region`, `colour`, `min_vintage`, `max_vintage`.

```bash
# Script d'ingestion (TypeScript / Bun)
# 1. Télécharger le CSV LWIN
# 2. Normaliser les champs (unicode NFC, trim, lowercase producer)
# 3. Insérer en PostgreSQL avec pg_trgm index
# 4. Exporter en SQLite pour bundle mobile

bun run scripts/ingest-lwin.ts --source ./data/lwin.csv \
  --pg-url $DATABASE_URL \
  --sqlite-out ./mobile/assets/lwin.db
```

**Schéma SQLite bundlé** (allégé, ~40 Mo compressé) :

```sql
CREATE TABLE lwin (
  lwin_id      TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  producer     TEXT,
  country      TEXT,
  region       TEXT,
  appellation  TEXT,
  wine_type    TEXT,
  name_norm    TEXT  -- version normalisée pour matching
);
CREATE INDEX lwin_name_norm ON lwin(name_norm);
-- Pas de trigrams dans SQLite natif : utiliser FTS5
CREATE VIRTUAL TABLE lwin_fts USING fts5(
  name, producer, region, appellation,
  content='lwin', content_rowid='rowid'
);
```

#### Algorithme de matching

```typescript
interface MatchResult {
  wine: LwinEntry;
  confidence: number;  // 0.0 - 1.0
  method: 'exact' | 'fuzzy_fts' | 'fuzzy_levenshtein';
}

function matchWine(ocrTokens: OcrTokens): MatchResult[] {
  const { producer, name, vintage, region, appellation } = ocrTokens;

  // 1. Exact match sur LWIN ID si vintage + appellation connus
  // 2. FTS5 match avec boost sur vintage (poids x2)
  // 3. Levenshtein distance si FTS < 0.5 (tolérance fautes OCR)
  // 4. Score composite = 0.4*name + 0.3*producer + 0.2*vintage + 0.1*region

  // Seuil décision : score >= 0.7 → match direct
  //                  score 0.4-0.7 → présenter top 3 à l'utilisateur
  //                  score < 0.4 → LLM fallback
}
```

**Tolérance aux erreurs OCR typiques :**
- `Château` → `Chateau`, `Châtleau`, `Ch£teau`
- `1er Cru` → `1cr Cru`, `Premier Cru`, `1 er cru`
- Millésime : regex `\b(19|20)\d{2}\b` sur tout le texte OCR

---

### 5.b Fallback LLM (Ollama LLaVA)

#### Prompt engineering pour la reconnaissance d'étiquette

```
SYSTEM:
Tu es un expert sommelier spécialisé dans la reconnaissance de vins français,
italiens et espagnols. Tu dois identifier un vin à partir d'une photo d'étiquette
et de fragments de texte OCR partiellement lisibles.

Réponds UNIQUEMENT en JSON valide. Ne hallucine pas de données : si tu n'es pas
sûr d'un champ, utilise null.

USER:
Photo d'étiquette : [IMAGE_BASE64]
Texte OCR extrait (peut contenir des erreurs) : "{ocrRawText}"

Identifie ce vin. JSON attendu :
{
  "producer": string | null,
  "wine_name": string | null,
  "vintage": number | null,
  "region": string | null,
  "appellation": string | null,
  "country": string | null,
  "wine_type": "red" | "white" | "rose" | "sparkling" | "dessert" | null,
  "confidence": number,  // 0.0 à 1.0 — ton niveau de certitude global
  "reasoning": string    // 2-3 phrases expliquant ton déduction
}
```

#### Gestion des scores de confiance et boucle de validation

```
Confidence LLM ≥ 0.8  → Afficher résultat avec "Identifié par IA" + bouton corriger
Confidence LLM 0.5-0.8 → Afficher résultat avec avertissement + champ éditable
Confidence LLM < 0.5  → Afficher "Vin non reconnu" + formulaire manuel complet
```

#### Stockage des corrections pour enrichissement

Chaque correction utilisateur (SCAN-3) alimente `scan_corrections`. Un job hebdomadaire analyse les corrections pour :
1. Mettre à jour `wines` avec la bonne correspondance (si > 3 corrections identiques)
2. Générer des règles de normalisation OCR supplémentaires
3. Identifier les vins absents de LWIN pour enrichissement manuel

---

### 5.c Modèle d'apogée V1 — Règles heuristiques

#### Structure YAML

```yaml
# apogee_rules.yaml
# Format : règles évaluées en ordre, première correspondance gagne
# Formule : apogee_min = vintage + years_min, etc.

rules:
  # ── FRANCE ────────────────────────────────────────────────
  - name: "Bordeaux Grand Cru Classé"
    match:
      region: "Bordeaux"
      appellation: ["Pauillac", "Saint-Estèphe", "Saint-Julien", "Margaux", "Pessac-Léognan"]
      wine_type: "red"
    years_min: 10
    years_peak: 18
    years_max: 35
    note: "Vins structurés, tanins fermes"

  - name: "Bordeaux générique"
    match:
      region: "Bordeaux"
      wine_type: "red"
    years_min: 3
    years_peak: 7
    years_max: 12

  - name: "Bourgogne Grand Cru rouge"
    match:
      region: "Bourgogne"
      appellation: ["Chambolle-Musigny", "Gevrey-Chambertin", "Vosne-Romanée", "Nuits-Saint-Georges"]
      wine_type: "red"
    years_min: 8
    years_peak: 15
    years_max: 25

  - name: "Bourgogne Village rouge"
    match:
      region: "Bourgogne"
      wine_type: "red"
    years_min: 3
    years_peak: 7
    years_max: 15

  - name: "Champagne millésimé"
    match:
      region: "Champagne"
      wine_type: "sparkling"
      has_vintage: true
    years_min: 5
    years_peak: 10
    years_max: 20

  - name: "Champagne non millésimé"
    match:
      region: "Champagne"
      wine_type: "sparkling"
      has_vintage: false
    years_min: 1
    years_peak: 3
    years_max: 5

  - name: "Côte du Rhône Nord (Syrah)"
    match:
      region: "Rhône"
      appellation: ["Hermitage", "Côte-Rôtie", "Cornas", "Saint-Joseph"]
      wine_type: "red"
    years_min: 8
    years_peak: 15
    years_max: 30

  - name: "Côte du Rhône Sud (Grenache)"
    match:
      region: "Rhône"
      appellation: ["Châteauneuf-du-Pape", "Gigondas", "Vacqueyras"]
      wine_type: "red"
    years_min: 5
    years_peak: 12
    years_max: 20

  # ── ITALIE ────────────────────────────────────────────────
  - name: "Barolo / Barbaresco"
    match:
      country: "Italy"
      appellation: ["Barolo", "Barbaresco"]
      wine_type: "red"
    years_min: 10
    years_peak: 18
    years_max: 30

  - name: "Brunello di Montalcino"
    match:
      country: "Italy"
      appellation: "Brunello di Montalcino"
      wine_type: "red"
    years_min: 10
    years_peak: 20
    years_max: 35

  - name: "Amarone della Valpolicella"
    match:
      country: "Italy"
      appellation: "Amarone della Valpolicella"
      wine_type: "red"
    years_min: 8
    years_peak: 15
    years_max: 25

  - name: "Chianti Classico Riserva"
    match:
      country: "Italy"
      appellation: "Chianti Classico"
      wine_type: "red"
    years_min: 5
    years_peak: 10
    years_max: 18

  # ── ESPAGNE ───────────────────────────────────────────────
  - name: "Rioja Gran Reserva"
    match:
      country: "Spain"
      region: "Rioja"
      classification: "Gran Reserva"
      wine_type: "red"
    years_min: 8
    years_peak: 15
    years_max: 25

  - name: "Ribera del Duero"
    match:
      country: "Spain"
      region: "Ribera del Duero"
      wine_type: "red"
    years_min: 5
    years_peak: 12
    years_max: 20

  # ── FALLBACK GÉNÉRAL ──────────────────────────────────────
  - name: "Vin rouge générique"
    match:
      wine_type: "red"
    years_min: 2
    years_peak: 5
    years_max: 10

  - name: "Vin blanc sec générique"
    match:
      wine_type: "white"
    years_min: 1
    years_peak: 3
    years_max: 7
```

#### Évolution vers V2

La V1 est purement basée sur règles. La V2 intégrera :
1. Les corrections utilisateurs comme signal de qualité
2. Un score de millésime par région (ex : Bordeaux 2015 = excellent → +3 ans sur les fenêtres)
3. Optionnellement : fine-tuning d'un modèle léger (regression sur les données collectées en Phase 2)

---

### 5.d Moteur de recommandation met/vin

#### Approche Phase 1 : LLM pur avec contexte cave

```typescript
// Prompt recommandation (Ollama Llama3.2 text)
const buildRecoPrompt = (query: string, cellar: CellarSummary) => `
Tu es un sommelier expert. L'utilisateur a la cave suivante :
${formatCellarForPrompt(cellar)}  // Top 20 bouteilles avec statut apogée

Sa demande : "${query}"

Propose 1 à 3 bouteilles de sa cave. Pour chaque proposition, donne :
- Le nom exact de la bouteille (tel qu'il apparaît dans sa cave)
- La raison du choix (1-2 phrases)
- Le statut d'apogée (idéal / acceptable / un peu jeune)

Si aucune bouteille ne convient parfaitement, dis-le et propose une alternative
pour le prochain achat.

Réponds en JSON :
{
  "recommendations": [{ "bottle_id": "...", "reason": "...", "apogee_status": "..." }],
  "comment": "..."  // commentaire général si pertinent
}
`;
```

#### Filtrage pré-LLM (optimisation coûts et pertinence)

Avant d'envoyer au LLM, filtrer la cave pour ne passer que les bouteilles pertinentes :
1. Statut `in_cellar` et `quantity > 0`
2. Vins dans leur fenêtre d'apogée (ou proches, ±2 ans)
3. Tri par pertinence potentielle (rouge → plats viande, blanc → poisson/fromage)
4. Limiter à 20-30 bouteilles max dans le prompt

---

## 6. Stratégie données et conformité

### 6.1 LWIN — Licence et modalités

LWIN est distribué sous **CC BY-SA 4.0** :
- Attribution obligatoire dans l'app ("Base de données LWIN © Liv-ex")
- Redistribution sous la même licence si l'app redistribue les données LWIN
- L'enrichissement propriétaire (corrections utilisateurs, apogées) peut être licencié séparément

**Action** : Mentionner LWIN dans les crédits de l'app et les CGU. Vérifier avec un avocat si le SQLite bundlé constitue une "distribution" au sens CC BY-SA.

### 6.2 Données propriétaires

Les données générées par les utilisateurs (corrections, notes de dégustation, positions cave) sont la propriété de l'utilisateur selon les CGU. Cave IA peut les utiliser de manière anonymisée et agrégée pour améliorer les modèles.

Stratégie de valeur : les corrections agrégées de milliers d'utilisateurs constituent une base propriétaire de matching vin → étiquette, impossible à répliquer par un concurrent.

### 6.3 RGPD — Checklist opérationnelle

| Obligation | Implémentation | Délai |
|-----------|---------------|-------|
| Consentement explicite | Écran onboarding avec CGU | Phase 1 |
| Droit d'accès | Export JSON de toutes les données user | Phase 1 |
| Droit à l'oubli | `DELETE /api/account` — soft delete 30j | Phase 1 |
| Portabilité | Export CSV cave + notes | Phase 2 |
| DPO | Porteur du projet (TPE < 250 employés, DPO non obligatoire) | Avant lancement |
| Registre des traitements | Document interne à tenir à jour | Avant lancement |
| Hébergement EU | Supabase Frankfurt (eu-central-1) | Phase 1 |
| Photos étiquettes | Stockées sur Supabase Storage EU, accessibles uniquement par le user | Phase 1 |
| Logs serveur | Rétention max 30 jours | Phase 1 |

### 6.4 Sécurité

- **Auth** : JWT Supabase, refresh tokens rotatiifs, durée session 30 jours
- **RLS (Row Level Security)** : Politiques PostgreSQL — chaque user ne voit que ses données
- **Photos** : URLs presignées avec expiration 1h (jamais d'URL publique permanente)
- **Secrets** : Variables d'environnement Supabase + `.env` local jamais commité (`.gitignore`)
- **API rate limiting** : Supabase Edge Functions limitées à 100 req/min/user
- **HTTPS** : Obligatoire partout (Supabase impose TLS)
- **Chiffrement at rest** : Supabase chiffre les données PostgreSQL et Storage par défaut

---

## 7. Tests et qualité

### 7.1 Stratégie de tests

```
Pyramide des tests Cave IA

        ┌───────┐
        │  E2E  │  Detox (iOS simulateur) — 5-10 parcours critiques
        └───┬───┘
       ┌────┴────┐
       │  Intég. │  Jest + Supertest — API routes, pipeline scan
       └────┬────┘
      ┌─────┴──────┐
      │  Unitaires │  Jest — parsers, matching, règles apogée, prompts
      └────────────┘
```

**Unitaires (Jest)**
- `matchWine()` : 50+ cas de test couvrant erreurs OCR typiques
- Règles YAML apogée : chaque règle testée sur 2-3 millésimes exemple
- Normalisation texte OCR : caractères spéciaux, accents, encodages
- Formatage prompts LLM : vérification que le JSON template est valide

**Intégration**
- Pipeline complet OCR → LWIN → résultat (avec mock Ollama)
- CRUD API cave (Supabase local avec `supabase start`)
- Authentification (magic link simulé)

**E2E Detox**
- Parcours 1 : Inscription → scan bouteille → validation → vue cave
- Parcours 2 : "Quel vin ce soir ?" → recommandation → feedback
- Parcours 3 : Ajouter note → historique dégustations
- Parcours 4 : Alerte apogée → ouvrir bouteille → consommer

### 7.2 Évaluation précision OCR + LLM

**Jeu de test étiquettes** (à constituer en Phase 0)
- 100 étiquettes photographiées avec un iPhone 13+ (conditions réelles)
- Mix : 40% Bordeaux/Bourgogne classiques, 30% petits producteurs français, 20% Italie/Espagne, 10% étiquettes difficiles (artisanales, vieilles, rétro-éclairées)
- Annotation manuelle : `{lwin_id, producer, name, vintage}` pour chaque étiquette

**Métriques**

```typescript
interface OcrEvalMetrics {
  // Reconnaissance complète (tous champs corrects)
  full_match_rate: number;          // cible Phase 0: >= 0.70
  // Reconnaissance partielle (millésime + producteur correct)
  partial_match_rate: number;       // cible Phase 0: >= 0.85
  // Temps moyen pipeline (OCR + matching, hors LLM)
  avg_latency_ms: number;           // cible: <= 1500ms
  // Précision LLM sur les cas rejetés par LWIN
  llm_correct_rate: number;         // cible: >= 0.60 sur rejetés
  // Taux de faux positifs (mauvais vin avec haute confiance)
  false_positive_rate: number;      // cible: <= 0.05
}
```

Script d'évaluation automatisé à lancer avec `bun run eval:ocr --dataset ./data/eval-100.json`.

### 7.3 CI/CD

```yaml
# GitHub Actions — .github/workflows/ci.yml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v1
      - run: bun install
      - run: bun test           # unitaires + intégration
      - run: bun run type-check # tsc --noEmit

  build-ios:
    runs-on: macos-latest
    needs: test
    steps:
      - uses: expo/expo-github-action@v8
      - run: eas build --platform ios --profile preview --non-interactive
```

**Déploiement**
- Preview builds : EAS Build sur chaque PR → lien TestFlight automatique
- Production builds : tag `v*.*.*` → EAS Submit → App Store Connect

---

## 8. Estimation budgétaire et ressources

### 8.1 Coûts fixes mensuels (Phase 0-1)

| Service | Tier | Coût/mois | Notes |
|---------|------|-----------|-------|
| Supabase | Free (puis Pro) | 0 € → 25 €/mois | Passage Pro si > 500 MB DB ou 2 GB storage |
| Expo / EAS | Free (puis Production) | 0 € → 99 $/an | EAS Build : 30 builds/mois gratuits |
| Apple Developer | Annuel | ~99 $/an (~8 €/mois) | Obligatoire pour TestFlight + App Store |
| Ollama NAS | Infra existante | ~0 € | Électricité uniquement (~2-5 €/mois) |
| Domaine + email | OVH / Porkbun | ~3 €/mois | `caveai.app` ou équivalent |
| **Total Phase 0** | | **~11 €/mois** | |
| **Total Phase 1** | | **~30-40 €/mois** | Avec Supabase Pro |

### 8.2 Coûts variables — Estimation à l'échelle

| Scénario | Utilisateurs actifs | Scans/mois | Coût LLM | Coût total |
|----------|--------------------|-----------|-----------| -----------|
| Phase 1 (beta) | 20 | ~200 | 0 € (NAS) | ~35 €/mois |
| Phase 2 (100 users) | 100 | ~1 000 | 0 € (NAS) ou ~10 € (Mistral API) | ~50 €/mois |
| Scale (1 000 users) | 1 000 | ~10 000 | ~100 € (API) | ~200 €/mois |

**Point de bascule NAS → API LLM** : quand le NAS devient un SPOF avec > 50 users actifs ou quand la latence dépasse 5s régulièrement.

### 8.3 Équipe Phase 1

| Rôle | Temps | Notes |
|------|-------|-------|
| Dev solo (toi) | Full time ou 60%+ | Mobile + backend + IA |
| Beta testeurs | 10-20 personnes | Réseau personnel, bénévoles |
| Sommelier conseil | Ponctuel (~2h) | Valider les règles d'apogée, idéalement un ami caviste |

---

## 9. Risques et plans de mitigation

| # | Risque | Niveau | Plan de mitigation |
|---|--------|--------|-------------------|
| **R1** | OCR insuffisant sur étiquettes artisanales, vieilles, mal éclairées | Élevé | Guides photo in-app (éclairage recommandé) + fallback manuel toujours disponible + collecte active des cas d'échec pour amélioration |
| **R2** | NAS Ollama indisponible (panne, réseau, surcharge) | Élevé | Timeout 8s strict + fallback "reconnaissance différée" : bouteille ajoutée manuellement, LLM tenté en background quand NAS revient. Prévoir migration API dès Phase 2 |
| **R3** | LWIN incomplet (vins nature, micro-domaines, étiquettes non standardisées) | Moyen | Accepter le taux d'échec, enrichir via corrections utilisateurs. Documenter le taux de couverture attendu (~85% sur vins commerciaux, ~50% sur vins de garage) |
| **R4** | Latence pipeline > 3s → abandon de l'utilisateur | Élevé | OCR on-device (< 500ms) + matching SQLite local (< 200ms) = résultat partiel affiché immédiatement. LLM en async non bloquant |
| **R5** | Qualité fenêtres d'apogée insuffisante → perte de confiance | Moyen | V1 heuristique avec incertitude explicite ("fourchette estimée"). Afficher la source ("estimation basée sur le cépage et la région"). Ne pas surprommettre |
| **R6** | Dev solo — épuisement ou arrêt | Élevé | Documentation technique à jour dès Phase 0. Code structuré pour onboarding d'un deuxième dev. Backlog Notion public si besoin de déléguer |
| **R7** | Violation licence LWIN lors de la redistribution SQLite | Moyen | Consulter un avocat avant le lancement public. Prévoir un fallback avec base maison si nécessaire. Attribution visible dans l'app |
| **R8** | Règlement App Store refus (IA, vie privée photos) | Moyen | Préparer Privacy Nutrition Labels. Photos traitées on-device, pas de visage, pas de biométrie. Data usage disclosure dans App Store Connect |
| **R9** | Coût passage à l'échelle LLM API sous-estimé | Faible Phase 1 | Surveiller le ratio "scans LLM / scans total" dès la beta. Si > 40% des scans passent par LLM, optimiser le matching LWIN avant de scaler |
| **R10** | Utilisateurs saisissent des données sensibles dans les notes | Faible | CGU claires. Ne pas traîner les notes dans les prompts LLM sans anonymisation. Chiffrement at rest via Supabase |

---

## 10. Questions ouvertes pour le porteur

Ces décisions ne peuvent pas être tranchées sans ton arbitrage :

1. **Supabase vs infra custom** : L'approche Supabase (tout-en-un, managed, EU) est fortement recommandée pour un dev solo. Es-tu à l'aise avec cette dépendance ? Si tu as déjà une infra VPS EU ou des préférences (Railway, Fly.io), cela change le plan backend.

2. **Oligopole NAS → production** : L'Ollama sur NAS est excellent pour la Phase 0 mais fragile pour la Phase 1 beta. As-tu envisagé un VPS GPU EU dès la Phase 1 (ex. Scaleway GPU ~150€/mois) ou préfères-tu passer directement à une API LLM payante (Mistral, Claude) dès que le budget le permet ?

3. **Modèle d'apogée : partenariat ou from-scratch ?** : As-tu des contacts dans le monde du vin (cavistes, sommeliers, critiques) qui pourraient valider ou co-construire les règles d'apogée ? Un partenariat avec un guide régional français (ex. La Revue du Vin de France) changerait fondamentalement la qualité perçue de l'app — mais ajoute une dépendance et un délai.

4. **Import CellarTracker dès le MVP ?** : Un utilisateur avec 500 bouteilles dans CellarTracker ne re-saisira pas manuellement. L'import CSV est en Phase 2 dans le plan — est-ce acceptable pour tes beta users Phase 1, ou faut-il l'avancer ?

5. **Monétisation freemium ou payant direct ?** : Le plan suppose un freemium (cave ≤ 50 bouteilles gratuit, illimité en payant). Veux-tu tester un modèle payant dès la beta (même à 1 €) pour valider la willingness to pay, ou rester gratuit jusqu'au lancement App Store ?

6. **Android Phase 2 : React Native ou natif ?** : Le choix React Native/Expo est fait pour iOS-first. Quand Android arrivera, le plan suppose de réutiliser ~80% du code avec Google ML Kit pour l'OCR. Es-tu à l'aise avec ce niveau de partage de code ou préfères-tu anticiper un fork natif Android ?

7. **Nom de domaine et identité de marque** : "Cave IA" est un nom de travail. As-tu déjà une identité de marque ou un nom de domaine enregistré ? Cela n'impacte pas le technique mais influe sur le naming des packages et du bundle identifier iOS.

---

## Annexe — Arborescence codebase suggérée

```
cave-ia/
├── apps/
│   └── mobile/                    # Expo React Native
│       ├── app/                   # Expo Router (file-based)
│       │   ├── (auth)/
│       │   │   ├── login.tsx
│       │   │   └── onboarding.tsx
│       │   ├── (tabs)/
│       │   │   ├── cellar/
│       │   │   │   ├── index.tsx  # liste cave
│       │   │   │   └── [id].tsx   # détail bouteille
│       │   │   ├── scan.tsx
│       │   │   ├── reco.tsx
│       │   │   └── notes.tsx
│       │   └── _layout.tsx
│       ├── modules/
│       │   └── ocr/               # module natif Expo Vision Framework
│       │       ├── index.ts
│       │       ├── OcrModule.ios.ts
│       │       └── OcrModule.android.ts  # ML Kit — Phase 2
│       ├── lib/
│       │   ├── lwin/              # client SQLite LWIN
│       │   ├── matching/          # algorithme fuzzy
│       │   ├── apogee/            # moteur règles YAML
│       │   └── supabase/          # client Supabase
│       └── stores/                # Zustand stores
│
├── supabase/
│   ├── migrations/                # SQL migrations
│   ├── functions/                 # Edge Functions
│   │   ├── scan-match/            # matching LWIN server-side
│   │   ├── reco/                  # recommandations LLM
│   │   └── apogee-batch/          # cron alertes apogée
│   └── seed/                      # données de test
│
├── scripts/
│   ├── ingest-lwin.ts             # import CSV LWIN
│   ├── build-sqlite.ts            # génération bundle mobile
│   └── eval-ocr.ts                # évaluation précision OCR
│
├── data/
│   ├── lwin.csv                   # source LWIN (gitignore si lourd)
│   ├── apogee_rules.yaml          # règles d'apogée
│   └── eval-100.json              # jeu d'évaluation annoté
│
└── docs/
    ├── PLAN.md                    # ce document
    └── ADR/                       # Architecture Decision Records
```

---

*Ce plan sera mis à jour après chaque phase. Les décisions architecturales importantes doivent être documentées dans `docs/ADR/`.*
