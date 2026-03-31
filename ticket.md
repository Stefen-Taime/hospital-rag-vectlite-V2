# VectLite — Observations & Feedback pour les développeurs

**Projet**: RAG Hôpitaux Montréal  
**Version VectLite testée**: 0.1.3  
**Date**: 2026-03-31  

---

## 1. Documentation API — Paramètre `sparse` non documenté clairement

**Sévérité**: Haute  
**Contexte**: Lors de l'implémentation de la recherche hybride (dense + sparse/BM25).

La documentation du site montre un exemple avec `sparse=vectlite.sparse_terms("auth guide")` dans `search()`, mais ne documente pas clairement :
- Que `upsert()` accepte aussi un paramètre `sparse`
- Que `sparse` attend un `dict[str, float]` (retour de `sparse_terms()`) et **non** un `str` directement
- Le message d'erreur `'str' object cannot be converted to 'PyDict'` est peu explicite

**Suggestion**: 
- Documenter explicitement que `sparse` attend le résultat de `sparse_terms()` dans `upsert()` et `search()`
- Améliorer le message d'erreur: `"sparse parameter expects dict[str, float] from sparse_terms(), got str"`

---

## 2. Ancien modèle `text-embedding-004` référencé dans les exemples

**Sévérité**: Moyenne  
**Contexte**: Les exemples de la documentation utilisent `dimension=384` qui correspond à des modèles anciens.

Google a déprécié `text-embedding-004` et le nouveau modèle `gemini-embedding-001` produit des vecteurs de dimension 3072. Les exemples de la doc VectLite devraient être mis à jour.

---

## 3. Erreur `dimension mismatch` peu guidante

**Sévérité**: Basse  
**Contexte**: `vectlite.VectLiteError: vector dimension mismatch: expected 768, found 3072`

Ce message apparaît quand on ouvre un fichier `.vdb` existant avec une dimension différente. L'erreur est correcte mais pourrait inclure un conseil:
- "Delete the existing .vdb file or use a different path to create a new database with the new dimension."

---

## 4. SDK deprecated `google.generativeai` vs `google.genai`

**Sévérité**: Info  
**Contexte**: Le site VectLite pourrait mentionner dans ses exemples d'intégration que le SDK `google-generativeai` est deprecated au profit de `google-genai`.

---

## 5. Absence de `db.count()` dans la documentation

**Sévérité**: Basse  
**Contexte**: La méthode `db.count()` existe et fonctionne mais n'est pas documentée sur le site. Idem pour `db.list()`.

---

## 6. Paramètre `sparse_text` vs `sparse` — confusion dans la doc web

**Sévérité**: Haute  
**Contexte**: La page principale du site mentionne "sparse BM25 keyword retrieval" mais les exemples de code ne sont pas cohérents avec la signature réelle de l'API Python.

- La doc web laisse croire qu'on peut passer du texte brut
- L'API Python requiert `vectlite.sparse_terms(text)` pour convertir en dict
- Il serait plus ergonomique que `upsert()` et `search()` acceptent directement un `str` pour le sparse et appellent `sparse_terms()` en interne

---

## 7. Performance `upsert()` avec `sparse` — dégradation O(n²)

**Sévérité**: Haute (contournable via `bulk_ingest`, voir #8)  
**Contexte**: Ingestion de ~11 000 documents (vecteurs 1536-dim, `text-embedding-3-small` OpenAI).

### Mesures observées avec `upsert()` en boucle (même dans une transaction)

| Batch (200 docs chacun) | Temps |
|--------------------------|-------|
| Batch 1 (0-200) | 13s |
| Batch 2 (200-400) | 80s (6x plus lent) |
| Batch 3 (400-600) | 175s (13x plus lent) |
| Batch 4 (600-800) | 285s (22x plus lent) |
| Estimation 11 058 docs | **> 2 heures** |

La dégradation est **non-linéaire**, ce qui suggère une reconstruction d'index sparse à chaque transaction.

### Suggestion

- Investiguer pourquoi `upsert()` avec `sparse` dégrade en O(n²)
- Documenter clairement que pour l'ingestion en volume, il faut utiliser `bulk_ingest()` et non `upsert()` en boucle

---

## 8. `bulk_ingest()` existe mais n'est PAS DOCUMENTÉ — c'est la solution

**Sévérité**: **CRITIQUE (documentation)**  
**Contexte**: Après avoir passé des heures à debugger la lenteur de `upsert()`, nous avons découvert `db.bulk_ingest()` via `dir(vectlite.Database)`.

### Résultat avec `bulk_ingest()`

| Méthode | 11 058 docs avec sparse | Ratio |
|---------|------------------------|-------|
| `upsert()` en boucle + transaction | **> 2 heures** (estimé) | 1x |
| `bulk_ingest(records)` | **22 secondes** | **~300x plus rapide** |

### Le problème

**`bulk_ingest()` n'apparaît nulle part** dans la documentation du site, ni dans les exemples, ni dans le README GitHub. C'est pourtant LA méthode à utiliser pour l'ingestion initiale. Un utilisateur qui suit la doc va naturellement écrire une boucle `upsert()` et conclure que VectLite est inutilisable pour des datasets > 1000 docs.

### Aussi non documentés

Méthodes trouvées via `dir()` mais absentes de la doc :
- `bulk_ingest(records, namespace=None, batch_size=10000)`
- `upsert_many(records, namespace=None)`
- `insert_many(records, namespace=None)`
- `compact()`
- `flush()`
- `snapshot()`
- `backup()`
- `delete_many()`
- `search_with_stats()`

### Suggestion

- **Priorité 1** : Documenter `bulk_ingest()` en page d'accueil comme méthode recommandée pour l'ingestion
- Documenter le format attendu pour `records` : `list[dict]` avec clés `id`, `vector`, `metadata`, `sparse`
- Ajouter un guide "Getting Started" qui utilise `bulk_ingest()` dès le départ

---

## 9. Bonne stabilité malgré la lenteur

**Sévérité**: Info (positif)  
**Contexte**: Malgré la lenteur, VectLite ne crash pas, ne corrompt pas les données, et les transactions par batch sont stables. Le fichier `.vdb` unique reste très pratique pour le déploiement et la portabilité.

---

## Résumé

| # | Observation | Sévérité |
|---|------------|----------|
| 1 | Paramètre `sparse` non documenté pour `upsert()` | Haute |
| 2 | Exemples avec modèle embedding obsolète | Moyenne |
| 3 | Message d'erreur dimension mismatch peu guidant | Basse |
| 4 | Référence au SDK Google deprecated | Info |
| 5 | `db.count()` et `db.list()` non documentés | Basse |
| 6 | Confusion `sparse_text` vs `sparse` dans la doc | Haute |
| 7 | Performance `upsert()` avec sparse — dégradation O(n²) | Haute |
| 8 | **`bulk_ingest()` non documenté — solution 300x plus rapide** | **CRITIQUE** |
| 9 | Nombreuses méthodes non documentées (upsert_many, compact, etc.) | Haute |
| 10 | Bonne stabilité des transactions (positif) | Info |
