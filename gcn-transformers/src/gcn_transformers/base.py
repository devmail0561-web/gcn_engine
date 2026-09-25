# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Classe de base abstraite pour encodeurs Transformer (gcn-transformers).

Cette classe factorise la logique PyTorch↔NumPy pour les encodeurs basés sur
des modèles Transformer pré-entraînés (CamemBERT, XLM-RoBERTa, CodeBERT).

Implémente le Protocol CausalEncoder de gcn-python pour compatibilité complète
avec CGNPipeline.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn


class TransformerEncoderBase:
    """
    Classe de base abstraite pour encodeurs Transformer → CausalEncoder Protocol.

    Factorise :
    - Conversion NumPy ↔ PyTorch avec requires_grad
    - backward_node_dx et backward_edge_dx via torch.autograd
    - Cache snapshots avec graphe autograd préservé
    - Optimizer AdamW unique pour node+edge (évite double step)
    - Gestion eval()/train() pour dropout

    Les sous-classes concrètes (XLMRobertaEncoder, CamembertEncoder, CodeBERTEncoder)
    doivent instancier self.model et self.tokenizer AVANT d'appeler super().__init__().
    """

    def __init__(
        self,
        d_clause: int,
        d_edge: int,
        freeze_layers: int = 10,
        learning_rate: float = 1e-5,
        device: str | None = None,
        n_node_types: int = 7,
        n_relation_types: int = 11,
    ):
        """
        Initialise l'encodeur Transformer.

        IMPORTANT : Les sous-classes doivent instancier self.model et self.tokenizer
        AVANT d'appeler super().__init__(), car __init__ accède à self.model.config.

        Args:
            d_clause: Dimension features UD (79 par défaut = vocab.d_clause_effective(0, False))
            d_edge: Dimension edge vectors (365 par défaut = vocab.d_edge_closed_loop(...))
            freeze_layers: Nombre de couches Transformer à geler (10/12 par défaut)
            learning_rate: Learning rate AdamW (1e-5 recommandé pour Transformers)
            device: "cuda", "cpu", ou None (auto-détection)
            n_node_types: Nombre de types de nœuds (7 par défaut : NODE_TYPES)
            n_relation_types: Nombre de relations causales (11 par défaut : RELATION_TYPES)

        Raises:
            RuntimeError: Si self.model n'existe pas (sous-classe doit l'instancier avant)
        """
        self.d_clause = d_clause
        self.d_edge = d_edge
        if d_clause == 79:
            warnings.warn(
                f"d_clause=79 est la valeur de base sans embedding. "
                "Avec --embedding-dim > 0, passez vocab.d_clause_effective(d_emb, False) "
                "(ex. d_clause=207 pour d_emb=128). Le modèle entraîné avec un autre d_clause "
                "crashera au chargement (shape mismatch proj_ud).",
                UserWarning, stacklevel=3,
            )
        self.freeze_layers = freeze_layers
        self.training = True
        # CRITIQUE : True pour activer l'auto-attention inter-clauses au forward.
        # Trade-off accepté : backward rejoue forward_node() individuel (approximation),
        # mais le gain d'attention au forward inference >> perte précision backward.
        # Alternative False → Transformer équivaut à un MLP de 280M params (code mort).
        self.prefers_batch_forward = True
        self.n_node_types = n_node_types
        self.n_relation_types = n_relation_types

        # Device auto-détecté
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Vérifier que self.model existe (instancié par sous-classe)
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}.__init__() : self.model doit être instancié "
                "par la sous-classe AVANT d'appeler super().__init__()"
            )

        # Vérifier que model.config.hidden_size existe (structure HuggingFace attendue)
        if not hasattr(self.model, 'config'):
            raise ValueError(
                f"{type(self).__name__}.__init__() : self.model doit avoir un attribut 'config' "
                "(structure HuggingFace attendue). Modèle fourni : {type(self.model).__name__}"
            )
        if not hasattr(self.model.config, 'hidden_size'):
            raise ValueError(
                f"{type(self).__name__}.__init__() : self.model.config doit avoir 'hidden_size'. "
                f"Config trouvé : {dir(self.model.config)}"
            )

        # Déplacer le modèle sur le device
        self.model = self.model.to(self._device)

        # Projection UD → hidden_size (persistant, optimisable)
        # Option A : projection des features syntaxiques UD
        self.proj_ud = nn.Linear(d_clause, self.model.config.hidden_size).to(self._device)

        # Têtes de classification (node/edge)
        self._node_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_node_types),
        ).to(self._device)

        # CORRECTION : edge head reçoit d_edge (365), PAS hidden*2
        self._edge_head = nn.Sequential(
            nn.Linear(d_edge, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_relation_types),
        ).to(self._device)

        # Geler les layers bas du Transformer
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            n_layers = len(self.model.encoder.layer)
            if n_layers == 0:
                warnings.warn(
                    f"{type(self).__name__} : model.encoder.layer est vide, "
                    f"impossible de geler {freeze_layers} couches.",
                    UserWarning
                )
            elif freeze_layers > n_layers:
                warnings.warn(
                    f"{type(self).__name__} : freeze_layers={freeze_layers} > {n_layers} couches disponibles. "
                    f"Gel de toutes les {n_layers} couches.",
                    UserWarning
                )
            actual_frozen = min(freeze_layers, n_layers)
            for layer_idx in range(actual_frozen):
                for param in self.model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
        else:
            if freeze_layers > 0:
                warnings.warn(
                    f"{type(self).__name__} : model ne suit pas la structure BERT/RoBERTa "
                    f"(model.encoder.layer attendu). Aucune couche gelée. "
                    f"Implémentez le gel manuellement dans la sous-classe si nécessaire.",
                    UserWarning
                )

        # Optimizer AdamW UNIQUE pour node+edge (évite double step)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params += list(self.proj_ud.parameters())
        trainable_params += list(self._node_head.parameters())
        trainable_params += list(self._edge_head.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Cache pour backward_node_dx et backward_edge_dx
        self._cached_input_tensor: torch.Tensor | None = None
        self._cached_output_logits: torch.Tensor | None = None
        self._cached_edge_input_tensor: torch.Tensor | None = None
        self._cached_edge_output_logits: torch.Tensor | None = None

        # Flag pour warning lr (émettre UNE SEULE FOIS)
        self._lr_warning_emitted = False

        # Flag pour warning contrat gradients (émettre UNE SEULE FOIS)
        self._grad_contract_warned = False

        # Flag pour tracker si zero_grad() est nécessaire après update
        self._needs_zero_grad = False

    def forward_node(self, x: np.ndarray) -> np.ndarray:
        """
        Single node forward — construit le graphe autograd même en training=False.

        Args:
            x: Vecteur UD clause (d_clause,)

        Returns:
            Logits node types (n_node_types,)
        """
        return self.forward_batch(x[np.newaxis, :])[0]

    def forward_batch(self, X: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
        """
        Batch forward (N, d_clause) → (N, n_node_types) node logits.

        Option A (par défaut) : Projection UD → hidden_size (agnosticisme langue)
        Option B (si texts fourni) : Tokenization texte → embeddings pré-entraînés

        IMPORTANT : PAS de torch.no_grad() — le graphe doit être construit toujours
        pour backward_node_dx, même si training=False (le pipeline désactive training
        avant forward puis le réactive).

        Args:
            X: Batch vecteurs UD (N, d_clause)
            texts: Textes bruts optionnels pour tokenization (Option B, future)

        Returns:
            Logits node types (N, n_node_types)
        """
        # Safety : zero_grad() si update_edge() n'a pas été appelé après update_node()
        if self._needs_zero_grad:
            self.optimizer.zero_grad()
            self._needs_zero_grad = False

        # Safety : vérifier batch non vide
        if X.shape[0] == 0:
            # Retourner logits vides (0, n_node_types)
            return np.zeros((0, self.n_node_types), dtype=np.float32)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        X_t.requires_grad_(True)  # TOUJOURS pour backward_node_dx

        if texts is not None:
            # Option B : tokenization réelle (recommandé, future v1.1)
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            H = outputs.last_hidden_state[:, 0, :]  # [CLS] token (N, hidden_size)
        else:
            # Option A : projection UD (fallback, v1.0)
            H = self.proj_ud(X_t)  # (N, hidden_size)

            # Forward Transformer (self-attention sur N nœuds)
            # PAS de torch.no_grad() — graphe requis
            outputs = self.model(inputs_embeds=H.unsqueeze(0))  # (1, N, hidden_size)
            H_enriched = outputs.last_hidden_state.squeeze(0)  # (N, hidden_size)
            H = H_enriched

        logits = self._node_head(H)  # (N, n_node_types)

        # Cache pour backward
        self._cached_input_tensor = X_t
        self._cached_output_logits = logits

        return logits.detach().cpu().numpy()

    def backward_node_dx(self, d_logits: np.ndarray) -> tuple[list[tuple], np.ndarray]:
        """
        Backward avec autograd PyTorch — retourne (grads, dx).

        Stratégie :
        1. Snapshot des .grad des têtes AVANT backward
        2. Backward autograd PyTorch (accumule .grad sur les paramètres)
        3. Extraction du DELTA (.grad - snapshot) au format gcn-python
        4. Gradient vers l'entrée dx (pour R-GCN)

        CONTRAT : le retour est la contribution de CET appel, pas le cumul.
        Le pipeline somme les retours sur N nœuds (cgnp.py:883-890) : retourner
        le cumul ferait compter les nœuds précédents 2, 3, ... fois (double
        comptage vérifié expérimentalement sur XLM-RoBERTa : 7701 au lieu de
        3802 pour 3 nœuds).

        IMPORTANT : NE PAS appeler zero_grad() ici — le pipeline accumule les
        gradients sur N nœuds puis appelle update_node() (step) /
        update_edge() (zero_grad).

        Args:
            d_logits: Gradients depuis la loss ((n_node_types,) ou (N, n_node_types))

        Returns:
            grads: Contribution node_head de cet appel au format [(dW, db), ...]
            dx: Gradient vers l'entrée (d_clause,) pour R-GCN

        Raises:
            RuntimeError: Si appelé sans forward préalable
        """
        if self._cached_output_logits is None:
            raise RuntimeError("backward_node_dx appelé sans forward_batch préalable")

        prev_grads = self._snapshot_head_grads(self._node_head)

        # Backward autograd
        d_logits_t = torch.as_tensor(d_logits, dtype=torch.float32, device=self._device)

        # Ajuster shape si nécessaire : si d_logits est (7,) mais output est (1, 7)
        if d_logits_t.dim() == 1 and self._cached_output_logits.dim() == 2:
            if self._cached_output_logits.shape[0] == 1:
                d_logits_t = d_logits_t.unsqueeze(0)  # (7,) → (1, 7)

        self._cached_output_logits.backward(d_logits_t, retain_graph=False)

        # Gradient vers l'entrée (pour R-GCN)
        if self._cached_input_tensor.grad is not None:
            dx = self._cached_input_tensor.grad.cpu().numpy().copy()
            # Si batch=1, squeeze pour retourner (d_clause,) au lieu de (1, d_clause)
            if dx.shape[0] == 1 and dx.ndim == 2:
                dx = dx[0]
        else:
            dx_shape = self._cached_input_tensor.shape
            if dx_shape[0] == 1 and len(dx_shape) == 2:
                dx = np.zeros(dx_shape[1], dtype=np.float32)
            else:
                dx = np.zeros(dx_shape, dtype=np.float32)

        # Extraction du delta node_head au format gcn-python
        grads = self._head_grad_deltas(self._node_head, prev_grads)

        # Clear cache (mais PAS zero_grad — fait dans update_node)
        self._cached_input_tensor = None
        self._cached_output_logits = None

        return grads, dx

    @staticmethod
    def _snapshot_head_grads(module: nn.Module) -> dict[int, tuple]:
        """Snapshot des .grad des couches Linear de `module` (clé = id(layer))."""
        snap: dict[int, tuple] = {}
        for layer in module:
            if isinstance(layer, nn.Linear):
                snap[id(layer)] = (
                    layer.weight.grad.clone() if layer.weight.grad is not None else None,
                    layer.bias.grad.clone()
                    if (layer.bias is not None and layer.bias.grad is not None)
                    else None,
                )
        return snap

    @staticmethod
    def _head_grad_deltas(
        module: nn.Module,
        prev: dict[int, tuple],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Delta (.grad - snapshot) des couches Linear, format gcn-python.

        Un grad manquant (None) est traité comme zéro — pas de crash (bug #7).
        """
        grads: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in module:
            if not isinstance(layer, nn.Linear):
                continue
            prev_w, prev_b = prev.get(id(layer), (None, None))
            w = layer.weight.grad
            b = layer.bias.grad if layer.bias is not None else None
            if w is None:
                dW = np.zeros_like(layer.weight.detach().cpu().numpy())
            elif prev_w is None:
                dW = w.cpu().numpy().copy()
            else:
                dW = (w - prev_w).cpu().numpy().copy()
            if layer.bias is None:
                db = np.zeros((0,), dtype=np.float32)
            elif b is None:
                db = np.zeros_like(layer.bias.detach().cpu().numpy())
            elif prev_b is None:
                db = b.cpu().numpy().copy()
            else:
                db = (b - prev_b).cpu().numpy().copy()
            grads.append((dW, db))
        return grads

    def forward_edge(self, x: np.ndarray) -> np.ndarray:
        """
        Edge forward (d_edge,) → (n_relation_types,) relation logits.

        Args:
            x: Vecteur edge (d_edge,) = features + enriched_src + enriched_dst

        Returns:
            Logits relations (n_relation_types,)
        """
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
        x_t.requires_grad_(True)  # Pour backward_edge_dx

        logits = self._edge_head(x_t)

        # Cache pour backward
        self._cached_edge_input_tensor = x_t
        self._cached_edge_output_logits = logits

        return logits.detach().cpu().numpy()

    def backward_edge_dx(self, d_logits: np.ndarray) -> tuple[list[tuple], np.ndarray]:
        """
        Backward edge — similaire à backward_node_dx (retour = DELTA de cet appel).

        Args:
            d_logits: Gradients depuis la loss (n_relation_types,)

        Returns:
            grads: Contribution edge_head de cet appel au format [(dW, db), ...]
            dx: Gradient vers l'entrée (d_edge,)

        Raises:
            RuntimeError: Si appelé sans forward_edge préalable
        """
        if self._cached_edge_output_logits is None:
            raise RuntimeError("backward_edge_dx appelé sans forward_edge préalable")

        prev_grads = self._snapshot_head_grads(self._edge_head)

        # Backward autograd
        d_logits_t = torch.as_tensor(d_logits, dtype=torch.float32, device=self._device)
        self._cached_edge_output_logits.backward(d_logits_t, retain_graph=False)

        # Gradient vers l'entrée
        if self._cached_edge_input_tensor.grad is not None:
            dx = self._cached_edge_input_tensor.grad.cpu().numpy().copy()
        else:
            dx = np.zeros(self._cached_edge_input_tensor.shape, dtype=np.float32)

        # Extraction du delta edge_head
        grads = self._head_grad_deltas(self._edge_head, prev_grads)

        # Clear cache
        self._cached_edge_input_tensor = None
        self._cached_edge_output_logits = None

        return grads, dx

    def parameters(self) -> list[np.ndarray]:
        """
        Retourne tous les paramètres trainables au format NumPy (pour checkpoint).

        Filtrage cohérent : UNIQUEMENT les paramètres avec requires_grad=True sont inclus.
        Cela garantit que le checkpoint reflète exactement ce qui est entraîné.

        Ordre : Transformer (non gelé) + proj_ud + node_head + edge_head

        Returns:
            Liste de tableaux NumPy (poids et biais trainables uniquement)
        """
        params = []

        # Transformer layers (non gelés seulement)
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p.detach().cpu().numpy())

        # Projection UD (si non gelé)
        if self.proj_ud.weight.requires_grad:
            params.append(self.proj_ud.weight.detach().cpu().numpy())
        if self.proj_ud.bias.requires_grad:
            params.append(self.proj_ud.bias.detach().cpu().numpy())

        # Têtes de classification (si non gelées)
        for head in [self._node_head, self._edge_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    if layer.weight.requires_grad:
                        params.append(layer.weight.detach().cpu().numpy())
                    if layer.bias.requires_grad:
                        params.append(layer.bias.detach().cpu().numpy())

        return params

    def load_parameters(self, params_list: list[np.ndarray]) -> None:
        """
        Charge les paramètres depuis une liste NumPy (checkpoint).

        CRITIQUE : copie vers les tensors PyTorch, pas vers les copies NumPy.
        Ordre DOIT correspondre à parameters() : Transformer → proj_ud → heads.

        Args:
            params_list: Liste d'arrays NumPy (depuis checkpoint)
        """
        idx = 0

        # Transformer layers (non gelés)
        for p in self.model.parameters():
            if p.requires_grad:
                if idx >= len(params_list):
                    raise ValueError(f"load_parameters: pas assez de paramètres (attendu >= {idx+1})")
                if p.shape != params_list[idx].shape:
                    raise ValueError(
                        f"load_parameters[{idx}]: shape mismatch "
                        f"tensor={p.shape} vs data={params_list[idx].shape}"
                    )
                p.data.copy_(torch.from_numpy(params_list[idx]).to(p.device))
                idx += 1

        # Projection UD
        if self.proj_ud.weight.requires_grad:
            if idx >= len(params_list):
                raise ValueError("load_parameters: pas assez de paramètres pour proj_ud.weight")
            self.proj_ud.weight.data.copy_(torch.from_numpy(params_list[idx]).to(self.proj_ud.weight.device))
            idx += 1
        if self.proj_ud.bias.requires_grad:
            if idx >= len(params_list):
                raise ValueError("load_parameters: pas assez de paramètres pour proj_ud.bias")
            self.proj_ud.bias.data.copy_(torch.from_numpy(params_list[idx]).to(self.proj_ud.bias.device))
            idx += 1

        # Têtes de classification
        for head_name, head in [("node_head", self._node_head), ("edge_head", self._edge_head)]:
            for layer_i, layer in enumerate(head):
                if isinstance(layer, nn.Linear):
                    if layer.weight.requires_grad:
                        if idx >= len(params_list):
                            raise ValueError(f"load_parameters: pas assez de paramètres pour {head_name}[{layer_i}].weight")
                        layer.weight.data.copy_(torch.from_numpy(params_list[idx]).to(layer.weight.device))
                        idx += 1
                    if layer.bias.requires_grad:
                        if idx >= len(params_list):
                            raise ValueError(f"load_parameters: pas assez de paramètres pour {head_name}[{layer_i}].bias")
                        layer.bias.data.copy_(torch.from_numpy(params_list[idx]).to(layer.bias.device))
                        idx += 1

        if idx != len(params_list):
            import warnings
            warnings.warn(
                f"load_parameters: {len(params_list) - idx} paramètres non utilisés "
                f"({idx} chargés sur {len(params_list)})",
                UserWarning
            )

    def update_node(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        """
        Contrat CausalEncoder : applique les gradients fournis puis optimizer.step().

        Le pipeline fournit les gradients node_head NORMALISÉS (somme / n_samples,
        cgnp.py:1224) alors que l'autograd a accumulé la somme brute. Cette méthode :

        1. recopie les (dW, db) fournis dans .grad des couches node_head —
           l'argument `grads` est donc réellement appliqué (contrat) ;
        2. met à l'échelle les AUTRES paramètres (backbone Transformer, proj_ud,
           edge_head) du même facteur, déduit du rapport
           (.grad autograd / gradients fournis) qui vaut exactement n_samples —
           toute la partie torch du réseau partage alors la convention
           "moyenne par échantillon" et plus aucun paramètre n'est mis à jour
           sur une somme non normalisée ;
        3. fait optimizer.step() — le lr du pipeline est ignoré (AdamW propre à
           l'encodeur), warning si différent.

        Le zero_grad() est DIFFÉRÉ à update_edge() : à cet instant le step
        consomme aussi les gradients edge accumulés. Si update_edge() n'est
        jamais appelé, forward_batch() fait le zero_grad() au forward suivant.

        Args:
            grads: Gradients node_head normalisés [(dW, db), ...]
            lr: Learning rate du pipeline (ignoré, warning si différent)
        """
        if not self._lr_warning_emitted:
            pipeline_lr = lr
            adamw_lr = self.optimizer.param_groups[0]['lr']
            if abs(pipeline_lr - adamw_lr) > 1e-6:
                warnings.warn(
                    f"Pipeline lr={pipeline_lr:.2e} != AdamW lr={adamw_lr:.2e}. "
                    "AdamW lr utilisé (comportement normal pour Transformers).",
                    UserWarning
                )
                self._lr_warning_emitted = True

        self._apply_pipeline_grads(self._node_head, grads, role="node")

        self.optimizer.step()
        # NE PAS zero_grad() ici : update_edge() est appelé juste après
        self._needs_zero_grad = True

    def update_edge(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        """
        Contrat CausalEncoder : prend en compte les gradients edge puis nettoie.

        Le optimizer.step() a DÉJÀ eu lieu dans update_node() : à cet appel les
        gradients edge fournis ont été consommés par ce step, avec la même
        échelle que les gradients node. Cette méthode vérifie donc leur
        cohérence avec les .grad effectivement consommés (les gradients fournis
        ne sont pas ignorés : ils servent de contrôle), puis fait zero_grad()
        pour empêcher qu'ils soient réutilisés au prochain backward.

        Args:
            grads: Gradients edge normalisés [(dW, db), ...]
            lr: Learning rate (ignoré)
        """
        self._check_edge_grads(grads)
        if self._needs_zero_grad:
            self.optimizer.zero_grad()
            self._needs_zero_grad = False

    def _warn_grad_contract(self, detail: str) -> None:
        """Émet UNE SEULE FOIS un warning d'incohérence de contrat de gradients."""
        if getattr(self, "_grad_contract_warned", False):
            return
        warnings.warn(
            "Contrat update_node()/update_edge() incomplet — "
            f"{detail}. Les .grad autograd accumulés sont conservés "
            "(comportement inchangé).",
            UserWarning,
            stacklevel=3,
        )
        self._grad_contract_warned = True

    @staticmethod
    def _linear_layers(module: nn.Module) -> list[nn.Linear]:
        return [m for m in module if isinstance(m, nn.Linear)]

    def _apply_pipeline_grads(
        self,
        module: nn.Module,
        grads: list[tuple[np.ndarray, np.ndarray]] | None,
        role: str,
    ) -> float:
        """
        Recopie les gradients (dW, db) du pipeline dans .grad de `module` et
        aligne l'échelle des autres paramètres sur celle-ci.

        Si le format/les shapes ne correspondent pas au contrat, aucun .grad
        n'est écrasé (les valeurs autograd sont conservées) et un warning est
        émis une fois.

        Returns:
            Facteur d'échelle appliqué aux autres paramètres (1.0 = inchangé)
        """
        layers = self._linear_layers(module)
        if not layers:
            return 1.0
        if grads is None or len(grads) != len(layers):
            self._warn_grad_contract(
                f"{role}: {0 if grads is None else len(grads)} gradients fournis "
                f"pour {len(layers)} couches Linear"
            )
            return 1.0

        parsed: list[tuple[nn.Linear, torch.Tensor, torch.Tensor]] = []
        for layer, item in zip(layers, grads):
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                self._warn_grad_contract(
                    f"{role}: entrée de type {type(item).__name__} au lieu de (dW, db)"
                )
                return 1.0
            dW_t = torch.as_tensor(
                np.asarray(item[0]), dtype=torch.float32, device=layer.weight.device
            )
            db_t = torch.as_tensor(
                np.asarray(item[1]), dtype=torch.float32, device=layer.weight.device
            )
            if tuple(dW_t.shape) != tuple(layer.weight.shape):
                self._warn_grad_contract(
                    f"{role}: shape {tuple(dW_t.shape)} != {tuple(layer.weight.shape)}"
                )
                return 1.0
            if layer.bias is None or tuple(db_t.shape) != tuple(layer.bias.shape):
                self._warn_grad_contract(
                    f"{role}: bias shape {tuple(db_t.shape)} incompatible"
                )
                return 1.0
            if not (bool(torch.isfinite(dW_t).all()) and bool(torch.isfinite(db_t).all())):
                self._warn_grad_contract(f"{role}: gradients non finis — .grad inchangés")
                return 1.0
            parsed.append((layer, dW_t, db_t))

        scale = self._pipeline_grad_scale(parsed[0][0], parsed[0][1])
        for layer, dW_t, db_t in parsed:
            layer.weight.grad = dW_t
            layer.bias.grad = db_t

        if scale != 1.0:
            excluded = {id(p) for p in module.parameters()}
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if id(p) in excluded or p.grad is None:
                        continue
                    p.grad = p.grad * (1.0 / scale)
        return scale

    def _pipeline_grad_scale(self, layer: nn.Linear, provided: torch.Tensor) -> float:
        """
        Déduit n_samples = (.grad autograd) / (gradients fournis).

        Hors accumulation le rapport vaut 1 ; avec apply_accumulated_gradients()
        il vaut exactement n_samples (entier >= 1). Tout autre rapport (bruit,
        gradients divergents) est refusé : garde-fou contre une mise à l'échelle
        erronée de l'ensemble du réseau.
        """
        cur = layer.weight.grad
        if cur is None or tuple(cur.shape) != tuple(provided.shape):
            return 1.0
        a = cur.detach().reshape(-1).to(torch.float64).cpu().numpy()
        b = provided.detach().reshape(-1).to(torch.float64).cpu().numpy()
        mask = np.abs(b) > 0
        if int(mask.sum()) < 8:
            return 1.0
        ratios = a[mask] / b[mask]
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size < 8:
            return 1.0
        r = float(np.median(ratios))
        if not np.isfinite(r) or r < 1.0:
            return 1.0
        n = int(round(r))
        if n == 1:
            return 1.0
        if n > 1_000_000 or abs(r - n) > 1e-3 * n:
            self._warn_grad_contract(
                f"rapport .grad/grads_fournis={r:.6g} non entier — normalisation "
                "par n_samples non appliquée aux paramètres hors node_head"
            )
            return 1.0
        return float(n)

    def _check_edge_grads(self, grads: list[tuple[np.ndarray, np.ndarray]] | None) -> None:
        """Contrôle que les gradients edge fournis correspondent aux .grad consommés."""
        if grads is None:
            return
        layers = self._linear_layers(self._edge_head)
        if len(grads) != len(layers):
            self._warn_grad_contract(
                f"edge: {len(grads)} gradients fournis pour {len(layers)} couches Linear"
            )
            return
        for layer, item in zip(layers, grads):
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                self._warn_grad_contract(
                    f"edge: entrée de type {type(item).__name__} au lieu de (dW, db)"
                )
                return
            cur = layer.weight.grad
            if cur is None:
                self._warn_grad_contract(
                    "edge: gradients fournis alors qu'aucun .grad edge n'a été accumulé "
                    "(le step de update_node() a eu lieu sans eux)"
                )
                return
            provided = torch.as_tensor(
                np.asarray(item[0]), dtype=torch.float32, device=cur.device
            )
            if tuple(provided.shape) != tuple(cur.shape):
                self._warn_grad_contract(
                    f"edge: shape {tuple(provided.shape)} != {tuple(cur.shape)}"
                )
                return
            if not bool(torch.allclose(cur, provided, rtol=1e-3, atol=1e-6)):
                self._warn_grad_contract(
                    "edge: gradients fournis incohérents avec les .grad consommés par "
                    "update_node() (normalisation ou double comptage côté pipeline)"
                )
                return

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """
        Compatibilité rétroactive — délègue à optimizer.step().

        ATTENTION : Ne pas appeler update() en plus de update_node()/update_edge().
        Cela causerait un double optimizer.step() dans la même itération.

        Args:
            grads: Gradients (non utilisés)
            lr: Learning rate (ignoré)
        """
        if self._needs_zero_grad:
            # update_node() a déjà été appelé — ne pas step() deux fois !
            warnings.warn(
                "update() appelé après update_node() — double optimizer.step() évité. "
                "N'appelez PAS update() si vous utilisez update_node()/update_edge().",
                UserWarning
            )
            self.optimizer.zero_grad()
            self._needs_zero_grad = False
        else:
            # Cas normal : update() utilisé seul (rétro-compatible)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def snapshot_node_cache(self) -> dict:
        """
        Snapshot avec graphe autograd préservé (node).

        Returns:
            Cache contenant input_tensor et output_logits avec .grad_fn
        """
        return {
            'input_tensor': self._cached_input_tensor,
            'output_logits': self._cached_output_logits,
        }

    def restore_node_cache(self, snapshot: dict) -> None:
        """
        Restaure le cache node.

        Args:
            snapshot: Cache depuis snapshot_node_cache()
        """
        self._cached_input_tensor = snapshot['input_tensor']
        self._cached_output_logits = snapshot['output_logits']

    def snapshot_edge_cache(self) -> dict:
        """
        Snapshot edge avec graphe autograd préservé.

        Returns:
            Cache contenant input_tensor et output_logits edge
        """
        return {
            'input_tensor': self._cached_edge_input_tensor,
            'output_logits': self._cached_edge_output_logits,
        }

    def restore_edge_cache(self, snapshot: dict) -> None:
        """
        Restaure le cache edge.

        Args:
            snapshot: Cache depuis snapshot_edge_cache()
        """
        self._cached_edge_input_tensor = snapshot['input_tensor']
        self._cached_edge_output_logits = snapshot['output_logits']

    def eval(self) -> None:
        """
        Mode eval : désactive dropout ET Transformer.

        IMPORTANT : Appeler model.eval() en plus de self.training = False
        pour désactiver correctement le dropout du Transformer.
        """
        self.training = False
        self.model.eval()
        self._node_head.eval()
        self._edge_head.eval()

    def train(self, mode: bool = True) -> None:
        """
        Mode train : active/désactive dropout ET Transformer.

        Args:
            mode: True pour train, False pour eval (compatible nn.Module)
        """
        self.training = mode
        if mode:
            self.model.train()
            self._node_head.train()
            self._edge_head.train()
        else:
            self.model.eval()
            self._node_head.eval()
            self._edge_head.eval()

    def __setattr__(self, name: str, value) -> None:
        """
        Synchronise self.training avec model.training pour éviter
        dropout actif à l'inférence quand encoder.training=False direct.
        """
        super().__setattr__(name, value)
        if name == 'training' and hasattr(self, 'model') and hasattr(self.model, 'train'):
            if value:
                self.model.train()
                if hasattr(self, '_node_head'):
                    self._node_head.train()
                if hasattr(self, '_edge_head'):
                    self._edge_head.train()
            else:
                self.model.eval()
                if hasattr(self, '_node_head'):
                    self._node_head.eval()
                if hasattr(self, '_edge_head'):
                    self._edge_head.eval()
