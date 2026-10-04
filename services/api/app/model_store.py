"""Holds the five champion models and swaps them atomically on /reload."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    compressor: object
    classifier_raw: object
    classifier_pca: object
    detector: object
    atlas: object
    versions: dict[str, str]
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    _maps_payload: dict | None = field(default=None, repr=False)

    def maps_payload(self) -> dict:
        if self._maps_payload is None:  # built once per bundle: ~2,000 points and thumbnails
            self._maps_payload = self.atlas.to_payload()
        return self._maps_payload


class ModelStore:
    def __init__(self, alias: str = "champion"):
        self.alias = alias
        self._bundle: ModelBundle | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle

    def set_bundle(self, bundle: ModelBundle) -> None:
        with self._lock:
            self._bundle = bundle
            self.last_error = None

    def load_from_registry(self) -> ModelBundle:
        """Load ``models:/<name>@<alias>`` for all five registered models."""
        from dimred_mlops import tracking

        models, versions = {}, {}
        for key, name in tracking.MODEL_NAMES.items():
            mv = tracking.get_alias_version(name, self.alias)
            if mv is None:
                raise LookupError(f"No '{self.alias}' version of {name} in the registry yet")
            # resolve the alias once and load that exact version, so the reported versions are
            # the ones actually served even if the alias moves meanwhile
            versions[key] = str(mv.version)
            models[key] = tracking.load_model(name, versions[key])
        bundle = ModelBundle(**models, versions=versions)
        self.set_bundle(bundle)
        log.info("Loaded champion models %s", versions)
        return bundle

    def registry_versions(self) -> dict[str, str] | None:
        from dimred_mlops import tracking

        out = {}
        for key, name in tracking.MODEL_NAMES.items():
            mv = tracking.get_alias_version(name, self.alias)
            if mv is None:
                return None
            out[key] = str(mv.version)
        return out

    def refresh_if_stale(self) -> bool:
        """Reload when the registry's alias points at different versions than the ones served.

        Lets several API replicas pick up a new champion without each receiving /reload."""
        try:
            current = self.registry_versions()
        except Exception as exc:
            log.warning("Registry poll failed: %s", exc)
            return False
        if not current or (self._bundle is not None and current == self._bundle.versions):
            return False
        if self._bundle is not None:
            changed = [k for k in current if current[k] != self._bundle.versions.get(k)]
            if len(changed) < len(current):
                # every training run registers all five models, so a partial change means a
                # promotion is in progress: wait for the next poll instead of mixing versions
                log.info("Promotion in progress (%s changed) - waiting", changed)
                return False
        log.info("Champion changed in the registry (%s) - reloading", current)
        return self.try_load()

    def try_load(self) -> bool:
        try:
            self.load_from_registry()
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("Models not loaded: %s", self.last_error)
            return False
