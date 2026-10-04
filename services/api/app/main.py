"""FastAPI model service + web UI.

* ``POST /compress``          - keep d principal components of a digit and rebuild it
* ``POST /predict``           - SVM on raw pixels vs SVM on PCA features, the unusual-input flag
                                and the digit's position on the 2D maps; logged to PostgreSQL
* ``GET  /maps``              - precomputed t-SNE / PCA / Kernel PCA / LLE / Isomap / MDS maps
* ``GET  /compress/info``     - explained-variance curve of the deployed PCA
* ``GET  /samples``           - a held-out test digit, optionally corrupted
* ``POST /reload``            - hot-swap to the current ``champion`` models in the MLflow registry
* ``GET  /monitoring/summary``- live unusual-input rate, model agreement and prediction mix
* ``GET  /``                  - single-page web UI

Run with ``uvicorn app.main:create_app --factory``.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func, select

from dimred_mlops.compression import reconstruct
from dimred_mlops.data import CORRUPTIONS, corrupt_digits
from dimred_mlops.db import Prediction, init_db, make_engine

from .model_store import ModelBundle, ModelStore
from .schemas import (
    ClassifierOutput,
    CompressRequest,
    CompressResponse,
    PredictRequest,
    PredictResponse,
    UnusualOutput,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"
EXPECTED_UNUSUAL_PCT = float(os.getenv("EXPECTED_UNUSUAL_PCT", "4.0"))


def _classify(model, x: np.ndarray) -> ClassifierOutput:
    t0 = time.perf_counter()
    scores = np.asarray(model.decision_function(x))[0]
    latency = (time.perf_counter() - t0) * 1000
    classes = np.asarray(model.classes_, dtype=int)
    full = np.full(10, float(scores.min()))
    full[classes] = scores
    order = np.argsort(full)[::-1]
    pca = getattr(model, "named_steps", {}).get("pca")
    return ClassifierOutput(
        digit=int(order[0]),
        scores=[round(float(s), 4) for s in full],
        margin=round(float(full[order[0]] - full[order[1]]), 4),
        n_features=int(pca.n_components_) if pca is not None else x.shape[1],
        latency_ms=round(latency, 3),
    )


def _unusual(detector, x: np.ndarray) -> UnusualOutput:
    err = float(detector.reconstruction_error(x)[0])
    thr = float(detector.threshold_)
    pct = float(detector.error_percentile(x)[0])
    flag = err > thr
    message = (
        "The PCA cannot rebuild this input well: it does not look like the training digits, "
        "so treat both predictions with caution."
        if flag
        else "Rebuilt well from the principal components: it looks like a training digit."
    )
    return UnusualOutput(
        is_unusual=flag,
        reconstruction_error=round(err, 2),
        threshold=round(thr, 2),
        error_percentile=round(pct, 1),
        message=message,
    )


def create_app(
    store: ModelStore | None = None, database_url: str | None = None, load_models: bool = True
) -> FastAPI:
    store = store or ModelStore(alias=os.getenv("MODEL_ALIAS", "champion"))
    engine = make_engine(database_url or os.getenv("DATABASE_URL", "sqlite:///app.db"))
    Session = init_db(engine)
    state = {"last_poll": time.time()}
    poll_seconds = float(os.getenv("REGISTRY_POLL_SECONDS", "60"))
    admin_token = os.getenv("ADMIN_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_models and store.bundle is None:
            store.try_load()
        yield
        engine.dispose()

    app = FastAPI(
        title="Dimensionality Reduction MLOps - MNIST service",
        version="1.0.0",
        description="PCA compression, SVM on raw vs PCA features, a reconstruction-error "
        "unusual-input flag and 2D maps (t-SNE, PCA, Kernel PCA, LLE) of handwritten digits.",
        lifespan=lifespan,
    )
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        # Drop the echoed input: it can contain NaN/inf, which is not valid JSON.
        detail = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def require_models() -> ModelBundle:
        # at most once per poll interval: pick up a new @champion (or the first one)
        if load_models and poll_seconds > 0 and time.time() - state["last_poll"] > poll_seconds:
            state["last_poll"] = time.time()
            store.refresh_if_stale()
        bundle = store.bundle
        if bundle is None:
            raise HTTPException(
                503, f"No models deployed yet ({store.last_error}). Run the pipeline first."
            )
        return bundle

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        b = store.bundle
        return {
            "status": "ok" if b else "no-models",
            "model_versions": b.versions if b else {},
            "loaded_at": b.loaded_at if b else None,
            "alias": store.alias,
            "detail": None if b else store.last_error,
        }

    @app.get("/ready")
    def ready():
        """Readiness probe: 200 only once champion models are loaded (``/health`` is liveness)."""
        if store.bundle is None:
            raise HTTPException(503, f"No models loaded: {store.last_error}")
        return {"status": "ready", "model_versions": store.bundle.versions}

    @app.post("/reload")
    def reload(x_admin_token: str | None = Header(default=None)):
        """Hot-swap to the registry's current champion. If ADMIN_TOKEN is set, the request must
        carry it in the ``X-Admin-Token`` header (the deploy flow sends it)."""
        given = (x_admin_token or "").encode("utf-8")
        if admin_token and not hmac.compare_digest(given, admin_token.encode("utf-8")):
            raise HTTPException(401, "Missing or invalid X-Admin-Token")
        try:
            b = store.load_from_registry()
        except Exception as exc:
            raise HTTPException(503, f"Reload failed: {exc}") from exc
        return {"status": "reloaded", "model_versions": b.versions, "loaded_at": b.loaded_at}

    @app.get("/compress/info")
    def compress_info():
        """Cumulative explained variance of the deployed PCA, for the UI's slider."""
        b = require_models()
        cum = np.cumsum(b.compressor.explained_variance_ratio_)
        marks = {f"{v:.0%}": int(np.argmax(cum >= v) + 1) for v in (0.5, 0.8, 0.9, 0.95, 0.99)}
        return {"cumulative_variance": [round(float(c), 5) for c in cum], "marks": marks}

    @app.post("/compress", response_model=CompressResponse)
    def compress(req: CompressRequest):
        b = require_models()
        x = np.asarray(req.pixels, dtype=np.float64).reshape(1, -1)
        d = min(req.n_components, b.compressor.n_components_)
        rec = reconstruct(b.compressor, x, d)[0]
        mse = float(np.mean((x[0] - rec) ** 2))
        return CompressResponse(
            n_components=d,
            variance_kept=round(float(b.compressor.explained_variance_ratio_[:d].sum()), 5),
            reconstruction=[int(v) for v in np.clip(np.rint(rec), 0, 255)],
            mse=round(mse, 2),
            rmse_grey_levels=round(float(np.sqrt(mse)), 2),
            bytes_uint8_pixels=784,
            bytes_float32_code=4 * d,
            pct_of_features=round(100 * d / 784, 2),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        b = require_models()
        t0 = time.perf_counter()
        x = np.asarray(req.pixels, dtype=np.float32).reshape(1, -1)
        raw = _classify(b.classifier_raw, x)
        pca = _classify(b.classifier_pca, x)
        unusual = _unusual(b.detector, x)
        positions = {k: v[0] for k, v in b.atlas.project(x).items()}
        latency = (time.perf_counter() - t0) * 1000
        pred_id = None
        try:
            with Session() as s:
                row = Prediction(
                    source=req.source,
                    pixels=[int(round(p)) for p in req.pixels],
                    digit_raw=raw.digit,
                    digit_pca=pca.digit,
                    latency_raw_ms=raw.latency_ms,
                    latency_pca_ms=pca.latency_ms,
                    reconstruction_error=unusual.reconstruction_error,
                    error_threshold=unusual.threshold,
                    is_unusual=unusual.is_unusual,
                    model_versions=b.versions,
                    latency_ms=latency,
                )
                s.add(row)
                s.commit()
                pred_id = row.id
        except Exception as exc:  # serving must not fail because logging failed
            log.error("Could not log prediction: %s", exc)
        return PredictResponse(
            prediction_id=pred_id,
            raw=raw,
            pca=pca,
            unusual=unusual,
            map_positions=positions,
            model_versions=b.versions,
            latency_ms=round(latency, 2),
        )

    @app.get("/maps")
    def maps():
        """2D maps of training digits: coordinates in [0, 1], labels, 28x28 thumbnails (base64)."""
        return require_models().maps_payload()

    @app.get("/samples")
    def sample(
        kind: str = Query("clean", description=f"clean or one of {', '.join(CORRUPTIONS)}"),
        seed: int | None = Query(None, ge=0, description="fix the choice for a reproducible demo"),
    ):
        """A random held-out test digit, optionally corrupted - handy for trying the API."""
        if kind != "clean" and kind not in CORRUPTIONS:
            raise HTTPException(422, f"kind must be 'clean' or one of {CORRUPTIONS}")
        atlas = require_models().atlas
        if atlas.samples is None or len(atlas.samples) == 0:
            raise HTTPException(404, "The deployed atlas has no sample digits.")
        rng = np.random.default_rng(seed)
        i = int(rng.integers(len(atlas.samples)))
        x = atlas.samples[i : i + 1].astype(np.float32)
        if kind != "clean":
            x = corrupt_digits(x, kind, int(rng.integers(1_000_000)))
        return {
            "kind": kind,
            "pixels": [int(round(float(v))) for v in x[0]],
            "true_label": int(atlas.sample_labels[i]),
        }

    @app.get("/monitoring/summary")
    def monitoring_summary(hours: float = Query(168, gt=0, le=24 * 365)):
        """Aggregates are computed in the database; only the 12 most recent rows are fetched."""
        since = datetime.now(UTC) - timedelta(hours=hours)
        window = Prediction.created_at >= since
        unusual = case((Prediction.is_unusual, 1), else_=0)
        agree = case((Prediction.digit_raw == Prediction.digit_pca, 1), else_=0)
        with Session() as s:
            n, n_unusual, n_agree, lat_raw, lat_pca = s.execute(
                select(
                    func.count(),
                    func.sum(unusual),
                    func.sum(agree),
                    func.avg(Prediction.latency_raw_ms),
                    func.avg(Prediction.latency_pca_ms),
                ).where(window)
            ).one()
            class_rows = s.execute(
                select(Prediction.digit_pca, func.count())
                .where(window)
                .group_by(Prediction.digit_pca)
            ).all()
            source_rows = s.execute(
                select(Prediction.source, func.count(), func.sum(unusual))
                .where(window)
                .group_by(Prediction.source)
                .order_by(func.count().desc())
            ).all()
            recent = (
                s.execute(select(Prediction).where(window).order_by(Prediction.id.desc()).limit(12))
                .scalars()
                .all()
            )
        counts = {str(d): 0 for d in range(10)}
        counts.update({str(d): int(c) for d, c in class_rows})
        return {
            "window_hours": hours,
            "n_predictions": int(n),
            "unusual_rate_pct": round(100 * float(n_unusual) / n, 2) if n else None,
            # the detector's own calibration target (4% in the shipped config)
            "expected_unusual_rate_pct": round(100 * float(store.bundle.detector.flag_rate), 2)
            if store.bundle is not None
            else EXPECTED_UNUSUAL_PCT,
            "models_agree_pct": round(100 * float(n_agree) / n, 1) if n else None,
            "mean_latency_raw_ms": round(float(lat_raw), 3) if n else None,
            "mean_latency_pca_ms": round(float(lat_pca), 3) if n else None,
            "class_counts": counts,
            "by_source": {src: {"n": int(c), "unusual": int(a or 0)} for src, c, a in source_rows},
            "recent": [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(timespec="seconds")
                    if r.created_at
                    else None,
                    "source": r.source,
                    "digit_raw": r.digit_raw,
                    "digit_pca": r.digit_pca,
                    "is_unusual": r.is_unusual,
                    "reconstruction_error": round(r.reconstruction_error, 1),
                    "pixels": r.pixels,
                }
                for r in recent
            ],
        }

    return app
