"""Request / response models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

Pixel = Annotated[float, Field(ge=0, le=255, allow_inf_nan=False)]
SAFE_TEXT = r"^[A-Za-z0-9 _.:@-]+$"  # labels shown in the UI: no markup characters
PIXELS_FIELD = Field(
    ...,
    min_length=784,
    max_length=784,
    description="784 pixel values (28x28 image, row-major), each a finite number in [0, 255], "
    "0 = background (MNIST format)",
)


class CompressRequest(BaseModel):
    pixels: list[Pixel] = PIXELS_FIELD
    n_components: int = Field(154, ge=1, le=784, description="principal components to keep")


class CompressResponse(BaseModel):
    n_components: int
    variance_kept: float  # share of the training variance the first d components explain
    reconstruction: list[int]  # 784 pixels, clipped to 0-255
    mse: float
    rmse_grey_levels: float
    bytes_uint8_pixels: int
    bytes_float32_code: int
    pct_of_features: float


class PredictRequest(BaseModel):
    pixels: list[Pixel] = PIXELS_FIELD
    source: str = Field(
        "api",
        min_length=1,
        max_length=32,
        pattern=SAFE_TEXT,
        description="where the input came from (canvas, sample-noise, ...)",
    )


class ClassifierOutput(BaseModel):
    digit: int
    scores: list[float]  # one-vs-one votes per digit (SVM), max 9
    margin: float  # votes of the winner minus the runner-up
    n_features: int
    latency_ms: float


class UnusualOutput(BaseModel):
    is_unusual: bool
    reconstruction_error: float
    threshold: float
    error_percentile: float
    message: str


class PredictResponse(BaseModel):
    prediction_id: int | None
    raw: ClassifierOutput
    pca: ClassifierOutput
    unusual: UnusualOutput
    map_positions: dict[str, list[float]]
    model_versions: dict[str, str]
    latency_ms: float
