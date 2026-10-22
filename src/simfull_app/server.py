"""FastAPI web server for simfull simulation software."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .models import (
    SimulateRequest,
    STLImportRequest,
    ThermalBridgeRequest,
    Wall1DRequest,
)
from .service import SimulationService, default_two_room_building


class SimulationApp:
    """Builds the FastAPI application for simfull."""

    STATIC = Path(__file__).resolve().parent / "static"

    def __init__(self, cors_origins: list[str] | None = None) -> None:
        self.service = SimulationService()
        self.app = FastAPI(
            title="simfull simulation software",
            version=__version__,
            description="Full building, thermal network, spatial fields, and CAD geometry simulation engine.",
        )
        origins = cors_origins or ["*"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        self._routes()

    def _routes(self) -> None:
        app, svc = self.app, self.service

        @app.get("/api/health")
        def health() -> dict[str, Any]:
            return svc.health()

        @app.get("/api/weather/presets")
        def weather_presets() -> list[dict[str, Any]]:
            return svc.weather_presets()

        @app.get("/api/materials")
        def materials() -> dict[str, Any]:
            return svc.materials()

        @app.get("/api/constructions")
        def constructions() -> dict[str, Any]:
            return svc.constructions()

        @app.get("/api/geometry/default")
        def default_geometry() -> dict[str, Any]:
            return default_two_room_building().to_dict()

        @app.post("/api/simulate/network")
        async def simulate_network(req: SimulateRequest) -> dict[str, Any]:
            try:
                return await run_in_threadpool(svc.simulate_network, req)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

        @app.post("/api/simulate/thermal-bridge")
        async def simulate_thermal_bridge(req: ThermalBridgeRequest) -> dict[str, Any]:
            try:
                return await run_in_threadpool(svc.simulate_thermal_bridge, req)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

        @app.post("/api/simulate/wall-1d")
        async def simulate_wall_1d(req: Wall1DRequest) -> dict[str, Any]:
            try:
                return await run_in_threadpool(svc.simulate_wall_1d, req)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

        @app.post("/api/geometry/from-stl")
        async def geometry_from_stl(req: STLImportRequest) -> dict[str, Any]:
            try:
                return await run_in_threadpool(svc.parse_stl, req)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e)) from e

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(self.STATIC / "index.html")

        app.mount("/static", StaticFiles(directory=self.STATIC), name="static")


def create_app() -> FastAPI:
    return SimulationApp().app


def main() -> None:
    import uvicorn
    uvicorn.run(
        create_app(),
        host=os.environ.get("SIMFULL_APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIMFULL_APP_PORT") or os.environ.get("PORT") or "8080"),
    )
