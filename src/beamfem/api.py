"""Small dependency-free WSGI REST API for validation and optimization."""

from __future__ import annotations

import argparse
from http import HTTPStatus
import json
import secrets
from typing import Any, Callable, Mapping
from uuid import uuid4
from wsgiref.simple_server import make_server

from .io import build_discrete_problem, to_serializable, validate_problem_spec
from .optimize.backends import ExactBackend, GreedyBackend, SolverLimits


API_VERSION = "v1"


def optimize_document(
    document: Mapping[str, Any], *, backend: str = "greedy",
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, construct, solve, and FEM-re-evaluate one portable problem."""

    settings = dict(settings or {})
    spec = validate_problem_spec(document)
    problem = build_discrete_problem(spec).problem
    limits = SolverLimits(
        max_evaluations=settings.get("max_evaluations"),
        max_iterations=settings.get("max_iterations"),
        time_limit=settings.get("time_limit"),
        memory_limit_mb=settings.get("memory_limit_mb"),
    )
    if backend == "greedy":
        solver = GreedyBackend(
            penalty=float(settings.get("penalty", 1.0e6)),
            pairwise=bool(settings.get("pairwise", True)),
        )
    elif backend == "exact":
        solver = ExactBackend(max_combinations=int(settings.get("max_combinations", 200_000)))
    else:
        raise ValueError("REST backend must be 'greedy' or 'exact'")
    return {"api_version": API_VERSION, "optimization": to_serializable(
        solver.solve(problem, limits=limits)
    )}


def create_wsgi_app(*, bearer_token: str | None = None, max_body_bytes: int = 5_000_000) -> Callable:
    """Create a production-server-compatible WSGI application.

    The bundled server is suitable for local/private deployment. Public
    deployment should place this WSGI callable behind an HTTPS reverse proxy
    with request limits, authentication, and process supervision.
    """

    if max_body_bytes < 1:
        raise ValueError("max_body_bytes must be positive")

    def app(environ, start_response):
        request_id = environ.get("HTTP_X_REQUEST_ID") or str(uuid4())

        def respond(status: HTTPStatus, payload: Mapping[str, Any]):
            body = json.dumps({**payload, "request_id": request_id}, ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
            start_response(f"{status.value} {status.phrase}", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("X-Request-ID", request_id),
                ("Cache-Control", "no-store"),
            ])
            return [body]

        try:
            if bearer_token is not None:
                supplied = environ.get("HTTP_AUTHORIZATION", "")
                expected = f"Bearer {bearer_token}"
                if not secrets.compare_digest(supplied, expected):
                    return respond(HTTPStatus.UNAUTHORIZED, {
                        "error": {"code": "unauthorized", "message": "valid bearer token required"}
                    })
            method, path = environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", "")
            if method == "GET" and path == "/health":
                return respond(HTTPStatus.OK, {"status": "ok", "api_version": API_VERSION})
            if method != "POST" or path not in {"/v1/validate", "/v1/optimize"}:
                return respond(HTTPStatus.NOT_FOUND, {
                    "error": {"code": "not_found", "message": "unknown endpoint"}
                })
            raw_length = environ.get("CONTENT_LENGTH", "0") or "0"
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 1 or length > max_body_bytes:
                return respond(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {
                    "error": {"code": "invalid_body_size", "message": "request body size is outside limits"}
                })
            payload = json.loads(environ["wsgi.input"].read(length).decode("utf-8"))
            if not isinstance(payload, Mapping) or not isinstance(payload.get("problem"), Mapping):
                raise ValueError("body.problem must be an object")
            if path == "/v1/validate":
                spec = validate_problem_spec(payload["problem"])
                return respond(HTTPStatus.OK, {"valid": True, "schema_version": spec.schema_version})
            backend = payload.get("backend", "greedy")
            settings = payload.get("settings", {})
            if not isinstance(backend, str) or not isinstance(settings, Mapping):
                raise ValueError("backend must be a string and settings must be an object")
            return respond(HTTPStatus.OK, optimize_document(
                payload["problem"], backend=backend, settings=settings
            ))
        except Exception as exc:
            return respond(HTTPStatus.BAD_REQUEST, {"error": {
                "code": type(exc).__name__, "message": str(exc),
            }})
    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve the beamfem REST API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bearer-token", help="optional static bearer token")
    parser.add_argument("--max-body-bytes", type=int, default=5_000_000)
    args = parser.parse_args(argv)
    with make_server(args.host, args.port, create_wsgi_app(
        bearer_token=args.bearer_token, max_body_bytes=args.max_body_bytes
    )) as server:
        print(f"beamfem API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
