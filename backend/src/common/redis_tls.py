"""Helpers for configuring Redis TLS/SSL options from environment variables."""

from __future__ import annotations

import os
import ssl
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


_TRUTHY = {"1", "true", "yes", "on"}
_SSL_KWARGS = {
    "ssl_cert_reqs",
    "ssl_ca_certs",
    "ssl_certfile",
    "ssl_keyfile",
    "ssl_check_hostname",
}

_CERT_REQS_INT_TO_FLAG = {
    ssl.CERT_NONE: "none",
    ssl.CERT_REQUIRED: "required",
}
if hasattr(ssl, "CERT_OPTIONAL"):
    _CERT_REQS_INT_TO_FLAG[getattr(ssl, "CERT_OPTIONAL")] = "optional"


def redis_connection_config(
    url: str,
) -> tuple[str, dict[str, object] | None]:
    """Return a sanitized Redis URL plus TLS options for redis-py/kombu."""

    sanitized_url, url_options = _ssl_options_from_url(url)
    env_options = _ssl_options_from_env()

    options: dict[str, object] = {**url_options, **env_options}
    use_tls_env = os.getenv("REDIS_USE_TLS")

    if use_tls_env is not None:
        if use_tls_env.lower() not in _TRUTHY:
            return sanitized_url, None
        options = _ensure_cert_reqs(options)
        return sanitized_url, options

    if options or _url_implies_tls(sanitized_url):
        options = _ensure_cert_reqs(options)
        return sanitized_url, options

    return sanitized_url, None


def _ssl_options_from_env() -> dict[str, object]:
    options: dict[str, object] = {}

    cert_reqs_env = os.getenv("REDIS_SSL_CERT_REQS")
    if cert_reqs_env:
        options["ssl_cert_reqs"] = _coerce_cert_reqs(cert_reqs_env)

    for key, env_name in (
        ("ssl_ca_certs", "REDIS_SSL_CA_CERTS"),
        ("ssl_certfile", "REDIS_SSL_CERTFILE"),
        ("ssl_keyfile", "REDIS_SSL_KEYFILE"),
        ("ssl_check_hostname", "REDIS_SSL_CHECK_HOSTNAME"),
    ):
        value = os.getenv(env_name)
        if value:
            options[key] = value

    return options


def _ssl_options_from_url(url: str) -> tuple[str, dict[str, object]]:
    parsed = urlparse(url)
    if not parsed.query:
        return url, {}

    options: dict[str, object] = {}
    retained_params: list[tuple[str, str]] = []

    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in _SSL_KWARGS:
            canonical = lowered
            if canonical == "ssl_cert_reqs":
                options[canonical] = _coerce_cert_reqs(value)
            elif canonical == "ssl_check_hostname":
                options[canonical] = _to_bool(value)
            else:
                options[canonical] = value
        else:
            retained_params.append((key, value))

    sanitized_query = urlencode(retained_params, doseq=True)
    sanitized_url = urlunparse(parsed._replace(query=sanitized_query))
    return sanitized_url, options


def _ensure_cert_reqs(options: dict[str, object] | None) -> dict[str, object]:
    options = dict(options or {})
    if "ssl_cert_reqs" not in options:
        options["ssl_cert_reqs"] = "required"
    return options


def _url_implies_tls(url: str) -> bool:
    return urlparse(url).scheme == "rediss"


def _coerce_cert_reqs(value: object) -> str:
    if value is None:
        return "required"

    text = str(value).strip()
    if not text:
        return "required"

    upper = text.upper()
    if upper in {"NONE", "DISABLED"}:
        return "none"
    elif upper in {"OPTIONAL"}:
        return "optional"
    elif upper in {"REQUIRED", "CERT_REQUIRED"}:
        return "required"

    if upper.startswith("CERT_"):
        upper = upper[len("CERT_") :]

    normalized = upper.lower()
    if normalized in {"none", "disabled"}:
        return "none"
    if normalized in {"optional"}:
        return "optional"
    if normalized in {"required"}:
        return "required"

    # Allow passing numeric VerifyMode constants directly
    if isinstance(value, int):
        mapped = _CERT_REQS_INT_TO_FLAG.get(value)
        if mapped:
            return mapped

    try:
        enum_value = getattr(ssl, f"CERT_{upper}")
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported REDIS SSL certificate requirement: {value}"
        ) from exc
    mapped = _CERT_REQS_INT_TO_FLAG.get(enum_value)
    if not mapped:
        raise ValueError(
            f"Unsupported REDIS SSL certificate requirement: {value}"
        )
    return mapped


def _to_bool(value: str) -> bool:
    truthy: Iterable[str] = _TRUTHY
    return str(value).strip().lower() in truthy
