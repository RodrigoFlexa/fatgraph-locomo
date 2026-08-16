"""Runtime settings loaded from ``.env`` and the process environment.

Precedence, highest first:

1. real environment variables (CI, ``export``, ``docker run -e``);
2. the ``.env`` file at the project root;
3. the defaults below.

Secrets never enter the YAML configs and never enter the results manifest --
:meth:`Settings.redacted` is what gets serialised.

The loader is deliberately dependency-free: ``python-dotenv`` would work but it
is one more thing to install for ~40 lines of parsing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from fgl.paths import project_root

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

#: Values shipped in ``.env.example``. A run must not start with these still in
#: place -- otherwise the first Azure call fails with an opaque DNS/auth error
#: instead of "you forgot to edit .env".
PLACEHOLDERS = frozenset(
    {
        "https://your-resource.openai.azure.com/",
        "https://your-resource.openai.azure.com",
        "your-resource",
        "changeme",
        "xxx",
        "<your-key>",
        "sk-...",
    }
)


def is_placeholder(value: str | None) -> bool:
    if not value:
        return False
    v = value.strip()
    return v in PLACEHOLDERS or "your-resource" in v or v.lower() in {"changeme", "todo"}


# --------------------------------------------------------------------------- #
# .env parsing                                                                 #
# --------------------------------------------------------------------------- #


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Supports ``export``, ``#`` comments, quotes."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #")[0].rstrip()
        out[key] = value
    return out


def load_dotenv(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``. Real env vars win unless ``override``."""
    p = Path(path) if path else project_root() / ".env"
    if not p.exists():
        return {}
    values = parse_dotenv(p.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


# --------------------------------------------------------------------------- #
# Settings                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    """Everything that comes from the environment rather than from a YAML."""

    # --- Azure OpenAI credentials (secret) --------------------------------
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = field(default=None, repr=False)
    azure_api_version: Optional[str] = None

    # --- model selection, overrides the YAML when set ---------------------
    llm_deployment: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_azure_deployment: Optional[str] = None

    # --- corporate gateway support ----------------------------------------
    #: private CA bundle for TLS (``FGL_CA_BUNDLE``), resolved to an absolute path
    ca_bundle: Optional[str] = None
    #: endpoint already contains the routing path -> pass it as ``base_url``
    use_base_url: bool = False

    dotenv_path: Optional[Path] = None
    dotenv_found: bool = False
    #: variables left at their ``.env.example`` value
    placeholders: tuple[str, ...] = ()
    #: ``config.ini`` the credentials came from, when not from the environment
    ini_path: Optional[str] = None

    # ------------------------------------------------------------------ io --
    @classmethod
    def load(cls, dotenv_path: str | Path | None = None, override: bool = False) -> "Settings":
        p = Path(dotenv_path) if dotenv_path else project_root() / ".env"
        found = bool(load_dotenv(p, override=override))
        env = os.environ.get

        # A corporate gateway often ships credentials in an .ini instead of the
        # environment. FGL_AZURE_CONFIG_INI points at it; the environment still
        # wins for anything it defines.
        ini_path, ini = _load_ini(env("FGL_AZURE_CONFIG_INI"), env("FGL_AZURE_CONFIG_SECTION"))

        raw = {
            "azure_endpoint": env("AZURE_OPENAI_ENDPOINT") or ini.get("endpoint"),
            "azure_api_key": env("AZURE_OPENAI_API_KEY") or ini.get("api_key"),
            "azure_api_version": env("AZURE_OPENAI_API_VERSION") or ini.get("api_version"),
            "llm_deployment": env("FGL_LLM_DEPLOYMENT") or None,
            "embedding_provider": env("FGL_EMBEDDING_PROVIDER") or None,
            "embedding_model": env("FGL_EMBEDDING_MODEL") or None,
            "embedding_azure_deployment": env("FGL_EMBEDDING_AZURE_DEPLOYMENT") or None,
        }
        env_names = {
            "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
            "azure_api_key": "AZURE_OPENAI_API_KEY",
            "azure_api_version": "AZURE_OPENAI_API_VERSION",
        }
        # a placeholder is worse than nothing: treat it as unset, but say so
        stale = tuple(
            env_names[k] for k, v in raw.items() if k in env_names and is_placeholder(v)
        )
        for k in list(raw):
            if is_placeholder(raw[k]):
                raw[k] = None

        from fgl.llm.azure import resolve_ca_bundle

        ca_raw = env("FGL_CA_BUNDLE") or ini.get("ca_bundle")
        ca = resolve_ca_bundle(ca_raw)
        if ca_raw and not ca:
            raise RuntimeError(
                f"FGL_CA_BUNDLE aponta para {ca_raw!r}, que não foi encontrado "
                f"(procurei em {project_root()}, no diretório atual e ao lado do módulo). "
                "Use um caminho absoluto ou coloque o arquivo na raiz do projeto."
            )

        endpoint = raw["azure_endpoint"] or ""
        explicit = env("FGL_AZURE_USE_BASE_URL")
        use_base_url = (
            explicit.strip().lower() in ("1", "true", "yes", "on")
            if explicit
            else _looks_like_base_url(endpoint)
        )

        return cls(
            **raw,
            ca_bundle=ca,
            use_base_url=use_base_url,
            dotenv_path=p,
            dotenv_found=found,
            placeholders=stale,
            ini_path=ini_path,
        )

    # ------------------------------------------------------------ validation -
    @property
    def azure_ready(self) -> bool:
        return bool(self.azure_endpoint and self.azure_api_key and self.azure_api_version)

    def missing_azure(self) -> list[str]:
        return [
            name
            for name, value in (
                ("AZURE_OPENAI_ENDPOINT", self.azure_endpoint),
                ("AZURE_OPENAI_API_KEY", self.azure_api_key),
                ("AZURE_OPENAI_API_VERSION", self.azure_api_version),
            )
            if not value
        ]

    def explain_missing(self) -> str:
        """Human-readable reason the credentials are not usable yet."""
        parts = []
        stale = set(self.placeholders)
        blank = [n for n in self.missing_azure() if n not in stale]
        if blank:
            parts.append("não definidas: " + ", ".join(blank))
        if stale:
            parts.append(
                "ainda com o valor de exemplo do .env.example: " + ", ".join(sorted(stale))
            )
        return "; ".join(parts) or "credenciais incompletas"

    def require_azure(self) -> None:
        if not self.azure_ready:
            raise RuntimeError(
                f"Azure não configurado — {self.explain_missing()}.\n"
                f"Edite {self.dotenv_path} (comece de .env.example) ou exporte as variáveis."
            )

    # --------------------------------------------------------------- apply ---
    def apply_to(self, cfg) -> None:
        """Overlay the env-provided model choices onto a :class:`Config`.

        Only non-empty values override the YAML, so a config file stays the
        source of truth for anything the environment does not mention.
        """
        if self.llm_deployment:
            cfg.llm.deployment = self.llm_deployment
        if self.embedding_provider:
            cfg.embeddings.provider = self.embedding_provider
        if self.embedding_model:
            cfg.embeddings.model = self.embedding_model
        if self.embedding_azure_deployment:
            cfg.embeddings.azure_deployment = self.embedding_azure_deployment

    # -------------------------------------------------------------- report ---
    def redacted(self) -> dict:
        """Safe to write into ``metrics.json``: keys become fingerprints."""
        out: dict[str, object] = {}
        for f in fields(self):
            if f.name in ("dotenv_path", "dotenv_found", "placeholders", "ini_path"):
                continue
            value = getattr(self, f.name)
            if value is None:
                out[f.name] = None
            elif "key" in f.name or "secret" in f.name:
                out[f.name] = f"<set:{_fingerprint(str(value))}>"
            elif "endpoint" in f.name:
                out[f.name] = _mask_host(str(value))
            else:
                out[f.name] = value
        out["dotenv_found"] = self.dotenv_found
        if self.ini_path:
            out["config_ini"] = self.ini_path
        if self.placeholders:
            out["placeholders"] = list(self.placeholders)
        return out


def load_settings(dotenv_path: str | Path | None = None, override: bool = False) -> Settings:
    return Settings.load(dotenv_path, override=override)


#: Keys we look for inside an ``.ini`` section, in order of preference.
INI_KEYS = {
    "api_key": ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "API_KEY"),
    "api_version": ("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION", "API_VERSION"),
    "endpoint": (
        "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT", "OPENAI_BASE_URL",
        "BASE_URL", "ENDPOINT",
    ),
    "ca_bundle": ("CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"),
}


def _load_ini(path: str | None, section: str | None) -> tuple[Optional[str], dict[str, str]]:
    """Read Azure credentials from an ``.ini`` (ConfigParser) file.

    Supports the shape used by corporate gateways::

        [OPENAI]
        OPENAI_API_KEY = ...
        OPENAI_API_VERSION = ...
        AZURE_OPENAI_BASE_URL = ...

    Section defaults to ``OPENAI`` (then ``AZURE``, then ``DEFAULT``).
    ``ExtendedInterpolation`` is enabled, matching the usual conventions.
    """
    if not path:
        return None, {}
    from configparser import ConfigParser, ExtendedInterpolation

    p = Path(path).expanduser()
    if not p.is_absolute():
        for base in (project_root(), Path.cwd()):
            if (base / p).exists():
                p = (base / p).resolve()
                break
    if not p.exists():
        raise RuntimeError(
            f"FGL_AZURE_CONFIG_INI aponta para {path!r}, que não existe "
            f"(resolvido como {p}). Use um caminho absoluto."
        )

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(p, encoding="UTF-8")

    candidates = [section] if section else ["OPENAI", "AZURE", "azure", "openai"]
    chosen = next((s for s in candidates if s and parser.has_section(s)), None)
    values = dict(parser[chosen]) if chosen else dict(parser.defaults())
    upper = {k.upper(): v for k, v in values.items()}

    out: dict[str, str] = {}
    for target, names in INI_KEYS.items():
        for name in names:
            if upper.get(name):
                out[target] = upper[name]
                break
    return str(p), out


def _looks_like_base_url(endpoint: str) -> bool:
    """A URL carrying a path is a gateway base_url, not a bare Azure resource."""
    m = re.match(r"^https?://[^/]+(/.*)?$", endpoint or "")
    if not m:
        return False
    path = (m.group(1) or "").strip("/")
    return bool(path)


def _fingerprint(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:8]


def _mask_host(url: str) -> str:
    m = re.match(r"^(https?://)([^/]+)(.*)$", url)
    if not m:
        return "<set>"
    host = m.group(2)
    head = host.split(".")[0]
    keep = head[:3] + "***" if len(head) > 3 else "***"
    rest = ".".join(host.split(".")[1:])
    return f"{m.group(1)}{keep}" + (f".{rest}" if rest else "") + m.group(3)
