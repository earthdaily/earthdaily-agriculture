"""
EarthDaily Data Studio (EDS) identity — the second identity provider.

Used by the vegetation time-series extractor that authenticates against EDS;
every other
extractor still authenticates through :class:`~earthdaily.agriculture.core.identity.EDAuthenticator`
against the Geosys identity server. Listed in ``release/public_allowlist.yml``
→ ``private_excluded`` alongside ``lrts_functions.py``, so it does not ship
while its only consumer is private.

──────────────────────────────────────────────────────────────────────────────
Why a separate module rather than a branch inside EDAuthenticator
──────────────────────────────────────────────────────────────────────────────
The two providers share nothing but the shape of the value they return:

=================  ==========================  ==============================
                   EDAuthenticator (Geosys)    EDSAuthenticator (this module)
=================  ==========================  ==============================
grant type         ``password``                ``client_credentials``
endpoint           ``identity.geosys-na.com``  ``…/api_tokens/exchange``
                   ``/v2.1/connect/token``
credentials        client id + secret (Basic)  one long-lived API token
                   + username + password
lifetime           1 hour                      24 hours
environments       ``prod`` / ``preprod``      ``PROD`` / ``BLEEDING``
=================  ==========================  ==============================

``EDAuthenticator`` also owns S3 client bootstrap, so folding a second provider
into it would deepen an existing conflation of identity and storage.

──────────────────────────────────────────────────────────────────────────────
The exchange
──────────────────────────────────────────────────────────────────────────────
The API token from console.earthdaily.com is long-lived but **cannot call the
API directly**. It is exchanged for a 24-hour access token::

    POST {base}/account_management/v1/authentication/api_tokens/exchange
    grant_type=client_credentials
    client_id=EARTHDAILY_API_TOKEN          ← a literal, not your client id
    client_secret=<EDS_API_TOKEN>

    -> {"access_token": "...", ...}

``client_id`` really is the fixed string ``EARTHDAILY_API_TOKEN``; the API token
travels in ``client_secret``. Easy to misread as a placeholder.

⚠️  **Tokens are environment-specific**, and one host serves both the exchange
and the data API for a given environment (:data:`EDS_BASE_URLS`). A PROD token
is refused by BLEEDING's exchange with a 403. To work against BLEEDING you need
an API token issued for BLEEDING.

⚠️  **Data access is a separate grant from authentication.** As of 2026-08-06
this account exchanges successfully on prod but gets 403 from prod ``/lrts`` —
the token is fine, the subscription does not include the endpoint.

Usage::

    from earthdaily.agriculture.core.identity_eds import EDSAuthenticator

    token, expires_at = EDSAuthenticator.get_new_token("prod")
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import requests

#: EDS host per repo environment. **The same host serves the token exchange and
#: the data API** — this is the vendor's documented model ("comment the PROD
#: line, uncomment the BLEEDING one, and use a BLEEDING API token"), and tokens
#: are issued per environment.
#:
#: Measured 2026-08-06 with a PROD token:
#:
#: ========================================  ============================
#: exchange at api.earthdaily.com            200, 24h access token
#: prod data with that token                 403 (no /lrts subscription)
#: exchange at bleeding-api.test-internal    403 (needs a BLEEDING token)
#: ========================================  ============================
#:
#: ⚠️  BLEEDING's *data* plane was also observed accepting a prod-issued access
#: token. That is NOT the documented contract and is not relied on here —
#: BLEEDING is internal test infrastructure, and a permissive data plane there is
#: far more likely a gap than a guarantee. Use a token issued for the
#: environment you are calling.
EDS_BASE_URLS = {
    "prod": "https://api.earthdaily.com",
    "preprod": "https://bleeding-api.test-internal.earthdaily.com",
}

#: Path of the exchange endpoint, appended to the base above.
TOKEN_EXCHANGE_PATH = "/account_management/v1/authentication/api_tokens/exchange"

#: The literal OAuth2 ``client_id`` the exchange expects. Not a placeholder.
API_TOKEN_CLIENT_ID = "EARTHDAILY_API_TOKEN"

#: Documented access-token lifetime, used when the response omits ``expires_in``.
DEFAULT_TOKEN_LIFETIME_SECONDS = 24 * 3600

#: Refresh this long before actual expiry, so a long bulk run cannot have a
#: token die mid-flight between the check and the request.
EXPIRY_SAFETY_MARGIN_SECONDS = 300


class EDSAuthenticator:
    """OAuth2 ``client_credentials`` exchange against EarthDaily Data Studio.

    Stateless by design — :class:`BaseExtractor` already owns the token and its
    expiry, so this only performs the exchange. That mirrors how
    ``EDAuthenticator.get_new_token`` is used and keeps both providers
    substitutable behind ``BaseExtractor.get_new_token()``.
    """

    REQUEST_TIMEOUT = 30

    @staticmethod
    def get_api_token(env: str, api_token: Optional[str] = None) -> Optional[str]:
        """Resolve the long-lived API token.

        Precedence: explicit argument > ``{ENV}_EDS_API_TOKEN`` > ``EDS_API_TOKEN``.

        There is deliberately no cross-environment fallback. Tokens are issued
        per environment, so silently reaching for a prod token when preprod was
        asked for would produce a 403 on the exchange with nothing to explain it.
        """
        if api_token:
            return api_token
        prefix = env.upper().replace("-", "")
        return os.getenv(f"{prefix}_EDS_API_TOKEN") or os.getenv("EDS_API_TOKEN")

    @staticmethod
    def check_token(
        expiration_date: Optional[datetime],
        access_token: Optional[str],
        env: str = "prod",
        api_token: Optional[str] = None,
    ) -> Tuple[str, datetime]:
        """Reuse the access token while it is valid, otherwise exchange a new one.

        The EDS counterpart of :meth:`~earthdaily.agriculture.core.identity.EDAuthenticator.check_token`,
        with the same contract so both providers refresh identically behind
        ``@requires_token`` -> ``ensure_token_valid()``.

        Args:
            expiration_date: Expiry of the token in hand, or None if there is none.
            access_token: The token in hand, or None.
            env: Repo environment, selecting host and API token.
            api_token: Explicit API token, bypassing the environment lookup.

        Returns:
            tuple: ``(access_token, expires_at)`` — the existing pair when still
            valid, otherwise a freshly exchanged one.
        """
        # `expires_at` already has EXPIRY_SAFETY_MARGIN_SECONDS subtracted at
        # exchange time, so no second margin is applied here — doing so would
        # discard a usable token twice over.
        if access_token and expiration_date and datetime.now() < expiration_date:
            return access_token, expiration_date

        return EDSAuthenticator.get_new_token(env, api_token=api_token)

    @staticmethod
    def get_new_token(env: str = "prod", api_token: Optional[str] = None) -> Tuple[str, datetime]:
        """Exchange the API token for a 24-hour access token.

        Args:
            env: Repo environment (``prod`` / ``preprod``), selecting both the
                host to exchange against and which API token to look up.
            api_token: Overrides the environment lookup. Mostly for tests.

        Returns:
            tuple: ``(access_token, expires_at)`` — the same shape
            ``EDAuthenticator.get_new_token`` returns, so ``BaseExtractor`` needs
            no special-casing. ``expires_at`` already has the safety margin
            subtracted.

        Raises:
            ValueError: No API token configured, or an unknown environment.
            requests.HTTPError: The exchange was refused. A 401 here means the
                token is revoked, mistyped, or from the wrong environment.
        """
        if env not in EDS_BASE_URLS:
            raise ValueError(f"❌ Unknown environment '{env}' for EDS — expected one of {sorted(EDS_BASE_URLS)}")

        token = EDSAuthenticator.get_api_token(env, api_token)
        if not token:
            prefix = env.upper().replace("-", "")
            raise ValueError(
                f"❌ No EarthDaily API token found for env='{env}'. Generate one at "
                f"console.earthdaily.com (account → API tokens; shown only once) and set "
                f"{prefix}_EDS_API_TOKEN or EDS_API_TOKEN in src/.env. Tokens are "
                f"environment-specific — a prod token will not work against preprod."
            )

        url = EDS_BASE_URLS[env] + TOKEN_EXCHANGE_PATH
        response = requests.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": API_TOKEN_CLIENT_ID,
                "client_secret": token,
            },
            timeout=EDSAuthenticator.REQUEST_TIMEOUT,
        )
        if response.status_code == 401:
            raise requests.HTTPError(
                f"❌ EDS rejected the API token (401) at {url}. Either it is revoked or "
                f"mistyped, or it belongs to a different environment — tokens are issued per "
                f"environment and this call used env='{env}' ({EDS_BASE_URLS[env]}). "
                f"Re-issue it at console.earthdaily.com if in doubt.",
                response=response,
            )
        response.raise_for_status()

        payload = response.json() or {}
        access_token = payload.get("access_token")
        if not access_token:
            raise ValueError(f"❌ EDS exchange returned no access_token. Body: {str(payload)[:200]}")

        lifetime = int(payload.get("expires_in") or DEFAULT_TOKEN_LIFETIME_SECONDS)
        expires_at = datetime.now() + timedelta(seconds=max(lifetime - EXPIRY_SAFETY_MARGIN_SECONDS, 60))

        print(f"🔑 EDS access token acquired for env={env}, valid until {expires_at}")
        return access_token, expires_at
